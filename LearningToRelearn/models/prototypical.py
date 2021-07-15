import logging
import math
import time
import pprint
from contextlib import nullcontext

import higher
import torch
from torch import nn, optim
import wandb
import yaml
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.experimental import initialize, compose

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as opt
from transformers import AdamW

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import ReplayMemory, TRANSFORMER_HDIM, TransformerClsModel, TransformerNeuromodulator, TransformerRLN
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, datasets_dict, n_samples_order
from LearningToRelearn.datasets.text_classification_dataset import get_datasets
from LearningToRelearn.models.base_models import TransformerRLN, TransformerClsModel, MemoryStore, ClassMemoryStore, LSTMDecoder


class PrototypicalNetwork(Learner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.inner_lr = config.learner.inner_lr
        self.meta_lr = config.learner.meta_lr
        self.mini_batch_size = config.training.batch_size

        self.pn = TransformerClsModel(model_name=config.learner.model_name,
                                      n_classes=config.data.n_classes,
                                      max_length=config.data.max_length,
                                      device=self.device)
        if config.wandb:
            wandb.watch(self.pn, log='all')

        self.memory = ClassMemoryStore(key_dim=TRANSFORMER_HDIM, device=self.device,
                                       class_discount=config.learner.class_discount, n_classes=config.data.n_classes,
                                       discount_method=config.learner.class_discount_method)
        self.loss_fn = nn.CrossEntropyLoss()

        meta_params = [p for p in self.pn.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pn.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)
        #TODO: remove below line
        self.episode_samples_seen = 0 # have to keep track of per-task samples seen as we might use replay as well

    def training(self, datasets, **kwargs):
        replay_freq, replay_steps = self.replay_parameters()
        self.logger.info("Replay frequency: {}".format(replay_freq))
        self.logger.info("Replay steps: {}".format(replay_steps))

        datas, order, n_samples, eval_train_dataset, eval_eval_dataset, eval_dataset = self.prepare_data(datasets)
        for i, (data, dataset_name, n_sample) in enumerate(zip(datas, order, n_samples)):
            self.logger.info(f"Observing dataset {dataset_name} for {n_sample} samples. "
                             f"Evaluation={dataset_name=='evaluation'}")
            if dataset_name == "evaluation" and self.config.testing.few_shot:
                self.few_shot_testing(train_dataset=eval_train_dataset, eval_dataset=eval_eval_dataset, split="test",
                                      increment_counters=False)
            else:
                train_dataloader = iter(DataLoader(data, batch_size=self.mini_batch_size, shuffle=False))
                self.episode_samples_seen = 0 # have to keep track of per-task samples seen as we might use replay as well
                # iterate over episodes
                while True:
                    self.set_train()
                    support_set, task = self.get_support_set(train_dataloader, n_sample)
                    # TODO: return flag that indicates whether the query set is from the memory. Don't include this in the online accuracy calc
                    query_set = self.get_query_set(train_dataloader, replay_freq, replay_steps, n_sample, write_memory=False)
                    if support_set is None or query_set is None:
                        break
                    self.training_step(support_set, query_set, task=task)

                    self.meta_training_log()
                    self.write_metrics()
                    self.current_iter += 1
                    if self.episode_samples_seen >= n_sample:
                        break
            if i == 0:
                self.metrics["eval_task_first_encounter_evaluation"] = \
                    self.evaluate(DataLoader(eval_dataset, batch_size=self.mini_batch_size))["accuracy"]
            if dataset_name == self.config.testing.eval_dataset:
                self.eval_task_first_encounter = False
                
    def training_step(self, support_set, query_set=None, task=None):
        self.inner_optimizer.zero_grad()

        self.logger.debug("-------------------- TRAINING STEP  -------------------")
        # with higher.innerloop_ctx(self.pn, self.inner_optimizer,
        #                           copy_initial_weights=False,
        #                           track_higher_grads=False) as (fpn, diffopt):
        do_memory_update = self.config.learner.prototype_update_freq > 0 and \
                        (self.current_iter % self.config.learner.prototype_update_freq) == 0
        ### GET SUPPORT SET REPRESENTATIONS ###
        self.logger.debug("----------------- SUPPORT SET ----------------- ")
        representations, all_labels = self.get_representations(support_set[:1])
        representations_merged = torch.cat(representations)
        class_means, unique_labels = self.get_class_means(representations_merged, all_labels)
        self._examples_seen += len(representations_merged)
        self.logger.debug(f"Examples seen increased by {len(representations_merged)}")

        ### UPDATE MEMORY ###
        if do_memory_update:
            updated_memory_representations = self.memory.update(class_means, unique_labels, logger=self.logger)
        ### DETERMINE WHAT'S SEEN AS PROTOTYPE ###
        if self.config.learner.prototypes == "class_means":
            prototypes = expand_class_representations(self.memory.class_representations, class_means, unique_labels)
        elif self.config.learner.prototypes == "memory":
            prototypes = updated_memory_representations
        else:
            raise AssertionError("Prototype type not in {'class_means', 'memory'}, fix config file.")

        ### INITIALIZE LINEAR LAYER WITH PROTOYPICAL-EQUIVALENT WEIGHTS ###
        # self.init_prototypical_classifier(prototypes, linear_module=fpn.linear)
        weight = 2 * prototypes # divide by number of dimensions, otherwise blows up
        bias = - (prototypes ** 2).sum(dim=1)

        self.logger.debug("----------------- QUERY SET  ----------------- ")
        ### EVALUATE ON QUERY SET AND UPDATE ENCODER ###
        # Outer loop
        if query_set is not None:
            for text, labels in query_set:
                labels = torch.tensor(labels).to(self.device)
                query_representations = self.forward(text, labels)["representation"]

                # distance query representations to prototypes (BATCH X N_PROTOTYPES)
                # distances = euclidean_dist(query_representations, prototypes)
                # logits = - distances
                logits = query_representations @ weight.T + bias
                loss = self.loss_fn(logits, labels)
                # log_probability = F.log_softmax(-distances, dim=1)
                # loss is negation of the log probability, index using the labels for each observation
                # loss = (- log_probability[torch.arange(len(log_probability)), labels]).mean()
                self.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_optimizer.step()

                predictions = model_utils.make_prediction(logits.detach())
                # predictions = torch.tensor([inv_label_map[p.item()] for p in predictions])
                # to_print = pprint.pformat(list(map(lambda x: (x[0].item(), x[1].item(),
                #                         [round(z, 3) for z in x[2].tolist()]),
                #                         list(zip(labels, predictions, distances)))))
                self.logger.debug(
                    f"Unique Labels: {unique_labels.tolist()}\n"
                    # f"Labels, Indices, Predictions, Distances:\n{to_print}\n"
                    f"Loss:\n{loss.item()}\n"
                    f"Predictions:\n{predictions}\n"
                )
                self.update_query_tracker(loss, predictions, labels)
                metrics = model_utils.calculate_metrics(predictions.tolist(), labels.tolist())
                online_metrics = {
                    "accuracy": metrics["accuracy"],
                    "examples_seen": self.examples_seen(),
                    "task": task if task is not None else "none"
                }
                self.metrics["online"].append(online_metrics)
                if task is not None and task == self.config.testing.eval_dataset and \
                    self.eval_task_first_encounter:
                    self.metrics["eval_task_first_encounter"].append(online_metrics)
                self._examples_seen += len(text)
                self.logger.debug(f"Examples seen increased by {len(text)}")

            # Meta optimizer step
            # self.meta_optimizer.step()
            # self.meta_optimizer.zero_grad()
        self.logger.debug("-------------------- TRAINING STEP END  -------------------")

    def get_representations(self, support_set, prediction_network=None):
        """
        Parameters
        ---
        support_set: List[Tuple[batch text, batch labels]]
        prediction network: pytorch module 

        Returns
        ---
        Tuple[List[Tensor], List[Int]] where the first result is the hidden representation and the second
        the labels.
        """
        representations = []
        all_labels = []
        for text, labels in support_set:
            labels = torch.tensor(labels).to(self.device)
            all_labels.extend(labels.tolist())
            # labels = labels.to(self.device)
            output = self.forward(text, labels, prediction_network=prediction_network)
            representations.append(output["representation"])
        return representations, all_labels

    def forward(self, text, labels, prediction_network=None, no_grad=False):
        if prediction_network is None:
            prediction_network = self.pn
        input_dict = self.pn.encode_text(text)
        context_manager = torch.no_grad() if no_grad else nullcontext()
        with context_manager:
            representation = prediction_network(input_dict, out_from="transformers")
            logits = prediction_network(representation, out_from="linear")
        return {"representation": representation, "logits": logits}
    
    def update_memory(self, class_means, unique_labels):
        to_update = unique_labels
        # selection of old class representations here
        old_class_representations = self.memory.class_representations[to_update]
        # if old class representations haven't been given values yet, don't bias towards 0 by exponential update
        if (old_class_representations == 0).bool().all():
            new_class_representations = class_means
        else:
            # memory update rule here
            new_class_representations = (1 - self.config.learner.class_discount) * old_class_representations + self.config.learner.class_discount * class_means
        self.logger.debug(f"Updating class representations for classes {unique_labels}.\n"
                         f"Distance old class representations and class means: {[round(z, 2) for z in (old_class_representations - class_means).norm(dim=1).tolist()]}\n"
                         f"Distance old and new class representations: {[round(z, 2) for z in (new_class_representations - old_class_representations).norm(dim=1).tolist()]}"
                         )
        # for returning new class representations while keeping gradients intact
        result = torch.clone(self.memory.class_representations)
        result[to_update] = new_class_representations
        # update memory
        self.memory.class_representations[to_update] = new_class_representations.detach()

        return result

    def init_prototypical_classifier(self, prototypes, linear_module=None):
        if linear_module is None:
            linear_module = self.pn.linear
        weight = 2 * prototypes / TRANSFORMER_HDIM # divide by number of dimensions, otherwise blows up
        bias = - (prototypes ** 2).sum(dim=1) / TRANSFORMER_HDIM
        # otherwise the bias of the classes observed in the support set is always smaller than 
        # not observed ones, which favors the unobserved ones. However, it is expected that labels
        # in the support set are more likely to be in the query set.
        bias_unchanged = bias == 0
        bias[bias_unchanged] = bias.min()
        self.logger.info(f"Prototype is zero vector for classes {bias_unchanged.nonzero(as_tuple=True)[0].tolist()}. "
                         f"Setting their bias entries to the minimum of the uninitialized bias vector.")
        # prototypical-equivalent network initialization
        linear_module.weight.data = weight
        linear_module.bias.data = bias
        self.logger.info(f"Classifier bias initialized to {bias}.")
        
        # a = mmaml.classifier.weight
        # # https://stackoverflow.com/questions/61279403/gradient-flow-through-torch-nn-parameter
        # # a = torch.nn.Parameter(torch.ones((10,)), requires_grad=True) 
        # b = a[:] # silly hack to convert in a raw tensor including the computation graph
        # # b.retain_grad() # Otherwise backward pass will not store the gradient since it is not a leaf 
        # it is necessary to do it this way to retain the gradient information on the classifier parameters
        # https://discuss.pytorch.org/t/non-leaf-variables-as-a-modules-parameters/65775
        # del self.classifier.weight
        # self.classifier.weight = 2 * prototypes
        # del self.classifier.bias
        # self.classifier.bias = bias
        # weight_copy = self.classifier.weight[:]
        # bias_copy = self.classifier.bias[:]

    def update_meta_gradients(self, loss, fpn):
        # PN meta gradients
        pn_params = [p for p in fpn.parameters() if p.requires_grad]
        meta_pn_grads = torch.autograd.grad(loss, pn_params, allow_unused=True)
        pn_params = [p for p in self.pn.parameters() if p.requires_grad]
        for param, meta_grad in zip(pn_params, meta_pn_grads):
            if meta_grad is not None:
                if param.grad is not None:
                    param.grad += meta_grad.detach()
                else:
                    param.grad = meta_grad.detach()

    def update_support_tracker(self, loss, pred, labels):
        self.tracker["support_loss"].append(loss.item())
        self.tracker["support_predictions"].extend(pred.tolist())
        self.tracker["support_labels"].extend(labels.tolist())

    def update_query_tracker(self, loss, pred, labels):
        self.tracker["query_loss"].append(loss.item())
        self.tracker["query_predictions"].extend(pred.tolist())
        self.tracker["query_labels"].extend(labels.tolist())

    def reset_tracker(self):
        self.tracker = {
            "support_loss": [],
            "support_predictions": [],
            "support_labels": [],
            "query_loss": [],
            "query_predictions": [],
            "query_labels": []
        }

    def evaluate(self, dataloader, prediction_network=None):
        # if self.config.learner.evaluation_support_set:
        #     support_set = []
        #     for _ in range(self.config.learner.updates):
        #         text, labels = self.memory.read_batch(batch_size=self.mini_batch_size)
        #         support_set.append((text, labels))

        # with higher.innerloop_ctx(self.pn, self.inner_optimizer,
        #                         copy_initial_weights=False,
        #                         track_higher_grads=False) as (fpn, diffopt):
        #     if self.config.learner.evaluation_support_set:
        #         self.set_train()
        #         support_prediction_network = fpn
        #         # Inner loop
        #         task_predictions, task_labels = [], []
        #         support_loss = []
        #         for text, labels in support_set:
        #             labels = torch.tensor(labels).to(self.device)
        #             # labels = labels.to(self.device)
        #             output = self.forward(text, labels, fpn)
        #             loss = self.loss_fn(output["logits"], labels)
        #             diffopt.step(loss)

        #             pred = model_utils.make_prediction(output["logits"].detach())
        #             support_loss.append(loss.item())
        #             task_predictions.extend(pred.tolist())
        #             task_labels.extend(labels.tolist())
        #         results = model_utils.calculate_metrics(task_predictions, task_labels)
        #         self.logger.info("Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
        #                     "F1 score = {:.4f}".format(np.mean(support_loss), results["accuracy"],
        #                     results["precision"], results["recall"], results["f1"]))
        #         self.set_eval()
        #     else:
        #         support_prediction_network = self.pn
        #     if prediction_network is None:
        #         prediction_network = support_prediction_network

        self.set_eval()
        prototypes = self.memory.class_representations
        weight = 2 * prototypes
        bias = - (prototypes ** 2).sum(dim=1)
        all_losses, all_predictions, all_labels = [], [], []
        for i, (text, labels, _) in enumerate(dataloader):
            labels = torch.tensor(labels).to(self.device)
            representations = self.forward(text, labels)["representation"]
            logits = representations @ weight.T + bias
            # labels = labels.to(self.device)
            loss = self.loss_fn(logits, labels)
            loss = loss.item()
            pred = model_utils.make_prediction(logits.detach())
            all_losses.append(loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())

        results = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.debug("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), results["accuracy"],
                    results["precision"], results["recall"], results["f1"]))
        return results

    def model_state(self):
        return {"pn": self.pn.state_dict()}

    def optimizer_state(self):
        return self.meta_optimizer.state_dict()

    def load_model_state(self, checkpoint):
        self.pn.load_state_dict(checkpoint["model_state"]["pn"])

    def load_optimizer_state(self, checkpoint):
        self.meta_optimizer.load_state_dict(checkpoint["optimizer"])

    def save_other_state_information(self, state):
        """Any learner specific state information is added here"""
        state["memory"] = self.memory
        return state

    def load_other_state_information(self, checkpoint):
        self.memory = checkpoint["memory"]

    def set_eval(self):
        self.pn.eval()

    def set_train(self):
        self.pn.train()

    def few_shot_testing(self, train_dataset, eval_dataset, increment_counters=False, split="test"):
        """
        Allow the model to train on a small amount of datapoints at a time. After every training step,
        evaluate on many samples that haven't been seen yet.

        Results are saved in learner's `metrics` attribute.

        Parameters
        ---
        train_dataset: Dataset
            Contains examples on which the model is trained before being evaluated
        eval_dataset: Dataset
            Contains examples on which the model is evaluated
        increment_counters: bool
            If True, update online metrics and current iteration counters.
        """
        self.logger.info(f"few shot testing on dataset {self.config.testing.eval_dataset} "
                         f"with {len(train_dataset)} samples")
        train_dataloader, eval_dataloader = self.few_shot_preparation(train_dataset, eval_dataset, split=split)
        all_predictions, all_labels = [], []
        def add_none(iterator):
            yield None
            for x in iterator:
                yield x
        shifted_dataloader = add_none(train_dataloader)
        prototypes = self.memory.class_representations
        for i, (support_set, (query_text, query_labels, datasets)) in enumerate(zip(shifted_dataloader, train_dataloader)):
            query_labels = torch.tensor(query_labels).to(self.device)
            # happens on the first one
            if support_set is None:
                prototypes = self.memory.class_representations
            else:
                support_text, support_labels, _ = support_set
                support_labels = torch.tensor(support_labels).to(self.device)
                support_representations = self.forward(support_text, support_labels)["representation"]
                support_class_means, unique_labels = self.get_class_means(support_representations, support_labels)
                updated_memory_representations = self.memory.update(support_class_means, unique_labels, logger=self.logger)
                prototypes = updated_memory_representations
                if self.config.learner.few_shot_detach_prototypes:
                    prototypes = prototypes.detach()
            weight = 2 * prototypes
            bias = - (prototypes ** 2).sum(dim=1)
            query_representations = self.forward(query_text, query_labels)["representation"]
            logits = query_representations @ weight.T + bias
            loss = self.loss_fn(logits, query_labels)

            self.meta_optimizer.zero_grad()
            loss.backward()
            self.meta_optimizer.step()

            predictions = model_utils.make_prediction(logits.detach())
            all_predictions.extend(predictions.tolist())
            all_labels.extend(query_labels.tolist())
            dataset_results = self.evaluate(dataloader=eval_dataloader)
            self.log_few_shot(all_predictions, all_labels, datasets, dataset_results,
                                increment_counters, query_text, i, split=split)
            if (i * self.config.testing.few_shot_batch_size) % self.mini_batch_size == 0 and i > 0:
                all_predictions, all_labels = [], []
        self.few_shot_end()

    def get_class_means(self, embeddings, labels):
        """Return class means and unique labels given neighbors.
        
        Parameters
        ---
        embeddings: Tensor, shape (batch_size, embed_size)
        labels: iterable of labels for each embedding
            
        Returns
        ---
        Tuple[List[Tensor], List[Tensor]]:
            class means and unique labels
        """
        class_means = []
        unique_labels = torch.tensor(labels).unique()
        for label_ in unique_labels:
            label_ixs = (label_ == torch.tensor(labels)).nonzero(as_tuple=False).flatten()
            same_class_embeddings = embeddings[label_ixs]
            class_means.append(same_class_embeddings.mean(axis=0))
        return torch.stack(class_means), unique_labels


def expand_class_representations(class_representations, class_means, unique_labels):
    expanded = torch.zeros_like(class_representations)
    expanded[unique_labels] = class_means
    return expanded

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
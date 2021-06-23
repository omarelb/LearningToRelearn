import logging
import math
import time
from contextlib import nullcontext

import higher
import torch
from torch import nn, optim
import wandb

import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import LinearPLN, ReplayMemory, TRANSFORMER_HDIM, TransformerClsModel, TransformerNeuromodulator, TransformerRLN
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, datasets_dict, n_samples_order


class OML(Learner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.inner_lr = config.learner.inner_lr
        self.meta_lr = config.learner.meta_lr
        self.mini_batch_size = config.training.batch_size

        self.rln = TransformerRLN(model_name=config.learner.model_name,
                                  max_length=config.data.max_length,
                                  device=self.device)
        self.pln = LinearPLN(in_dim=TRANSFORMER_HDIM, out_dim=config.data.n_classes, device=self.device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.loss_fn = nn.CrossEntropyLoss()

        self.logger.info("Loaded {} as RLN".format(self.rln.__class__.__name__))
        self.logger.info("Loaded {} as PLN".format(self.pln.__class__.__name__))

        meta_params = [p for p in self.rln.parameters() if p.requires_grad] + \
                      [p for p in self.pln.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pln.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def training(self, datasets, **kwargs):
        replay_freq, replay_steps = self.replay_parameters()
        self.logger.info("Replay frequency: {}".format(replay_freq))
        self.logger.info("Replay steps: {}".format(replay_steps))

        datas, order, n_samples, eval_train_dataset, eval_eval_dataset = self.prepare_data(datasets)
        for data, dataset_name, n_sample in zip(datas, order, n_samples):
            self.logger.info(f"Observing dataset {dataset_name} for {n_sample} samples. "
                             f"Evaluation={dataset_name=='evaluation'}")
            if dataset_name == "evaluation":
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
                    query_set = self.get_query_set(train_dataloader, replay_freq, replay_steps, n_sample)
                    if support_set is None or query_set is None:
                        break

                    self.training_step(support_set, query_set, task=task)

                    self.meta_training_log()
                    self.write_metrics()
                    self.current_iter += 1
                    if self.episode_samples_seen >= n_sample:
                        break
            if dataset_name == self.config.testing.eval_dataset:
                self.eval_task_first_encounter = False

    def training_step(self, support_set, query_set=None, task=None):
        self.inner_optimizer.zero_grad()
        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                copy_initial_weights=False,
                                track_higher_grads=False) as (fpln, diffopt):
            # Inner loop
            for text, labels in support_set:
                labels = torch.tensor(labels).to(self.device)
                # labels = labels.to(self.device)
                output = self.forward(text, labels, fpln)
                loss = self.loss_fn(output["logits"], labels)
                diffopt.step(loss)
                self.memory.write_batch(text, labels)

                predictions = model_utils.make_prediction(output["logits"].detach())
                self.update_support_tracker(loss, predictions, labels)
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

            # Outer loop
            if query_set is not None:
                for text, labels in query_set:
                    labels = torch.tensor(labels).to(self.device)
                    # labels = labels.to(self.device)
                    output = self.forward(text, labels, fpln)
                    loss = self.loss_fn(output["logits"], labels)
                    self.update_meta_gradients(loss, fpln)

                    predictions = model_utils.make_prediction(output["logits"].detach())
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

                # Meta optimizer step
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

    def forward(self, text, labels, prediction_network=None, no_grad=False):
        if prediction_network is None:
            prediction_network = self.pln
        input_dict = self.rln.encode_text(text)
        if prediction_network is None:
            prediction_network = self.pln
        context_manager = torch.no_grad() if no_grad else nullcontext()
        with context_manager:
            representation = self.rln(input_dict)
            logits = prediction_network(representation)
        return {"logits": logits}

    def update_meta_gradients(self, loss, fpln):
        # rln meta gradients
        rln_params = [p for p in self.rln.parameters() if p.requires_grad]
        meta_rln_grads = torch.autograd.grad(loss, rln_params, retain_graph=True, allow_unused=True)
        for param, meta_grad in zip(rln_params, meta_rln_grads):
            if meta_grad is not None:
                if param.grad is not None:
                    param.grad += meta_grad.detach()
                else:
                    param.grad = meta_grad.detach()
        # PLN meta gradients
        pln_params = [p for p in fpln.parameters() if p.requires_grad]
        meta_pln_grads = torch.autograd.grad(loss, pln_params, allow_unused=True)
        pln_params = [p for p in self.pln.parameters() if p.requires_grad]
        for param, meta_grad in zip(pln_params, meta_pln_grads):
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
        if self.config.learner.evaluation_support_set:
            support_set = []
            for _ in range(self.config.learner.updates):
                text, labels = self.memory.read_batch(batch_size=self.mini_batch_size)
                support_set.append((text, labels))

        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                copy_initial_weights=False,
                                track_higher_grads=False) as (fpln, diffopt):
            if self.config.learner.evaluation_support_set:
                self.set_train()
                support_prediction_network = fpln
                # Inner loop
                task_predictions, task_labels = [], []
                support_loss = []
                for text, labels in support_set:
                    labels = torch.tensor(labels).to(self.device)
                    # labels = labels.to(self.device)
                    output = self.forward(text, labels, fpln)
                    loss = self.loss_fn(output["logits"], labels)
                    diffopt.step(loss)

                    pred = model_utils.make_prediction(output["logits"].detach())
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(labels.tolist())
                results = model_utils.calculate_metrics(task_predictions, task_labels)
                self.logger.info("Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                            "F1 score = {:.4f}".format(np.mean(support_loss), results["accuracy"],
                            results["precision"], results["recall"], results["f1"]))
                self.set_eval()
            else:
                support_prediction_network = self.pln
            if prediction_network is None:
                prediction_network = support_prediction_network

            self.set_eval()
            all_losses, all_predictions, all_labels = [], [], []
            for i, (text, labels, datasets) in enumerate(dataloader):
                labels = torch.tensor(labels).to(self.device)
                # labels = labels.to(self.device)
                output = self.forward(text, labels, prediction_network, no_grad=True)
                loss = self.loss_fn(output["logits"], labels)
                loss = loss.item()
                pred = model_utils.make_prediction(output["logits"].detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                # if i % 20 == 0:
                #     self.logger.info(f"Batch {i + 1}/{len(dataloader)} processed")

        results = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.debug("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), results["accuracy"],
                    results["precision"], results["recall"], results["f1"]))
        return results

    def model_state(self):
        return {"rln": self.rln.state_dict(),
                "pln": self.pln.state_dict()}

    def optimizer_state(self):
        return self.meta_optimizer.state_dict()

    def load_model_state(self, checkpoint):
        self.rln.load_state_dict(checkpoint["model_state"]["rln"])
        self.pln.load_state_dict(checkpoint["model_state"]["pln"])

    def load_optimizer_state(self, checkpoint):
        self.meta_optimizer.load_state_dict(checkpoint["optimizer"])

    def save_other_state_information(self, state):
        """Any learner specific state information is added here"""
        state["memory"] = self.memory
        return state

    def load_other_state_information(self, checkpoint):
        self.memory = checkpoint["memory"]

    def set_eval(self):
        self.pln.eval()

    def set_train(self):
        self.pln.train()

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
        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                copy_initial_weights=False,
                                track_higher_grads=False) as (fpln, diffopt):
            self.pln.train()
            self.rln.eval()
            # Inner loop
            for i, (text, labels, datasets) in enumerate(train_dataloader):
                labels = torch.tensor(labels).to(self.device)
                output = self.forward(text, labels, fpln)
                loss = self.loss_fn(output["logits"], labels)
                diffopt.step(loss)

                predictions = model_utils.make_prediction(output["logits"].detach())
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())
                dataset_results = self.evaluate(dataloader=eval_dataloader, prediction_network=fpln)
                self.log_few_shot(all_predictions, all_labels, datasets, dataset_results,
                                  increment_counters, text, i, split=split)
                if (i * self.config.testing.few_shot_batch_size) % self.mini_batch_size == 0 and i > 0:
                    all_predictions, all_labels = [], []
        self.few_shot_end()


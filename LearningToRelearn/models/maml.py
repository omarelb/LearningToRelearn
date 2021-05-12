import logging
import math
import time

import higher
import torch
from torch import nn, optim
import wandb

import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import ReplayMemory, TransformerClsModel, TransformerNeuromodulator
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, datasets_dict, n_samples_order


class MAML(Learner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.inner_lr = config.learner.inner_lr
        self.meta_lr = config.learner.meta_lr
        self.write_prob = config.learner.write_prob
        self.replay_rate = config.learner.replay_rate
        self.replay_every = config.learner.replay_every
        self.mini_batch_size = config.training.batch_size

        self.pn = TransformerClsModel(model_name=config.learner.model_name,
                                      n_classes=config.data.n_classes,
                                      max_length=config.data.max_length,
                                      device=self.device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.loss_fn = nn.CrossEntropyLoss()

        self.logger.info("Loaded {} as PN".format(self.pn.__class__.__name__))

        meta_params = [p for p in self.pn.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pn.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def training(self, datasets, **kwargs):
        train_datasets = datasets_dict(datasets["train"], datasets["order"])
        val_datasets = datasets_dict(datasets["val"], datasets["order"])
        eval_dataset = val_datasets[self.config.testing.eval_dataset]
        eval_dataset = eval_dataset.sample(min(self.config.testing.few_shot_validation_size, len(eval_dataset)))

        replay_freq, replay_steps = self.replay_parameters()
        self.logger.info("Replay frequency: {}".format(replay_freq))
        self.logger.info("Replay steps: {}".format(replay_steps))

        n_samples, order = n_samples_order(self.config.learner.samples_per_task, self.config.task_order, datasets["order"])
        datas = get_continuum(train_datasets, order=order, n_samples=n_samples,
                             eval_dataset=self.config.testing.eval_dataset, merge=False)
        for data, dataset_name, n_sample in zip(datas, order, n_samples):
            self.logger.info(f"Observing dataset {dataset_name} for {n_sample} samples. "
                             f"Evaluation={dataset_name=='evaluation'}")
            if dataset_name == "evaluation":
                self.few_shot_testing(train_dataset=data, eval_dataset=eval_dataset, increment_counters=True)
            else:
                train_dataloader = iter(DataLoader(data, batch_size=self.mini_batch_size, shuffle=False))
                # iterate over episodes
                while True:
                    self.set_train()
                    support_set, task = self.get_support_set(train_dataloader)
                    # TODO: return flag that indicates whether the query set is from the memory. Don't include this in the online accuracy calc
                    query_set = self.get_query_set(train_dataloader, replay_freq, replay_steps)
                    if support_set is None or query_set is None:
                        break

                    self.training_step(support_set, query_set, task=task)

                    self.log()
                    self.write_metrics()
                    self.current_iter += 1

    def training_step(self, support_set, query_set=None, task=None):
        self.inner_optimizer.zero_grad()
        with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                copy_initial_weights=False,
                                track_higher_grads=False) as (fpn, diffopt):
            # Inner loop
            for text, labels in support_set:
                labels = torch.tensor(labels).to(self.device)
                # labels = labels.to(self.device)
                output = self.forward(text, labels, fpn)
                loss = self.loss_fn(output["logits"], labels)
                diffopt.step(loss)
                self.memory.write_batch(text, labels)

                predictions = model_utils.make_prediction(output["logits"].detach())
                self.update_support_tracker(loss, predictions, labels)
                online_metrics = model_utils.calculate_metrics(predictions.tolist(), labels.tolist())
                self.metrics["online"].append({
                    "accuracy": online_metrics["accuracy"],
                    "examples_seen": self.examples_seen(),
                    "task": task if task is not None else "none"
                })
                self._examples_seen += len(text)

            # Outer loop
            if query_set is not None:
                for text, labels in query_set:
                    labels = torch.tensor(labels).to(self.device)
                    # labels = labels.to(self.device)
                    output = self.forward(text, labels, fpn)
                    loss = self.loss_fn(output["logits"], labels)
                    self.update_meta_gradients(loss, fpn)

                    predictions = model_utils.make_prediction(output["logits"].detach())
                    self.update_query_tracker(loss, predictions, labels)
                    online_metrics = model_utils.calculate_metrics(predictions.tolist(), labels.tolist())
                    self.metrics["online"].append({
                        "accuracy": online_metrics["accuracy"],
                        "examples_seen": self.examples_seen(),
                        "task": task if task is not None else "none"
                    })
                    self._examples_seen += len(text)

                # Meta optimizer step
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

    def forward(self, text, labels, prediction_network, no_grad=False):
        input_dict = self.pn.encode_text(text)
        if no_grad:
            with torch.no_grad():
                logits = prediction_network(input_dict)
        else:
            logits = prediction_network(input_dict)

        return {"logits": logits}

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

    def log(self):
        support_loss = np.mean(self.tracker["support_loss"])
        query_loss = np.mean(self.tracker["query_loss"])
        support_metrics = model_utils.calculate_metrics(self.tracker["support_predictions"], self.tracker["support_labels"])
        query_metrics = model_utils.calculate_metrics(self.tracker["query_predictions"], self.tracker["query_labels"])

        self.logger.info(
            f"Episode {self.current_iter + 1} Support set: Loss = {support_loss:.4f}, "
            f"accuracy = {support_metrics['accuracy']:.4f}, precision = {support_metrics['precision']:.4f}, "
            f"recall = {support_metrics['recall']:.4f}, F1 score = {support_metrics['f1']:.4f}"
        )
        self.logger.info(
            f"Episode {self.current_iter + 1} Query set: Loss = {query_loss:.4f}, "
            f"accuracy = {query_metrics['accuracy']:.4f}, precision = {query_metrics['precision']:.4f}, "
            f"recall = {query_metrics['recall']:.4f}, F1 score = {query_metrics['f1']:.4f}"
        )
        if self.config.wandb:
            wandb.log({
                "support_accuracy": support_metrics['accuracy'],
                "support_precision": support_metrics['precision'],
                "support_recall": support_metrics['recall'],
                "support_f1": support_metrics['f1'],
                "support_loss": support_loss,
                "query_accuracy": query_metrics['accuracy'],
                "query_precision": query_metrics['precision'],
                "query_recall": query_metrics['recall'],
                "query_f1": query_metrics['f1'],
                "query_loss": query_loss,
                "examples_seen": self.examples_seen()
            })
        self.reset_tracker()

    def evaluate(self, dataloader, prediction_network=None):
        if self.config.learner.evaluation_support_set:
            support_set = []
            for _ in range(self.config.learner.updates):
                text, labels = self.memory.read_batch(batch_size=self.mini_batch_size)
                support_set.append((text, labels))

        with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                copy_initial_weights=False,
                                track_higher_grads=False) as (fpn, diffopt):
            if self.config.learner.evaluation_support_set:
                self.set_train()
                support_prediction_network = fpn
                # Inner loop
                task_predictions, task_labels = [], []
                support_loss = []
                for text, labels in support_set:
                    labels = torch.tensor(labels).to(self.device)
                    # labels = labels.to(self.device)
                    output = self.forward(text, labels, fpn)
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
                support_prediction_network = self.pn
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
        self.logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
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

    def get_support_set(self, data_iterator):
        """Return a list of datapoints, and return None if the end of the data is reached."""
        support_set = []
        for _ in range(self.config.learner.updates):
            try:
                text, labels, datasets = next(data_iterator)
                support_set.append((text, labels))
            except StopIteration:
                # self.logger.info("Terminating training as all the data is seen")
                return None
        return support_set, datasets[0]

    def get_query_set(self, data_iterator, replay_freq, replay_steps):
        """Return a list of datapoints, and return None if the end of the data is reached."""
        query_set = []
        if self.replay_rate != 0 and (self.current_iter + 1) % replay_freq == 0:
            # now we replay from memory
            self.logger.debug("query set read from memory")
            for _ in range(replay_steps):
                text, labels = self.memory.read_batch(batch_size=self.mini_batch_size)
                query_set.append((text, labels))
        else:
            # otherwise simply use next batch from data stream as query set
            try:
                text, labels, _ = next(data_iterator)
                query_set.append((text, labels))
                self.memory.write_batch(text, labels)
                self._examples_seen += self.mini_batch_size
            except StopIteration:
                # self.logger.info("Terminating training as all the data is seen")
                return None
        return query_set

    def set_eval(self):
        self.pn.eval()

    def set_train(self):
        self.pn.train()

    def few_shot_testing(self, train_dataset, eval_dataset, increment_counters=False):
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
        train_dataloader, eval_dataloader = self.few_shot_preparation(train_dataset, eval_dataset)
        all_predictions, all_labels = [], []
        with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                copy_initial_weights=False,
                                track_higher_grads=False) as (fpn, diffopt):
            # Inner loop
            for i, (text, labels, datasets) in enumerate(train_dataloader):
                self.set_train()
                labels = torch.tensor(labels).to(self.device)
                output = self.forward(text, labels, fpn)
                loss = self.loss_fn(output["logits"], labels)
                diffopt.step(loss)

                predictions = model_utils.make_prediction(output["logits"].detach())
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())
                dataset_results = self.evaluate(dataloader=eval_dataloader, prediction_network=fpn)
                self.log_few_shot(all_predictions, all_labels, datasets, dataset_results, increment_counters, text, i)
                if (i * self.config.testing.few_shot_batch_size) % self.mini_batch_size == 0 and i > 0:
                    all_predictions, all_labels = [], []
        self.few_shot_counter += 1


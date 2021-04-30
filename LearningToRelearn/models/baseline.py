import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW
import wandb

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import TransformerClsModel
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, alternating_order, datasets_dict


class Baseline(Learner):
    def __init__(self, config, **kwargs):
        """
        Baseline models: sequential and multitask setup.
        """
        super().__init__(config, **kwargs)
        self.lr = config.learner.lr
        self.type = config.learner.type
        self.n_epochs = config.training.epochs
        self.log_freq = config.training.log_freq
        self.model = TransformerClsModel(model_name=config.learner.model_name,
                                         n_classes=config.data.n_classes,
                                         max_length=config.data.max_length,
                                         device=self.device)
        self.logger.info("Loaded {} as model".format(self.model.__class__.__name__))
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def training(self, datasets, **kwargs):
        # train_datasets = {dataset_name: dataset for dataset_name, dataset in zip(datasets["order"], datasets["train"])}
        train_datasets = datasets_dict(datasets["train"], datasets["order"])
        samples_per_task = self.config.learner.samples_per_task

        order = self.config.task_order if self.config.task_order is not None else datasets["order"]
        n_samples = [samples_per_task] * len(order) if samples_per_task is not None else samples_per_task
        if self.type == "sequential":
            # if task_order is specified, use that instead of datasets["order"]
            self.logger.info(f"Using task order {order}")
            for train_dataset in get_continuum(train_datasets, order=order, n_samples=n_samples, merge=False):
                self.logger.info("Training on {}".format(train_dataset.__class__.__name__))
                train_dataloader = DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=False)
                self.train(dataloader=train_dataloader, datasets=datasets)
        elif self.type == "multitask":
            self.logger.info("Training multi-task model on all datasets")
            data = get_continuum(train_datasets, order=order, n_samples=n_samples)
            train_dataloader = DataLoader(data, batch_size=self.mini_batch_size, shuffle=True)
            self.train(dataloader=train_dataloader, datasets=datasets)
        elif self.type == "single":
            # train on a single task / dataset
            self.logger.info(f"Training single model on dataset {self.config.learner.dataset}")
            n_samples = [samples_per_task] if samples_per_task is not None else None
            data = get_continuum(train_datasets, order=[self.config.learner.dataset], n_samples=n_samples)
            train_dataloader = DataLoader(data, batch_size=self.mini_batch_size, shuffle=True)
            self.train(dataloader=train_dataloader, datasets=datasets)
        elif self.type == "alternating":
            order, n_samples = alternating_order(train_datasets, tasks=self.config.data.alternating_tasks,
                                                 n_samples_per_switch=self.config.data.alternating_n_samples_per_switch,
                                                 relative_frequencies=self.config.data.alternating_relative_frequencies)
            dataset = get_continuum(train_datasets, order=order, n_samples=n_samples)
            train_dataloader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False)
            self.train(dataloader=train_dataloader, datasets=datasets)
        else:
            raise ValueError("Invalid training mode")

    def train(self, dataloader=None, datasets=None):
        val_datasets = datasets_dict(datasets["val"], datasets["order"])

        for epoch in range(self.n_epochs):
            all_losses, all_predictions, all_labels = [], [], []

            for text, labels, datasets in dataloader:
                output = self.training_step(text, labels)
                task = datasets[0]

                predictions = model_utils.make_prediction(output["logits"].detach())
                self.update_tracker(output, predictions, labels)

                metrics = model_utils.calculate_metrics(self.tracker["predictions"], self.tracker["labels"])
                self.metrics["online"].append({
                    "accuracy": metrics["accuracy"],
                    "examples_seen": self.examples_seen(),
                    "task": task
                })
                if self.current_iter % self.log_freq == 0:
                    self.log()
                    self.write_metrics()
                if self.current_iter % self.validate_freq == 0:
                    self.validate(val_datasets, n_samples=self.config.training.n_validation_samples)
                self.current_iter += 1

    def training_step(self, text, labels):
        self.set_train()
        labels = torch.tensor(labels).to(self.device)

        input_dict = self.model.encode_text(text)
        logits = self.model(input_dict)
        loss = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()

        return {"logits": logits, "loss": loss}

    def log(self):
        metrics = model_utils.calculate_metrics(self.tracker["predictions"], self.tracker["labels"])
        self.logger.info(
            "Iteration {} - Metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
            "F1 score = {:.4f}".format(self.current_iter + 1,
                                        np.mean(self.tracker["losses"]),
                                        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]))
        if self.config.wandb:
            wandb.log({
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "loss": np.mean(self.tracker["losses"]),
                "examples_seen": self.examples_seen()
            })
        self.reset_tracker()

    def examples_seen(self):
        return (self.current_iter + 1) * self.mini_batch_size

    def evaluate(self, dataloader):
        self.set_eval()
        all_losses, all_predictions, all_labels = [], [], []

        for i, (text, labels, _) in enumerate(dataloader):
            labels = torch.tensor(labels).to(self.device)
            input_dict = self.model.encode_text(text)
            with torch.no_grad():
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)
            loss = loss.item()
            pred = model_utils.make_prediction(output.detach())
            all_losses.append(loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())
            if i % 20 == 0:
                self.logger.info(f"Batch {i + 1}/{len(dataloader)} processed")

        metrics = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), metrics["accuracy"], metrics["precision"],
                                               metrics["recall"], metrics["f1"]))

        return {"accuracy": metrics["accuracy"], "precision": metrics["precision"],
                "recall": metrics["recall"], "f1": metrics["f1"]}

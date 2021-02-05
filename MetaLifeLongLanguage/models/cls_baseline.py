import logging
import time

import numpy as np
import torch
from torch import nn
from torch.utils import data
from transformers import AdamW
import wandb

import MetaLifeLongLanguage.datasets.utils as dataset_utils
import MetaLifeLongLanguage.models.utils as model_utils
from MetaLifeLongLanguage.models.base_models import TransformerClsModel
from MetaLifeLongLanguage.learner import Learner

# logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# logger = logging.getLogger("Baseline-Log")


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
        self.log_freq = config.training.log_freq
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def training(self, datasets, **kwargs):
        train_datasets = datasets["train"]
        if self.type == "sequential":
            data_length = sum([len(train_dataset) for train_dataset in train_datasets]) // self.mini_batch_size
            for train_dataset in train_datasets:
                self.logger.info("Training on {}".format(train_dataset.__class__.__name__))
                train_dataloader = data.DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=False,
                                                   collate_fn=dataset_utils.batch_encode)
                self.train(dataloader=train_dataloader, data_length=data_length)
        elif self.type == "multitask":
            train_dataset = data.ConcatDataset(train_datasets)
            self.logger.info("Training multi-task model on all datasets")
            train_dataloader = data.DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=True,
                                               collate_fn=dataset_utils.batch_encode)
            self.train(dataloader=train_dataloader)
        elif self.type == "single":
            # train on a single task / dataset
            order = datasets["order"]
            train_dataset = train_datasets[order.index(self.config.learner.dataset)]
            self.logger.info(f"Training single model on dataset {self.config.learner.dataset}")
            train_dataloader = data.DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=True,
                                               collate_fn=dataset_utils.batch_encode)
            self.train(dataloader=train_dataloader)
        else:
            raise ValueError("Invalid training mode")

    def train(self, dataloader, data_length=None):
        self.model.train()
        if data_length is None:
            data_length = len(dataloader) * self.n_epochs
        for epoch in range(self.n_epochs):
            all_losses, all_predictions, all_labels = [], [], []

            for text, labels in dataloader:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.model.encode_text(text)
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                self.logger.debug(f"Loss: {loss}")
                pred = model_utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())

                if self.current_iter % self.log_freq == 0:
                    acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
                    time_per_iteration, estimated_time_left = self.time_metrics(data_length)
                    self.logger.info(
                        "Iteration {}/{} ({:.2f}%) -- {:.3f} (sec/it) -- Time Left: {}\nMetrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                        "F1 score = {:.4f}".format(self.current_iter + 1, data_length, (self.current_iter + 1) / data_length * 100,
                                                time_per_iteration, estimated_time_left,
                                                np.mean(all_losses), acc, prec, rec, f1))
                    if self.config.wandb:
                        wandb.log({
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1": f1,
                            "loss": np.mean(all_losses),
                            "examples_seen": (self.current_iter + 1) * self.mini_batch_size
                        })
                    all_losses, all_predictions, all_labels = [], [], []
                    self.start_time = time.time()
                self.time_checkpoint()
                self.current_iter += 1

    def testing(self, datasets):
        """
        Parameters
        ---
        datasets: List[Dataset]
            Test datasets.
        """
        accuracies, precisions, recalls, f1s = [], [], [], []
        results = {}
        # only have one dataset if type is single
        if self.type == "single":
            order = datasets["order"]
            train_dataset = train_datasets[order.index(self.config.learner.dataset)]
            datasets = [train_dataset]
        for dataset in datasets:
            dataset_name = dataset.__class__.__name__
            self.logger.info("Testing on {}".format(dataset_name))
            test_dataloader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False,
                                         collate_fn=batch_encode)
            dataset_results = self.evaluate(dataloader=test_dataloader)
            accuracies.append(dataset_results["accuracy"])
            precisions.append(dataset_results["precision"])
            recalls.append(dataset_results["recall"])
            f1s.append(dataset_results["f1"])
            results[dataset_name] = dataset_results

        mean_results = {
            "accuracy": np.mean(accuracies),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s)
        }
        self.logger.info("Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(
                        mean_results["accuracy"], mean_results["precision"], mean_results["recall"],
                        mean_results["f1"]
                    ))
        return results, mean_results

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels = [], [], []

        self.model.eval()

        for i, (text, labels) in enumerate(dataloader):
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

        acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), acc, prec, rec, f1))

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

import logging
import torch
from torch import nn

import numpy as np

from torch.utils import data
from transformers import AdamW

import MetaLifeLongLanguage.datasets.utils as dataset_utils
import MetaLifeLongLanguage.models.utils as model_utils
from MetaLifeLongLanguage.models.base_models import TransformerClsModel
from MetaLifeLongLanguage.learner import Learner

logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Baseline-Log")


class Baseline(Learner):
    def __init__(self, config):
        """
        Baseline models: sequential and multitask setup.
        """
        self.lr = config.learner.lr
        self.type = config.learner.type
        self.mini_batch_size = config.training.batch_size
        self.model = TransformerClsModel(model_name=config.learner.model_name,
                                         n_classes=config.data.n_classes,
                                         max_length=config.data.max_length,
                                         device=self.device)
        logger.info("Loaded {} as model".format(self.model.__class__.__name__))
        self.loss_fn = nn.CrossEntropyLoss()
        self.log_freq = config.training.log_freq
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def training(self, datasets, **kwargs):
        train_datasets = datasets["train"]
        if self.type == "sequential":
            for train_dataset in train_datasets:
                logger.info("Training on {}".format(train_dataset.__class__.__name__))
                train_dataloader = data.DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=False,
                                                   collate_fn=dataset_utils.batch_encode)
                self.train(dataloader=train_dataloader)
        elif self.type == "multitask":
            train_dataset = data.ConcatDataset(train_datasets)
            logger.info("Training multi-task model on all datasets")
            train_dataloader = data.DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=True,
                                               collate_fn=dataset_utils.batch_encode)
            self.train(dataloader=train_dataloader)
        else:
            raise ValueError("Invalid training mode")

    def train(self, dataloader):
        self.model.train()

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
                pred = model_utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                self.current_iter += 1

                if self.current_iter % self.log_freq == 0:
                    acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
                    logger.info(
                        "Epoch {} metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                        "F1 score = {:.4f}".format(epoch + 1, np.mean(all_losses), acc, prec, rec, f1))
                    self.writer.add_scalar("Train/Accuracy", acc, self.current_iter)
                    self.writer.add_scalar("Train/Precision", prec, self.current_iter)
                    self.writer.add_scalar("Train/Recall", rec, self.current_iter)
                    self.writer.add_scalar("Train/F1-Score", f1, self.current_iter)
                    self.writer.add_scalar("Train/Loss", np.mean(all_losses), self.current_iter)
                    all_losses, all_predictions, all_labels = [], [], []

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels = [], [], []

        self.model.eval()

        for text, labels in dataloader:
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

        acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
        logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), acc, prec, rec, f1))

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

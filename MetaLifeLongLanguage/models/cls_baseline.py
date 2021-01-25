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

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Baseline-Log')


class Baseline(Learner):
    def __init__(self,
                 lr,
                 type,
                 model_name,
                 n_classes,
                 max_length,
                 device="cuda"):
        """
        Baseline models: sequential and multitask setup.

        Parameters
        ---
        lr: float
            Learning rate.
        type: str
            One of {"sequential", "multitask"}
        model_name: str
            One of {"bert", "albert"}
        n_classes: int
            Number of classes.
        max_length: int
            Max length of sentences fed into the model.
        device: str
            One of {"cpu", "cuda"}.
        """
        self.lr = lr
        self.device = device
        self.type = type
        self.model = TransformerClsModel(model_name=model.name,
                                         n_classes=n_classes,
                                         max_length=max_length,
                                         device=device)
        logger.info('Loaded {} as model'.format(self.model.__class__.__name__))
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def save_model(self, model_path):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)

    def train(self, dataloader, n_epochs, log_freq):

        self.model.train()

        for epoch in range(n_epochs):
            all_losses, all_predictions, all_labels = [], [], []
            iter = 0

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
                iter += 1

                if iter % log_freq == 0:
                    acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
                    logger.info(
                        'Epoch {} metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                        'F1 score = {:.4f}'.format(epoch + 1, np.mean(all_losses), acc, prec, rec, f1))
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
        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def training(self, train_datasets, val_datasets, **kwargs):
        #TODO: use val_datasets
        n_epochs = kwargs.get('n_epochs', 1)
        log_freq = kwargs.get('log_freq', 500)
        mini_batch_size = kwargs.get('mini_batch_size')
        if self.training_mode == 'sequential':
            for train_dataset in train_datasets:
                logger.info('Training on {}'.format(train_dataset.__class__.__name__))
                train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=False,
                                                   collate_fn=dataset_utils.batch_encode)
                self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)
        elif self.training_mode == 'multi_task':
            train_dataset = data.ConcatDataset(train_datasets)
            logger.info('Training multi-task model on all datasets')
            train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True,
                                               collate_fn=dataset_utils.batch_encode)
            self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)
        else:
            raise ValueError('Invalid training mode')

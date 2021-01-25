import logging
import torch
from torch import nn

import numpy as np

from torch.utils import data
from transformers import AdamW

import MetaLifeLongLanguage.datasets.utils as dataset_utils
import MetaLifeLongLanguage.models.utils as model_utils
from MetaLifeLongLanguage.models.base_models import TransformerClsModel, ReplayMemory
from MetaLifeLongLanguage.learner import Learner

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Replay-Log')


class Replay(Learner):

    def __init__(self, device, n_classes, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.device = device

        self.model = TransformerClsModel(model_name=kwargs.get('model'),
                                         n_classes=n_classes,
                                         max_length=kwargs.get('max_length'),
                                         device=device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        logger.info('Loaded {} as model'.format(self.model.__class__.__name__))

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    # TODO: save_model and load_model should be general functions for any model, not model methods
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

                # TODO: This doesn't have to be recomputed all the time
                mini_batch_size = len(labels)
                replay_freq = self.replay_every // mini_batch_size
                replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)

                if self.replay_rate != 0 and (iter + 1) % replay_freq == 0:
                    self.optimizer.zero_grad()
                    for _ in range(replay_steps):
                        ref_text, ref_labels = self.memory.read_batch(batch_size=mini_batch_size)
                        ref_labels = torch.tensor(ref_labels).to(self.device)
                        ref_input_dict = self.model.encode_text(ref_text)
                        ref_output = self.model(ref_input_dict)
                        ref_loss = self.loss_fn(ref_output, ref_labels)
                        ref_loss.backward()

                    # TODO: Ask Nithin why he does this
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    # TODO: turn into hyperparam
                    torch.nn.utils.clip_grad_norm(params, 25)
                    self.optimizer.step()

                loss = loss.item()
                pred = model_utils.make_prediction(output.detach())
                # TODO: Add writing to tensorboard functionality
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                iter += 1
                self.memory.write_batch(text, labels)

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

    def training(self, train_datasets, **kwargs):
        n_epochs = kwargs.get('n_epochs', 1)
        log_freq = kwargs.get('log_freq', 50)
        mini_batch_size = kwargs.get('mini_batch_size')
        train_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=False,
                                           collate_fn=dataset_utils.batch_encode)
        self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)

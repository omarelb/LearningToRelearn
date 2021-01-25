import logging
import torch
from torch import nn

import numpy as np

from torch.utils import data
from transformers import AdamW

import MetaLifeLongLanguage.datasets
import MetaLifeLongLanguage.models.utils as model_utils
from MetaLifeLongLanguage.models.base_models import TransformerClsModel, ReplayMemory
from MetaLifeLongLanguage.learner import Learner
from MetaLifeLongLanguage.datasets.utils import batch_encode

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AGEM-Log')


class AGEM(Learner):
    def __init__(self, config):
        super().__init__(config)
        self.lr = config.learner.lr
        self.write_prob = config.write_prob
        self.replay_rate = config.replay_rate
        self.replay_every = config.replay_every
        self.device = config.training.device
        self.n_epochs = config.training.epochs
        self.mini_batch_size = config.training.batch_size
        self.log_freq = config.training.log_freq

        self.model = TransformerClsModel(model_name=config.learner.model_name,
                                         n_classes=config.data.n_classes,
                                         max_length=config.data.max_length,
                                         device=self.device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.logger.info('Loaded {} as model'.format(self.model.__class__.__name__))

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def training(self, datasets, **kwargs):
        train_datasets = data.ConcatDataset(datasets["train"])
        dataloaders = {
            "train": data.DataLoader(train_datasets, batch_size=self.mini_batch_size, shuffle=False,
                                     collate_fn=batch_encode),
        }
        self.train(dataloaders=dataloaders)

    def train(self, dataloaders):
        self.model.train()
        dataloader = dataloaders["train"]

        for epoch in range(self.n_epochs):
            epoch_iteration = 0
            all_losses, all_predictions, all_labels = [], [], []

            for text, labels in dataloader:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.model.encode_text(text)
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)

                self.update_parameters(loss, mini_batch_size=len(labels), epoch_iteration=epoch_iteration)

                loss = loss.item()
                pred = model_utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                self.current_iter += 1
                self.memory.write_batch(text, labels)

                if self.current_iter % self.log_freq == 0:
                    self.write_log(all_predictions, all_labels, all_losses)
                    all_losses, all_predictions, all_labels = [], [], []
                if self.current_iter % self.config.training.save_freq == 0:
                    self.save_checkpoint()
                epoch_iteration += 1
            self.current_epoch += 1

    def update_parameters(self, loss, mini_batch_size, epoch_iteration):
        """Update parameters of model"""
        self.optimizer.zero_grad()

        params = [p for p in self.model.parameters() if p.requires_grad]
        orig_grad = torch.autograd.grad(loss, params)

        replay_freq = self.replay_every // mini_batch_size
        replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)

        if self.replay_rate != 0 and (epoch_iteration + 1) % replay_freq == 0:
            ref_grad_sum = None
            for _ in range(replay_steps):
                ref_text, ref_labels = self.memory.read_batch(batch_size=mini_batch_size)
                ref_labels = torch.tensor(ref_labels).to(self.device)
                ref_input_dict = self.model.encode_text(ref_text)
                ref_output = self.model(ref_input_dict)
                ref_loss = self.loss_fn(ref_output, ref_labels)
                ref_grad = torch.autograd.grad(ref_loss, params)
                if ref_grad_sum is None:
                    ref_grad_sum = ref_grad
                else:
                    ref_grad_sum = [x + y for (x, y) in zip(ref_grad, ref_grad_sum)]
            final_grad = self.compute_grad(orig_grad, ref_grad_sum)
        else:
            final_grad = orig_grad

        for param, grad in zip(params, final_grad):
            param.grad = grad.data
        self.optimizer.step()

    def write_log(self, all_predictions, all_labels, all_losses):
        acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.info(
            'Iteration {} metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
            'F1 score = {:.4f}'.format(self.current_iter + 1, np.mean(all_losses), acc, prec, rec, f1))
        self.writer.add_scalar('Train/Accuracy', acc, self.current_iter)
        self.writer.add_scalar('Train/Precision', prec, self.current_iter)
        self.writer.add_scalar('Train/Recall', rec, self.current_iter)
        self.writer.add_scalar('Train/F1-Score', f1, self.current_iter)
        self.writer.add_scalar('Train/Loss', np.mean(all_losses), self.current_iter)

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

    def testing(self, datasets):
        accuracies, precisions, recalls, f1s = [], [], [], []
        results = {}
        for dataset in datasets:
            dataset_name = dataset.__class__.__name__
            logger.info('Testing on {}'.format(dataset_name))
            test_dataloader = data.DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False,
                                              collate_fn=batch_encode)
            dataset_results = self.evaluate(dataloader=test_dataloader)
            accuracies.append(results["accuracy"])
            precisions.append(results["precision"])
            recalls.append(results["recall"])
            f1s.append(results["f1"])
            results[dataset_name] = dataset_results

        logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))
        return results

    def compute_grad(self, orig_grad, ref_grad):
        """Computes gradient according to the AGEM method"""
        with torch.no_grad():
            flat_orig_grad = torch.cat([torch.flatten(x) for x in orig_grad])
            flat_ref_grad = torch.cat([torch.flatten(x) for x in ref_grad])
            dot_product = torch.dot(flat_orig_grad, flat_ref_grad)
            if dot_product >= 0:
                return orig_grad
            proj_component = dot_product / torch.dot(flat_ref_grad, flat_ref_grad)
            modified_grad = [o - proj_component * r for (o, r) in zip(orig_grad, ref_grad)]
            return modified_grad

    def save_model(self, model_path):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
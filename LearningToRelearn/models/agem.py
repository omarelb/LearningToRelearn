import logging
import time

import wandb
import numpy as np
import torch
from torch import nn
from torch.utils import data
from transformers import AdamW

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import TransformerClsModel, ReplayMemory
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.utils import batch_encode

# logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# logger = logging.getLogger("AGEM-Log")

class AGEM(Learner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.lr = config.learner.lr
        self.write_prob = config.write_prob
        self.replay_rate = config.replay_rate
        self.replay_every = config.replay_every
        self.n_epochs = config.training.epochs

        self.model = TransformerClsModel(model_name=config.learner.model_name,
                                         n_classes=config.data.n_classes,
                                         max_length=config.data.max_length,
                                         device=self.device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.logger.info("Loaded {} as model".format(self.model.__class__.__name__))

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
        data_length = len(dataloader) * self.n_epochs

        for epoch in range(self.n_epochs):
            all_losses, all_predictions, all_labels = [], [], []

            for text, labels in dataloader:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.model.encode_text(text)
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)

                self.update_parameters(loss, mini_batch_size=len(labels))

                loss = loss.item()
                pred = model_utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                self.memory.write_batch(text, labels)

                if self.current_iter % self.log_freq == 0:
                    self.write_log(all_predictions, all_labels, all_losses, data_length=data_length)
                    self.start_time = time.time() # time from last log
                    all_losses, all_predictions, all_labels = [], [], []
                # if self.current_iter % self.config.training.save_freq == 0:
                self.time_checkpoint()
                self.current_iter += 1
            self.current_epoch += 1

    def update_parameters(self, loss, mini_batch_size):
        """Update parameters of model"""
        self.optimizer.zero_grad()

        params = [p for p in self.model.parameters() if p.requires_grad]
        orig_grad = torch.autograd.grad(loss, params)

        replay_freq = self.replay_every // mini_batch_size
        replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)

        if self.replay_rate != 0 and (self.current_iter + 1) % replay_freq == 0:
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

    def write_log(self, all_predictions, all_labels, all_losses, data_length):
        acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
        time_per_iteration, estimated_time_left = self.time_metrics(data_length)
        self.logger.info(
            "Iteration {}/{} ({:.2f}%) -- {:.3f} (sec/it) -- Time Left: {}\nMetrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
            "F1 score = {:.4f}".format(self.current_iter + 1, data_length, (self.current_iter + 1) / data_length * 100,
                                       time_per_iteration, estimated_time_left,
                                       np.mean(all_losses), acc, prec, rec, f1))
        if self.config.wandb:
            n_examples_seen = (self.current_iter + 1) * self.mini_batch_size
            wandb.log({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "loss": np.mean(all_losses),
                "examples_seen": n_examples_seen
            })

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels = [], [], []

        self.set_eval()

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

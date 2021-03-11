import logging
import math

import higher
import torch
from torch import nn, optim

import numpy as np

from torch.utils import data
from transformers import AdamW

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import ReplayMemory, TransformerClsModel
from LearningToRelearn.learner import Learner

logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MAML-Log")


class MAML(Learner):

    def __init__(self, device, n_classes, **kwargs):
        super().__init__(config)
        self.inner_lr = config.learner.inner_lr
        self.meta_lr = config.learner.meta_lr
        self.write_prob = config.write_prob
        self.replay_rate = config.replay_rate
        self.replay_every = config.replay_every
        self.device = config.training.device
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
        train_datasets = datasets["train"]

        replay_freq, replay_steps = self.replay_parameters()
        self.logger.info("Replay frequency: {}".format(replay_freq))
        self.logger.info("Replay steps: {}".format(replay_steps))

        concat_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = iter(data.DataLoader(concat_dataset, batch_size=self.mini_batch_size, shuffle=False,
                                                collate_fn=dataset_utils.batch_encode))

        while True:
            self.inner_optimizer.zero_grad()
            support_loss, support_acc, support_prec, support_rec, support_f1 = [], [], [], [], []

            with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpn, diffopt):

                # Inner loop
                support_set = []
                task_predictions, task_labels = [], []
                for _ in range(updates):
                    try:
                        text, labels = next(train_dataloader)
                        support_set.append((text, labels))
                    except StopIteration:
                        self.logger.info("Terminating training as all the data is seen")
                        return

                for text, labels in support_set:
                    labels = torch.tensor(labels).to(self.device)
                    input_dict = self.pn.encode_text(text)
                    output = fpn(input_dict)
                    loss = self.loss_fn(output, labels)
                    diffopt.step(loss)
                    pred = model_utils.make_prediction(output.detach())
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(labels.tolist())
                    self.memory.write_batch(text, labels)

                acc, prec, rec, f1 = model_utils.calculate_metrics(task_predictions, task_labels)

                self.logger.info("Episode {} support set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, "
                            "recall = {:.4f}, F1 score = {:.4f}".format(self.current_iter + 1,
                                                                        np.mean(support_loss), acc, prec, rec, f1))

                # Outer loop
                query_loss, query_acc, query_prec, query_rec, query_f1 = [], [], [], [], []
                query_set = []

                if self.replay_rate != 0 and (self.current_iter + 1) % replay_freq == 0:
                    for _ in range(replay_steps):
                        text, labels = self.memory.read_batch(batch_size=mini_batch_size)
                        query_set.append((text, labels))
                else:
                    try:
                        text, labels = next(train_dataloader)
                        query_set.append((text, labels))
                        self.memory.write_batch(text, labels)
                    except StopIteration:
                        self.logger.info("Terminating training as all the data is seen")
                        return

                for text, labels in query_set:
                    labels = torch.tensor(labels).to(self.device)
                    input_dict = self.pn.encode_text(text)
                    output = fpn(input_dict)
                    loss = self.loss_fn(output, labels)
                    query_loss.append(loss.item())
                    pred = model_utils.make_prediction(output.detach())

                    acc, prec, rec, f1 = model_utils.calculate_metrics(pred.tolist(), labels.tolist())
                    query_acc.append(acc)
                    query_prec.append(prec)
                    query_rec.append(rec)
                    query_f1.append(f1)

                    # PN meta gradients
                    pn_params = [p for p in fpn.parameters() if p.requires_grad]
                    meta_pn_grads = torch.autograd.grad(loss, pn_params)
                    pn_params = [p for p in self.pn.parameters() if p.requires_grad]
                    for param, meta_grad in zip(pn_params, meta_pn_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                # Meta optimizer step
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

                self.logger.info("Episode {} query set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, "
                            "recall = {:.4f}, F1 score = {:.4f}".format(self.current_iter + 1,
                                                                        np.mean(query_loss), np.mean(query_acc),
                                                                        np.mean(query_prec), np.mean(query_rec),
                                                                        np.mean(query_f1)))

                self.current_iter += 1

    def evaluate(self, dataloader, updates, mini_batch_size):

        support_set = []
        for _ in range(updates):
            text, labels = self.memory.read_batch(batch_size=mini_batch_size)
            support_set.append((text, labels))

        with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpn, diffopt):

            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for text, labels in support_set:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.pn.encode_text(text)
                output = fpn(input_dict)
                loss = self.loss_fn(output, labels)
                diffopt.step(loss)
                pred = model_utils.make_prediction(output.detach())
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                task_labels.extend(labels.tolist())

            acc, prec, rec, f1 = model_utils.calculate_metrics(task_predictions, task_labels)

            self.logger.info("Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, "
                        "recall = {:.4f}, F1 score = {:.4f}".format(np.mean(support_loss), acc, prec, rec, f1))

            all_losses, all_predictions, all_labels = [], [], []

            for text, labels in dataloader:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.pn.encode_text(text)
                with torch.no_grad():
                    output = fpn(input_dict)
                    loss = self.loss_fn(output, labels)
                loss = loss.item()
                pred = model_utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())

        acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), acc, prec, rec, f1))

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def model_state(self, model_path):
        return self.pn.state_dict()

    def optimizer_state(self):
        return self.meta_optimizer.state_dict()

    def load_model_state(self, checkpoint):
        self.pn.load_state_dict(checkpoint["model_state"])

    def load_optimizer_state(self, checkpoint):
        self.meta_optimizer.load_state_dict(checkpoint["optimizer"])
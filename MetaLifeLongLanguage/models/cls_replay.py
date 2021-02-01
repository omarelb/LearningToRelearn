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
from MetaLifeLongLanguage.models.base_models import TransformerClsModel, ReplayMemory
from MetaLifeLongLanguage.learner import Learner

# logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# logger = logging.getLogger("Replay-Log")


class Replay(Learner):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.lr = config.learner.lr
        self.write_prob = config.write_prob
        self.replay_rate = config.replay_rate
        self.replay_every = config.replay_every
        self.n_epochs = config.training.epochs
        self.log_freq = config.training.log_freq

        self.model = TransformerClsModel(model_name=config.learner.model_name,
                                      n_classes=config.data.n_classes,
                                      max_length=config.data.max_length,
                                      device=self.device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.logger.info("Loaded {} as model".format(self.model.__class__.__name__))

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def training(self, datasets, **kwargs):
        train_datasets = datasets["train"]
        train_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = data.DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=False,
                                           collate_fn=dataset_utils.batch_encode)
        self.train(dataloader=train_dataloader)

    def train(self, dataloader):
        self.model.train()
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

                # TODO: This doesn"t have to be recomputed all the time
                mini_batch_size = len(labels)
                replay_freq = self.replay_every // mini_batch_size
                replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)

                if self.replay_rate != 0 and (self.current_iter + 1) % replay_freq == 0:
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
                    torch.nn.utils.clip_grad_norm(params, self.config.clip_grad_norm)
                    self.optimizer.step()

                loss = loss.item()
                pred = model_utils.make_prediction(output.detach())
                # TODO: Add writing to tensorboard functionality
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                self.memory.write_batch(text, labels)

                if self.current_iter % self.log_freq == 0:
                    acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
                    time_per_iteration, estimated_time_left = self.time_metrics(data_length)
                    self.logger.info(
                        "Iteration {}/{} ({:.2f}%) -- {:.3f} (sec/it) -- Time Left: {}\nMetrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                        "F1 score = {:.4f}".format(self.current_iter + 1, data_length, (self.current_iter + 1) / data_length * 100,
                                                time_per_iteration, estimated_time_left,
                                                np.mean(all_losses), acc, prec, rec, f1))
                    self.writer.add_scalar("Train/Accuracy", acc, self.current_iter)
                    self.writer.add_scalar("Train/Precision", prec, self.current_iter)
                    self.writer.add_scalar("Train/Recall", rec, self.current_iter)
                    self.writer.add_scalar("Train/F1-Score", f1, self.current_iter)
                    self.writer.add_scalar("Train/Loss", np.mean(all_losses), self.current_iter)
                    if self.config.wandb:
                        wandb.log({
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1": f1,
                            "loss": np.mean(all_losses),
                            "examples_seen": (self.current_iter + 1) * self.mini_batch_size
                        })
                    self.start_time = time.time()
                    all_losses, all_predictions, all_labels = [], [], []
                self.time_checkpoint()
                self.current_iter += 1

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
        if self.config.wandb:
            wandb.log({"test_accuracy": acc, "test_precision": prec, "test_recall": rec, "test_f1": f1})

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

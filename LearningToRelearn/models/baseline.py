import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from transformers import AdamW
import wandb

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import ReplayMemory, TransformerClsModel
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, alternating_order, datasets_dict, n_samples_order


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
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)

    def training(self, datasets, **kwargs):
        # train_datasets = {dataset_name: dataset for dataset_name, dataset in zip(datasets["order"], datasets["train"])}
        train_datasets = datasets_dict(datasets["train"], datasets["order"])
        val_datasets = datasets_dict(datasets["test"], datasets["order"])
        eval_dataset = val_datasets[self.config.testing.eval_dataset]
        eval_dataset = eval_dataset.sample(min(self.config.testing.few_shot_validation_size, len(eval_dataset)))

        if self.config.data.alternating_order:
            order, n_samples = alternating_order(train_datasets, tasks=self.config.data.alternating_tasks,
                                                 n_samples_per_switch=self.config.data.alternating_n_samples_per_switch,
                                                 relative_frequencies=self.config.data.alternating_relative_frequencies)
        else:
            n_samples, order = n_samples_order(self.config.learner.samples_per_task, self.config.task_order, datasets["order"])
        datas = get_continuum(train_datasets, order=order, n_samples=n_samples,
                             eval_dataset=self.config.testing.eval_dataset, merge=False)
        if self.config.learner.multitask:
            data = ConcatDataset(datas)
            train_dataloader = DataLoader(data, batch_size=self.mini_batch_size, shuffle=True)
            self.train(dataloader=train_dataloader, datasets=datasets)

        for data, dataset_name, n_sample in zip(datas, order, n_samples):
            self.logger.info(f"Observing dataset {dataset_name} for {n_sample} samples. "
                             f"Evaluation={dataset_name=='evaluation'}")
            if dataset_name == "evaluation":
                self.few_shot_testing(train_dataset=data, eval_dataset=eval_dataset, increment_counters=False)
            else:
                train_dataloader = DataLoader(data, batch_size=self.mini_batch_size, shuffle=False)
                self.train(dataloader=train_dataloader, datasets=datasets, dataset_name=dataset_name, max_samples=n_sample)
            if dataset_name == self.config.testing.eval_dataset:
                self.eval_task_first_encounter = False

    def train(self, dataloader=None, datasets=None, dataset_name=None, max_samples=None):
        val_datasets = datasets_dict(datasets["val"], datasets["order"])
        replay_freq, replay_steps = self.replay_parameters(metalearner=False)

        episode_samples_seen = 0 # have to keep track of per-task samples seen as we might use replay as well
        for _ in range(self.n_epochs):
            for text, labels, datasets in dataloader:
                output = self.training_step(text, labels)
                task = datasets[0]

                predictions = model_utils.make_prediction(output["logits"].detach())
                self.update_tracker(output, predictions, labels)

                metrics = model_utils.calculate_metrics(self.tracker["predictions"], self.tracker["labels"])
                online_metrics = {
                    "accuracy": metrics["accuracy"],
                    "examples_seen": self.examples_seen(),
                    "task": task
                }
                self.metrics["online"].append(online_metrics)
                if dataset_name is not None and dataset_name == self.config.testing.eval_dataset and \
                    self.eval_task_first_encounter:
                    self.metrics["eval_task_first_encounter"].append(online_metrics)
                if self.current_iter % self.log_freq == 0:
                    self.log()
                    self.write_metrics()
                if self.current_iter % self.validate_freq == 0:
                    self.validate(val_datasets, n_samples=self.config.training.n_validation_samples)
                if self.replay_rate != 0 and (self.current_iter + 1) % replay_freq == 0:
                    self.replay_training_step(replay_steps, episode_samples_seen, max_samples)
                self.memory.write_batch(text, labels)
                self._examples_seen += len(text)
                episode_samples_seen += len(text)
                self.current_iter += 1
                if max_samples is not None and episode_samples_seen >= max_samples:
                    break

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

    def replay_training_step(self, replay_steps, episode_samples_seen, max_samples):
        self.optimizer.zero_grad()
        for _ in range(replay_steps):
            text, labels = self.memory.read_batch(batch_size=self.mini_batch_size)
            labels = torch.tensor(labels).to(self.device)
            input_dict = self.model.encode_text(text)
            output = self.model(input_dict)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self._examples_seen += len(text)
            self.metrics["replay_samples_seen"] += len(text)
            episode_samples_seen += len(text)
            if max_samples is not None and episode_samples_seen >= max_samples:
                break

        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm(params, self.config.learner.clip_grad_norm)
        self.optimizer.step()

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
            # if i % 20 == 0:
            #     self.logger.info(f"Batch {i + 1}/{len(dataloader)} processed")

        metrics = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), metrics["accuracy"], metrics["precision"],
                                               metrics["recall"], metrics["f1"]))

        return {"accuracy": metrics["accuracy"], "precision": metrics["precision"],
                "recall": metrics["recall"], "f1": metrics["f1"]}

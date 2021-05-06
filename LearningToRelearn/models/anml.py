import logging
import math
import time

import higher
import torch
from torch import nn, optim
import wandb

import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import ReplayMemory, TransformerClsModel, TransformerNeuromodulator
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, datasets_dict

# logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# logger = logging.getLogger("ANML-Log")

class ANML(Learner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.inner_lr = config.learner.inner_lr
        self.meta_lr = config.learner.meta_lr
        self.write_prob = config.write_prob
        self.replay_rate = config.replay_rate
        self.replay_every = config.replay_every
        self.mini_batch_size = config.training.batch_size

        self.nm = TransformerNeuromodulator(model_name=config.learner.model_name,
                                            device=self.device)
        self.pn = TransformerClsModel(model_name=config.learner.model_name,
                                      n_classes=config.data.n_classes,
                                      max_length=config.data.max_length,
                                      device=self.device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.loss_fn = nn.CrossEntropyLoss()

        self.logger.info("Loaded {} as NM".format(self.nm.__class__.__name__))
        self.logger.info("Loaded {} as PN".format(self.pn.__class__.__name__))

        meta_params = [p for p in self.nm.parameters() if p.requires_grad] + \
                      [p for p in self.pn.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pn.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def training(self, datasets, **kwargs):
        train_datasets = datasets_dict(datasets["train"], datasets["order"])
        replay_freq, replay_steps = self.replay_parameters()
        self.logger.info("Replay frequency: {}".format(replay_freq))
        self.logger.info("Replay steps: {}".format(replay_steps))

        samples_per_task = self.config.learner.samples_per_task
        order = self.config.task_order if self.config.task_order is not None else datasets["order"]
        n_samples = samples_per_task
        if samples_per_task is not None and isinstance(samples_per_task, int):
            n_samples = [samples_per_task] * len(order)
        data = get_continuum(train_datasets, order=order, n_samples=n_samples)
        train_dataloader = iter(DataLoader(data, batch_size=self.mini_batch_size, shuffle=False))
        n_episodes = len(train_dataloader) // self.config.updates

        # iterate over episodes
        while True:
            self.set_train()
            support_set = self.get_support_set(train_dataloader)
            query_set = self.get_query_set(train_dataloader, replay_freq, replay_steps)
            if support_set is None or query_set is None:
                return

            self.training_step(support_set, query_set)
            # self.metrics["online"].append({
            #     "accuracy": online_metrics["accuracy"],
            #     "examples_seen": self.examples_seen(),
            #     "task": datasets[0]  # assumes whole batch is from same task
            # })

            self.log()
            self.write_metrics()
            self.current_iter += 1

    def training_step(self, support_set, query_set=None):
        self.inner_optimizer.zero_grad()
        with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                copy_initial_weights=False,
                                track_higher_grads=False) as (fpn, diffopt):
            # Inner loop
            for text, labels in support_set:
                labels = torch.tensor(labels).to(self.device)
                output = self.forward(text, labels, fpn)
                loss = self.loss_fn(output["logits"], labels)
                diffopt.step(loss)
                self.memory.write_batch(text, labels)

                predictions = model_utils.make_prediction(output["logits"].detach())
                self.update_support_tracker(loss, predictions, labels)
                online_metrics = model_utils.calculate_metrics(predictions.tolist(), labels.tolist())
                self.metrics["online"].append({
                    "accuracy": online_metrics["accuracy"],
                    "examples_seen": self.examples_seen(),
                })
                self._examples_seen += len(text)

            # Outer loop
            if query_set is not None:
                for text, labels in query_set:
                    labels = torch.tensor(labels).to(self.device)
                    # labels = labels.clone().detach().to(self.device)
                    output = self.forward(text, labels, fpn)
                    loss = self.loss_fn(output["logits"], labels)
                    self.update_meta_gradients(loss, fpn)

                    predictions = model_utils.make_prediction(output["logits"].detach())
                    self.update_query_tracker(loss, predictions, labels)
                    online_metrics = model_utils.calculate_metrics(predictions.tolist(), labels.tolist())
                    self.metrics["online"].append({
                        "accuracy": online_metrics["accuracy"],
                        "examples_seen": self.examples_seen(),
                    })
                    self._examples_seen += len(text)

                # Meta optimizer step
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

    def forward(self, text, labels, prediction_network, no_grad=False):
        if no_grad:
            with torch.no_grad():
                input_dict = self.pn.encode_text(text)
                representation = prediction_network(input_dict, out_from="transformers")
                modulation = self.nm(input_dict)
                logits = prediction_network(representation * modulation, out_from="linear")
        else:
            input_dict = self.pn.encode_text(text)
            representation = prediction_network(input_dict, out_from="transformers")
            modulation = self.nm(input_dict)
            modulated_representation = representation * modulation
            logits = prediction_network(modulated_representation, out_from="linear")

        return {"logits": logits}

    def update_meta_gradients(self, loss, fpn):
        # NM meta gradients
        nm_params = [p for p in self.nm.parameters() if p.requires_grad]
        meta_nm_grads = torch.autograd.grad(loss, nm_params, retain_graph=True, allow_unused=True)
        for param, meta_grad in zip(nm_params, meta_nm_grads):
            if param.grad is not None:
                param.grad += meta_grad.detach()
            else:
                param.grad = meta_grad.detach()

        # PN meta gradients
        pn_params = [p for p in fpn.parameters() if p.requires_grad]
        meta_pn_grads = torch.autograd.grad(loss, pn_params, allow_unused=True)
        pn_params = [p for p in self.pn.parameters() if p.requires_grad]
        for param, meta_grad in zip(pn_params, meta_pn_grads):
            if param.grad is not None:
                param.grad += meta_grad.detach()
            else:
                param.grad = meta_grad.detach()

    def update_support_tracker(self, loss, pred, labels):
        self.tracker["support_loss"].append(loss.item())
        self.tracker["support_predictions"].extend(pred.tolist())
        self.tracker["support_labels"].extend(labels.tolist())

    def update_query_tracker(self, loss, pred, labels):
        self.tracker["query_loss"].append(loss.item())
        self.tracker["query_predictions"].extend(pred.tolist())
        self.tracker["query_labels"].extend(labels.tolist())

    def reset_tracker(self):
        self.tracker = {
            "support_loss": [],
            "support_predictions": [],
            "support_labels": [],
            "query_loss": [],
            "query_predictions": [],
            "query_labels": []
        }

    def log(self):
        support_loss = np.mean(self.tracker["support_loss"])
        query_loss = np.mean(self.tracker["query_loss"])
        support_metrics = model_utils.calculate_metrics(self.tracker["support_predictions"], self.tracker["support_labels"])
        query_metrics = model_utils.calculate_metrics(self.tracker["query_predictions"], self.tracker["query_labels"])

        self.logger.info(
            f"Episode {self.current_iter + 1} Support set: Loss = {support_loss:.4f}, "
            f"accuracy = {support_metrics['accuracy']:.4f}, precision = {support_metrics['precision']:.4f}, "
            f"recall = {support_metrics['recall']:.4f}, F1 score = {support_metrics['f1']:.4f}"
        )
        self.logger.info(
            f"Episode {self.current_iter + 1} Query set: Loss = {query_loss:.4f}, "
            f"accuracy = {query_metrics['accuracy']:.4f}, precision = {query_metrics['precision']:.4f}, "
            f"recall = {query_metrics['recall']:.4f}, F1 score = {query_metrics['f1']:.4f}"
        )
        if self.config.wandb:
            wandb.log({
                "support_accuracy": support_metrics['accuracy'],
                "support_precision": support_metrics['precision'],
                "support_recall": support_metrics['recall'],
                "support_f1": support_metrics['f1'],
                "support_loss": support_loss,
                "query_accuracy": query_metrics['accuracy'],
                "query_precision": query_metrics['precision'],
                "query_recall": query_metrics['recall'],
                "query_f1": query_metrics['f1'],
                "query_loss": query_loss,
                "examples_seen": self.examples_seen()
            })
        self.reset_tracker()

    def evaluate(self, dataloader):
        if self.config.learner.evaluation_support_set:
            support_set = []
            for _ in range(self.config.updates):
                text, labels = self.memory.read_batch(batch_size=self.mini_batch_size)
                support_set.append((text, labels))

        with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                copy_initial_weights=False,
                                track_higher_grads=False) as (fpn, diffopt):
            if self.config.learner.evaluation_support_set:
                self.set_train()
                prediction_network = fpn
                # Inner loop
                task_predictions, task_labels = [], []
                support_loss = []
                for text, labels in support_set:
                    labels = torch.tensor(labels).to(self.device)
                    output = self.forward(text, labels, fpn)
                    loss = self.loss_fn(output["logits"], labels)
                    diffopt.step(loss)

                    pred = model_utils.make_prediction(output["logits"].detach())
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(labels.tolist())
                results = model_utils.calculate_metrics(task_predictions, task_labels)
                self.logger.info("Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                            "F1 score = {:.4f}".format(np.mean(support_loss), results["accuracy"],
                            results["precision"], results["recall"], results["f1"]))
                self.set_eval()
            else:
                prediction_network = self.pn

            all_losses, all_predictions, all_labels = [], [], []
            for i, (text, labels, datasets) in enumerate(dataloader):
                labels = torch.tensor(labels).to(self.device)
                output = self.forward(text, labels, prediction_network, no_grad=True)
                loss = self.loss_fn(output["logits"], labels)
                loss = loss.item()
                pred = model_utils.make_prediction(output["logits"].detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                # if i % 20 == 0:
                #     self.logger.info(f"Batch {i + 1}/{len(dataloader)} processed")

        results = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), results["accuracy"],
                    results["precision"], results["recall"], results["f1"]))
        return results

    def model_state(self):
        return {"nm": self.nm.state_dict(),
                "pn": self.pn.state_dict()}

    def optimizer_state(self):
        return self.meta_optimizer.state_dict()

    def load_model_state(self, checkpoint):
        self.nm.load_state_dict(checkpoint["model_state"]["nm"])
        self.pn.load_state_dict(checkpoint["model_state"]["pn"])

    def load_optimizer_state(self, checkpoint):
        self.meta_optimizer.load_state_dict(checkpoint["optimizer"])

    def save_other_state_information(self, state):
        """Any learner specific state information is added here"""
        state["memory"] = self.memory
        return state

    def load_other_state_information(self, checkpoint):
        self.memory = checkpoint["memory"]

    def get_support_set(self, data_iterator):
        """Return a list of datapoints, and return None if the end of the data is reached."""
        support_set = []
        for _ in range(self.config.updates):
            try:
                text, labels, _ = next(data_iterator)
                support_set.append((text, labels))
            except StopIteration:
                self.logger.info("Terminating training as all the data is seen")
                return None
        return support_set

    def get_query_set(self, data_iterator, replay_freq, replay_steps):
        """Return a list of datapoints, and return None if the end of the data is reached."""
        query_set = []
        if self.replay_rate != 0 and (self.current_iter + 1) % replay_freq == 0:
            # now we replay from memory
            for _ in range(replay_steps):
                text, labels = self.memory.read_batch(batch_size=self.mini_batch_size)
                query_set.append((text, labels))
        else:
            # otherwise simply use next batch from data stream as query set
            try:
                text, labels, _ = next(data_iterator)
                query_set.append((text, labels))
                self.memory.write_batch(text, labels)
                self._examples_seen += self.mini_batch_size
            except StopIteration:
                self.logger.info("Terminating training as all the data is seen")
                return None
        return query_set

    def set_eval(self):
        self.nm.eval()
        self.pn.eval()

    def set_train(self):
        self.nm.train()
        self.pn.train()

    def few_shot_testing(self, datasets):
        """
        Allow the model to train on a small amount of datapoints at a time. After every training step,
        evaluate on many samples that haven't been seen yet.

        Results are saved in learner's `metrics` attribute.

        Parameters
        ---
        datasets: Dict[str, Dataset]
            Maps a dataset name to its corresponding dataset object. Should not contain training data.
        """
        all_predictions, all_labels = [], []
        # TODO: evaluate on all datasets instead of just one.
        dataset = datasets[self.config.testing.eval_dataset]

        self.logger.info(f"few shot testing on dataset {self.config.testing.eval_dataset} "
                         f"with {self.config.testing.n_samples} samples")

        # split into training and testing point, assumes there is no meaningful difference in dataset order
        train_dataset = dataset.new(0, self.config.testing.n_samples)
        test_dataset = dataset.new(self.config.testing.n_samples, -1)
        # sample a subset so validation doesn't take too long
        test_dataset = test_dataset.sample(min(self.config.testing.few_shot_validation_size, len(test_dataset)))
        self.logger.info(f"Validating with test set of size {len(test_dataset)}")
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.testing.few_shot_batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.mini_batch_size, shuffle=False)

        all_predictions, all_labels = [], []

        zero_shot = {
            # zero shot accuracy
            "examples_seen": 0,
            "accuracy": self.evaluate(dataloader=test_dataloader)["accuracy"],
            "task": self.config.testing.eval_dataset
        }
        if self.config.wandb:
            wandb.log({
                "few_shot_accuracy": zero_shot["accuracy"],
                "examples_seen": 0
            })
        self.metrics["evaluation"]["few_shot"] = [zero_shot]
        self.metrics["evaluation"]["few_shot_training"] = []

        # Inner loop
        for i, (text, labels, datasets) in enumerate(train_dataloader):
            labels = torch.tensor(labels).to(self.device)
            output = self.forward(text, labels, self.pn)
            loss = self.loss_fn(output["logits"], labels)
            self.inner_optimizer.zero_grad()
            loss.backward()
            self.inner_optimizer.step()

            predictions = model_utils.make_prediction(output["logits"].detach())

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            online_metrics = model_utils.calculate_metrics(predictions.tolist(), labels.tolist())
            dataset_results = self.evaluate(dataloader=test_dataloader)

            train_results = {
                "examples_seen": i * self.config.testing.few_shot_batch_size,
                "accuracy": online_metrics["accuracy"],
                "task": datasets[0]  # assume whole batch is from same task
            }
            test_results = {
                "examples_seen": (i + 1) * self.config.testing.few_shot_batch_size,
                "accuracy": dataset_results["accuracy"],
                "task": datasets[0]
            }
            self.metrics["evaluation"]["few_shot_training"].append(train_results)
            self.metrics["evaluation"]["few_shot"].append(test_results)
            if self.config.wandb:
                wandb.log(train_results)
                wandb.log(test_results)
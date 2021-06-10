import os
from pickle import load
import sys
import logging
from typing import Tuple
from pathlib import Path
import time
import random
import math
import collections
from filelock import FileLock
import json

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, open_dict
import wandb

import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.datasets.text_classification_dataset import SAMPLE_SEED, alternating_order, get_continuum, get_datasets, ClassificationDataset,\
                                                                   datasets_dict, n_samples_order

# plt.style.use("seaborn-paper")
CHECKPOINTS = Path("model-checkpoints/")
LOGS = Path("tensorboard-logs/")
RESULTS_DIR = Path("results")
EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_IDS = Path(hydra.utils.to_absolute_path("experiment_ids.csv"))
METRICS_FILE = "metrics.json"
BEST_MODEL_FNAME = "best-model.pt"


class Learner:
    """Base Learner class.

    A learner implements a specific learning behavior. This is implemented in the specific
    learner's training and testing functions. This base class is used to initialize the experiment,
    and also contains logic for standard operations such as validating, checkpointing, etc., but this
    can be overridden by subclasses.
    All the parameters that drive the experiment behaviour are specified in a config dictionary.
    """

    def __init__(self, config, experiment_path=None):
        """Instantiate a trainer for autopunctuation models.

        Parameters
        ----------
        config:
            dict of parameters that drive the training behaviour.
        experiment_dir:
            path to experiment directory if it already exists
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # experiment_path is only explicitly supplied during an evaluation run
        if experiment_path is None:
            experiment_path = os.getcwd()  # hydra changes the runtime to the experiment folder
        # Experiment output directory
        self.exp_dir = Path(experiment_path)

        # weights and biases
        if config.wandb:
            with open_dict(config):
                config["exp_dir"] = self.exp_dir.as_posix()
            if config.name is None:
                config.name = "unnamed"
            experiment_id = update_experiment_ids(config)
            while True:
                try:
                    self.wandb_run = wandb.init(project="relearning", config=flatten_dict(config),
                               name=f"{experiment_id['name']}-{experiment_id['id']}", reinit=True)
                    break
                except Exception as e:
                    self.logger.info("wandb initialization failed. Retrying..")
                    time.sleep(10)

        # Checkpoint directory to save models
        self.checkpoint_dir = self.exp_dir / CHECKPOINTS
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_exists = len(list(self.checkpoint_dir.glob("*"))) > 0

        # data directory using original working directory
        self.data_dir = hydra.utils.to_absolute_path("data")

        # Tensorboard log directory
        self.log_dir = self.exp_dir / LOGS
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Test and evaluation results saved here
        self.results_dir = self.exp_dir / RESULTS_DIR
        if (self.results_dir / "results.json").is_file() and not "evaluate" in self.config and not self.config.name == "test":
            raise Exception(f"This experiment directory, {self.exp_dir}, already exists. Use a different experiment name "
                            "or different seed. If evaluating, specify the evaluate flag when running the program."
            )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if config.debug_logging:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info("-" * 50)
        self.logger.info("TRAINING LOG")
        self.logger.info("-" * 50)

        self.logger.info("-" * 50 + "\n" + f"CONFIG:\n{self.config}\n" + "-" * 50)

        # if checkpoint_exists:
        #     self.logger.info(f"Checkpoint for {self.exp_dir.name} ALREADY EXISTS. Continuing training.")

        self.logger.info(f"Setting seed: {self.config.seed}")
        np.random.seed(SAMPLE_SEED) # numpy only used for sampling data, which needs to stay equal over different runs
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self._examples_seen = 0
        self.best_accuracy = 0.
        self.best_loss = float("inf")

        self.device = config.training.device
        self.logger.info(f"Using device: {self.config.training.device}")
        self.mini_batch_size = config.training.batch_size
        self.log_freq = config.training.log_freq
        self.validate_freq = config.training.validate_freq
        self.type = config.learner.type

        # replay attributes
        self.write_prob = config.learner.write_prob
        self.replay_rate = config.learner.replay_rate
        self.replay_every = config.learner.replay_every

        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        # if checkpoint_exists:
        #     last_checkpoint = get_last_checkpoint_path(self.checkpoint_dir)
        #     self.logger.info(f"Loading model checkpoint from {last_checkpoint}")
        #     self.load_checkpoint(last_checkpoint.name)

        # this is used to track metrics of different tasks during training
        self.metrics = collections.defaultdict(dict)
        self.metrics["online"] = []
        # used to know when to log to first encounter metrics
        self.eval_task_first_encounter = True
        self.metrics["eval_task_first_encounter"] = []
        self.metrics["replay_samples_seen"] = 0
        # keeps track of how many times we have performed few shot testing, for logging purposes
        self.few_shot_counter = 0
        self.reset_tracker()

        # specifies which task the learner is currently learning
        # it is up to the specific learner to update this
        self.previous_task = None
        self.current_task = None

    def validate(self, datasets, n_samples=100, log=True):
        """
        Evaluate model performance on a dataset.

        Can be called throughout the course of training. Writes results
        to the learner's attribute `self.metrics`.

        Parameters
        ---
        datasets: Dict[str, Dataset]
            Maps task to a validation dataset.
        log: bool
            If true, log to metrics attribute and wandb
        """
        to_log = {"examples_seen": self.examples_seen()}
        result = {}
        for task, dataset in datasets.items():
            subset = ClassificationDataset(name=task, data=datasets[task].data.sample(n_samples))
            dl = DataLoader(subset, batch_size=self.mini_batch_size)
            performance = self.evaluate(dl)
            result[task] = performance
            if log:
                if "performance" not in self.metrics[task]:
                    self.metrics[task]["performance"] = []
                self.metrics[task]["performance"].append({
                    "performance": performance,
                    "examples_seen": self.examples_seen()
                })
            to_log[task + "_" + "train_val_acc"] = performance["accuracy"]
        if self.config.wandb:
            wandb.log(to_log)
        return result

    def testing(self, datasets, order):
        """
        Evaluate the learner after training.

        Parameters
        ---
        datasets: Dict[str, List[Dataset]]
            Test datasets.
        order: List[str]
            Specifies order of encountered datasets
        """
        self.logger.info("Testing..")
        train_datasets = datasets_dict(datasets["train"], order)
        for split in ("test", "val"):
            self.logger.info(f"Validating on split {split}")
            eval_datasets = datasets[split]
            eval_datasets = datasets_dict(eval_datasets, order)
            self.set_eval()

            if self.config.testing.average_accuracy:
                self.logger.info("getting average accuracy")
                self.average_accuracy(eval_datasets, split=split, train_datasets=train_datasets)

            if self.config.testing.few_shot:
                # split into training and testing point, assumes there is no meaningful difference in dataset order
                dataset = eval_datasets[self.config.testing.eval_dataset]
                train_dataset = dataset.new(0, self.config.testing.n_samples)
                eval_dataset = dataset.new(self.config.testing.n_samples, -1)
                # sample a subset so validation doesn't take too long
                eval_dataset = eval_dataset.sample(min(self.config.testing.few_shot_validation_size, len(eval_dataset)))
                self.logger.info(f"Few shot eval dataset size: {len(eval_dataset)}")
                self.few_shot_testing(train_dataset=train_dataset, eval_dataset=eval_dataset, split=split)

    def average_accuracy(self, datasets, split, train_datasets=None):
        results = {}
        accuracies, precisions, recalls, f1s = [], [], [], []

        # if train_datasets is not None and self.config.testing.n_samples_before_average_evaluate > 0:
        #     train_dataloader = DataLoader(ConcatDataset(train_datasets.values()), shuffle=True)
        #     # TODO: finish this
        for dataset_name, dataset in datasets.items():
            self.logger.info("Testing on {}".format(dataset_name))
            if self.config.testing.average_validation_size is not None:
                dataset = dataset.sample(min(len(dataset), self.config.testing.average_validation_size))
            test_dataloader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False)
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
        metrics_name = split + "_evaluation"
        self.metrics[metrics_name]["individual"] = results
        self.metrics[metrics_name]["average"] = mean_results
        if self.config.wandb:
            wandb.log({
                split + "_testing_average_accuracy": mean_results["accuracy"]
            })

    def few_shot_testing(self, train_dataset, eval_dataset, increment_counters=False, split="test"):
        """
        Allow the model to train on a small amount of datapoints at a time. After every training step,
        evaluate on many samples that haven't been seen yet.

        Results are saved in learner's `metrics` attribute.

        Parameters
        ---
        train_dataset: Dataset
            Contains examples on which the model is trained before being evaluated
        eval_dataset: Dataset
            Contains examples on which the model is evaluated
        increment_counters: bool
            If True, update online metrics and current iteration counters.
        split: str, one of {"val", "test"}.
            Which data split is used. For logging purposes.
        """
        self.logger.info(f"few shot testing on dataset {self.config.testing.eval_dataset} "
                         f"with {len(train_dataset)} samples")
        # whenever we do few shot evaluation, we reset the learning to before the evaluation started
        temp_checkpoint = "few_shot_temp_checkpoint.pt"
        self.save_checkpoint(file_name=temp_checkpoint, save_optimizer_state=True, delete_previous=False)
        train_dataloader, eval_dataloader = self.few_shot_preparation(train_dataset, eval_dataset, split=split)
        all_predictions, all_labels = [], []

        for i, (text, labels, datasets) in enumerate(train_dataloader):
            output = self.training_step(text, labels)
            predictions = model_utils.make_prediction(output["logits"].detach())
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

            dataset_results = self.evaluate(dataloader=eval_dataloader)
            self.log_few_shot(all_predictions, all_labels, datasets, dataset_results, increment_counters,
                              text, i, split=split)
            if (i * self.config.testing.few_shot_batch_size) % self.mini_batch_size == 0 and i > 0:
                all_predictions, all_labels = [], []
        self.load_checkpoint(temp_checkpoint, load_optimizer_state=True)
        # delete temp checkpoint
        (self.checkpoint_dir / temp_checkpoint).unlink()
        self.few_shot_counter += 1

    def few_shot_preparation(self, train_dataset, eval_dataset, split="test"):
        """Few shot preparation code that isn't specific to any learner"""
        self.logger.info(f"Few shot evaluation number {self.few_shot_counter}")
        metrics_entry = split + "_evaluation"
        if "few_shot" not in self.metrics[metrics_entry]:
            self.metrics[metrics_entry]["few_shot"] = []
            self.metrics[metrics_entry]["few_shot_training"] = []
        # TODO: evaluate on all datasets instead of just one.
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.testing.few_shot_batch_size, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.mini_batch_size, shuffle=False)

        # split into training and testing point, assumes there is no meaningful difference in dataset order
        self.logger.info(f"Validating with test set of size {len(eval_dataset)}")
        self.metrics[metrics_entry]["few_shot"].append([])
        self.metrics[metrics_entry]["few_shot_training"].append([])
        self.logger.info(f"Length of few shot metrics {len(self.metrics[metrics_entry]['few_shot'])}")
        
        return train_dataloader, eval_dataloader

    def log_few_shot(self, all_predictions, all_labels, datasets, dataset_results, increment_counters, text,
                     few_shot_batch, split="test"):
        """Few shot preparation code that isn't specific to any learner"""
        metrics_entry = split + "_evaluation"
        test_results = {
            "examples_seen": few_shot_batch * self.config.testing.few_shot_batch_size,
            "examples_seen_total": self.examples_seen(),
            "accuracy": dataset_results["accuracy"],
            "task": datasets[0]
        }
        if (few_shot_batch * self.config.testing.few_shot_batch_size) % self.mini_batch_size == 0 and few_shot_batch > 0:
            online_metrics = model_utils.calculate_metrics(all_predictions, all_labels)
            train_results = {
                "examples_seen": few_shot_batch * self.config.testing.few_shot_batch_size,
                "examples_seen_total": self.examples_seen(),
                "accuracy": online_metrics["accuracy"],
                "task": datasets[0]  # assume whole batch is from same task
            }
            self.metrics[metrics_entry]["few_shot_training"][-1].append(train_results)
            if increment_counters:
                self.metrics["online"].append({
                    "accuracy": online_metrics["accuracy"],
                    "examples_seen": self.examples_seen(),
                    "task": datasets[0]
                })
        if increment_counters:
            self._examples_seen += len(text)
        self.metrics[metrics_entry]["few_shot"][-1].append(test_results)
        self.write_metrics()
        if self.config.wandb:
            # replace with new name
            test_results = test_results.copy()
            test_results[f"few_shot_{split}_accuracy_{self.few_shot_counter}"] = test_results.pop("accuracy")
            wandb.log(test_results)

    def reset_tracker(self):
        """Initialize dictionary that stores information about loss, predictions, and labels
        during training, for logging purposes.
        """
        self.tracker = {
            "losses": [],
            "predictions": [],
            "labels": []
        }

    def update_tracker(self, output, predictions, labels):
        self.tracker["losses"].append(output["loss"])
        self.tracker["predictions"].extend(predictions.tolist())
        self.tracker["labels"].extend(labels.tolist())

    def replay_parameters(self, metalearner=True):
        """Calculate replay frequency and number of steps"""
        if self.replay_rate != 0:
            replay_freq = self.replay_every // self.mini_batch_size
            if metalearner:
                replay_freq = int(math.ceil((replay_freq + 1) / (self.config.learner.updates + 1)))
            replay_steps = max(int(self.replay_every * self.replay_rate / self.mini_batch_size), 1)
        else:
            replay_freq = 0
            replay_steps = 0
        return replay_freq, replay_steps

    def prepare_data(self, datasets):
        """Deal with making data ready for consumption.
        
        Parameters
        ---
        datasets: Dict[str, List of dataset names]

        Returns:
            tuple:

        """
        # train_datasets = {dataset_name: dataset for dataset_name, dataset in zip(datasets["order"], datasets["train"])}
        train_datasets = datasets_dict(datasets["train"], datasets["order"])
        val_datasets = datasets_dict(datasets["test"], datasets["order"])
        eval_dataset = val_datasets[self.config.testing.eval_dataset]
        if self.config.testing.few_shot:
            # split into training and testing point, assumes there is no meaningful difference in dataset order
            eval_train_dataset = eval_dataset.new(0, self.config.testing.n_samples)
            eval_eval_dataset = eval_dataset.new(self.config.testing.n_samples, -1)
            # sample a subset so validation doesn't take too long
            eval_eval_dataset = eval_eval_dataset.sample(min(self.config.testing.few_shot_validation_size, len(eval_dataset)))

        if self.config.data.alternating_order:
            order, n_samples = alternating_order(train_datasets, tasks=self.config.data.alternating_tasks,
                                                 n_samples_per_switch=self.config.data.alternating_n_samples_per_switch,
                                                 relative_frequencies=self.config.data.alternating_relative_frequencies)
        else:
            n_samples, order = n_samples_order(self.config.learner.samples_per_task, self.config.task_order, datasets["order"])
        datas = get_continuum(train_datasets, order=order, n_samples=n_samples,
                             eval_dataset=self.config.testing.eval_dataset, merge=False)
        return datas, order, n_samples, eval_train_dataset, eval_eval_dataset

    def get_support_set(self, data_iterator, max_sample):
        """Return a list of datapoints, and return None if the end of the data is reached.
        
        max_sample indicates the maximum amount of samples able to be drawn within one task observation.
        If it is reached, don't add more samples."""
        support_set = []
        for _ in range(self.config.learner.updates):
            try:
                text, labels, datasets = next(data_iterator)
                support_set.append((text, labels))
                self.episode_samples_seen += len(text)
            except StopIteration:
                # self.logger.info("Terminating training as all the data is seen")
                return None, None
            if self.episode_samples_seen >= max_sample:
                break
        return support_set, datasets[0]

    def get_query_set(self, data_iterator, replay_freq, replay_steps, max_sample):
        """Return a list of datapoints, and return None if the end of the data is reached."""
        query_set = []
        if self.replay_rate != 0 and (self.current_iter + 1) % replay_freq == 0:
            # now we replay from memory
            self.logger.debug("query set read from memory")
            for _ in range(replay_steps):
                text, labels = self.memory.read_batch(batch_size=self.mini_batch_size)
                query_set.append((text, labels))
                self.episode_samples_seen += len(text)
                self.metrics["replay_samples_seen"] += len(text)
                if self.episode_samples_seen >= max_sample:
                    break
        else:
            # otherwise simply use next batch from data stream as query set
            try:
                text, labels, _ = next(data_iterator)
                query_set.append((text, labels))
                self.memory.write_batch(text, labels)
                self.episode_samples_seen += len(text)
            except StopIteration:
                # self.logger.info("Terminating training as all the data is seen")
                return None
        return query_set

    def meta_training_log(self):
        """Logs data during training for meta learners."""
        support_loss = np.mean(self.tracker["support_loss"])
        query_loss = np.mean(self.tracker["query_loss"])
        support_metrics = model_utils.calculate_metrics(self.tracker["support_predictions"], self.tracker["support_labels"])
        query_metrics = model_utils.calculate_metrics(self.tracker["query_predictions"], self.tracker["query_labels"])

        self.logger.debug(
            f"Episode {self.current_iter + 1} Support set: Loss = {support_loss:.4f}, "
            f"accuracy = {support_metrics['accuracy']:.4f}, precision = {support_metrics['precision']:.4f}, "
            f"recall = {support_metrics['recall']:.4f}, F1 score = {support_metrics['f1']:.4f}"
        )
        self.logger.debug(
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

    def save_checkpoint(self, file_name: str = None, save_optimizer_state=None, delete_previous=False):
        """Save checkpoint in the checkpoint directory.

        Checkpoint directory and checkpoint file need to be specified in the configs.

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        if file_name is None:
            file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"

        file_name = self.checkpoint_dir / file_name
        state = {
            "iter": self.current_iter,
            "best_accuracy": self.best_accuracy,
            "best_loss": self.best_loss,
            "model_state": self.model_state()
        }
        if (save_optimizer_state is not None and save_optimizer_state) or self.config.save_optimizer_state:
            state["optimizer"] = self.optimizer_state()
        state = self.save_other_state_information(state)
        # delete previous checkpoint to avoid hogging space
        if delete_previous or self.config.delete_previous_checkpoint:
            previous_checkpoint = self.get_last_checkpoint_path()
            if previous_checkpoint is not None and previous_checkpoint.is_file():
                previous_checkpoint.unlink()
        torch.save(state, file_name)
        self.write_metrics()
        self.logger.info(f"Checkpoint saved @ {file_name}")

    def save_other_state_information(self, state):
        """Any learner specific state information is added here"""
        return state

    def load_checkpoint(self, file_name: str = None, load_optimizer_state=False):
        """Load the checkpoint with the given file name

        Checkpoint must contain:
            - current epoch
            - current iteration
            - model state
            - best accuracy achieved so far
            - optimizer state

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        try:
            if file_name is None:
                file_name = self.get_last_checkpoint_path()
            else:
                file_name = self.checkpoint_dir / file_name
            self.logger.info(f"Loading checkpoint from {file_name}")
            checkpoint = torch.load(file_name, self.config.training.device)

            # self.current_epoch = checkpoint["epoch"]
            self.current_iter = checkpoint["iter"]
            self.best_accuracy = checkpoint["best_accuracy"]
            self.best_loss = checkpoint["best_loss"]
            self.load_model_state(checkpoint)
            if load_optimizer_state or self.config.save_optimizer_state:
                self.load_optimizer_state(checkpoint)
            self.load_other_state_information(checkpoint)
        except OSError:
            self.logger.error(f"No checkpoint exists @ {self.checkpoint_dir}")

    def model_state(self):
        return self.model.state_dict()

    def optimizer_state(self):
        return self.optimizer.state_dict()

    def load_model_state(self, checkpoint):
        self.model.load_state_dict(checkpoint["model_state"])

    def load_optimizer_state(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def load_other_state_information(self, checkpoint):
        pass

    def get_datasets(self, data_dir, order):
        return get_datasets(data_dir, order)

    def time_checkpoint(self):
        """Save a checkpoint when a certain amount of time has elapsed."""
        curtime = time.time()
        deltatime = curtime - self.last_checkpoint_time
        # if current time is more than save_freq minutes from last checkpoint, save a checkpoint
        if self.config.checkpoint_while_training and deltatime >= self.config.save_freq * 60:
            self.logger.info(f"{deltatime / 60:.1f} minutes have elapsed, saving checkpoint")
            self.save_checkpoint()
            self.last_checkpoint_time = curtime

    def get_last_checkpoint_path(self, checkpoint_dir=None):
        """Return path of the latest checkpoint in a given checkpoint directory."""
        checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else self.checkpoint_dir
        paths = list(checkpoint_dir.glob("Epoch*"))
        if len(paths) > 0:
            # parse epochs and steps from path names
            epochs = []
            steps = []
            for path in paths:
                epoch, step = path.stem.split("-")
                epoch = int(epoch.split("[")[-1][:-1])
                step = int(step.split("[")[-1][:-1])
                epochs.append(epoch)
                steps.append(step)
            # sort first by epoch, then by step
            last_model_ix = np.lexsort((steps, epochs))[-1]
            return paths[last_model_ix]

    def examples_seen(self):
        return self._examples_seen

    def write_metrics(self):
        with open(self.results_dir / METRICS_FILE, "w") as f:
            json.dump(self.metrics, f)

    def set_train(self):
        """Set underlying pytorch network to train mode.
        
        If learner has multiple models, this method should be overwritten.
        """
        self.model.train()

    def set_eval(self):
        """Set underlying pytorch network to evaluation mode.
        
        If learner has multiple models, this method should be overwritten.
        """
        self.model.eval()


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = str(parent_key) + str(sep) + str(k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def update_experiment_ids(config):
    """Make sure each named wandb run has a counter so as to avoid duplicates."""
    # lock to avoid concurrent reading
    lock = FileLock(EXPERIMENT_IDS.as_posix() + ".lock", timeout=10)

    with lock:
        if not EXPERIMENT_IDS.exists():
            with open(EXPERIMENT_IDS, "w") as f:
                f.write("name,id\n")
                f.write(f"{config.name}, 0\n")
        experiment_ids_df = pd.read_csv(EXPERIMENT_IDS)
        in_data = (experiment_ids_df["name"] == config.name).any()  # check if name is in experiments list
        if in_data:
            # increment counter for this experiment name to avoid duplicate names
            id = experiment_ids_df.loc[experiment_ids_df["name"] == config.name, "id"].values[0] + 1
            experiment_ids_df.loc[experiment_ids_df["name"] == config.name, "id"] = id
            experiment_id = {"name": config.name, "id": id}
        else:
            experiment_id = {"name": config.name, "id": 1}
            experiment_ids_df = experiment_ids_df.append(experiment_id, ignore_index=True)
        experiment_ids_df.to_csv(EXPERIMENT_IDS, index=False)
        return experiment_id

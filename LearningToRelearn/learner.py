import os
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
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, open_dict
import wandb

import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.datasets.text_classification_dataset import get_datasets, ClassificationDataset,\
                                                                   datasets_dict
from LearningToRelearn.datasets.utils import batch_encode

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
        np.random.seed(self.config.seed)
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

        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        # if checkpoint_exists:
        #     last_checkpoint = get_last_checkpoint_path(self.checkpoint_dir)
        #     self.logger.info(f"Loading model checkpoint from {last_checkpoint}")
        #     self.load_checkpoint(last_checkpoint.name)

        # this is used to track metrics of different tasks during training
        self.metrics = collections.defaultdict(dict)
        self.metrics["online"] = []
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

    def testing(self, datasets, order, split="val"):
        """
        Evaluate the learner after training.

        Parameters
        ---
        datasets: Dict[str, List[Dataset]]
            Test datasets.
        order: List[str]
            Specifies order of encountered datasets
        """
        datasets = datasets[split]
        eval_datasets = datasets_dict(datasets, order)
        self.set_eval()

        if self.config.testing.average_accuracy:
            self.average_accuracy(eval_datasets)
        if self.config.testing.few_shot:
            # split into training and testing point, assumes there is no meaningful difference in dataset order
            dataset = eval_datasets[self.config.testing.eval_dataset]
            train_dataset = dataset.new(0, self.config.testing.n_samples)
            eval_dataset = dataset.new(self.config.testing.n_samples, -1)
            # sample a subset so validation doesn't take too long
            eval_dataset = eval_dataset.sample(min(self.config.testing.few_shot_validation_size, len(eval_dataset)))
            self.few_shot_testing(train_dataset=train_dataset, eval_dataset=eval_dataset)

    def average_accuracy(self, datasets):
        results = {}
        accuracies, precisions, recalls, f1s = [], [], [], []
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
        self.metrics["evaluation"]["individual"] = results
        self.metrics["evaluation"]["average"] = mean_results
        if self.config.wandb:
            wandb.log({
                "testing_average_accuracy": mean_results["accuracy"]
            })

    def few_shot_testing(self, train_dataset, eval_dataset, increment_counters=False):
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
        """
        if "few_shot" not in self.metrics["evaluation"]:
            self.metrics["evaluation"]["few_shot"] = []
            self.metrics["evaluation"]["few_shot_training"] = []
        all_predictions, all_labels = [], []
        # TODO: evaluate on all datasets instead of just one.
        self.logger.info(f"few shot testing on dataset {self.config.testing.eval_dataset} "
                         f"with {len(train_dataset)} samples")

        self.logger.info(f"Validating with test set of size {len(eval_dataset)}")
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.testing.few_shot_batch_size, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.mini_batch_size, shuffle=False)

        all_predictions, all_labels = [], []

        zero_shot = {
            # zero shot accuracy
            "examples_seen": 0,
            "accuracy": self.evaluate(dataloader=eval_dataloader)["accuracy"],
            "task": self.config.testing.eval_dataset
        }
        if self.config.wandb:
            wandb.log({
                "few_shot_accuracy": zero_shot["accuracy"],
                "examples_seen": 0
            })
        self.metrics["evaluation"]["few_shot"].append([zero_shot])
        self.metrics["evaluation"]["few_shot_training"].append([])

        for i, (text, labels, datasets) in enumerate(train_dataloader):
            output = self.training_step(text, labels)
            predictions = model_utils.make_prediction(output["logits"].detach())

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            online_metrics = model_utils.calculate_metrics(predictions.tolist(), labels.tolist())
            dataset_results = self.evaluate(dataloader=eval_dataloader)

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
            if increment_counters:
                self._examples_seen += len(text)
                self.metrics["online"].append({
                    "accuracy": online_metrics["accuracy"],
                    "examples_seen": self.examples_seen(),
                })
            self.metrics["evaluation"]["few_shot_training"][-1].append(train_results)
            self.metrics["evaluation"]["few_shot"][-1].append(test_results)
            if self.config.wandb:
                train_results[f"few_shot_training_accuracy_{self.few_shot_counter}"] = train_results.pop("accuracy")
                test_results[f"few_shot_test_accuracy_{self.few_shot_counter}"] = test_results.pop("accuracy")
                wandb.log(train_results)
                wandb.log(test_results)
        self.few_shot_counter += 1

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

    def replay_parameters(self):
        if self.replay_rate != 0:
            replay_batch_freq = self.replay_every // self.mini_batch_size
            replay_freq = int(math.ceil((replay_batch_freq + 1) / (self.config.learner.updates + 1)))
            replay_steps = int(self.replay_every * self.replay_rate / self.mini_batch_size)
        else:
            replay_freq = 0
            replay_steps = 0
        return replay_freq, replay_steps

    def save_checkpoint(self, file_name: str = None):
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
        if self.config.save_optimizer_state:
            state["optimizer"] = self.optimizer_state()
        state = self.save_other_state_information(state)
        # delete previous checkpoint to avoid hogging space
        if self.config.delete_previous_checkpoint:
            previous_checkpoint = self.get_last_checkpoint_path()
            if previous_checkpoint is not None and previous_checkpoint.is_file():
                previous_checkpoint.unlink()
        torch.save(state, file_name)
        self.logger.info(f"Checkpoint saved @ {file_name}")

    def save_other_state_information(self, state):
        """Any learner specific state information is added here"""
        return state

    def load_checkpoint(self, file_name: str = None):
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
            if self.config.save_optimizer_state:
                self.load_optimizer_state(checkpoint)
            self.load_other_state_information(checkpoint)
        except OSError:
            self.logger.error(f"No checkpoint exists @ {self.checkpoint_dir}")
        try:
            with open(self.results_dir / METRICS_FILE) as f:
                self.metrics = json.load(f)
        except OSError:
            self.logger.error("Failed loading metrics file")

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

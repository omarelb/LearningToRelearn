import os
import sys
import logging
from typing import Tuple
from pathlib import Path
import time
import random
import math
import collections

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import wandb

from MetaLifeLongLanguage.datasets.text_classification_dataset import get_datasets
from MetaLifeLongLanguage.datasets.utils import batch_encode

# plt.style.use("seaborn-paper")
CHECKPOINTS = Path("model-checkpoints/")
LOGS = Path("tensorboard-logs/")
RESULTS = Path("results")
EXPERIMENT_DIR = Path("experiments")
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

        # weights and biases
        if config.wandb:
            while True:
                try:
                    wandb.init(project="relearning", config=flatten_dict(config), name=config.name)
                    break
                except:
                    self.logger.info("wandb initialization failed. Retrying..")
                    time.sleep(10)

        if experiment_path is None:
            experiment_path = os.getcwd() # hydra changes the runtime to the experiment folder
        # Experiment output directory
        self.exp_dir = Path(experiment_path)

        # Checkpoint directory to save models        
        self.checkpoint_dir = self.exp_dir / CHECKPOINTS
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_exists = len(list(self.checkpoint_dir.glob("*"))) > 0

        # data directory using original working directory
        self.data_dir = hydra.utils.to_absolute_path("data")

        # Tensorboard log directory
        self.log_dir = self.exp_dir / LOGS
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.results_dir = self.exp_dir / RESULTS
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if config.debug:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info("-"*50)
        self.logger.info("TRAINING LOG")
        self.logger.info("-"*50)

        self.logger.info("-"*50 + "\n" + f"CONFIG:\n{self.config}\n" + "-"*50)

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
        self.best_accuracy = 0.
        self.best_loss = float("inf")

        self.device = config.training.device
        self.logger.info(f"Using device: {self.config.training.device}")
        self.mini_batch_size = config.training.batch_size

        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        # if checkpoint_exists:
        #     last_checkpoint = get_last_checkpoint_path(self.checkpoint_dir)
        #     self.logger.info(f"Loading model checkpoint from {last_checkpoint}")
        #     self.load_checkpoint(last_checkpoint.name)

    def testing(self, datasets):
        """
        Parameters
        ---
        datasets: List[Dataset]
            Test datasets.
        """
        accuracies, precisions, recalls, f1s = [], [], [], []
        results = {}
        for dataset in datasets:
            dataset_name = dataset.__class__.__name__
            self.logger.info("Testing on {}".format(dataset_name))
            test_dataloader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False,
                                         collate_fn=batch_encode)
            dataset_results = self.evaluate(dataloader=test_dataloader)
            accuracies.append(dataset_results["accuracy"])
            precisions.append(dataset_results["precision"])
            recalls.append(dataset_results["recall"])
            f1s.append(dataset_results["f1"])
            results[dataset_name] = dataset_results

        self.logger.info("Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))
        return results

    def replay_parameters(self):
        if self.replay_rate != 0:
            replay_batch_freq = self.replay_every // self.mini_batch_size
            replay_freq = int(math.ceil((replay_batch_freq + 1) / (self.config.updates + 1)))
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
            "model_state": self.model_state(),
            "optimizer": self.optimizer_state(),
        }
        state = self.save_other_state_information(state)
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

    def time_metrics(self, data_length):
        time_elapsed = time.time() - self.start_time
        time_per_iteration = time_elapsed / self.log_freq # seconds per iteration
        estimated_time_left = time.strftime('%H:%M:%S', time.gmtime(time_per_iteration * (data_length - (self.current_iter + 1))))
        return time_per_iteration, estimated_time_left

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


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
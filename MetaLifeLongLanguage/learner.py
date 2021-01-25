import os
import sys
import logging
from typing import Tuple
from pathlib import Path
import time
import random

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

from MetaLifeLongLanguage.datasets.text_classification_dataset import get_datasets

# plt.style.use('seaborn-paper')
CHECKPOINTS = Path("model-checkpoints/")
LOGS = Path("tensorboard-logs/")
BEST_MODEL_FNAME = "best-model.pt"


class Learner:
    """Base Learner class.

    A learner implements a specific learning behavior. This is implemented in the specific
    learner's training and testing functions. This base class is used to initialize the experiment,
    and also contains logic for standard operations such as validating, checkpointing, etc., but this
    can be overridden by subclasses.
    All the parameters that drive the experiment behaviour are specified in a config dictionary.
    """

    def __init__(self, config):
        """Instantiate a trainer for autopunctuation models.

        Parameters
        ----------
        config: 
            dict of parameters that drive the training behaviour.
        """
        self.config = config

        experiment_path = os.getcwd() # hydra changes the runtime to the experiment folder
        # Experiment output directory
        self.exp_dir = Path(experiment_path)
        self.logger = logging.getLogger(__name__)

        # Checkpoint directory to save models        
        self.checkpoint_dir = self.exp_dir / CHECKPOINTS
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_exists = len(list(self.checkpoint_dir.glob('*'))) > 0

        # data directory using original working directory
        self.data_dir = hydra.utils.to_absolute_path("data")

        # Tensorboard log directory
        # self.log_dir = self.exp_dir / LOGS
        # self.log_dir.mkdir(parents=True, exist_ok=True)
        # self.writer = SummaryWriter(log_dir=self.log_dir)

        self.logger.info("-"*50)
        self.logger.info("TRAINING LOG")
        self.logger.info("-"*50)

        self.logger.info("-"*50 + '\n' + f'CONFIG:\n{self.config}\n' + "-"*50)

        if checkpoint_exists:
            self.logger.info(f"Checkpoint for {self.exp_dir.name} ALREADY EXISTS. Continuing training.")

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
        self.best_loss = float('inf')

        self.logger.info(f"Using device: {self.config.training.device}")
        # if checkpoint_exists:
        #     last_checkpoint = get_last_checkpoint_path(self.checkpoint_dir)
        #     self.logger.info(f"Loading model checkpoint from {last_checkpoint}")
        #     self.load_checkpoint(last_checkpoint.name)

    def run(self):
        """Run the train-eval loop
        
        If the loop is interrupted manually, finalization will still be executed
        """
        try:
            self.logger.info(f"Begin training for {self.config.training.epochs} epochs")
            self.train()
        except KeyboardInterrupt:
            self.logger.warning("Manual interruption registered. Please wait to finalize...")
            self.finalize()

    def train(self):
        """ Main training loop """
        num_batches = len(self.train_dl)
        n_samples_per_batch = self.config.data.batch_size * self.config.data.block_size
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.logger.info(f'Current epoch: {self.current_epoch + 1} / {self.config.training.epochs}')
            self.current_epoch = epoch
            for i, batch in enumerate(self.train_dl):
                self.current_iter += 1
                t0 = time.time()
                results = self._batch_iteration(batch, training=True)
                time_spent = time.time() - t0

                self.writer.add_scalar('Train/Accuracy', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Train/F1-Score', results['f1_score'], self.current_iter)
                self.writer.add_scalar('Train/Loss', results['loss'], self.current_iter)
                report = (f"EPOCH:{epoch + 1} STEP:{i}/{num_batches}\t"
                          f"Accuracy: {results['accuracy']:.3f} "
                          f"F1-Score: {results['f1_score']:.3f} "
                          f"Speed: {n_samples_per_batch / time_spent :.1f} tokens/s ")
                if i % self.config.training.train_report_freq == 0:
                    self.logger.info(report)
                if i % self.config.training.valid_freq == 0:
                    self.validate()
                if i % self.config.training.save_freq == 0:
                    self.save_checkpoint()
            self.lr_scheduler.step()

    def validate(self):
        """ Main validation loop """
        self.model.eval()
        losses = []
        accuracies = []
        f1_scores = []
        Y, Y_pred = [], []

        self.logger.debug("Begin evaluation over validation set")
        with torch.no_grad():
            for i, batch in enumerate(self.valid_dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
                f1_scores.append(results['f1_score'])
                Y.append(results['y'])
                Y_pred.append(results['y_pred'])
#        "classification_report": metrics.classification_report(
#            y_true=[self.tokenizer.punctuation_decoder[int(x)][1:] for x in y],
#            y_pred=[self.tokenizer.punctuation_decoder[int(x)][1:] for x in y_pred],
#            labels=[punct[1:] for punct in self.tokenizer.punctuation_vocab],
#            output_dict=True)
            
        Y, Y_pred = np.concatenate(Y), np.concatenate(Y_pred)
        punctuation_labels = list(self.tokenizer.punctuation_vocab.keys())
        classification_report = metrics.classification_report(Y, Y_pred, target_names=punctuation_labels, output_dict=True)
        for label_name, numbers in classification_report.items():
            if not isinstance(numbers, dict):
                break
            for metric, number in numbers.items():
                self.writer.add_scalar(f'Valid/{label_name[1:]}/{metric.capitalize()}', number, self.current_iter)
        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)
        mean_f1_score = np.mean(f1_scores)
        # if mean_accuracy > self.best_accuracy:
        #     self.best_accuracy = mean_accuracy
        #     self.save_checkpoint(BEST_MODEL_FNAME)
        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            self.save_checkpoint(BEST_MODEL_FNAME)
        
        self.writer.add_scalar('Valid/Accuracy', mean_accuracy, self.current_iter)
        self.writer.add_scalar('Valid/F1-Score', mean_f1_score, self.current_iter)
        self.writer.add_scalar('Valid/Loss', mean_loss, self.current_iter)
        report = (f"[Validation]\t"
                  f"Accuracy: {mean_accuracy:.3f} "
                  f"F1-Score: {mean_f1_score:.3f} "
                  f"Total Loss: {mean_loss:.3f}")
        self.logger.info(report)

    def test(self):
        """ Model testing and evaluation """

        print("Loading best model checkpoint... ")
        self.load_checkpoint(BEST_MODEL_FNAME)
        self.model.eval()
        losses = []
        Y, Y_pred = [], []

        print("Begin testing...")
        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                results = self._batch_iteration(batch, training=False)
                x = batch[0].to(self.config.training.device)
                y = batch[1].to(self.config.training.device).contiguous().view(-1)
                logits, _, _ = self.model(x)
                logits = logits.contiguous().view(-1, logits.size(-1))
                y_pred = torch.argmax(logits, dim=-1)
                losses.append(self.loss_fn(logits, y).item())
                Y.append(y.cpu().detach().numpy())
                Y_pred.append(y_pred.cpu().detach().numpy())
    
        Y, Y_pred = np.concatenate(Y), np.concatenate(Y_pred)
        loss = np.mean(losses)
        accuracy = metrics.accuracy_score(Y, Y_pred)
        f1_score = metrics.f1_score(Y, Y_pred, average='weighted')
        punctuation_labels = list(self.tokenizer.punctuation_vocab.keys())
        report = metrics.classification_report(Y, Y_pred, target_names=punctuation_labels)
        summary = (f"\n[Test Report]\n"
                   f"Accuracy: {accuracy:.3f} "
                   f"F1-Score: {f1_score:.3f} "
                   f"Loss: {loss:.3f}\n")
        print(summary)
        print(report)

    def _batch_iteration(self, batch: Tuple, training: bool):
        """ Iterate over one batch """
        x = batch[0].to(self.config.training.device)
        y = batch[1].to(self.config.training.device).contiguous().view(-1)
        
        if training:
            self.model.train()
            self.opt.zero_grad()
            logits, _, _ = self.model(x)
            logits = logits.contiguous().view(-1, logits.size(-1))
            loss = self.loss_fn(logits, y)
            loss.backward()
            if self.config.training.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.max_grad_norm)
            self.opt.step()

        else:
            self.model.eval()
            with torch.no_grad():
                logits, _, _ = self.model(x)
                logits = logits.contiguous().view(-1, logits.size(-1))
                loss = self.loss_fn(logits, y)

        y_pred = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        results = {
            "loss": loss.item(),
            "accuracy": metrics.accuracy_score(y, y_pred),
            "f1_score":  metrics.f1_score(y, y_pred, average='weighted'),
            "y": y,
            "y_pred": y_pred
        }
        return results

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
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, file_name)
        self.logger.info(f"Checkpoint saved @ {file_name}")

    def load_checkpoint(self, file_name: str):
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
            file_name = self.checkpoint_dir / file_name
            self.logger.info(f"Loading checkpoint from {file_name}")
            checkpoint = torch.load(file_name, self.config.training.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.best_accuracy = checkpoint['best_accuracy']
            self.best_loss = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        except OSError:
            self.logger.error(f"No checkpoint exists @ {self.checkpoint_dir}")
        
    def finalize(self):
        """Finalize all necessary operations before stopping
        
        Saves checkpoint
        TODO: Upload to sharepoint
        """
        self.save_checkpoint()
        # upload_sharepoint(self.model)

    def get_datasets(self, data_dir, order):
        return get_datasets(data_dir, order)


def get_last_checkpoint_path(checkpoint_dir):
    """Return path of the latest checkpoint in a given checkpoint directory."""
    paths = list(checkpoint_dir.glob('Epoch*'))
    if len(paths) > 0:
        # parse epochs and steps from path names
        epochs = []
        steps = []
        for path in paths:
            epoch, step = path.stem.split('-')
            epoch = int(epoch.split('[')[-1][:-1])
            step = int(step.split('[')[-1][:-1])
            epochs.append(epoch)
            steps.append(step)
        # sort first by epoch, then by step
        last_model_ix = np.lexsort((steps, epochs))[-1]
        return paths[last_model_ix]
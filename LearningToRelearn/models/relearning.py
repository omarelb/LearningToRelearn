import logging
import time
import json

import numpy as np
import torch
from torch import nn
from torch.utils import data
from transformers import AdamW
import wandb

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import TransformerClsModel
from LearningToRelearn.learner import Learner, METRICS_FILE
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, alternating_order, datasets_dict

# logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# logger = logging.getLogger("Baseline-Log")
OTHER_TASKS = "other_tasks"

class Relearner(Learner):
    def __init__(self, config, **kwargs):
        """
        Baseline models: sequential and multitask setup.
        """
        super().__init__(config, **kwargs)
        self.lr = config.learner.lr
        self.n_epochs = config.training.epochs
        self.model = TransformerClsModel(model_name=config.learner.model_name,
                                         n_classes=config.data.n_classes,
                                         max_length=config.data.max_length,
                                         device=self.device)
        self.logger.info("Loaded {} as model".format(self.model.__class__.__name__))
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        # assume for now that we only look at one task at a time
        self.relearning_task = config.learner.relearning_task
        # increments each time the relearning task is observed
        self.relearning_iter = 0
        self.relative_performance_threshold_lower = self.config.learner.relative_performance_threshold_lower
        self.relative_performance_threshold_upper = self.config.learner.relative_performance_threshold_upper

        # TODO: set through config
        # Parameter deciding at which percentage of performance relearning is evaluated
        self.relearning_evaluation_alphas = (0.75, 0.8, 0.85, 0.9, 0.95)
        self.smooth_alpha = self.config.learner.smooth_alpha
        self.first_encounter = True
        self.n_samples_slope = self.config.learner.n_samples_slope
        self.saturated_patience = self.config.learner.n_samples_saturated_patience
        self.saturated_threshold = self.config.learner.saturated_threshold

        # patience counter for saturation check relearning task
        self.not_improving = 0
        # initialize metrics
        for task in (self.relearning_task, OTHER_TASKS):
            if "performance" not in self.metrics[task]:
                self.metrics[task]["performance"] = []

    def training(self, datasets, **kwargs):
        train_datasets = datasets_dict(datasets["train"], datasets["order"])
        val_datasets = datasets_dict(datasets["val"], datasets["order"])
        self.relearning_task_dataset = {self.relearning_task: val_datasets[self.relearning_task]}

        self.dataloaders = {
            self.relearning_task: data.DataLoader(train_datasets[self.relearning_task],
                                                  batch_size=self.mini_batch_size,
                                                  shuffle=True),
            # for now, pi;e all other tasks on one stack
            OTHER_TASKS: data.DataLoader(
                data.ConcatDataset([dataset for task, dataset in train_datasets.items() if task != self.relearning_task]),
                batch_size=self.mini_batch_size,
                shuffle=True
            )
        }
        self.metrics[self.relearning_task]["performance"].append([])
        # write performance of initial encounter (before training) to metrics
        self.metrics[self.relearning_task]["performance"][0].append(
            self.validate(self.relearning_task_dataset, log=False,
                            n_samples=self.config.training.n_validation_samples)[self.relearning_task]
        )
        self.metrics[self.relearning_task]["performance"][0][0]["examples_seen"] = 0
        # first encounter relearning task
        self.train(dataloader=self.dataloaders[self.relearning_task], datasets=datasets)

    def train(self, dataloader=None, datasets=None, data_length=None):
        val_datasets = datasets_dict(datasets["val"], datasets["order"])

        if data_length is None:
            data_length = len(dataloader) * self.n_epochs

        all_losses, all_predictions, all_labels = [], [], []

        for text, labels, tasks in dataloader:
            self._examples_seen += len(text)
            self.model.train()
            # assumes all data in batch is from same task
            self.current_task = self.relearning_task if tasks[0] == self.relearning_task else OTHER_TASKS
            loss, predictions = self._train_batch(text, labels)
            all_losses.append(loss)
            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

            if self.current_iter % self.log_freq == 0:
                acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
                time_per_iteration, estimated_time_left = self.time_metrics(data_length)
                self.logger.info(
                    "Iteration {}/{} ({:.2f}%) -- {:.3f} (sec/it) -- Time Left: {}\nMetrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(self.current_iter + 1, data_length, (self.current_iter + 1) / data_length * 100,
                                            time_per_iteration, estimated_time_left,
                                            np.mean(all_losses), acc, prec, rec, f1))
                if self.config.wandb:
                    wandb.log({
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1": f1,
                            "loss": np.mean(all_losses),
                            "examples_seen": self.examples_seen(),
                            "task": self.current_task
                        })
                all_losses, all_predictions, all_labels = [], [], []
                self.start_time = time.time()
            if self.current_iter % self.validate_freq == 0:
                # only evaluate relearning task when training on relearning task
                validation_datasets = self.relearning_task_dataset if self.current_task == self.relearning_task else val_datasets
                
                validate_results = self.validate(validation_datasets,
                                        n_samples=self.config.training.n_validation_samples, log=False)
                self.write_results(validate_results)
                relearning_task_performance = validate_results[self.relearning_task]["accuracy"]
                if not self.first_encounter:
                    # TODO: make this a weighted average as well
                    relearning_task_relative_performance = self.relative_performance(
                                                            performance=relearning_task_performance,
                                                            task=self.relearning_task)
                    self.logger.info((f"Examples seen: {self.examples_seen()} -- Relative performance of task '{self.relearning_task}':" +
                                      f"{relearning_task_relative_performance}. Thresholds: {self.relative_performance_threshold_lower}"
                                      f"-{self.relative_performance_threshold_upper}"))
                    if self.config.wandb:
                        wandb.log({"relative_performance": relearning_task_relative_performance,
                                   "examples_seen": self.examples_seen()})
                if self.current_task == self.relearning_task:
                    self.logger.debug(f"first encounter: {self.first_encounter}")
                    # relearning stops either when either one of two things happen:
                    # the relearning task is first encountered and it is saturated (doesn't improve)
                    if ((self.first_encounter and self.learning_saturated(task=self.relearning_task, 
                                                                        n_samples_slope=self.n_samples_slope,
                                                                        patience=self.saturated_patience,
                                                                        threshold=self.saturated_threshold,
                                                                        smooth_alpha=self.smooth_alpha))
                    or (
                        # the relearning task is re-encountered and relative performance reaches some threshold
                        not self.first_encounter and
                        relearning_task_relative_performance >= self.relative_performance_threshold_upper
                    )):
                        # write metrics, reset, and train the other tasks
                        self.write_relearning_metrics()
                        self.logger.info(f"Task {self.current_task} saturated at iteration {self.current_iter}")
                        # each list element in performance refers to one consecutive learning event of the relearning task
                        self.metrics[self.relearning_task]["performance"].append([])
                        self.not_improving = 0
                        if self.first_encounter:
                            self.logger.info(f"\n-----------FIRST ENCOUNTER RELEARNING TASK '{self.relearning_task}' FINISHED.----------\n")
                        self.first_encounter = False
                        self.logger.info("TRAINING ON OTHER TASKS")
                        self.train(dataloader=self.dataloaders[OTHER_TASKS], datasets=datasets)
                else:
                    # calculate relative performance relearning task
                    # if it reaches some threshold, train on relearning task again
                    # TODO: make performance measure attribute of relearner 
                    if relearning_task_relative_performance <= self.relative_performance_threshold_lower:
                        self.logger.info(f"Relative performance on relearning task {self.relearning_task} below threshold. Evaluating relearning..")
                        # this needs to be done because we want a fresh list of performances when we start
                        # training the relearning task again. The first item in this list is simply the zero
                        # shot performance after the relative performance threshold is reached
                        # this means that every odd list in the relearning_task metrics is when training on the relearning task
                        relearning_task_performance = self.metrics[self.relearning_task]["performance"]
                        relearning_task_performance.append([])
                        # copy the last entry of the performance while training on the other tasks to the new list
                        relearning_task_performance[-1].append(relearning_task_performance[-2][-1])
                        self.train(dataloader=self.dataloaders[self.relearning_task], datasets=datasets)
                with open(self.results_dir / METRICS_FILE, "w") as f:
                    json.dump(self.metrics, f)

            self.time_checkpoint()
            self.current_iter += 1

    def _train_batch(self, text, labels):
        labels = torch.tensor(labels).to(self.device)
        input_dict = self.model.encode_text(text)
        output = self.model(input_dict)
        loss = self.loss_fn(output, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        self.logger.debug(f"Loss: {loss}")
        pred = model_utils.make_prediction(output.detach())
        return loss, pred.tolist()

    def relative_performance(self, performance, task, metric="accuracy"):
        """Calculate relative performance of a task compared to first encounter.
        
        For now assumes that task is the relearning task, since it assumes `max_score` and
        `initial_score` attributes in the metrics attribute.
        """
        max_score = self.metrics[task]["max_score"]
        initial_score = self.metrics[task]["initial_score"]
        return (performance - initial_score) / (max_score - initial_score)


    def write_results(self, validate_results):
        """Write validation results to self.metrics"""
        for task in validate_results.keys():
            if "performance" not in self.metrics[task]:
                self.metrics[task]["performance"] = []
            task_performance = self.metrics[task]["performance"]
            if len(task_performance) == 0:
                task_performance.append([])
            validate_results[task]["examples_seen"] = self.examples_seen()
            task_performance[-1].append(
                validate_results[task]
            )

    def learning_saturated(self, task, n_samples_slope=500, threshold=0.0002, patience=800,
                            smooth_alpha=0.3, metric="accuracy"):
        # threshold = 0.0002 # equivalent to 0.1% increase per 500 samples
        """Return true if validation performance of a task is deemed not to increase anymore.
        
        This is done by checking the slope of the performance curve over some amount of samples.

        Parameters
        ---
        task: str
            Which task to consider.
        patience: int
            Measured in terms of examples seen.

        Returns
        ---
        bool: whether performance on task is saturated.
        """
        validate_freq = self.validate_freq
        batch_size = self.mini_batch_size
        n_samples_per_validate = self.mini_batch_size * self.validate_freq
        self.logger.debug("-----------------START SATURATION CHECK----------------")
        self.logger.debug(f"n_samples_per_validate: {n_samples_per_validate}")

        # + 1 because we are looking at intervals
        window_size = n_samples_slope // n_samples_per_validate + 1
        self.logger.debug(f"window_size: {window_size}")

        # extract specific performance metric from last recorded task performance
        performance = [performance_metrics[metric] for performance_metrics in
                        self.metrics[task]["performance"][-1]]
        self.logger.debug(f"performances: {performance}")
        if len(performance) >= window_size:
            # use moving average to smooth out noise
            moving_average = model_utils.ewma(performance, alpha=smooth_alpha)
            self.logger.debug(f"moving average: {moving_average}")
            # measured in percent points
            slope = 100 * (moving_average[-1] - moving_average[-window_size]) / \
                          ((window_size - 1) * n_samples_per_validate)
            self.logger.debug(f"Iteration {self.current_iter} -- Slope {slope} -- Threshold {threshold} -- not_improving: {self.not_improving}")
            if slope < threshold:
                self.not_improving += 1
            else:
                self.not_improving = 0
        self.logger.debug("-----------------END SATURATION CHECK----------------")
        if self.not_improving * n_samples_per_validate >= patience:
            return True
        return False

    def write_relearning_metrics(self, metric="accuracy"):
        """Perform calculations necessary to get relearning metrics and write to self.metrics"""
        self.logger.info("-----------------START RELEARNING EVALUATION----------------")

        performance = [performance_metrics[metric] for performance_metrics in
                        self.metrics[self.relearning_task]["performance"][-1]]
        moving_average = model_utils.ewma(performance, alpha=self.smooth_alpha)
        if self.first_encounter:
            max_score = max(moving_average)
            self.metrics[self.relearning_task]["relearning"] = []
            self.metrics[self.relearning_task]["max_score"] = max_score
            self.metrics[self.relearning_task]["initial_score"] = performance[0]
            self.logger.info(f"first encounter max score: {max_score} -- initial score: {performance[0]}")
        else:
            # for logging purposes
            max_score = self.metrics[self.relearning_task]["max_score"]
            initial_score = self.metrics[self.relearning_task]["initial_score"]

        k_zero_log = True # to avoid duplicate log messsages when k_alpha == 0
        relearning_metrics = {}
        for alpha in self.relearning_evaluation_alphas:
            # alpha_max_score = alpha * max_score
            self.logger.info(f"Evaluating with alpha {alpha}")
            # first index that reaches performance higher than alpha * max score
            try:
                i_alpha_save = next(i for i, v in enumerate(moving_average) if
                                    self.relative_performance(performance=v, task=self.relearning_task) >= alpha)
            except StopIteration:
                self.logger.info(f"This run didn't reach a relative performance of at least {alpha}, skipping..")
                if k_zero_log:
                    relative_performances = [self.relative_performance(performance=v, task=self.relearning_task) for v in moving_average]
                    summary = list(zip(performance, moving_average, relative_performances))
                    self.logger.info(f"Showing run statistics")
                    self.logger.info(f"First encounter max score: {max_score} -- initial score: {initial_score}")
                    self.logger.info(f"(Performance, Moving average, Relative performance): {summary}")
                    k_zero_log = False
                continue
            # number of examples seen in just this encounter, can be calculated as offset using the examples seen
            k_alpha = (self.metrics[self.relearning_task]["performance"][-1][i_alpha_save]["examples_seen"]
                      - self.metrics[self.relearning_task]["performance"][-1][0]["examples_seen"])
            if k_alpha == 0:
                # special value to avoid dividing by 0
                self.logger.info("warning: k_alpha equals 0, may skew results")
                if k_zero_log:
                    relative_performances = [self.relative_performance(performance=v, task=self.relearning_task) for v in moving_average]
                    summary = list(zip(performance, moving_average, relative_performances))
                    self.logger.info(f"Showing run statistics")
                    self.logger.info(f"First encounter max score: {max_score} -- initial score: {initial_score}")
                    self.logger.info(f"(Performance, Moving average, Relative performance): {summary}")
                    k_zero_log = False
                learning_speed = np.NaN
            else:
                # TODO: look at this, now the accuracy could be when relative performance is higher than alpha
                alpha_performance = self.metrics[self.relearning_task]["performance"][-1][i_alpha_save]["accuracy"]
                # scale 1-100
                learning_speed = 100 * (alpha_performance - performance[0]) / \
                                    k_alpha
            relearning_metrics[f"k_alpha_{alpha}"] = k_alpha
            relearning_metrics[f"learning_speed_alpha_{alpha}"] = learning_speed
            self.logger.info(f"Reached relative performance of {alpha} after k_{alpha} = {k_alpha} examples")
            self.logger.info(f"learning_speed_alpha_{alpha}: {learning_speed}")
            # we have already had the initial task encounter, now we can record relearning speed
            if not self.first_encounter: 
                relearning_slope_alpha = learning_speed / self.metrics[self.relearning_task]["relearning"][0][f"learning_speed_alpha_{alpha}"]
                relearning_sample_alpha = (
                    self.metrics[self.relearning_task]["relearning"][0][f"k_alpha_{alpha}"] / k_alpha
                    if k_alpha != 0 else np.nan
                )
                relearning_metrics[f"relearning_slope_alpha_{alpha}"] = relearning_slope_alpha
                relearning_metrics[f"relearning_sample_alpha_{alpha}"] = relearning_sample_alpha
                self.logger.info("Relearning metrics:")
                self.logger.info("------------------")
                self.logger.info(f"relearning_slope_alpha_{alpha}: {relearning_slope_alpha}")
                self.logger.info(f"relearning_sample_alpha_{alpha}: {relearning_sample_alpha}")

        self.metrics[self.relearning_task]["relearning"].append(relearning_metrics)
        if self.config.wandb:
            wandb.log({
                "examples_seen": self.examples_seen(),
                f"k_{alpha}": k_alpha,
                f"learning_speed_alpha_{alpha}": learning_speed,
                f"relearning_slope_alpha_{alpha}": relearning_slope_alpha if not self.first_encounter else None,
                f"relearning_sample_alpha_{alpha}": relearning_sample_alpha if not self.first_encounter else None
            })
        self.logger.info("-----------------END RELEARNING EVALUATION----------------")

    def examples_seen(self):
        return self._examples_seen

    # def testing(self, datasets, order):
    #     """
    #     Parameters
    #     ---
    #     datasets: List[Dataset]
    #         Test datasets.
    #     order: List[str]
    #         Specifies order of encountered datasets
    #     """
    #     accuracies, precisions, recalls, f1s = [], [], [], []
    #     results = {}
    #     # only have one dataset if type is single
    #     if self.type == "single":
    #         train_dataset = datasets[order.index(self.config.learner.dataset)]
    #         datasets = [train_dataset]
    #     for dataset in datasets:
    #         dataset_name = dataset.__class__.__name__
    #         self.logger.info("Testing on {}".format(dataset_name))
    #         test_dataloader = data.DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False)
    #         dataset_results = self.evaluate(dataloader=test_dataloader)
    #         accuracies.append(dataset_results["accuracy"])
    #         precisions.append(dataset_results["precision"])
    #         recalls.append(dataset_results["recall"])
    #         f1s.append(dataset_results["f1"])
    #         results[dataset_name] = dataset_results

    #     mean_results = {
    #         "accuracy": np.mean(accuracies),
    #         "precision": np.mean(precisions),
    #         "recall": np.mean(recalls),
    #         "f1": np.mean(f1s)
    #     }
    #     self.logger.info("Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
    #                 "F1 score = {:.4f}".format(
    #                     mean_results["accuracy"], mean_results["precision"], mean_results["recall"],
    #                     mean_results["f1"]
    #                 ))
    #     return results, mean_results

    def evaluate(self, dataloader, **kwargs):
        all_losses, all_predictions, all_labels = [], [], []

        self.model.eval()

        for i, (text, labels, task) in enumerate(dataloader):
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

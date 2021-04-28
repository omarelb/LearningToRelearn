import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import pandas as pd
import wandb

# command line management
from dataclasses import dataclass, field
import hydra
from omegaconf import DictConfig, OmegaConf

from LearningToRelearn.datasets.text_classification_dataset import get_datasets
from LearningToRelearn.learner import EXPERIMENT_DIR, METRICS_FILE, flatten_dict
from LearningToRelearn.models.agem import AGEM
from LearningToRelearn.models.anml import ANML
from LearningToRelearn.models.baseline import Baseline
from LearningToRelearn.models.maml import MAML
from LearningToRelearn.models.oml import OML
from LearningToRelearn.models.replay import Replay
from LearningToRelearn.models.relearning import Relearner
from LearningToRelearn.models.basic_memory import BasicMemory
from result_analysis import analyze_results

RESULTS_FILE = Path(hydra.utils.to_absolute_path("results.csv"))

logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ContinualLearningLog")

def get_learner(config, **kwargs):
    """
    Return instantiation of a model depending on its type specified in a config.
    """
    if config.learner.type in ("sequential", "multitask", "single", "alternating"):
        return Baseline(config, **kwargs)
    if config.learner.type == "agem":
        learner = AGEM(config, **kwargs)
    elif config.learner.type == "replay":
        learner = Replay(config, **kwargs)
    elif config.learner.type == "maml":
        learner = MAML(config, **kwargs)
    elif config.learner.type == "oml":
        learner = OML(config, **kwargs)
    elif config.learner.type == "anml":
        learner = ANML(config, **kwargs)
    elif config.learner.type == "relearning":
        learner = Relearner(config, **kwargs)
    elif config.learner.type == "basic_memory":
        learner = BasicMemory(config, **kwargs)
    else:
        raise NotImplementedError
    return learner

def write_results(config, mean_validation_results, mean_test_results):
    """Write results from run to a results file for analysis purposes."""
    run_info = flatten_dict(config)
    for key in list(run_info.keys()):
        if key.startswith("hydra"):
            del run_info[key]
    for k, v in list(mean_validation_results.items()) + list(mean_test_results.items()):
        run_info[k] = v
    run_info["finish_time"] = datetime.now()

    df = pd.read_csv(RESULTS_FILE) if RESULTS_FILE.exists() else pd.DataFrame()
    df = df.append(run_info, ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)

@hydra.main(config_path="config", config_name="defaults.yaml")
def main(config):
    # to load a checkpoint and perform inference, add +evaluate=<dir_of_experiment>
    if "evaluate" not in config:
        learner = get_learner(config)
        datasets = get_datasets(learner.data_dir, config.data.order, debug=config.debug_data)
        learner.training(datasets)
        learner.write_metrics()
        learner.save_checkpoint()
    else:
        # no training, just evaluate
        experiment_path = Path(hydra.utils.to_absolute_path(EXPERIMENT_DIR / config.evaluate))
        config_file = experiment_path / '.hydra' / 'config.yaml'
        config = OmegaConf.load(config_file)
        config.wandb = False
        learner = get_learner(config, experiment_path=experiment_path)
        learner.load_checkpoint()
        datasets = get_datasets(learner.data_dir, config.data.order, debug=config.debug_data)

    # validation set
    logger.info("----------Validation starts here----------")
    learner.testing(datasets["val"], order=datasets["order"])
    learner.write_metrics()
    validation_results = analyze_results(metrics_path=learner.results_dir / METRICS_FILE,
                                         use_wandb=config.wandb)

    # validation_results = {"validation_" + k : v for k, v in mean_results.items()}
    # pd.DataFrame.from_dict(results, orient="index").to_csv(learner.results_dir / "validation_results.csv")

    # test set
    # logger.info("----------Testing starts here----------")
    # results, mean_results = learner.testing(datasets["test"], order=datasets["order"])
    # mean_test_results = {"test_" + k : v for k, v in mean_results.items()}
    # pd.DataFrame.from_dict(results, orient="index").to_csv(learner.results_dir / "test_results.csv")

    if config.wandb:
        wandb.run.summary.update(validation_results)
        # wandb.log(validation_results)
        # wandb.log(mean_test_results)
        learner.wandb_run.finish()

    # write results to a file
    # write_results(config, mean_validation_results, mean_test_results)

if __name__ == "__main__":
    main()

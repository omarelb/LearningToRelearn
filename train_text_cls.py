import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import pandas as pd

# command line management
from dataclasses import dataclass, field
import hydra
from omegaconf import DictConfig, OmegaConf

from MetaLifeLongLanguage.datasets.text_classification_dataset import get_datasets
from MetaLifeLongLanguage.learner import EXPERIMENT_DIR
from MetaLifeLongLanguage.models.cls_agem import AGEM
from MetaLifeLongLanguage.models.cls_anml import ANML
from MetaLifeLongLanguage.models.cls_baseline import Baseline
from MetaLifeLongLanguage.models.cls_maml import MAML
from MetaLifeLongLanguage.models.cls_oml import OML
from MetaLifeLongLanguage.models.cls_replay import Replay

logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ContinualLearningLog")

def get_learner(config, **kwargs):
    """
    Return instantiation of a model depending on its type specified in a config.
    """
    if config.learner.type in ("sequential", "multitask"):
        return Baseline(config, **kwargs)
    elif config.learner.type == "agem":
        learner = AGEM(config, **kwargs)
    elif config.learner.type == "replay":
        learner = Replay(config, **kwargs)
    elif config.learner.type == "maml":
        learner = MAML(config, **kwargs)
    elif config.learner.type == "oml":
        learner = OML(config, **kwargs)
    elif config.learner.type == "anml":
        learner = ANML(config, **kwargs)
    else:
        raise NotImplementedError
    return learner

@hydra.main(config_path="config", config_name="defaults.yaml")
def main(config):
    # to load a checkpoint and perform inference, add +evaluate=<dir_of_experiment>
    if "evaluate" not in config: 
        learner = get_learner(config)
        datasets = get_datasets(learner.data_dir, config.data.order)
        learner.training(datasets)

        learner.save_checkpoint()
    else:
        experiment_path = Path(hydra.utils.to_absolute_path(EXPERIMENT_DIR / config.evaluate))
        config_file = experiment_path / '.hydra' / 'config.yaml'
        config = OmegaConf.load(config_file)
        config.wandb = False
        learner = get_learner(config, experiment_path=experiment_path)
        learner.load_checkpoint()
        datasets = get_datasets(learner.data_dir, config.data.order)

    # validation set
    logger.info("----------Validation starts here----------")
    results = learner.testing(datasets["val"])
    pd.DataFrame.from_dict(results, orient="index").to_csv(learner.results_dir / "validation_results.csv")

    # test set
    logger.info("----------Testing starts here----------")
    results = learner.testing(datasets["test"])
    pd.DataFrame.from_dict(results, orient="index").to_csv(learner.results_dir / "test_results.csv")

if __name__ == "__main__":
    main()

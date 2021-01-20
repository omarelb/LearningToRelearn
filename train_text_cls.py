import logging
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch

# command line management
from dataclasses import dataclass, field
import hydra
from omegaconf import DictConfig, OmegaConf

import MetaLifeLongLanguage.datasets.utils as dataset_utils
from MetaLifeLongLanguage.datasets.text_classification_dataset import get_dataset
from MetaLifeLongLanguage.models.cls_agem import AGEM
from MetaLifeLongLanguage.models.cls_anml import ANML
from MetaLifeLongLanguage.models.cls_baseline import Baseline
from MetaLifeLongLanguage.models.cls_maml import MAML
from MetaLifeLongLanguage.models.cls_oml import OML
from MetaLifeLongLanguage.models.cls_replay import Replay

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ContinualLearningLog')

@hydra.main(config_path="config", config_name="defaults.yaml")
def main(cfg):
    # Define the ordering of the datasets
    dataset_order_mapping = {
        1: [2, 0, 3, 1, 4],
        2: [3, 4, 0, 1, 2],
        3: [2, 4, 1, 3, 0],
        4: [0, 2, 1, 4, 3]
    }
    n_classes = 33

    logger.info('Using configuration: {}'.format(vars(args)))

    # Set base path
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = Path(base_path) / "data"

    # Set random seed
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load the datasets
    logger.info('Loading the datasets')
    train_datasets, val_datasets, test_datasets = [], [], []
    for dataset_id in dataset_order_mapping[cfg.data.order]:
        train_dataset, val_dataset, test_dataset = dataset_utils.get_dataset(data_path, dataset_id)
        logger.info('Loaded {}'.format(train_dataset.__class__.__name__))
        # the same model is used for all tasks, so we need to shift labels of tasks
        train_dataset = dataset_utils.offset_labels(train_dataset)
        val_dataset = dataset_utils.offset_labels(val_dataset)
        test_dataset = dataset_utils.offset_labels(test_dataset)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
    logger.info('Finished loading all the datasets')

    # Load the model
    # TODO: add hydra
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.learner.type == 'sequential':
        learner = Baseline(device=device, n_classes=n_classes, training_mode='sequential', cfg)
    elif cfg.learner.type == 'multi_task':
        learner = Baseline(device=device, n_classes=n_classes, training_mode='multi_task', cfg)
    elif cfg.learner.type == 'agem':
        learner = AGEM(device=device, n_classes=n_classes, cfg)
    elif cfg.learner.type == 'replay':
        learner = Replay(device=device, n_classes=n_classes, cfg)
    elif cfg.learner.type == 'maml':
        learner = MAML(device=device, n_classes=n_classes, cfg)
    elif cfg.learner.type == 'oml':
        learner = OML(device=device, n_classes=n_classes, cfg)
    elif cfg.learner.type == 'anml':
        learner = ANML(device=device, n_classes=n_classes, cfg)
    else:
        raise NotImplementedError
    logger.info('Using {} as learner'.format(learner.__class__.__name__))

    # Training
    model_file_name = learner.__class__.__name__ + '-' + str(datetime.now()).replace(':', '-').replace(' ', '_') + '.pt'
    model_dir = os.path.join(base_path, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    logger.info('----------Training starts here----------')
    learner.training(train_datasets, val_datasets, cfg)
    learner.save_model(os.path.join(model_dir, model_file_name))
    logger.info('Saved the model with name {}'.format(model_file_name))

    # Testing
    logger.info('----------Testing starts here----------')
    learner.testing(test_datasets, **vars(args))

if __name__ == '__main__':
    main()

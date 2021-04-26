import re
from pathlib import Path
from typing import Union, Tuple, List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils import data

import logging

MAX_TRAIN_SIZE = 115000
MAX_VAL_SIZE = 5000
MAX_TEST_SIZE = 7600
MAX_DEBUG_SIZE = 4

SAMPLE_SEED = 42

DATASET_SETTINGS = {
    "agnews": {
        "n_classes": 4,
        "path": "ag_news_csv"
    },  # 0
    "amazon": {
        "n_classes": 5,
        "path": "amazon_review_full_csv"
    },  # 1
    "yelp": {
        "n_classes": 5,
        "path": "yelp_review_full_csv"
    },  # 2
    "dbpedia": {
        "n_classes": 14,
        "path": "dbpedia_csv"
    },  # 3
    "yahoo": {
        "n_classes": 10,
        "path": "yahoo_answers_csv"
    }
}

# Define the ordering of the datasets
DATASET_ORDER_MAPPING = {
    1: ["yelp", "agnews", "dbpedia", "amazon", "yahoo"],
    2: ["dbpedia", "yahoo", "agnews", "amazon", "yelp"],
    3: ["yelp", "yahoo", "amazon", "dbpedia", "agnews"],
    4: ["agnews", "yelp", "amazon", "yahoo", "dbpedia"]
}


def preprocess(text):
    """
    Preprocesses the text
    """
    text = text.lower()
    # remove unnecessary char sequence
    text = re.sub(r"nbsp;", "", text)
    # less than greater than brackets
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\\\$", "$", text)
    # removes "\n" present explicitly
    text = re.sub(r"(\\n)+", " ", text)
    # removes "\\"
    text = re.sub(r"(\\\\)+", "", text)
    # replaces repeated punctuation marks with single punctuation followed by a space
    # e.g, what???? -> what?
    text = re.sub(r"([.?!]){2,}", r"\1", text)
    # quotation marks are wrongly encoded as this sequence
    text = re.sub(r" #39;", "'", text)
    # quotation marks are wrongly encoded as this sequence
    text = re.sub(r"quot;", "\"", text)
    # # replace decimal of the type x.y with x since decimal digits after "." do not affect, e.g, 1.25 -> 1
    # text = re.sub(r"(\d+)\.(\d+)", r"\1", text)
    # removes hyperlinks
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text)
    # removes unnecessary space
    text = re.sub(r"(\s){2,}", u" ", text)
    # appends space to $ which will help during tokenization
    text = text.replace(u"$", u"$ ")
    return str(text)


def cache_filename(file_path, split, ext=".csv", debug=False):
    """Create filename of preprocessed file from dataset settings."""
    file_path = Path(file_path)
    folder = file_path.parent
    name = file_path.stem
    n = MAX_TRAIN_SIZE if split == "train" else MAX_VAL_SIZE if split == "val" else MAX_TEST_SIZE
    if split == "val": name = "val"
    new_name = f"preprocessed-{name}-seed_{SAMPLE_SEED}-n_{n}" + ("_debug" if debug else "") + ext
    return folder / new_name


class ClassificationDataset(data.Dataset):
    """
    Represents a classification dataset.

    Input
    -----
    name: name of dataset
    data: pandas dataframe
        If given, don't load from path.
    file_path:
        Path to underlying data.
    split: str
        Indicates whether it is a train or test split.
        One of {"train", "val", "test"}
    reduce:
        Whether to take only a subsample of data. Subsample size is defined by constants
        above.
    load_preprocessed_from_cache:
        Whether to read and write preprocessed file from hard disk.

    Attributes
    ---
    data:
        Pandas dataframe of underlying data in memory.
    n_classes:
        Number of classes in dataset.
    """
    def __init__(self,
                 name: str,
                 data=None,
                 data_path: Union[str, Path]=None,
                 split: bool=None,
                 reduce: bool = False,
                 load_preprocessed_from_cache: bool = True,
                 debug=False):
        settings = DATASET_SETTINGS[name]
        self.name = name
        self.n_classes = settings["n_classes"]
        if data is not None:
            self.data = data
            return
        assert data_path is not None and split is not None
        assert split in ("train", "val", "test"), "specify correct split"
        file_path = {
            "train": Path(data_path) / settings["path"] / "train.csv",
            "val": Path(data_path) / settings["path"] / "train.csv",
            "test": Path(data_path) / settings["path"] / "test.csv"
        }[split]
        cache_file = cache_filename(file_path, split=split, debug=debug)
        if load_preprocessed_from_cache and cache_file.is_file():
            # load file
            self.data = pd.read_csv(cache_file)
        else:
            self.data = self.read_data(file_path)
            if reduce:
                train_size = MAX_DEBUG_SIZE if debug else MAX_TRAIN_SIZE
                val_size = MAX_DEBUG_SIZE if debug else MAX_VAL_SIZE
                test_size = MAX_DEBUG_SIZE if debug else MAX_TEST_SIZE
                if split in ("train", "val"):
                    self.data = self.data.sample(n=train_size + val_size, random_state=SAMPLE_SEED)
                    # not using train test split immediately to use same samples as Nithin
                    train, val = train_test_split(self.data, train_size=train_size,
                                                  test_size=val_size, random_state=SAMPLE_SEED)
                    self.data = train if split == "train" else val
                else:
                    self.data = self.data.sample(n=test_size, random_state=SAMPLE_SEED)
            self.data["text"] = self.data["text"].apply(preprocess)
        if load_preprocessed_from_cache and not cache_file.is_file():
            self.data.to_csv(cache_file, index=False)

    def read_data(self, file_path):
        """
        Implements how data should be read from a file.
        """
        raise NotImplementedError(f"The method read_data should be implemented for subclass of {type(self)}")

    def sample(self, n, **kwargs):
        return ClassificationDataset(name=self.name, data=self.data.sample(n, **kwargs))

    def new(self, ix_low, ix_high):
        """Return new dataset with data given by indices"""
        return ClassificationDataset(name=self.name, data=self.data.iloc[ix_low:ix_high])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data["text"].iloc[index]
        label = self.data["labels"].iloc[index]
        return text, label, self.name


class AGNewsDataset(ClassificationDataset):
    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",", names=["labels", "title", "description"],
                                index_col=False)
        data["text"] = data["title"] + ". " + data["description"]
        data["labels"] = data["labels"] - 1
        data.drop(columns=["title", "description"], inplace=True)
        data.dropna(inplace=True)
        return data


class AmazonDataset(ClassificationDataset):
    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",", names=["labels", "title", "description"],
                                index_col=False)
        data.dropna(inplace=True)
        data["text"] = data["title"] + ". " + data["description"]
        data["labels"] = data["labels"] - 1
        data.drop(columns=["title", "description"], inplace=True)
        return data

class DBPediaDataset(ClassificationDataset):
    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",", names=["labels", "title", "description"],
                                index_col=False)
        data["text"] = data["title"] + ". " + data["description"]
        data["labels"] = data["labels"] - 1
        data.drop(columns=["title", "description"], inplace=True)
        data.dropna(inplace=True)
        return data


class YelpDataset(ClassificationDataset):
    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",", names=["labels", "text"],
                                index_col=False)
        data.dropna(inplace=True)
        data["labels"] = data["labels"] - 1
        return data


class YahooAnswersDataset(ClassificationDataset):
    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",",
                                names=["labels", "question_title", "question_content", "best_answer"],
                                index_col=False)
        data.dropna(inplace=True)
        data["text"] = data["question_title"] + data["question_content"] + data["best_answer"]
        data["labels"] = data["labels"] - 1
        data.drop(columns=["question_title", "question_content", "best_answer"], inplace=True)
        return data


# this dict should stay in this position since it depends on the classes defined above, and is used in functions below
DATASET_MAPPING = {
    "agnews": AGNewsDataset,  # 0
    "amazon": AmazonDataset,  # 1
    "yelp": YelpDataset,  # 2
    "dbpedia": DBPediaDataset,  # 3
    "yahoo": YahooAnswersDataset  # 4
}


def get_dataset(data_path, dataset_id, debug=False):
    """Return a single dataset given its id and path.

    Train, validation,.and test sets are returned.

    If debug is set to True, only load a small subset of the data.
    """
    assert dataset_id in ("yelp", "agnews", "dbpedia", "amazon", "yahoo"), "invalid dataset id"
    dataset = DATASET_MAPPING[dataset_id]
    return (
        dataset(name=dataset_id, data_path=data_path, split="train", reduce=True, debug=debug),
        dataset(name=dataset_id, data_path=data_path, split="val", reduce=True, debug=debug),
        dataset(name=dataset_id, data_path=data_path, split="test", reduce=True, debug=debug),
    )


def get_datasets(data_path, data_order=1, debug=False):
    """
    Load multiple datasets according to an order index, where the order
    is defined by DATASET_ORDER_MAPPING in this file.
    """
    logging.info("Loading data...")
    train_datasets, val_datasets, test_datasets = [], [], []
    for dataset_id in DATASET_ORDER_MAPPING[data_order]:
        train_dataset, val_dataset, test_dataset = get_dataset(data_path, dataset_id, debug=debug)
        logging.info("Loaded {}".format(train_dataset.__class__.__name__))
        # the same model is used for all tasks, so we need to shift labels of tasks
        train_dataset = offset_labels(train_dataset)
        val_dataset = offset_labels(val_dataset)
        test_dataset = offset_labels(test_dataset)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
    logging.info("Finished loading all the datasets")
    return {
        "train": train_datasets,
        "val": val_datasets,
        "test": test_datasets,
        "order": DATASET_ORDER_MAPPING[data_order]
    }


def get_continuum(datasets, order=None, n_samples=None, shuffle=False, merge=True):
    """
    Generate a dataset representing a continuum of tasks.

    The structure of the continuum of tasks can be manipulated by the order and n_samples parameters.

    To ensure that each batch of data is from the same task, make sure that each entry in n_samples is a
    multiple of the batch size.

    Parameters
    ---
    datasets: Dict[str, Dataset]
        Maps dataset names to pytorch datasets.
    order: List[str]
        Specifies in which order tasks are seen. If None, defaults to data dict's order, which may be random
        for some versions python. The order can contain the same dataset multiple times.
        However, no duplicate samples are allowed.
    n_samples: List[int]
        Specifies how many samples are seen for every step of the order. If None, use all samples of datasets
        specified in the order.
    shuffle: bool
        If true, the *order* is shuffled (not all datapoints).
    merge: bool
        If true, merge all the datasets into one ConcatDataset. Otherwise, keep it as a list of Subset datasets.

    Returns
    ---
    Union[Pytorch ConcatDataset, List[Pytorch Subset Dataset]]:
        Continuum of data.
    """
    if order is None or len(order) == 0:
        order = [dataset_name for dataset_name in datasets.keys()]
    data_lengths = {dataset_name: len(datasets[dataset_name]) for dataset_name in datasets.keys()}
    # if not specified, use all samples of this dataset
    if n_samples is None or len(n_samples) == 0:
        n_samples = [data_lengths[dataset_name] for dataset_name in order]
    assert len(n_samples) == len(order), "order and n_samples must be same length"
    # check input correctness
    n_samples_per_task = defaultdict(int)
    for dataset_name, n_sample in zip(order, n_samples):
        n_samples_per_task[dataset_name] += n_sample
        if n_samples_per_task[dataset_name] > data_lengths[dataset_name]:
            raise AssertionError(
                "The number of specified samples exceeds the number of samples in data, check n_samples"
            )

    result = []
    permutation = list(range(len(order)))
    if shuffle:
        # shuffle through the order (not through all samples, as is done with a dataloader)
        np.random.shuffle(permutation)
    # keep track of which indices are already used for each task/dataset
    ixs_occupied = {}
    for i in permutation:
        dataset_name, n_sample = order[i], n_samples[i]
        data_len = len(datasets[dataset_name])
        if dataset_name not in ixs_occupied:
            ixs_occupied[dataset_name] = []
        ixs = np.random.choice(list(set(range(data_len)) - set(ixs_occupied[dataset_name])), size=n_sample,
                               replace=False)
        ixs_occupied[dataset_name].extend(ixs)
        result.append(data.Subset(datasets[dataset_name], indices=ixs))

    if merge:
        return data.ConcatDataset(result)
    else:
        return result

# alternating
def alternating_order(datasets, n_samples_per_switch, tasks=None, relative_frequencies=None):
    """
    Return an alternating task ordering scheme that can be used as input to the `get_continuum` function.

    Example: ["yelp", "amazon", "yelp", "amazon"], [10, 10, 10, 10]

    Input
    ---
    datasets: Dict[str, Dataset]
        Maps dataset names to pytorch datasets.
    n_samples_per_switch: int
        Specifies the number of samples used for one task before switching
    tasks: List[str]
        Names of tasks/datasets that should be alternated between. If None, use all tasks
        given in datasets.
    relative_frequences: List[int]
        Specifies the relative sample frequency of each task. This is multiplied by `n_samples_per_swithch`.
        Example:
            relative_frequences = [2, 1] --> [20, 10, 20, 10]

    Returns
    ---
    Tuple[List[str], List[int]]:
        order, n_samples
        Can be used as input for the `get_continuum` function.
    """
    if tasks is None or len(tasks) == 0:
        tasks = list(datasets.keys())
    if relative_frequencies is None or len(relative_frequencies) == 0:
        relative_frequencies = [1 for _ in tasks]
    assert len(tasks) == len(relative_frequencies), "make sure relative frequencies has same length as tasks"
    n_tasks = len(tasks)
    smallest_data_length = min([len(datasets[dataset_name]) / relative_frequency for dataset_name, relative_frequency in
                                zip(tasks, relative_frequencies)])

    order = []
    n_samples = []
    n_batches = int(smallest_data_length // n_samples_per_switch)
    for i in range(n_batches * n_tasks):
        task_ix = i % len(tasks)
        task = tasks[task_ix]
        order.append(task)
        n_samples.append(relative_frequencies[task_ix] * n_samples_per_switch)
    return order, n_samples


def offset_labels(dataset):
    """Shift labels of dataset depending on the dataset"""
    if isinstance(dataset, AmazonDataset) or isinstance(dataset, YelpDataset):
        offset_by = 0
    elif isinstance(dataset, AGNewsDataset):
        offset_by = 5
    elif isinstance(dataset, DBPediaDataset):
        offset_by = 5 + 4
    elif isinstance(dataset, YahooAnswersDataset):
        offset_by = 5 + 4 + 14
    dataset.data["labels"] = dataset.data["labels"] + offset_by
    return dataset


def datasets_dict(datasets, order):
    """
    Create a dict from a list of datasets and its order which specifies the name of each dataset.

    Parameters
    ---
    datasets: List[Dataset]
    order: List[str]
    """
    return {dataset_name: dataset for dataset_name, dataset in zip(order, datasets)}

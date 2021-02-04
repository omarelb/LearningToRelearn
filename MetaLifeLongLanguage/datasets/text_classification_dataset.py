import re
from pathlib import Path
from typing import Union, Tuple, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils import data

import logging

MAX_TRAIN_SIZE = 115000
MAX_VAL_SIZE = 5000
MAX_TEST_SIZE = 7600
MAX_DEBUG_SIZE = 4

SAMPLE_SEED = 42

# Define the ordering of the datasets
DATASET_ORDER_MAPPING = {
    1: [2, 0, 3, 1, 4],
    2: [3, 4, 0, 1, 2],
    3: [2, 4, 1, 3, 0],
    4: [0, 2, 1, 4, 3]
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
    file_path:
        Path to underlying data.
    split: str
        Indicates whether it is a train or test split.
        One of {"train", "val", "test"}
    n_classes:
        Number of classes.
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
                 file_path: Union[str, Path],
                 split: bool,
                 n_classes: int,
                 reduce: bool = False,
                 load_preprocessed_from_cache: bool = True,
                 debug=False):
        assert split in ("train", "val", "test"), "specify correct split"
        file_path = Path(file_path)
        self.n_classes = n_classes
        cache_file = cache_filename(file_path, split=split, debug=debug)
        if load_preprocessed_from_cache and cache_file.is_file():
                # load file
                self.data = pd.read_csv(cache_file)
        else:
            # TODO: add debug mode
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data["text"].iloc[index]
        label = self.data["labels"].iloc[index]
        return text, label


class AGNewsDataset(ClassificationDataset):
    def __init__(self, data_path, split, reduce=False, load_preprocessed_from_cache=True, debug=False):
        """
        data_path: str
            Location of the data folder.
        """
        paths = {
            "train": Path(data_path) / "ag_news_csv/train.csv",
            "val": Path(data_path) / "ag_news_csv/train.csv",
            "test": Path(data_path) / "ag_news_csv/test.csv"
        }
        super().__init__(paths[split], split, n_classes=4, reduce=reduce, load_preprocessed_from_cache=load_preprocessed_from_cache, debug=debug)

    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",", names=["labels", "title", "description"],
                                index_col=False)
        data["text"] = data["title"] + ". " + data["description"]
        data["labels"] = data["labels"] - 1
        data.drop(columns=["title", "description"], inplace=True)
        data.dropna(inplace=True)
        return data


class DBPediaDataset(ClassificationDataset):
    def __init__(self, data_path, split, reduce=False, load_preprocessed_from_cache=True, debug=False):
        paths = {
            "train": Path(data_path) / "dbpedia_csv/train.csv",
            "val": Path(data_path) / "dbpedia_csv/train.csv",
            "test": Path(data_path) / "dbpedia_csv/test.csv"
        }
        super().__init__(paths[split], split, n_classes=14, reduce=reduce, load_preprocessed_from_cache=load_preprocessed_from_cache, debug=debug)

    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",", names=["labels", "title", "description"],
                                index_col=False)
        data["text"] = data["title"] + ". " + data["description"]
        data["labels"] = data["labels"] - 1
        data.drop(columns=["title", "description"], inplace=True)
        data.dropna(inplace=True)
        return data

class AmazonDataset(ClassificationDataset):
    def __init__(self, data_path, split, reduce=False, load_preprocessed_from_cache=True, debug=False):
        paths = {
            "train": Path(data_path) / "amazon_review_full_csv/train.csv",
            "val": Path(data_path) / "amazon_review_full_csv/train.csv",
            "test": Path(data_path) / "amazon_review_full_csv/test.csv"
        }
        super().__init__(paths[split], split, n_classes=5, reduce=reduce, load_preprocessed_from_cache=load_preprocessed_from_cache, debug=debug)

    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",", names=["labels", "title", "description"],
                                index_col=False)
        data.dropna(inplace=True)
        data["text"] = data["title"] + ". " + data["description"]
        data["labels"] = data["labels"] - 1
        data.drop(columns=["title", "description"], inplace=True)
        return data


class YelpDataset(ClassificationDataset):
    def __init__(self, data_path, split, reduce=False, load_preprocessed_from_cache=True, debug=False):
        paths = {
            "train": Path(data_path) / "yelp_review_full_csv/train.csv",
            "val": Path(data_path) / "yelp_review_full_csv/train.csv",
            "test": Path(data_path) / "yelp_review_full_csv/test.csv"
        }
        super().__init__(paths[split], split, n_classes=5, reduce=reduce, load_preprocessed_from_cache=load_preprocessed_from_cache, debug=debug)

    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",", names=["labels", "text"],
                                index_col=False)
        data.dropna(inplace=True)
        data["labels"] = data["labels"] - 1
        return data


class YahooAnswersDataset(ClassificationDataset):
    def __init__(self, data_path, split, reduce=False, load_preprocessed_from_cache=True, debug=False):
        paths = {
            "train": Path(data_path) / "yahoo_answers_csv/train.csv",
            "val": Path(data_path) / "yahoo_answers_csv/train.csv",
            "test": Path(data_path) / "yahoo_answers_csv/test.csv"
        }
        super().__init__(paths[split], split, n_classes=10, reduce=reduce, load_preprocessed_from_cache=load_preprocessed_from_cache, debug=debug)

    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None, sep=",",
                                names=["labels", "question_title", "question_content", "best_answer"],
                                index_col=False)
        data.dropna(inplace=True)
        data["text"] = data["question_title"] + data["question_content"] + data["best_answer"]
        data["labels"] = data["labels"] - 1
        data.drop(columns=["question_title", "question_content", "best_answer"], inplace=True)
        return data


DATASET_MAPPING = [
    AGNewsDataset, # 0
    AmazonDataset, # 1
    YelpDataset, # 2
    DBPediaDataset, # 3
    YahooAnswersDataset # 4
]


def get_dataset(data_path, dataset_id, debug=False):
    """Return a single dataset given its id and path.
    
    Train, validation,.and test sets are returned.

    If debug is set to True, only load a small subset of the data.
    """
    assert 0 <= dataset_id <= 4, "invalid dataset id"
    dataset = DATASET_MAPPING[dataset_id]
    return (
        dataset(data_path, "train", reduce=True, debug=debug),
        dataset(data_path, "val", reduce=True, debug=debug),
        dataset(data_path, "test", reduce=True, debug=debug),
    ) 


def get_datasets(data_path, data_order, debug=False):
    """
    Load multiple datasets according to an order index, where the order
    is defined by DATASET_ORDER_MAPPING in this file.
    """
    logging.info(f"Loading data...")
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
    }

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
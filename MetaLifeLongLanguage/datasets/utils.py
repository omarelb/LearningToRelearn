import os
import random
import re

import torch
import numpy as np
from sklearn.cluster import KMeans

# from .lifelong_fewrel_dataset import LifelongFewRelDataset
                                                                      

def batch_encode(batch):
    text, labels = [], []
    for txt, lbl in batch:
        text.append(txt)
        labels.append(lbl)
    return text, labels


def remove_return_sym(str):
    return str.split('\n')[0]

def get_max_len(text_list):
    return max([len(x) for x in text_list])

def glove_vectorize(text, glove, dim=300):
    max_len = get_max_len(text)
    lengths = []
    vec = torch.ones((len(text), max_len, dim))
    for i, sent in enumerate(text):
        sent_emb = glove.get_vecs_by_tokens(sent, lower_case_backup=True)
        vec[i, :len(sent_emb)] = sent_emb
        lengths.append(len(sent))
    lengths = torch.tensor(lengths)
    return vec, lengths
import torch
import numpy as np
from sklearn import metrics

# from LearningToRelearn.models.cls_baseline import Baseline 
# from LearningToRelearn.models.cls_agem import AGEM 
# from LearningToRelearn.models.cls_anml import ANML 
# from LearningToRelearn.models.cls_maml import MAML
# from LearningToRelearn.models.cls_oml import OML 
# from LearningToRelearn.models.cls_replay import Replay 

def calculate_metrics(predictions, labels, binary=False):
    averaging = "binary" if binary else "macro"
    predictions = np.array(predictions)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    recall = metrics.recall_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    f1_score = metrics.f1_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    return accuracy, precision, recall, f1_score


def calculate_accuracy(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    return accuracy


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (output > 0).int()
        else:
            pred = output.max(-1)[1]
    return pred


def make_rel_prediction(cosine_sim, ranking_label):
    pred = []
    with torch.no_grad():
        pos_idx = [i for i, lbl in enumerate(ranking_label) if lbl == 1]
        if len(pos_idx) == 1:
            pred.append(torch.argmax(cosine_sim))
        else:
            for i in range(len(pos_idx) - 1):
                start_idx = pos_idx[i]
                end_idx = pos_idx[i+1]
                subset = cosine_sim[start_idx: end_idx]
                pred.append(torch.argmax(subset))
    pred = torch.tensor(pred)
    true_labels = torch.zeros_like(pred)
    return pred, true_labels


def ewma(series=None, prev_value=None, new_value=None, alpha=0.5):
    """
    Exponentially weighted moving average.
    
    Can be calculated over a series at once, or incrementially by supplying a new value.
    """
    if series is not None:
        assert len(series) > 0, "need at least one observation"
        if len(series) == 1:
            return [series[0]]
        else:
            if prev_value is None:
                res = []
                res.append(series[0])
                for x in series[1:]:
                    res.append(res[-1] * (1 - alpha) + x * alpha)
                return res
            else:
                return prev_value * (1 - alpha) + series[-1] * alpha
    elif prev_value is None and new_value is not None:
        return new_value
    elif prev_value is not None:
        assert new_value is not None, "new_value must be specified if no series is given"
        return prev_value * (1 - alpha) + new_value * alpha
    else:
        raise ValueError("incompatible parameter configuration")
import torch
import numpy as np
from sklearn import metrics

def calculate_metrics(predictions, labels, binary=False):
    averaging = "binary" if binary else "macro"
    predictions = np.array(predictions)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    recall = metrics.recall_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    f1_score = metrics.f1_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score}


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


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def get_class_means(embeddings, labels):
    """Return class means and unique labels given neighbors.
    
    Parameters
    ---
    embeddings: Tensor, shape (batch_size, embed_size)
    labels: iterable of labels for each embedding
        
    Returns
    ---
    Tuple[List[Tensor], List[Tensor]]:
        class means and unique labels
    """
    class_means = []
    unique_labels = torch.tensor(labels).unique()
    for label_ in unique_labels:
        label_ixs = (label_ == torch.tensor(labels)).nonzero(as_tuple=False).flatten()
        same_class_embeddings = embeddings[label_ixs]
        class_means.append(same_class_embeddings.mean(axis=0))
    return torch.stack(class_means), unique_labels

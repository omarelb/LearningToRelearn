"""
Analyzes training and testing runs, writing summary statistics to a file and generating any plots.
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess

from LearningToRelearn.learner import flatten_dict

def learning_curve_area(performances, batch_wise=False, zero_shot_difference=False):
    """
    Return area under learning curve given performance measurements.

    Parameters
    ---
    performances: List[Dict]
        Each entry contains a dictionary with at least keys {'examples_seen', 'accuracy'}.
    batch_wise: bool
        If true, normalization is done batch wise instead of using the 'examples_seen' information.
        Set this to true if the batch size used during evaluation is > 1.
    zero_shot_difference: bool
        If true, the zero shot accuracy is subtracted from every measurement, giving an indication of
        how fast it is learning compared to its baseline.

    Returns
    ---
    Dict[int, float] mapping learning curve area at k to its corresponding measurement.
    """
    result = {}
    area = 0
    for i, performance in enumerate(performances):
        area += performance["accuracy"]
        if batch_wise:
            normalization = i
        else:
            normalization = performance["examples_seen"]
        result[normalization] = (area - performances[0]["accuracy"] * zero_shot_difference) / (normalization + 1)
    return result

def learning_slope(performances):
    """
    Return slope of learning curve @ k given performance measurements.

    Parameters
    ---
    performances: List[Dict]
        Each entry contains a dictionary with at least keys {'examples_seen', 'accuracy'}.

    Returns
    ---
    Dict[int, float] mapping learning curve area at k to its corresponding measurement.
    """
    result = {}
    for i, performance in enumerate(performances[1:]):
        result[performance["examples_seen"]] = (performance["accuracy"] - performances[0]["accuracy"]) / \
                                                performance["examples_seen"]
    return result


def collect_results(metrics):
    """
    Collect results contained in a metrics dictionary.

    Parameters
    ---
    metrics_path: pathlike
        Path to json file containing training run data.
    metrics: Dict
        Contains training run data to be analyzed.

    Returns
    ---
    dictionary of information extracted from metrics dictionary.
    """
    results = {}
    # return online accuracy, the average accuracy during training.
    results["online_accuracy"] = pd.DataFrame(metrics["online"]).accuracy.mean()
    # mean accuracy over all evaluation datasets
    results["average_accuracy"] = metrics["evaluation"]["average"]["accuracy"]
    # accuracy over individual evaluation datasets
    results["individual_accuracy"] = metrics["evaluation"]["individual"]
    # task that was evaluated
    results["eval_task"] = metrics["evaluation"]["few_shot"][0]["task"]
    results["few_shot_learning_curve_area"] = learning_curve_area(metrics["evaluation"]["few_shot"], batch_wise=False)
    results["few_shot_learning_speed"] = learning_slope(metrics["evaluation"]["few_shot"])
    # validation accuracy after training on k samples, for multiple k
    # results["few_shot_accuracy"] = metrics["evaluation"]["few_shot"]
    return results

def analyze_results(metrics_path=None, metrics=None, write_path=None):
    """
    Collect results contained in a metrics dictionary.

    Parameters
    ---
    metrics_path: pathlike
        Path to json file containing training run data.
    metrics: Dict
        Contains training run data to be analyzed.

    Returns
    ---
    dictionary of information extracted from metrics dictionary.
    """
    assert (metrics_path is None) ^ (metrics is None), "only one of `metrics_path` or `metrics` can be specidied."
    if metrics is not None:
        assert write_path is not None, "write_path should be specified if path of the metrics file is not specified"
    if write_path is not None:
        write_path = Path(write_path)
    if metrics_path is not None:
        # set write path to same folder as metrics file if its path is supplied
        if write_path is None:
            write_path = Path(metrics_path).parent
        with open(metrics_path) as f:
            metrics = json.load(f)
    results = collect_results(metrics)
    with open(write_path / "results.json", "w") as f:
        json.dump(results, f)
    results_flattened = flatten_dict(results)
    # also write to csv
    pd.DataFrame({k: [v] for k, v in results_flattened.items()}).to_csv(write_path / "results.csv", index=False)

    # generate plots
    img_path = write_path / "img"
    img_path.mkdir(exist_ok=True, parents=True)
    online_df = pd.DataFrame(metrics["online"])

    figsize = (20, 10)
    # few shot accuracy
    plt.figure(figsize=figsize)
    y = [result["accuracy"] for result in metrics["evaluation"]["few_shot"]]
    plt.plot(y)
    img = img_path / "few_shot_accuracy.pdf"
    plt.savefig(img)
    subprocess.call(f"pdfcrop {img} {img}", shell=True)

    # Online accuracy
    plt.figure(figsize=figsize)
    plt.plot(online_df["accuracy"])
    img = img_path / "online_accuracy.pdf"
    plt.savefig(img)
    # crop the image
    subprocess.call(f"pdfcrop {img} {img}", shell=True)

    # Few shot learning curve area
    plt.figure(figsize=figsize)
    x = list(results["few_shot_learning_curve_area"].keys())
    y = list(results["few_shot_learning_curve_area"].values())
    plt.plot(x, y)
    img = img_path / "few_shot_learning_curve_area.pdf"
    plt.savefig(img)
    subprocess.call(f"pdfcrop {img} {img}", shell=True)

    # Few shot learning speed
    plt.figure(figsize=figsize)
    x = list(results["few_shot_learning_speed"].keys())
    y = list(results["few_shot_learning_speed"].values())
    plt.plot(x, y)
    img = img_path / "few_shot_learning_speed.pdf"
    plt.savefig(img)
    subprocess.call(f"pdfcrop {img} {img}", shell=True)

    #  task shifts
    plt.figure(figsize=figsize)
    task_shifts = online_df[online_df.task.shift() != online_df.task].examples_seen - 16

    ax = online_df.plot(x="examples_seen", y="accuracy")
    ax.vlines(x=task_shifts, ymin=0, ymax=1, color="black", linestyles="dashed")
    img = img_path / "task_shifts.pdf"
    plt.savefig(img)
    subprocess.call(f"pdfcrop {img} {img}", shell=True)

    return results_flattened

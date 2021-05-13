"""
Analyzes training and testing runs, writing summary statistics to a file and generating any plots.
"""

from pathlib import Path
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import wandb

from LearningToRelearn.learner import flatten_dict

def learning_curve_area(performances, zero_shot_difference=False):
    """
    Return area under learning curve given performance measurements.

    Parameters
    ---
    performances: List[Dict]
        Each entry contains a dictionary with at least keys {'examples_seen', 'accuracy'}.
    zero_shot_difference: bool
        If true, the zero shot accuracy is subtracted from every measurement, giving an indication of
        how fast it is learning compared to its baseline.

    Returns
    ---
    Dict[int, float] mapping learning curve area at k to its corresponding measurement.
    """
    # normalization is done batch wise instead of using the 'examples_seen' information, when
    # few shot evaluation batch size > 1
    batch_wise = (len(performances) > 1) and (performances[1]["examples_seen"] - performances[0]["examples_seen"] > 1)
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
    online_df = pd.DataFrame(metrics["online"])
    if "accuracy" in online_df.columns:
        results["online_accuracy"] = online_df.accuracy.mean()
    # mean accuracy over all evaluation datasets
    if "average" in metrics["evaluation"]:
        results["average_accuracy"] = metrics["evaluation"]["average"]["accuracy"]
    # accuracy over individual evaluation datasets
    if "individual" in metrics["evaluation"]:
        results["individual_accuracy"] = metrics["evaluation"]["individual"]
    # task that was evaluated
    if "few_shot" in metrics["evaluation"]:
        results["eval_task"] = metrics["evaluation"]["few_shot"][0][0]["task"]
        for i, few_shot_metrics in enumerate(metrics["evaluation"]["few_shot"]):
            results[f"few_shot_learning_curve_area_{i}"] = learning_curve_area(few_shot_metrics)
            results[f"few_shot_learning_curve_area_difference_{i}"] = learning_curve_area(few_shot_metrics,
                                                                                    zero_shot_difference=True)
            results[f"few_shot_learning_speed_{i}"] = learning_slope(few_shot_metrics)
    # validation accuracy after training on k samples, for multiple k
    # results["few_shot_accuracy"] = metrics["evaluation"]["few_shot"]
    return results

def analyze_results(metrics_path=None, metrics=None, write_path=None, use_wandb=False):
    """
    Collect results contained in a metrics dictionary.

    Parameters
    ---
    metrics_path: pathlike
        Path to json file containing training run data.
    metrics: Dict
        Contains training run data to be analyzed.
    use_wandb: bool
        Whether using wandb. If so, log plots to wandb.

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
    if "few_shot" in metrics["evaluation"]:
        # few shot accuracy
        plt.figure(figsize=figsize)
        for i, few_shot_metrics in enumerate(metrics["evaluation"]["few_shot"]):
            y = [result["accuracy"] for result in few_shot_metrics]
            plt.plot(y)
            if use_wandb:
                wandb.log({f"chart_few_shot_accuracy_{i}": plt})
            img = img_path / f"few_shot_accuracy_{i}.pdf"
            plt.savefig(img)
            subprocess.call(f"pdfcrop {img} {img}", shell=True)

    # Online accuracy
    if "accuracy" in online_df.columns:
        plt.figure(figsize=figsize)
        plt.plot(online_df["examples_seen"], online_df["accuracy"])
        if use_wandb:
            wandb.log({"chart_online_accuracy": plt})
        img = img_path / "online_accuracy.pdf"
        plt.savefig(img)
        # crop the image
        subprocess.call(f"pdfcrop {img} {img}", shell=True)

    if "few_shot_learning_curve_area_0" in results:
        # Few shot learning curve area
        for i, _ in enumerate(metrics["evaluation"]["few_shot"]):
            plt.figure(figsize=figsize)
            x = list(results[f"few_shot_learning_curve_area_{i}"].keys())
            y = list(results[f"few_shot_learning_curve_area_{i}"].values())
            plt.plot(x, y)
            if use_wandb:
                wandb.log({f"chart_few_shot_lca_{i}": plt})
            img = img_path / f"few_shot_learning_curve_area_{i}.pdf"
            plt.savefig(img)
            subprocess.call(f"pdfcrop {img} {img}", shell=True)

            # Few shot learning curve area
            plt.figure(figsize=figsize)
            x = list(results[f"few_shot_learning_curve_area_difference_{i}"].keys())
            y = list(results[f"few_shot_learning_curve_area_difference_{i}"].values())
            plt.plot(x, y)
            if use_wandb:
                wandb.log({f"chart_few_shot_lca_zero_shot_difference_{i}": plt})
            img = img_path / f"few_shot_learning_curve_area_difference_{i}.pdf"
            plt.savefig(img)
            subprocess.call(f"pdfcrop {img} {img}", shell=True)

    if "few_shot_learning_speed_0" in results:
        # Few shot learning speed
        plt.figure(figsize=figsize)
        x = list(results[f"few_shot_learning_speed_{i}"].keys())
        y = list(results[f"few_shot_learning_speed_{i}"].values())
        plt.plot(x, y)
        if use_wandb:
            wandb.log({f"chart_learning_speed_{i}": plt})
        img = img_path / f"few_shot_learning_speed_{i}.pdf"
        plt.savefig(img)
        subprocess.call(f"pdfcrop {img} {img}", shell=True)

    #  task shifts
    # plt.figure(figsize=figsize)
    # task_shifts = online_df[online_df.task.shift() != online_df.task].examples_seen - 16

    # ax = online_df.plot(x="examples_seen", y="accuracy")
    # ax.vlines(x=task_shifts, ymin=0, ymax=1, color="black", linestyles="dashed")
    # img = img_path / "task_shifts.pdf"
    # plt.savefig(img)
    # subprocess.call(f"pdfcrop {img} {img}", shell=True)

    return results_flattened

if __name__ == "__main__":
    analyze_results(metrics_path=Path('experiments') / sys.argv[1])

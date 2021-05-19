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
    batch_wise = False
    batch_size = 1
    if len(performances) > 1:
        batch_size = performances[1]["examples_seen"] - performances[0]["examples_seen"]
        if batch_size > 1:
            batch_wise = True
    result = {}
    area = 0
    for i, performance in enumerate(performances):
        area += performance["accuracy"]
        result[i * batch_size] = area / (i + 1) - performances[0]["accuracy"] * zero_shot_difference
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

def expand_dict(d):
    """
    Fills in gaps in dictionary returned by learning_curve_area if batch_size > 1.
    
    Gaps are simply filled with the last previously seen value, and therefore underestimates
    the true learning curve area in most cases.
    
    Parameters
    ---
    d: Dict[int, float]
        Maps examples seen to a value
    
    Returns
    ---
    Dict[int, float]
        Same as input, with Gaps filled in.
    """
    new_d = {}
    for i, value in d.items():
        new_d[i] = value
        if i == sorted(d)[-1]: break
        j = i + 1
        while j not in d.keys():
            new_d[j] = value
            j += 1
    # sort
    return {key: new_d[key] for key in sorted(new_d)}


def get_relearning(first_encounter_dict, few_shot_learning_dict):
    """Utility function"""
    result = {}
    for examples_seen, value in few_shot_learning_dict.items():
        try:
            fraction = value / first_encounter_dict[examples_seen]
        except ZeroDivisionError:
            fraction = np.nan
        except KeyError:
            continue
        result[examples_seen] = fraction
    return result

def get_forgetting(few_shot_metrics, metrics):
    forgetting, forgetting_normalized = None, None
    first_encounter_accuracy = metrics["eval_task_first_encounter"][0]["accuracy"]
    first_evaluation = few_shot_metrics[0]
    max_examples_seen = first_evaluation["examples_seen_total"]
    # get entries of task before this evaluation
    filtered = [entry for entry in metrics["online"] if entry["examples_seen"] < max_examples_seen and
                entry["task"] == first_evaluation["task"]]
    if len(filtered) > 1:
        best_previous_accuracy = max(entry["accuracy"] for entry in filtered)
        first_eval_accuracy = first_evaluation["accuracy"]
        forgetting = best_previous_accuracy - first_eval_accuracy
        try:
            forgetting_normalized = forgetting / (best_previous_accuracy - first_encounter_accuracy)
        except ZeroDivisionError as e:
            print(e)
            forgetting_normalized = None
    return forgetting, forgetting_normalized

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
        # results["eval_task"] = metrics["evaluation"]["few_shot"][0][0]["task"]
        results["first_encounter_initial_accuracy"] = metrics["eval_task_first_encounter"][0]["accuracy"]
        results["first_encounter_best_accuracy"] = max(entry["accuracy"] for entry in metrics["eval_task_first_encounter"])
        results[f"first_encounter_learning_curve_area"] = learning_curve_area(metrics["eval_task_first_encounter"])
        results[f"first_encounter_learning_curve_area_difference"] = learning_curve_area(metrics["eval_task_first_encounter"],
                                                                                         zero_shot_difference=True)
        results[f"first_encounter_learning_speed"] = learning_slope(metrics["eval_task_first_encounter"])
        results[f"first_encounter_learning_curve_area_expanded"] = expand_dict(results[f"first_encounter_learning_curve_area"])
        results[f"first_encounter_learning_curve_area_difference_expanded"] = expand_dict(results[f"first_encounter_learning_curve_area_difference"])
        results[f"first_encounter_learning_speed_expanded"] = expand_dict(learning_slope(metrics["eval_task_first_encounter"]))
        for i, few_shot_metrics in enumerate(metrics["evaluation"]["few_shot"]):
            results[f"few_shot_learning_curve_area_{i}"] = learning_curve_area(few_shot_metrics)
            results[f"few_shot_learning_curve_area_difference_{i}"] = learning_curve_area(few_shot_metrics,
                                                                                    zero_shot_difference=True)
            results[f"few_shot_learning_speed_{i}"] = learning_slope(few_shot_metrics)

            results[f"few_shot_relearning_curve_area_{i}"] = get_relearning(results[f"first_encounter_learning_curve_area_expanded"],
                                                                            results[f"few_shot_learning_curve_area_{i}"])
            results[f"few_shot_relearning_curve_area_difference_{i}"] = get_relearning(results[f"first_encounter_learning_curve_area_difference_expanded"],
                                                                            results[f"few_shot_learning_curve_area_difference_{i}"])
            results[f"few_shot_relearning_speed_{i}"] = get_relearning(results[f"first_encounter_learning_speed_expanded"],
                                                                       results[f"few_shot_learning_speed_{i}"])
            results[f"forgetting_{i}"], results[f"forgetting_normalized_{i}"] = get_forgetting(few_shot_metrics, metrics)
    # validation accuracy after training on k samples, for multiple k
    # results["few_shot_accuracy"] = metrics["evaluation"]["few_shot"]
    return results

def analyze_results(metrics_path=None, metrics=None, write_path=None, use_wandb=False, make_plots=True):
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
    pd.DataFrame.from_dict(results_flattened, orient="index", columns=["value"]).to_csv(write_path / "results.csv")

    if not make_plots:
        return

    # generate plots
    img_path = write_path / "img"
    img_path.mkdir(exist_ok=True, parents=True)
    online_df = pd.DataFrame(metrics["online"])

    figsize = (20, 10)
    if "few_shot" in metrics["evaluation"]:
        # few shot accuracy
        plt.figure(figsize=figsize)
        for i, few_shot_metrics in enumerate(metrics["evaluation"]["few_shot"]):
            x = [result["examples_seen"] for result in few_shot_metrics]
            y = [result["accuracy"] for result in few_shot_metrics]
            plt.plot(x, y)
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
        # first encounter learning curve area difference
        plt.figure(figsize=figsize)
        x = list(results[f"first_encounter_learning_curve_area_difference"].keys())
        y = list(results[f"first_encounter_learning_curve_area_difference"].values())
        plt.plot(x, y)
        if use_wandb:
            wandb.log({f"chart_first_encounter_lca": plt})
        img = img_path / f"first_encounter_learning_curve_area_difference.pdf"
        plt.savefig(img)
        subprocess.call(f"pdfcrop {img} {img}", shell=True)

        # first encounter learning speed
        plt.figure(figsize=figsize)
        x = list(results[f"first_encounter_learning_speed"].keys())
        y = list(results[f"first_encounter_learning_speed"].values())
        plt.plot(x, y)
        if use_wandb:
            wandb.log({f"chart_first_encounter_learning_speed": plt})
        img = img_path / f"first_encounter_learning_speed.pdf"
        plt.savefig(img)
        subprocess.call(f"pdfcrop {img} {img}", shell=True)

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

            # ---------------------------------------
            # Few shot learning curve area
            plt.figure(figsize=figsize)
            x = list(results[f"few_shot_relearning_curve_area_{i}"].keys())
            y = list(results[f"few_shot_relearning_curve_area_{i}"].values())
            plt.plot(x, y)
            if use_wandb:
                wandb.log({f"chart_few_shot_relearning_curve_area_{i}": plt})
            img = img_path / f"few_shot_relearning_curve_area_{i}.pdf"
            plt.savefig(img)
            subprocess.call(f"pdfcrop {img} {img}", shell=True)

            # Few shot learning curve area
            plt.figure(figsize=figsize)
            x = list(results[f"few_shot_relearning_curve_area_difference_{i}"].keys())
            y = list(results[f"few_shot_relearning_curve_area_difference_{i}"].values())
            plt.plot(x, y)
            if use_wandb:
                wandb.log({f"chart_few_shot_relearning_curve_area_zero_shot_difference_{i}": plt})
            img = img_path / f"few_shot_relearning_curve_area_difference_{i}.pdf"
            plt.savefig(img)
            subprocess.call(f"pdfcrop {img} {img}", shell=True)

            # Few shot learning curve area
            plt.figure(figsize=figsize)
            x = list(results[f"few_shot_learning_speed_{i}"].keys())
            y = list(results[f"few_shot_learning_speed_{i}"].values())
            plt.plot(x, y)
            if use_wandb:
                wandb.log({f"chart_few_shot_learning_speed_{i}": plt})
            img = img_path / f"few_shot_learning_speed_{i}.pdf"
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
    analyze_results(metrics_path=Path('experiments') / sys.argv[1], make_plots=False)

import argparse
import csv
import itertools
import os

import torch

from . import calibrate  # noqa: E402
from . import evaluate  # noqa: E402
from . import util  # noqa: E402
from . import grouping  # noqa: E402
from . import loader  # noqa: E402

TECHNIQUES = {"Histogram binning": calibrate.histogram_binning,
              "Isotonic regression": calibrate.isotonic_regression,
              "Isotonic scaling binning": calibrate.isotonic_scaling_binning,
              }

UGLY_NAMES = {"Histogram binning": "histogram_binning",
              "Isotonic regression": "isotonic_regression",
              "Isotonic scaling binning": "isotonic_scaling_binning"}


def run_experiment(task, scores_re: torch.Tensor, scores_ev: torch.Tensor,
                   labels_re: torch.Tensor, labels_ev: torch.Tensor,
                   training_label_counts: dict, results_path: str,
                   num_groups: int = 5, threshold=.01,
                   recalibration_bins: int = 10, eval_bins: int = 10,
                   tfg=False):
    """Runs a calibration experiment and logs results.
    Args:
        scores_re: Confidence scores for REcalibration
        scores_ev: Confidence scores for EValuation
    """

    tag_groups = grouping.tag_frequency_grouping(training_label_counts, num_groups)

    for technique_name, technique in TECHNIQUES.items():
        print(technique_name)
        recal_set = {}
        eval_set = {}
        calibrated_scores_by_group = {}

        for group_num, tag_group in tag_groups.items():
            recal_set[group_num] = {"scores": scores_re[:, tag_group],
                                    "labels": labels_re[:, tag_group]}
            eval_set[group_num] = {"scores": scores_ev[:, tag_group],
                                   "labels": labels_ev[:, tag_group]}

            t_x, t_y = util.apply_threshold(recal_set[group_num]["scores"],
                                            recal_set[group_num]["labels"],
                                            threshold)
            recal_set[group_num]["thresholded_scores"], recal_set[group_num]["thresholded_labels"] = t_x, t_y

            t_x, t_y = util.apply_threshold(eval_set[group_num]["scores"],
                                            eval_set[group_num]["labels"],
                                            threshold)
            eval_set[group_num]["thresholded_scores"],  eval_set[group_num]["thresholded_labels"] = t_x, t_y

            if tfg:
                # Calibrate each group of scores independently
                if technique_name == "Isotonic regression":
                    calibrated_scores_by_group[group_num] = technique(recal_set[group_num]["thresholded_scores"],
                                                                      eval_set[group_num]["thresholded_scores"],
                                                                      recal_set[group_num]["thresholded_labels"],
                                                                      eval_set[group_num]["thresholded_labels"])
                else:
                    calibrated_scores_by_group[group_num] = technique(recal_set[group_num]["thresholded_scores"],
                                                                      eval_set[group_num]["thresholded_scores"],
                                                                      recal_set[group_num]["thresholded_labels"],
                                                                      eval_set[group_num]["thresholded_labels"],
                                                                      num_bins=recalibration_bins)

        if not tfg:
            # Calibrate all scores together
            if technique_name == "Isotonic regression":
                calibrated_scores = technique(torch.cat([x["thresholded_scores"] for x in recal_set.values()]).cuda(),
                                              torch.cat([x["thresholded_scores"] for x in eval_set.values()]).cuda(),
                                              torch.cat([x["thresholded_labels"] for x in recal_set.values()]).cuda(),
                                              torch.cat([x["thresholded_labels"] for x in eval_set.values()]).cuda())
            else:
                calibrated_scores = technique(torch.cat([x["thresholded_scores"] for x in recal_set.values()]).cuda(),
                                              torch.cat([x["thresholded_scores"] for x in eval_set.values()]).cuda(),
                                              torch.cat([x["thresholded_labels"] for x in recal_set.values()]).cuda(),
                                              torch.cat([x["thresholded_labels"] for x in eval_set.values()]).cuda(),
                                              num_bins=recalibration_bins)

            # Break the scores back into their groups for evaluation
            lower_boundary = 0
            for group_num in range(len(tag_groups)):
                upper_boundary = lower_boundary + eval_set[group_num]["thresholded_scores"].numel()
                calibrated_scores_by_group[group_num] = calibrated_scores[lower_boundary: upper_boundary]
                lower_boundary = upper_boundary

        # Evaluate by group
        # Make this a list of lists for easy chaining when writing results
        metrics = []
        for group_num in range(len(tag_groups)):
            group_metrics = evaluate.evaluate_group(eval_set[group_num]["thresholded_scores"],
                                                    calibrated_scores_by_group[group_num],
                                                    eval_set[group_num]["thresholded_labels"],
                                                    num_bins=eval_bins)
            metrics.append(group_metrics)

        collective_metrics = evaluate.evaluate_group(
            torch.cat([x["thresholded_scores"] for x in eval_set.values()]).cuda(),
            torch.cat(list(calibrated_scores_by_group.values())).cuda(),
            torch.cat([x["thresholded_labels"] for x in eval_set.values()]).cuda(),
            num_bins=eval_bins)
        metrics.append(collective_metrics)

        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([UGLY_NAMES[technique_name], recalibration_bins, eval_bins] +
                            list(itertools.chain(*(metrics))))


def main(args):
    if args.task == "lsr":
        scores_dev, scores_test, labels_dev, labels_test, label_counts = loader.load_lsr()
    else:
        scores_dev, scores_test, labels_dev, labels_test, label_counts = loader.load_ccg()

    # Set up results file
    if args.tfg:
        prefix = "tfg"
    else:
        prefix = "scw"

    groups_header = []

    for i in range(args.num_groups):
        groups_header.append(f"group{i}_rmse_uncal")
        groups_header.append(f"group{i}_rmse_cal")
        groups_header.append(f"group{i}_absolute_change")
        groups_header.append(f"group{i}_relative_change")
        groups_header.append(f"group{i}_num_scores")

    results_folder = "../results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_path = f"{results_folder}/{args.task}_{prefix}_{args.num_groups}_groups.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["technique", "recalibration_bins", "evaluation_bins"] + groups_header +
                        ["collective_pre", "collective_post", "absolute change", "relative change", "numel"])

    # Run experiment
    run_experiment(args.task,  scores_dev, scores_test,  labels_dev, labels_test, label_counts,
                   results_path=results_path, recalibration_bins=args.recalibration_bins, eval_bins=args.eval_bins,
                   num_groups=args.num_groups, threshold=args.threshold, tfg=args.tfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a calibration experiment and saves outcomes in results folder")
    parser.add_argument("--task", choices=["lsr", "ccg"], default="lsr", help="Task to run experiment on")
    parser.add_argument("--num-groups", type=int, default=5,
                        help="Number of groups for TFG recalibration and evaluation")
    parser.add_argument("--threshold", "-t", type=int, default=.01,
                        help="Scores below threshold excluded from recalibration models")
    parser.add_argument("--recalibration-bins", "-r", type=int, default=10, help="Number of bins for recalibration")
    parser.add_argument("--eval-bins", "-e", type=int, default=10, help="Number of evaluation bins")
    parser.add_argument("--tfg", action="store_true")
    args = parser.parse_args()
    main(args)

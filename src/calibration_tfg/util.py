"""Miscellaneous pre- and post-processing utilities for calibration."""

import math

import torch


def get_fixed_width_bin_boundaries(num_bins: int = 10):
    """Calculates bin boundaries for num_bins evenly-spaced bins.

    This can be done without having any confidence scores.

    Args:
        num_bins: Number of evenly-spaced bins to create.
    """

    boundaries = torch.linspace(start=0, end=1, steps=num_bins + 1)
    lower_boundaries = boundaries[:-1]
    upper_boundaries = boundaries[1:]
    return lower_boundaries, upper_boundaries


def get_adaptive_bin_boundaries(scores, bin_size: int = 200, num_bins: int = None):
    """Determines bin boundaries by putting an equal number of items in each bin.

    Args:
        scores: A 1-D or 2-D tensor of confidence scores in the range [0,1].
        bin_size: Number of scores to put in each bin.
        num_bins: Number of bins to create. If num_bins provided, it overrides bin_size.
    """

    with torch.no_grad():
        scores = scores.flatten().sort().values
        if num_bins is not None:
            bin_size = scores.shape[0] / num_bins
        lower_indexes = [i * bin_size for i in range(math.floor(len(scores) / bin_size))]
        lower_boundaries = scores[lower_indexes]

        # Manually set lowest boundary to 0
        lower_boundaries[0] = 0.0

        # Upper boundaries are lower boundaries shifted once, with the highest upper boundary being 1.0
        upper_boundaries = torch.cat((lower_boundaries[1:], torch.tensor(1.0, device="cuda").unsqueeze(0)))

        return lower_boundaries, upper_boundaries


def apply_threshold(scores: torch.Tensor, labels: torch.Tensor, threshold: float = .01):
    """Returns scores (and corresponding labels) greater than or equal to a specified threshold.

    Args:
        scores: A 1-D or 2-D tensor of confidence scores in the range [0,1].
        labels: A tensor containing the same number of elements as scores.
        threshold: The threshold to apply.
    """

    if scores.numel() != labels.numel():
        raise ValueError(f"Number of scores ({scores.numel()}) does not match number of labels ({labels.numel()}).")
    threshold_mask = scores.ge(threshold)
    return scores[threshold_mask], labels[threshold_mask]

"""Evaluation for calibration."""


import torch

from . import util


def ece(scores: torch.Tensor, labels: torch.Tensor,
        bin_size: int = 200, num_bins=None):
    """Estimates expected calibration error (l1).

    Args:
        scores: [n, t] Tensor of confidence scores.
        labels: [n, t] One-hot tensor of labels.
        bin_size: Number of scores to put in each bin.
        num_bins: Number of bins to create. If num_bins provided, it overrides bin_size.
    """
    lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(scores, bin_size, num_bins)

    ece = 0
    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask = scores.ge(lower) & scores.lt(upper)
        proportion_in_bin = bin_mask.double().mean()
        if proportion_in_bin.item() > 0:
            average_label = labels[bin_mask].double().mean()
            average_confidence = scores[bin_mask].double().mean()
            abs_error = torch.abs(average_confidence - average_label)
            weighted_abs_error = proportion_in_bin * abs_error
            ece += weighted_abs_error
    return ece.item()


def rmse(scores: torch.Tensor, labels: torch.Tensor, bin_size: int = 200, num_bins=None):
    """Estimates RMSE-based calibration error (l2).

    Args:
        scores: [n, t] Tensor of confidence scores.
        labels: [n, t] One-hot tensor of labels.
        bin_size: Number of scores to put in each bin.
        num_bins: Number of bins to create. If num_bins provided, it overrides bin_size.
    """

    lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(scores, bin_size, num_bins)

    rmse = 0
    scores = scores.flatten()
    labels = labels.flatten()

    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask = scores.ge(lower) & scores.lt(upper)
        proportion_in_bin = bin_mask.double().mean()
        if proportion_in_bin.item() > 0:
            average_label = labels[bin_mask].double().mean()
            average_confidence = scores[bin_mask].double().mean()
            squared_error = (average_confidence - average_label) ** 2
            weighted_square_error = proportion_in_bin * squared_error
            rmse += weighted_square_error
    rmse = torch.sqrt(rmse).item()
    return rmse


def evaluate_group(uncalibrated_scores, calibrated_scores, labels, num_bins):
    digits = 6
    error_pre = rmse(uncalibrated_scores, labels, num_bins=num_bins)
    error_post = rmse(calibrated_scores, labels, num_bins=num_bins)
    absolute_change = round((error_post - error_pre), digits)
    relative_change = round((error_post - error_pre) / error_pre, digits)
    group_metrics = [error_pre, error_post, absolute_change, relative_change, uncalibrated_scores.numel()]
    return group_metrics

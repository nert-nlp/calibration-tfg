"""Implementation of four calibration methods:
    - histogram binning
    - isotonic regression
    - temperature scaling
    - scaling binning

Temperature scaling adapted from:
https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

Definitions:
    n: number of inputs with predictions
    t: number of possible labels/tags for each input
"""

import logging
import math

import torch

from sklearn.isotonic import IsotonicRegression

from . import util

logger = logging.getLogger("calibration")


def histogram_binning(scores_dev: torch.Tensor, scores_test: torch.Tensor,
                      labels_dev: torch.Tensor, labels_test: torch.Tensor,
                      bin_size: int = 10_000, num_bins=None):
    """Calibrates confidence scores using histogram binning.

    Args:
        scores_dev: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for dev set.
        scores_test: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for test set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        labels_test: [n, t] One-hot tensor of labels for test set.
        bin_size: Number of scores to put in each bin.
        num_bins: Number of bins to create. If num_bins provided, it overrides bin_size.
    """

    logger.info("Starting histogram binning...")

    lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(scores_dev, bin_size, num_bins)

    flattened_scores_dev = scores_dev.reshape(-1)
    flattened_labels_dev = labels_dev.reshape(-1)
    flattened_scores_test = scores_test.clone().reshape(-1)

    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask_dev = flattened_scores_dev.ge(lower) & flattened_scores_dev.lt(upper)
        average_dev_label = flattened_labels_dev[bin_mask_dev].float().mean()
        bin_mask_test = flattened_scores_test.ge(lower) & flattened_scores_test.lt(upper)
        flattened_scores_test[bin_mask_test] = average_dev_label

    calibrated_scores_test = flattened_scores_test

    return calibrated_scores_test


def isotonic_regression(scores_dev: torch.Tensor, scores_test: torch.Tensor,
                        labels_dev: torch.Tensor, labels_test: torch.Tensor):

    """Calibrates confidence scores using scikit-learn implementation of isotonic regression.

        Isotonic regression does not require bins for recalibration.

    Args:
        scores_dev: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for dev set.
        scores_test: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for test set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        labels_test: [n, t] One-hot tensor of labels for test set.
    """

    logger.info("Starting isotonic regression...")

    # Scores need to be moved to CPU for sklearn model
    flattened_scores_dev = scores_dev.reshape(-1).cpu()
    flattened_labels_dev = labels_dev.reshape(-1).cpu()
    flattened_scores_test = (scores_test.reshape(-1)).cpu()

    model = IsotonicRegression(y_min=0, y_max=1)
    model.fit(X=flattened_scores_dev, y=flattened_labels_dev)
    predictions = torch.Tensor(model.predict(flattened_scores_test)).cuda()

    calibrated_scores_test = predictions

    return calibrated_scores_test


def isotonic_scaling_binning(scores_dev: torch.Tensor, scores_test: torch.Tensor,
                             labels_dev: torch.Tensor, labels_test: torch.Tensor,
                             num_bins=None):
    """Calibrates confidence scores using scaling binning (Kumar et al., 2019).

        There's a lot going on here, so we use the same notation from Kumar et al. (2019) and defer to
        the algorithm explanation in the paper for details. Isotonic regression is used as the scaling function.

        Note: Kumar et al. advise against fixed-width binning for scaling binning,
        and instead recommend fixed-size bins.
        We do not test scaling binning with fixed-width binning.

    Args:
        logits_dev: [n, t] Tensor of pre-softmax logits for dev set.
        logits_test: [n, t] Tensor of pre-softmax logits for test set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        labels_test: [n, t] One-hot tensor of labels for test set.
        num_bins: Number of bins to create.
    """

    logger.info("Starting isotonic scaling binning...")

    # Split dev logits in half; t1-t3 notation comes from Kumar (2019)
    halfway = math.floor(scores_dev.shape[0] / 2)

    # Isotonic regression model gets fit to t1
    t1 = scores_dev[:halfway]
    y1 = labels_dev[:halfway]

    # Binning is done with t2
    t2 = scores_dev[halfway:]

    # Scores need to be moved to CPU for sklearn model
    flattened_t1 = t1.cpu()
    flattened_y1 = y1.cpu()
    flattened_t2 = t2.cpu()
    flattened_t3 = scores_test.cpu()

    model = IsotonicRegression(y_min=0, y_max=.99999)
    model.fit(X=flattened_t1, y=flattened_y1)

    # Bin t2 after the scores have been adjusted based on t1 model
    t2_predictions = torch.Tensor(model.predict(flattened_t2)).cuda()
    lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(t2_predictions, num_bins=num_bins)

    t3_predictions = torch.Tensor(model.predict(flattened_t3)).cuda()

    # Make sure none of the predictions over 1
    assert (t3_predictions >= 1.).sum(dim=0) == 0

    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask_test = t3_predictions.ge(lower) & t3_predictions.lt(upper)
        average_iso_scaled_score = t3_predictions[bin_mask_test].float().mean()
        t3_predictions[bin_mask_test] = average_iso_scaled_score

    calibrated_scores_test = t3_predictions

    return calibrated_scores_test

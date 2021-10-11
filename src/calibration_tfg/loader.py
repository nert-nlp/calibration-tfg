"""Functions for loading saved confidence scores and labels."""

import json

import torch


def load_lsr():
    # Load dev scores and labels
    with open("../data/lsr/dev_scores.pt", "rb") as f:
        scores_dev = torch.load(f).cuda()

    with open("../data/lsr/dev_labels.pt", "rb") as f:
        labels_dev = torch.load(f).cuda()

    # Load test scores and labels
    with open("../data/lsr/test_scores.pt", "rb") as f:
        scores_test = torch.load(f).cuda()
    with open("../data/lsr/test_labels.pt", "rb") as f:
        labels_test = torch.load(f).cuda()

    # Get training tag frequencies
    with open("../data/lsr/training_tag_frequencies.json", "r") as f:
        training_tag_frequencies = json.load(f)
    with open("../data/lsr/tag_indexes.json", "r") as f:
        tag_indexes = json.load(f)
    indexed_training_tag_counts = {}
    for (tag, count) in training_tag_frequencies.items():
        tag_index = tag_indexes[tag]
        indexed_training_tag_counts[tag_index] = count

    return scores_dev, scores_test, labels_dev, labels_test, indexed_training_tag_counts


def load_ccg():
    # Load dev scores and labels
    with open("../data/ccg/dev_scores.pt", "rb") as f:
        scores_dev = torch.load(f).cuda()
    with open("../data/ccg/dev_labels.pt", "rb") as f:
        labels_dev = torch.nn.functional.one_hot(torch.load(f), 426).cuda()

    # Load test scores and labels
    with open("../data/ccg/test_scores.pt", "rb") as f:
        scores_test = torch.load(f).cuda()
    with open("../data/ccg/test_labels.pt", "rb") as f:
        labels_test = torch.nn.functional.one_hot(torch.load(f), 426).cuda()

    # Get training tag frequencies
    with open("../data/ccg/training_tag_frequencies.json", "r") as f:
        training_tag_frequencies = json.load(f)
    with open("../data/ccg/tag_indexes.json", "r") as f:
        tag_indexes = json.load(f)
    indexed_training_tag_counts = {}
    for (tag, count) in training_tag_frequencies.items():
        tag_index = tag_indexes[tag]
        indexed_training_tag_counts[tag_index] = count

    return scores_dev, scores_test, labels_dev, labels_test, indexed_training_tag_counts

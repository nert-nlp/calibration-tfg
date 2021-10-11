"""Techniques for grouping tags for calibration."""

from collections import Counter


def tag_frequency_grouping(label_counts: dict, num_groups: int) -> list:
    """Creates groups of similarly frequent labels to calibrate together.

    Make sure label_counts has zeros for possible labels that didn't appear in training.

    Args:
        labels_counts: A dict of label frequencies in training data.
        num_groups: The number of groups to partition labels into.

    Returns:
        A list of lists, where each list is a group of label indexes to calibrate together.
    """

    equal_proportion = 1 / num_groups

    total = sum(label_counts.values())

    groups = {}

    current_proportion = 0
    group_num = 0

    groups[group_num] = []
    for key, count in Counter(label_counts).most_common():
        label_proportion = count / total
        current_proportion += label_proportion
        groups[group_num].append(int(key))
        if current_proportion > equal_proportion and group_num < (num_groups - 1):
            # Create a new group
            group_num += 1
            groups[group_num] = []
            current_proportion = 0

    return groups

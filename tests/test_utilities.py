"""
test_utilities.py
-----------------

Test utility functions.

"""

import numpy as np
from netrd.utilities.entropy import categorized_data
from netrd.utilities.entropy import entropy, joint_entropy, conditional_entropy


def test_categorized_data():
    """Test the function that turn continuous data into categorical."""
    raw = np.array([[1.0, 1.4, 3.0], [2.0, 2.2, 5.0]]).T
    n_bins = 2
    data = categorized_data(raw, n_bins)

    data_true = np.array([[0, 0, 1], [0, 0, 1]]).T
    assert np.array_equal(data, data_true)


def test_entropies():
    """
    Test functions computing entropy, joint entropy and conditional entropy.

    """
    data = np.array([[1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0]]).T
    H = entropy(data[:, 0])
    H_joint = joint_entropy(data)
    H_cond = conditional_entropy(data[:, 1, np.newaxis],
                                 data[:, 0, np.newaxis])

    H_true = 1.0
    H_joint_true = 1 + 1.5 * np.log2(4/3)
    H_cond_true = H_joint - H
    assert H == H_true
    assert H_joint == H_joint_true
    assert H_cond == H_cond_true

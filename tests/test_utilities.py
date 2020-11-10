"""
test_utilities.py
-----------------

Test utility functions.

"""

import numpy as np
from netrd.utilities.entropy import categorized_data
from netrd.utilities.entropy import entropy_from_seq, joint_entropy, conditional_entropy


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
    H = entropy_from_seq(data[:, 0])
    H_joint = joint_entropy(data)
    H_cond = conditional_entropy(data[:, 1, np.newaxis], data[:, 0, np.newaxis])

    H_true = 1.0
    H_joint_true = 3 / 4 + 3 / 4 * np.log2(8 / 3)
    H_cond_true = H_joint - H

    assert np.isclose(H, H_true)
    assert np.isclose(H_joint, H_joint_true)
    assert np.isclose(H_cond, H_cond_true)

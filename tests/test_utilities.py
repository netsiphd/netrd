"""
test_utilities.py
-----------------

Test utility functions.

"""

import numpy as np
from netrd.utilities.entropy import categorized_data
from netrd.utilities.entropy import entropy_from_seq, joint_entropy, conditional_entropy
from netrd.utilities import threshold


def test_thresholds():
    """
    Test the threshold function by testing three underlying thresholding
    methods: range, quantile, and degree.
    """

    mat = np.arange(1, 17, 1).reshape((4, 4))

    for k in range(5):
        thresholded_mat = threshold(mat, 'degree', avg_k=k, remove_self_loops=False)
        assert (thresholded_mat != 0).sum() == 4 * k

    for n in range(17):
        thresholded_mat = threshold(
            mat, 'quantile', quantile=n / 16, remove_self_loops=False
        )
        print(n)
        assert (thresholded_mat != 0).sum() == 16 - n

    thresholded_mat = threshold(
        mat, 'range', cutoffs=[(0, np.inf)], remove_self_loops=False
    )
    assert (thresholded_mat >= 0).all()

    thresholded_mat = threshold(
        mat, 'range', cutoffs=[(-np.inf, 0)], remove_self_loops=False
    )
    assert (thresholded_mat <= 0).all()

    target_mat = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    assert np.array_equal(
        threshold(mat, 'range', cutoffs=[(9, 16)], remove_self_loops=False), target_mat
    )
    assert np.array_equal(
        threshold(mat, 'degree', avg_k=2, remove_self_loops=False), target_mat
    )
    assert np.array_equal(
        threshold(mat, 'quantile', quantile=0.5, remove_self_loops=False), target_mat
    )

    target_mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])

    assert np.array_equal(
        threshold(
            mat, 'range', cutoffs=[(9, 16)], binary=True, remove_self_loops=False
        ),
        target_mat,
    )
    assert np.array_equal(
        threshold(mat, 'degree', avg_k=2, binary=True, remove_self_loops=False),
        target_mat,
    )
    assert np.array_equal(
        threshold(mat, 'quantile', quantile=0.5, binary=True, remove_self_loops=False),
        target_mat,
    )


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

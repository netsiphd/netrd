"""
test_reconstruction.py
----------------------

Test reconstruction algorithms.

"""

import numpy as np
from netrd import reconstruction
from netrd.reconstruction import ConvergentCrossMapping
from netrd.reconstruction import BaseReconstructor


def test_graph_size():
    """
    The number of nodes in a reconstructed graph should be
    equal to the number of sensors in the time series data
    used to reconstruct the graph.
    """
    size = 50
    for label, obj in reconstruction.__dict__.items():
        if label in [
            'PartialCorrelationMatrix',
            'NaiveTransferEntropy' 'OptimalCausationEntropy',
        ]:
            continue
        if isinstance(obj, type) and BaseReconstructor in obj.__bases__:
            TS = np.random.random((size, 125))
            G = obj().fit(TS).threshold_in_range([(-np.inf, np.inf)]).to_graph()
            assert G.order() == size


def test_naive_transfer_entropy():
    """
    Use a smaller data set to test the NaiveTransferEntropy,
    because it is very slow.

    """
    size = 25
    TS = np.random.random((size, 100))
    R = reconstruction.NaiveTransferEntropy()
    G = R.fit(TS).threshold_in_range([(-np.inf, np.inf)]).to_graph()
    assert G.order() == size


def test_oce():
    """
    Test optimal causation entropy using a smaller dataset.
    """

    size = 25
    TS = np.random.random((size, 50))
    R = reconstruction.OptimalCausationEntropy()
    G = R.fit(TS).threshold_in_range([(-np.inf, np.inf)]).to_graph()
    assert G.order() == size


def test_convergent_cross_mapping():
    """
    Examine the outcome of ConvergentCrossMapping with synthetic
    time series data generated from a two-species Lotka-Vottera model.

    """
    filepath = '../data/two_species_coupled_time_series.dat'
    edgelist = {(1, 0), (0, 1)}
    keys = ['weights_matrix', 'pvalues_matrix']

    TS = np.loadtxt(filepath, delimiter=',')
    recon = ConvergentCrossMapping()
    G = (
        recon.fit(TS)
        .remove_self_loops()
        .threshold_in_range(cutoffs=[(-np.inf, np.inf)])
        .to_graph()
    )
    el = set(G.edges())
    res = recon.results.keys()

    assert G.is_directed()
    assert el == edgelist
    assert all(k in res for k in keys)


def test_partial_correlation():
    """
    The PartialCorrelationMatrix has many parameterizations
    that ought to be tested differently. Otherwise, this should be
    equivalent to `test_graph_size`.
    """
    for resid in [True, False]:
        for index in [0, None]:
            for size in [10, 100]:
                if index is None and resid is True:
                    pass  # this shouldn't be a valid parameterization
                else:
                    TS = np.random.random((size, 50))
                    R = reconstruction.PartialCorrelationMatrix()
                    R = R.fit(TS, index=index, of_residuals=resid)
                    G = R.threshold_in_range([(-np.inf, np.inf)]).to_graph()
                    if index is None:
                        assert G.order() == size
                    else:
                        assert G.order() == (size - 1)


def test_thresholds():
    """
    Test the threshold function by testing three underlying thresholding
    methods: range, quantile, and degree.
    """

    R = reconstruction.BaseReconstructor()
    R._matrix = np.arange(1, 17, 1).reshape((4, 4))

    for k in range(5):
        R._matrix = np.arange(1, 17, 1).reshape((4, 4))
        thresholded_mat = R.threshold('degree', avg_k=k).to_dense()
        assert (thresholded_mat != 0).sum() == 4 * k

    for n in range(17):
        R._matrix = np.arange(1, 17, 1).reshape((4, 4))
        thresholded_mat = R.threshold('quantile', quantile=n / 16).to_dense()
        assert (thresholded_mat != 0).sum() == 16 - n

    R._matrix = np.arange(1, 17, 1).reshape((4, 4))
    thresholded_mat = R.threshold('range', cutoffs=[(0, np.inf)]).to_dense()
    assert (thresholded_mat >= 0).all()

    R._matrix = np.arange(1, 17, 1).reshape((4, 4))
    thresholded_mat = R.threshold('range', cutoffs=[(-np.inf, 0)]).to_dense()
    assert (thresholded_mat <= 0).all()

    target_mat = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    R._matrix = np.arange(1, 17, 1).reshape((4, 4))
    assert np.array_equal(
        R.threshold('range', cutoffs=[(9, 16)]).to_dense(), target_mat
    )

    R._matrix = np.arange(1, 17, 1).reshape((4, 4))
    assert np.array_equal(R.threshold('degree', avg_k=2).to_dense(), target_mat)

    R._matrix = np.arange(1, 17, 1).reshape((4, 4))
    assert np.array_equal(R.threshold('quantile', quantile=0.5).to_dense(), target_mat)

    target_mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    R._matrix = np.arange(1, 17, 1).reshape((4, 4))
    assert np.array_equal(
        R.threshold('range', cutoffs=[(9, 16)]).binarize().to_dense(), target_mat,
    )

    R._matrix = np.arange(1, 17, 1).reshape((4, 4))
    assert np.array_equal(
        R.threshold('degree', avg_k=2, binary=True).binarize().to_dense(), target_mat,
    )

    R._matrix = np.arange(1, 17, 1).reshape((4, 4))
    assert np.array_equal(
        R.threshold('quantile', quantile=0.5).binarize().to_dense(), target_mat,
    )

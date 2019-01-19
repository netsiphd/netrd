"""
test_reconstruction.py
----------------------

Test reconstruction algorithms.

"""

import numpy as np
from netrd import reconstruction
from netrd.reconstruction import ConvergentCrossMappingReconstructor
from netrd.reconstruction import BaseReconstructor


def test_graph_size():
    """
    The number of nodes in a reconstructed graph should be
    equal to the number of sensors in the time series data
    used to reconstruct the graph.
    """
    for label, obj in reconstruction.__dict__.items():
        if label in [
                'PartialCorrelationMatrixReconstructor',
                'NaiveTransferEntropyReconstructor'
        ]:
            continue
        if isinstance(obj, type) and BaseReconstructor in obj.__bases__:
            for size in [10, 100]:
                TS = np.random.random((size, 250))
                G = obj().fit(TS)
                assert G.order() == size


def test_naive_transfer_entropy():
    """
    Use a smaller data set to test the NaiveTransferEntropyReconstructor,
    because it is very slow.

    """
    size = 50
    TS = np.random.random((size, 100))
    G = reconstruction.NaiveTransferEntropyReconstructor().fit(TS, delay_max=2)
    assert G.order() == size


def test_convergent_cross_mapping():
    """
    Examine the outcome of ConvergentCrossMappingReconstructor with synthetic
    time series data generated from a two-species Lotka-Vottera model.

    """
    filepath = '../data/two_species_coupled_time_series.dat'
    edgelist = {(1, 0), (0, 1)}
    keys = ['graph', 'correlation', 'pvalue']

    TS = np.loadtxt(filepath, delimiter=',')
    recon = ConvergentCrossMappingReconstructor()
    G = recon.fit(TS)
    el = set(G.edges())
    res = recon.results.keys()

    assert el == edgelist
    assert all(k in res for k in keys)


def test_partial_correlation():
    """
    The PartialCorrelationMatrixReconstructor has many parameterizations
    that ought to be tested differently. Otherwise, this should be 
    equivalent to `test_graph_size`.
    """
    for resid in [True, False]:
        for index in [0, None]:
            for size in [10, 100]:
                if index is None and resid is True:
                    pass # this shouldn't be a valid parameterization
                else:
                    TS = np.random.random((size, 50))
                    G = reconstruction.PartialCorrelationMatrixReconstructor().fit(TS, index=index)
                    if index is None:
                        assert G.order() == size
                    else:
                        assert G.order() == (size - 1)

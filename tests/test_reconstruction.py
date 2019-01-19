"""
test_reconstruction.py
----------------------

Test reconstruction algorithms.

"""

import numpy as np
from netrd import reconstruction
from netrd.reconstruction import ConvergentCrossMappingReconstructor


def test_graph_size():
    """
    The number of nodes in a reconstructed graph should be
    equal to the number of sensors in the time series data
    used to reconstruct the graph.
    """
    for size in [10, 100, 1000]:
        TS = np.random.random((size, 500))
        G = reconstruction.CorrelationMatrixReconstructor().fit(TS)
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

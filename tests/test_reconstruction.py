"""
test_reconstruction.py
----------------------

Test reconstruction algorithms.

"""

import numpy as np
from netrd import reconstruction


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

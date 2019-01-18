"""
test_reconstruction.py
----------------------

Test reconstruction algorithms.

"""

import numpy as np
from netrd import reconstruction
from netrd.reconstruction import BaseReconstructor


def test_graph_size():
    """
    The number of nodes in a reconstructed graph should be
    equal to the number of sensors in the time series data
    used to reconstruct the graph.
    """
    for label, obj in reconstruction.__dict__.items():
        if label == 'PartialCorrelationMatrixReconstructor':
            continue
        if isinstance(obj, type) and BaseReconstructor in obj.__bases__:
            for size in [10, 100, 1000]:
                TS = np.random.random((size, 500))
                G = obj().fit(TS)
                assert G.order() == size

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
                    TS = np.random.random((size, 200))
                    G = reconstruction.PartialCorrelationMatrixReconstructor().fit(TS, index=index)
                    if index is None:
                        assert G.order() == size
                    else:
                        assert G.order() == (size - 1)


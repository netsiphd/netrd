"""
random.py
---------

Reconstruct a network from a random matrix
not taking the time series into account.

author: Brennan Klein
email: klein.br@husky.neu.edu
Submitted as part of the 2019 NetSI Collabathon.

"""

from .base import BaseReconstructor
import networkx as nx
import numpy as np
from ..utilities import create_graph, threshold


class RandomReconstructor(BaseReconstructor):
    """Returns a random graph (dummy class)."""

    def fit(self, TS, threshold_type='range', **kwargs):
        """Return a random correlation matrix with a threshold.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            array consisting of :math:`L` observations from :math:`N` sensors.

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        -------
        G (nx.Graph)
            a reconstructed graph with :math:`N` nodes.

        """
        N, L = TS.shape
        W = np.random.rand(N, N)
        A = threshold(W, threshold_type, **kwargs)
        G = create_graph(A)
        self.results['graph'] = G
        self.results['weights_matrix'] = W
        self.results['thresholded_matrix'] = A
        return G

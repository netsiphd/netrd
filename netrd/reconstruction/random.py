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
import numpy as np


class RandomReconstructor(BaseReconstructor):
    """Returns a random graph (dummy class)."""

    def fit(self, TS):
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

        self.results['weights_matrix'] = W
        self.update_matrix(W)
        return self

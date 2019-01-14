"""
correlation_matrix.py
---------------------

Reconstruction of graphs using the correlation matrix.

author: Stefan McCabe
email: stefanmccabe at gmail dot com
submitted as part of the 2019 NetSI Collabathon
"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx


class CorrelationMatrixReconstructor(BaseReconstructor):
    def fit(self, T, num_eigs=10, quantile=0.9):
        """
        Reconstruct a network from time series data using a regularized
        form of the precision matrix. After [this tutorial](
        https://bwlewis.github.io/correlation-regularization/) in R.

        Params
        ------
        T (np.ndarray): Array consisting of $T$ observations from $N$ sensors
        num_eigs (int): The number of eigenvalues to use. This corresponds
        to the amount of regularization.
        quantile (float): The threshold above which to create an edge, e.g.,
        only create edges between elements above the 90th quantile of the
        correlation matrix.

        Returns
        -------
        G: a reconstructed graph.

        """

        num_sensors = T.shape[0]

        # get the correlation matrix
        X = np.corrcoef(T)

        # get its eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(X)
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        # construct the precision matrix
        P = (vecs[:, :num_eigs]) @\
            (1 / (vals[:num_eigs]).reshape(num_eigs, 1) * (vecs[:, :num_eigs]).T)
        P = P / (np.sqrt(np.diag(P)).reshape(num_sensors, 1) @\
                 np.sqrt(np.diag(P)).reshape(1, num_sensors))

        # threshold the precision matrix
        A = P * (P > np.percentile(P, quantile * 100))

        # construct the network
        self.results['graph'] = nx.from_numpy_array(A)
        G = self.results['graph']

        return G

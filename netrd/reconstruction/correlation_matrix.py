"""
correlation_matrix.py
---------------------

Reconstruction of graphs using the correlation matrix.

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon

"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx


class CorrelationMatrixReconstructor(BaseReconstructor):
    def fit(self, TS, num_eigs=10, quantile=0.9):
        """
        Reconstruct a network from time series data using a regularized
        form of the precision matrix. After [this tutorial](
        https://bwlewis.github.io/correlation-regularization/) in R.

        Params
        ------
        TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors
        num_eigs (int): The number of eigenvalues to use. (This corresponds
        to the amount of regularization.) The number of eigenvalues used must
        be less than $N$.
        quantile (float): The threshold above which to create an edge, e.g.,
        only create edges between elements above the 90th quantile of the
        correlation matrix.

        Returns
        -------
        G: a reconstructed graph.

        """

        N = TS.shape[0]

        if num_eigs > N:
            raise ValueError("The number of eigenvalues used must be less "
                             "than the number of sensors.")

        # get the correlation matrix
        X = np.corrcoef(TS)

        # get its eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(X)
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        # construct the precision matrix and store it
        P = (vecs[:, :num_eigs]) @\
            (1 / (vals[:num_eigs]).reshape(num_eigs, 1) * (vecs[:, :num_eigs]).T)
        P = P / (np.sqrt(np.diag(P)).reshape(N, 1) @\
                 np.sqrt(np.diag(P)).reshape(1, N))
        self.results['matrix'] = P

        # threshold the precision matrix
        A = P * (P > np.percentile(P, quantile * 100))

        # construct the network
        self.results['graph'] = nx.from_numpy_array(A)
        G = self.results['graph']

        return G

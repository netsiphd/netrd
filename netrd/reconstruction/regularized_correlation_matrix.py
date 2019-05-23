"""
regularized_correlation_matrix.py
---------------------

Reconstruction of graphs using the correlation matrix.

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon

"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx
from ..utilities import create_graph, threshold


class RegularizedCorrelationMatrixReconstructor(BaseReconstructor):
    """Uses a regularized form of the precision matrix."""

    def fit(self, TS, num_eigs=10, threshold_type='quantile', **kwargs):
        """Uses a regularized form of the precision matrix.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`. For details see [1]_.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N` sensors

        num_eigs (int)
            The number of eigenvalues to use. (This corresponds to the
            amount of regularization.) The number of eigenvalues used must
            be less than :math:`N`.

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using '`**kwargs`'.

        Returns
        -------
        G (nx.Graph)
            a reconstructed graph.


        References
        ----------

        .. [1] https://bwlewis.github.io/correlation-regularization/

        """
        N = TS.shape[0]

        if num_eigs > N:
            raise ValueError(
                "The number of eigenvalues used must be less "
                "than the number of sensors."
            )

        # get the correlation matrix
        X = np.corrcoef(TS)

        # get its eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(X)
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        # construct the precision matrix and store it
        P = (vecs[:, :num_eigs]) @ (
            1 / (vals[:num_eigs]).reshape(num_eigs, 1) * (vecs[:, :num_eigs]).T
        )
        P = P / (np.sqrt(np.diag(P)).reshape(N, 1) @ np.sqrt(np.diag(P)).reshape(1, N))
        self.results['weights_matrix'] = P

        # threshold the precision matrix
        A = threshold(P, threshold_type, **kwargs)
        self.results['thresholded_matrix'] = A

        # construct the network
        self.results['graph'] = create_graph(A)
        G = self.results['graph']

        return G

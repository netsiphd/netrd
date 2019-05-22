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
from ..utilities import create_graph, threshold


class CorrelationMatrix(BaseReconstructor):
    """Uses the correlation matrix."""

    def fit(self, TS, num_eigs=None, threshold_type='range', **kwargs):
        """Uses the correlation matrix.

        If ``num_eigs`` is `None`, perform the reconstruction using the
        unregularized correlation matrix. Otherwise, construct a regularized
        precision matrix using ``num_eigs`` eigenvectors and eigenvalues of the
        correlation matrix. For details on the regularization method, see [1].
        The results dictionary also stores the raw correlation matrix
        (potentially regularized) as `'weights_matrix'` and the thresholded
        version of the correlation matrix as `'thresholded_matrix'`. For
        details see [2]_.

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
            using `**kwargs`.

        Returns
        -------
        G (nx.Graph)
            a reconstructed graph.

        References
        ----------
        .. [1] https://bwlewis.github.io/correlation-regularization/

        .. [2] https://github.com/valeria-io/visualising_stocks_correlations/blob/master/corr_matrix_viz.ipynb

        """
        # get the correlation matrix
        cor = np.corrcoef(TS)

        if num_eigs:
            N = TS.shape[0]
            if num_eigs > N:
                raise ValueError(
                    "The number of eigenvalues used must be less "
                    "than the number of sensors."
                )

            # get eigenvalues and eigenvectors of the correlation matrix
            vals, vecs = np.linalg.eigh(cor)
            idx = vals.argsort()[::-1]
            vals = vals[idx]
            vecs = vecs[:, idx]

            # construct the precision matrix and store it
            P = (vecs[:, :num_eigs]) @ (
                1 / (vals[:num_eigs]).reshape(num_eigs, 1) * (vecs[:, :num_eigs]).T
            )
            P = P / (
                np.sqrt(np.diag(P)).reshape(N, 1) @ np.sqrt(np.diag(P)).reshape(1, N)
            )
            mat = P
        else:
            mat = cor

        # store the appropriate source matrix
        self.results['weights_matrix'] = mat

        # threshold the correlation matrix
        A = threshold(mat, threshold_type, **kwargs)
        self.results['thresholded_matrix'] = A

        # construct the network
        self.results['graph'] = create_graph(A)
        G = self.results['graph']

        return G

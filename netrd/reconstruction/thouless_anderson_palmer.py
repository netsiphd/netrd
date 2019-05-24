"""
thouless_anderson_palmer.py
---------------------
Reconstruction of graphs using a Thouless-Anderson-Palmer
mean field approximation
author: Brennan Klein
email: brennanjamesklein at gmail dot com
submitted as part of the 2019 NetSI Collabathon
"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx
import scipy as sp
from scipy import linalg
from ..utilities import create_graph, threshold


class ThoulessAndersonPalmer(BaseReconstructor):
    """Uses Thouless-Anderson-Palmer mean field approximation."""

    def fit(self, TS, threshold_type='range', **kwargs):
        """Infer inter-node coupling weights using a Thouless-Anderson-Palmer mean
        field approximation.

        From the paper: "Similar to naive mean field, TAP works well only
        in the regime of large sample sizes and small coupling variability.
        However, this method leads to poor inference results in the regime
        of small sample sizes and/or large coupling variability." For
        details see [1]_.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N`
            sensors.

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        -------
        G (nx.Graph or nx.DiGraph)
            a reconstructed graph.

        References
        -----------

        .. [1] https://github.com/nihcompmed/network-inference/blob/master/sphinx/codesource/inference.py

        """

        N, L = np.shape(TS)  # N nodes, length L
        m = np.mean(TS, axis=1)  # empirical value

        # A matrix
        A = 1 - m ** 2
        A_inv = np.diag(1 / A)
        A = np.diag(A)
        ds = TS.T - m  # equal time correlation
        C = np.cov(ds, rowvar=False, bias=True)
        C_inv = linalg.inv(C)

        s1 = TS[:, 1:]  # one-step-delayed correlation

        ds1 = s1.T - np.mean(s1, axis=1)
        D = cross_cov(ds1, ds[:-1])

        # predict naive mean field W:
        B = np.dot(D, C_inv)
        W_NMF = np.dot(A_inv, B)

        # TAP part: solving for Fi in the following equation
        # F(1-F)**2) = (1-m**2)sum_j W_NMF**2(1-m**2) ==> 0<F<1

        step = 0.001
        nloop = int(0.33 / step) + 2

        W2_NMF = W_NMF ** 2

        temp = np.empty(N)
        F = np.empty(N)

        for i in range(N):
            temp[i] = (1 - m[i] ** 2) * np.sum(W2_NMF[i, :] * (1 - m[:] ** 2))

            y = -1.0
            iloop = 0

            while y < 0 and iloop < nloop:
                x = iloop * step
                y = x * (1 - x) ** 2 - temp[i]
                iloop += 1

            F[i] = x

        # A_TAP matrix
        A_TAP = np.empty(N)
        for i in range(N):
            A_TAP[i] = A[i, i] * (1 - F[i])

        A_TAP_inv = np.diag(1 / A_TAP)

        # predict W:
        W = np.dot(A_TAP_inv, B)
        self.results['weights_matrix'] = W

        # threshold the network
        W_thresh = threshold(W, threshold_type, **kwargs)
        self.results['thresholded_matrix'] = W_thresh

        # construct the network
        self.results['graph'] = create_graph(W_thresh)
        G = self.results['graph']

        return G


def cross_cov(a, b):
    """
    cross_covariance
    a,b -->  <(a - <a>)(b - <b>)>  (axis=0)
    """
    da = a - np.mean(a, axis=0)
    db = b - np.mean(b, axis=0)

    return np.matmul(da.T, db) / a.shape[0]

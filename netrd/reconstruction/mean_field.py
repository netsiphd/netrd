"""
exact_mean_field.py
---------------------
Reconstruction of graphs using the exact mean field
author: Brennan Klein
email: brennanjamesklein at gmail dot com
submitted as part of the 2019 NetSI Collabathon
"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx
import scipy as sp
from scipy import linalg
from scipy.integrate import quad
from scipy.optimize import fsolve
from ..utilities import create_graph, threshold


class MeanField(BaseReconstructor):
    def fit(
        self, TS, exact=True, stop_criterion=True, threshold_type='range', **kwargs
    ):
        """Infer inter-node coupling weights using a mean field approximation.

        From the paper: "Exact mean field (eMF) is another mean field
        approximation, similar to naive mean field and thouless anderson
        palmer. We can improve the performance of this method by adding our
        stopping criterion. In general, eMF outperforms nMF and TAP, but it
        is still worse than FEM and MLE, especially in the limit of small
        sample sizes and large coupling variability." For details see [1]_.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N` sensors.

        exact (bool)
            If True, use the exact mean field approximation. If False, use the
            naive mean field approximation.

        stop_criterion (bool)
            If True, prevent overly-long runtimes. Only applies for exact mean
            field.

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
        ----------

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

        if exact:
            # ---------------------------------------------------------------
            fun1 = (
                lambda x, H: (1 / np.sqrt(2 * np.pi))
                * np.exp(-x ** 2 / 2)
                * np.tanh(H + x * np.sqrt(delta))
            )

            fun2 = (
                lambda x: (1 / np.sqrt(2 * np.pi))
                * np.exp(-x ** 2 / 2)
                * (1 - np.square(np.tanh(H + x * np.sqrt(delta))))
            )

            W = np.empty((N, N))

            nloop = 100

            for i0 in range(N):
                cost = np.zeros(nloop + 1)
                delta = 1.0

                def integrand(H):
                    """
                    Return the integrand of this function
                    """
                    y, err = quad(fun1, -np.inf, np.inf, args=(H,))

                    return y - m[i0]

                for iloop in range(1, nloop):
                    H = fsolve(integrand, 0.0)
                    H = float(H)

                    a, err = quad(fun2, -np.inf, np.inf)
                    a = float(a)

                    if a != 0:
                        delta = (1 / (a ** 2)) * np.sum(
                            (B[i0, :] ** 2) * (1 - m[:] ** 2)
                        )
                        W_temp = B[i0, :] / a

                    H_temp = np.dot(TS[:, :-1].T, W_temp)
                    cost[iloop] = np.mean((s1.T[:, i0] - np.tanh(H_temp)) ** 2)

                    if stop_criterion and cost[iloop] >= cost[iloop - 1]:
                        break

                W[i0, :] = W_temp[:]
        else:
            W = np.dot(A_inv, B)

        # threshold the network
        W_thresh = threshold(W, threshold_type, **kwargs)

        # construct the network

        self.results['graph'] = create_graph(W_thresh)
        self.results['weights_matrix'] = W
        self.results['thresholded_matrix'] = W_thresh
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

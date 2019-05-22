"""
maximum_likelihood_estimation.py
---------------------
Reconstruction of graphs using maximum likelihood estimation
author: Brennan Klein
email: brennanjamesklein at gmail dot com
submitted as part of the 2019 NeTSI Collabathon
"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx
from ..utilities import create_graph, threshold


class MaximumLikelihoodEstimation(BaseReconstructor):
    """Uses maximum likelihood estimation."""

    def fit(self, TS, rate=1.0, stop_criterion=True, threshold_type='degree', **kwargs):
        """Infer inter-node coupling weights using maximum likelihood estimation
        methods.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N` sensors.

        rate (float)
            rate term in maximum likelihood

        stop_criterion (bool)
            if True, prevent overly-long runtimes

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using '`**kwargs`'.

        Returns
        -------
        G (nx.Graph or nx.DiGraph)
            a reconstructed graph.

        References
        ----------

        .. [1] https://github.com/nihcompmed/network-inference/blob/master/sphinx/codesource/inference.py

        """

        N, L = np.shape(TS)  # N nodes, length L
        rate = rate / L

        s1 = TS[:, :-1]
        W = np.zeros((N, N))

        nloop = 10000
        for i0 in range(N):
            st1 = TS[i0, 1:]  # time series activity of single node

            w = np.zeros(N)
            h = np.zeros(L - 1)
            cost = np.full(nloop, 100.0)

            for iloop in range(nloop):
                dw = np.dot(s1, (st1 - np.tanh(h)))

                w += rate * dw
                h = np.dot(s1.T, w)

                cost[iloop] = ((st1 - np.tanh(h)) ** 2).mean()

                if stop_criterion and cost[iloop] >= cost[iloop - 1]:
                    break

            W[i0, :] = w

        # threshold the network
        W_thresh = threshold(W, threshold_type, **kwargs)

        # construct the network

        self.results['graph'] = create_graph(W_thresh)
        self.results['weights_matrix'] = W
        self.results['thresholded_matrix'] = W_thresh
        G = self.results['graph']

        return G

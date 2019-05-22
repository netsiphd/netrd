"""
free_energy_minimization.py
---------------------------
Reconstruction of graphs by minimizing a free energy of your data
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


class FreeEnergyMinimization(BaseReconstructor):
    """Applies free energy principle."""

    def fit(self, TS, threshold_type='degree', **kwargs):
        """Infer inter-node coupling weights by minimizing a free energy over the
        data structure.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`. For details see [1]_.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math.`N`
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
        ----------

        .. [1] https://github.com/nihcompmed/network-inference/blob/master/sphinx/codesource/inference.py

        """

        N, L = np.shape(TS)  # N nodes, length L
        m = np.mean(TS[:, :-1], axis=1)  # model average
        ds = TS[:, :-1].T - m  # discrepancy
        t1 = L - 1  # time limit

        # covariance of the discrepeancy
        c = np.cov(ds, rowvar=False, bias=True)

        c_inv = linalg.inv(c)  # inverse
        dst = ds.T  # discrepancy at time t

        # empty matrix to populate w/ inferred couplings
        W = np.empty((N, N))

        nloop = 10000  # failsafe

        for i0 in range(N):  # for each node

            TS1 = TS[i0, 1:]  # take its entire time series
            h = TS1  # calculate the the local field

            cost = np.full(nloop, 100.0)

            for iloop in range(nloop):

                h_av = np.mean(h)  # average local field
                hs_av = np.dot(dst, h - h_av) / t1  # deltaE_i delta\sigma_k
                w = np.dot(hs_av, c_inv)  # expectation under model

                h = np.dot(TS[:, :-1].T, w[:])  # estimate of local field
                TS_model = np.tanh(h)  # under kinetic Ising model

                # discrepancy cost
                cost[iloop] = np.mean((TS1[:] - TS_model[:]) ** 2)

                if cost[iloop] >= cost[iloop - 1]:
                    break  # if it increases, break

                # complicated, but this seems to be the estimate of W_i
                h *= np.divide(
                    TS1, TS_model, out=np.ones_like(TS1), where=TS_model != 0
                )

            W[i0, :] = w[:]

        # threshold the network
        W_thresh = threshold(W, threshold_type, **kwargs)

        # construct the network

        self.results['graph'] = create_graph(W_thresh)
        self.results['weights_matrix'] = W
        self.results['thresholded_matrix'] = W_thresh
        G = self.results['graph']

        return G

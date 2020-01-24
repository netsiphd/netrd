"""
granger_causality.py
--------------

Graph reconstruction algorithm based on [1].

[1] P. Desrosiers, S. Labrecque, M. Tremblay, M. Bélanger, B. De Dorlodot,
D. C. Côté, "Network inference from functional experimental data", Proc. SPIE
9690, Clinical and Translational Neurophotonics; Neural Imaging and Sensing;
and Optogenetics and Optical Manipulation, 969019 (2016);

author: Charles Murphy
email: charles.murphy.1@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.
"""

import numpy as np

from .base import BaseReconstructor
from sklearn.linear_model import LinearRegression
from ..utilities import create_graph, threshold


class GrangerCausality(BaseReconstructor):
    """Uses the Granger causality between nodes."""

    def fit(self, TS, lag=1, threshold_type="range", **kwargs):
        r"""Reconstruct a network based on the Granger causality. To evaluate
        the effect of a time series :math:`j` over another, :math:`i`, it first
        evaluates the error :math:`e_1` given by an autoregressive model fit
        with :math:`i` alone. Then, it evaluates another error :math:`e_2`
        given by an autoregressive model trained to correlate the future of
        :math:`i` with the past of :math:`i` and :math:`j`. The Granger
        causality of node :math:`j` over :math:`i` is simply given by
        :math:`log(var(e_1) / var(e_2))``.

        It reconstructs the network by calculating the Granger
        causality for each pair of nodes.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N`
            sensors.

        lag (int)
            Time lag to consider.

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        --------

        G (nx.Graph)
            A reconstructed graph with :math:`N` nodes.

        """

        n = TS.shape[0]
        W = np.zeros([n, n])

        for i in range(n):
            xi, yi = GrangerCausality.split_data(TS[i, :], lag)

            for j in range(n):
                xj, yj = GrangerCausality.split_data(TS[j, :], lag)
                xij = np.concatenate([xi, xj], axis=-1)
                reg1 = LinearRegression().fit(xi, yi)
                reg2 = LinearRegression().fit(xij, yi)
                err1 = yi - reg1.predict(xi)
                err2 = yi - reg2.predict(xij)

                std_i = np.std(err1)
                std_ij = np.std(err2)

                if std_i == 0:
                    W[j, i] = -99999999
                elif std_ij == 0:
                    W[j, i] = 99999999
                else:
                    W[j, i] = np.log(std_i) - np.log(std_ij)

        self.results["weights_matrix"] = W
        # threshold the network
        W_thresh = threshold(W, threshold_type, **kwargs)
        self.results["thresholded_matrix"] = W_thresh

        # construct the network
        self.results["graph"] = create_graph(W_thresh)
        G = self.results["graph"]

        return G

    @staticmethod
    def split_data(TS, lag):
        """From a single node time series, return a training dataset with
        corresponding targets.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N`
            sensors.

        lag (int)
            Time lag to consider.

        Returns
        -------

        inputs (np.ndarray)
            Training data for the inputs.

        targets (np.ndarray)
            Training data for the targets.

        """
        T = len(TS)
        inputs = np.zeros([T - lag - 1, lag])
        targets = np.zeros(T - lag - 1)

        for t in range(T - lag - 1):
            inputs[t, :] = TS[t : lag + t]
            targets[t] = TS[t + lag]

        return inputs, targets

"""
time_granger_causality.py
--------------

Graph reconstruction algorithm based on [1].

[1] P. Desrosiers, S. Labrecque, M. Tremblay, M. Bélanger, B. De Dorlodot,
D. C. Côté, "Network inference from functional experimental data", Proc. SPIE
9690, Clinical and Translational Neurophotonics; Neural Imaging and Sensing; and
Optogenetics and Optical Manipulation, 969019 (2016);

author: Charles Murphy
email: charles.murphy.1@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.
"""

import networkx as nx
import numpy as np

from .base import BaseReconstructor
from sklearn.linear_model import LinearRegression
from ..utilities import create_graph, threshold


class TimeGrangerCausality(BaseReconstructor):
    """Infers Granger causality."""

    def fit(self, TS, lag=1, threshold_type='range', **kwargs):
        """Reconstruct a network based on the Granger causality.

        To evaluate the effect of a time series (j) over another (i), it
        first evaluates the error e2 given by an autoregressive model
        fitted to (i) alone. Then, it evaluates another error e2 given by
        an autoregressive model training to correlate the future of (i)
        with the past of (i) and (j). The Granger causality of node (j)
        over (i) is simply given by :math:`\log(var(e1) / var(e2))`. It
        constructs the network by calculating the Granger causality for
        each pair of nodes.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`.

        Parameters
        ----------
        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N`
            sensors.

        lag (int)
            Time lag.

        threshold_type (str)
            Which thresholding function to use on the matrix of

        weights
            See `netrd.utilities.threshold.py` for documentation. Pass
            additional arguments to the thresholder using ``**kwargs``.

        Returns
        -------

        G (nx.Graph)
            A reconstructed graph with :math:`N` nodes.

        References
        ----------
        .. [1] P. Desrosiers, S. Labrecque, M. Tremblay, M. Bélanger, B. De
               Dorlodot, D. C. Côté, "Network inference from functional
               experimental data", Proc. SPIE 9690, Clinical and
               Translational Neurophotonics; Neural Imaging and Sensing;
               and Optogenetics and Optical Manipulation, 969019 (2016);

        """
        n = TS.shape[0]
        self.results['weights_matrix'] = np.zeros([n, n])
        for i in range(n):
            xi, yi = get_training_data(TS[i, :], lag)
            for j in range(n):
                xj, yj = get_training_data(TS[j, :], lag)
                X, Y = np.concatenate([xi, xj]), np.concatenate([yi, yj])
                reg1 = LinearRegression().fit(xi, yi)
                reg2 = LinearRegression().fit(X, Y)
                err1 = yi - reg1.predict(xi)
                err2 = Y - reg2.predict(X)
                self.results['weights_matrix'][j, i] = np.log(
                    np.std(err1) / np.std(err2)
                )

        # threshold the network
        W_thresh = threshold(self.results['weights_matrix'], threshold_type, **kwargs)
        self.results['thresholded_matrix'] = W_thresh

        # construct the network
        self.results['graph'] = create_graph(W_thresh)
        G = self.results['graph']

        return G


def get_training_data(TS, lag):
    """From a single node time series, it returns a training dataset with
    corresponding targets.

    Parameters
    ----------

    TS (np.ndarray)
        Array consisting of :math:`L` observations from :math:`N` sensors.

    lag (int): Time lag.

    Returns
    -------

    inputs (np.ndarray): Training data for the inputs.

    targets (np.ndarray): Training data for the targets.

    """
    T = len(TS)
    inputs = np.zeros([T - lag - 1, lag])
    targets = np.zeros(T - lag - 1)

    for t in range(T - lag - 1):
        inputs[t, :] = TS[t : lag + t]
        targets[t] = TS[t + lag + 1]

    return inputs, targets

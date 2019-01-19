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

class TimeGrangerCausalityReconstructor(BaseReconstructor):
    def fit(self, TS, lag=1):
        """A brief one-line description of the algorithm goes here.

        A short paragraph may follow. The paragraph may include $latex$ by
        enclosing it in dollar signs $\textbf{like this}$.

        Params
        ------
        TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.

        lag (int): Time lag.

        Returns
        -------
        G (nx.Graph): A reconstructed graph with $N$ nodes.

        """

        n = TS.shape[0]
        self.results['weights'] = np.zeros([n, n])
        for i in range(n):
            xi, yi = get_training_data(TS[i, :], lag)
            for j in range(n):
                xj, yj = get_training_data(TS[j, :], lag)
                X, Y = np.concatenate([xi, xj]), np.concatenate([yi, yj])
                reg1 = LinearRegression().fit(xi, yi)
                reg2 = LinearRegression().fit(X, Y)
                err1 = yi - reg1.predict(xi)
                err2 = Y - reg2.predict(X)
                self.results['weights'][i, j] = np.log(np.std(err2) / np.std(err1))


        G = nx.from_numpy_array(self.results['weights'])

        self.results['graph'] = G

        return G

def get_training_data(TS, lag):
    """From a single node time series, it returns a training dataset with
    corresponding targets.

    Params
    ------

    TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.

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
        inputs[t, :] = TS[t:lag + t]
        targets[t] = TS[t + lag + 1]

    return inputs, targets


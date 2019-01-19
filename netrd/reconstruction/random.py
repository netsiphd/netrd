"""
random.py
---------

Reconstruct a network from a random matrix 
not taking the time series into account.

author: Brennan Klein
email: klein.br@husky.neu.edu
Submitted as part of the 2019 NetSI Collabathon.

"""

from .base import BaseReconstructor
import networkx as nx
import numpy as np


class RandomReconstructor(BaseReconstructor):
    def fit(self, TS, tau=0.6):
        """
        Reconstruct a network from a time serues -- just kidding, simply return 
        a random correlation matrix with a threshold.

        Params
        ------
        TS (np.ndarray): array consisting of $L$ observations from $N$ sensors.
        tau (float): threshold

        Returns
        -------
        G (nx.Graph): a reconstructed graph with $N$ nodes.
        """

        N, L = TS.shape
        W = np.random.rand(N, N)
        A = np.array(W > tau, dtype=int)
        G = nx.from_numpy_array(A)

        self.results['graph'] = G

        self.results['random_matrix'] = W

        self.results['adjacency_matrix'] = A

        return G

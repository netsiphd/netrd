"""
voter.py
--------

Implementation of voter model dynamics on a network.

author: Stefan McCabe

Submitted as part of the 2019 NetSI Collabathon.

"""

from netrd.dynamics import BaseDynamics
import numpy as np
import networkx as nx
from ..utilities import unweighted


class VoterModel(BaseDynamics):
    """Voter dynamics."""

    @unweighted
    def simulate(self, G, L, noise=None):
        r"""Simulate voter-model-style dynamics on a network.

        Nodes are randomly assigned a state in :math:`\{-1, 1\}`; at each
        time step all nodes asynchronously update by choosing their new
        state uniformly from their neighbors. Generates an :math:`N \times
        L` time series.

        The results dictionary also stores the ground truth network as
        `'ground_truth'`.

        Parameters
        ----------
        G (nx.Graph)
            the input (ground-truth) graph with `N` nodes.

        L (int)
            the length of the desired time series.

        noise (float, str or None)
            if noise is present, with this probability a node's state will
            be randomly redrawn from :math:`\{-1, 1\}` independent of its
            neighbors' states. If 'automatic', set noise to :math:`1/N`.

        Returns
        -------
        TS (np.ndarray)
            an :math:`N \times L` array of synthetic time series data.

        """

        N = G.number_of_nodes()

        if noise is None:
            noise = 0
        elif noise == 'automatic' or noise == 'auto':
            noise = 1 / N
        elif not isinstance(noise, (int, float)):
            raise ValueError("noise must be a number, 'automatic', or None")

        transitions = nx.to_numpy_array(G)
        transitions = transitions / np.sum(transitions, axis=0)

        TS = np.zeros((N, L))
        TS[:, 0] = [1 if x < 0.5 else -1 for x in np.random.rand(N)]
        indices = np.arange(N)

        for t in range(1, L):
            np.random.shuffle(indices)
            TS[:, t] = TS[:, t - 1]
            for i in indices:
                TS[i, t] = np.random.choice(TS[:, t], p=transitions[:, i])
                if np.random.rand() < noise:
                    TS[i, t] = 1 if np.random.rand() < 0.5 else -1

        self.results['ground_truth'] = G
        self.results['TS'] = TS
        return TS

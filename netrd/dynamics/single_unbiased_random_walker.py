"""
single_unbiased_random_walker.py
--------------------------------

Simulate a lonely walker on a network.

"""
from .base import BaseDynamics
import networkx as nx
import numpy as np


class SingleUnbiasedRandomWalker(BaseDynamics):
    """Random walk dynamics."""

    def __init__(self):
        self.results = {}

    def simulate(self, G, L, initial_node=None):
        r"""Simulate single random-walker dynamics on a ground truth network.

        Generates an :math:`N \times L` time series `TS` with
        ``TS[j,t]==1`` if the walker is at node :math:`j` at time
        :math:`t`, and ``TS[j,t]==0`` otherwise.

        The results dictionary also stores the ground truth network as
        `'ground_truth'`.

        Examples
        --------
        .. code:: python

            G = nx.ring_of_cliques(4,16)
            L = 2001
            dynamics = SingleUnbiasedRandomWalker()
            TS = dynamics.simulate(G, L)


        Parameters
        ----------
        G (nx.Graph)
            The input (ground-truth) graph with :math:`N` nodes.

        L (int)
            The length of the desired time series.

        Returns
        -------
        TS (np.ndarray)
            An :math:`N \times L` array of synthetic time series data.

        """
        # get adjacency matrix and set up vector of indices
        A = nx.to_numpy_array(G)
        N = G.number_of_nodes()
        W = np.zeros(L, dtype=int)
        # place walker at initial location
        if initial_node:
            W[0] = initial_node
        else:
            W[0] = np.random.randint(N)

        # run dynamical process
        for t in range(L - 1):
            W[t + 1] = np.random.choice(np.where(A[W[t], :])[0])
        self.results['node_index_sequence'] = W
        # turn into a binary-valued
        TS = np.zeros((N, L))
        for t, w in enumerate(W):
            TS[w, t] = 1
        self.results['TS'] = TS
        self.results['ground_truth'] = G
        return TS

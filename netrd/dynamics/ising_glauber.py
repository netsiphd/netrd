"""
ising_glauber.py
----------------

Implementation to simulate the Ising-Glauber model on a network.

author: Chia-Hung Yang
Submitted as part of the 2019 NetSI Collabathon.
"""

from netrd.dynamics import BaseDynamics
import numpy as np
import networkx as nx
from numpy.random import rand


class IsingGlauber(BaseDynamics):
    """Ising-Glauber model."""

    def simulate(self, G, L, init=None, beta=2):
        r"""Simulate time series on a network from the Ising-Glauber model.

        In the Ising Glauber model, each node has a binary state. At every
        time step, nodes switch their state with certain probability. For
        inactive nodes, this probaility is :math:`1 / (1 + e^{\beta (k -
        2m) / k})` where :math:`\beta` is a parameter tuning the likelihood
        of switching state, :math:`k` is degree of the node and :math:`m`
        is the number of its active neighbors; for active nodes the
        switch-state probability is :math:`1 - 1 / (1 + e^{\beta (k - 2m) /
        k})` instead.

        The results dictionary also stores the ground truth network as
        `'ground_truth'`.

        Parameters
        ----------
        G (nx.Graph)
            Underlying ground-truth network of simulated time series which
            has :math:`N` nodes.

        L (int)
            Length of time series.

        init (np.ndarray)
            Length-:math:`N` 1D array of nodes' initial condition, which
            must have binary value (0 or 1).

        beta (float)
            Inversed temperature tuning the likelihood that a node switches
            its state. Default to :math:`2`.

        Returns
        -------
        TS (np.ndarray)
            :math:`N \times L` array of :math:`L` observations on :math:`N`
            nodes.

        """

        N = G.number_of_nodes()
        adjmat = nx.to_numpy_array(G, dtype=float)
        degs = adjmat.sum(axis=0)

        # Randomly initialize an initial condition if not specified
        TS = np.zeros((N, L), dtype=int)
        if init is None:
            init = rand(N)
        TS[:, 0] = np.round(init).astype(int)

        # Simulate the time series
        for t in range(L - 1):
            state = TS[:, t].copy()  # State for each node
            num_act_nei = np.dot(state, adjmat)  # Number of active neighbors

            hamltn = (degs - 2 * num_act_nei) / degs
            thrds = 1 / (1 + np.exp(beta * hamltn))
            # Probability of switching state
            probs = np.where(state == 0, thrds, 1 - thrds)

            _next = np.where(rand(N) < probs, 1 - state, state)
            TS[:, t + 1] = _next

        self.results['ground_truth'] = G
        self.results['TS'] = TS
        return TS

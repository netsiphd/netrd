"""
kuramoto.py
-----------

author: Harrison Hartle
"""

from .base import BaseDynamics
import networkx as nx
import numpy as np
import scipy.integrate as it


class Kuramoto(BaseDynamics):
    """Kuramoto model of oscillators."""

    def __init__(self):
        self.results = {}

    def simulate(self, G, L, dt=0.01, K=1, ICs=[], freqs=[]):
        r"""Simulate Kuramoto model on a ground truth network.

        The results dictionary also stores the ground truth network as
        `'ground_truth'` and the internal frequencies of the process as
        `'internal_frequencies'`.

        Parameters
        ----------
        G (nx.Graph)
            the input (ground-truth) graph with :math:`N` nodes.

        L (int)
            the length of the desired time series.

        dt (float)
            size of timestep for numerical integration.

        K (float)
            coupling strength (prefactor for interaction terms).

        ICs (np.ndarray)
            an :math:`N \times 1` array of initial phase-angles.

        freqs (np.ndarray)
            an :math:`N \times 1` array of internal frequencies.

        Returns
        -------
        TS (np.ndarray)
            an :math:`N \times L` array of synthetic time series data.

        Examples
        --------
        .. code:: python

            G = nx.ring_of_cliques(4,16)
            L = int(1e4)
            dynamics = Kuramoto()
            TS = dynamics.simulate(G, L, dt=0.04, K=0.3)


        """
        # get network features
        A = nx.to_numpy_array(G)
        N = G.number_of_nodes()

        # setup for running the evolution equations
        psi0 = (
            ICs if len(ICs) > 0 else 2.0 * np.pi * np.random.rand(N)
        )  # set initial conditions
        g = (
            freqs if len(freqs) > 0 else 0.1 + 0.100 * np.random.rand(N)
        )  # set internal frequency vector

        t = np.linspace(dt, L * dt, L)  # time-vector
        o = np.ones(N)  # define a rate of change function

        ddt_psis = lambda psi, t, g, K, A: g + (K / N) * (
            A * np.sin(np.outer(o, psi) - np.outer(psi, o))
        ).dot(o)

        argu = (g, K, A)  # arguments in a tuple for it.integrate
        # integrate the equations of motion numerically
        TS_T = it.odeint(ddt_psis, psi0, t, args=argu)

        # odeint returns LxN result - transposing yields reversed-order nodes => apply flipud.
        TS = np.flipud(TS_T.T)

        self.results[
            'internal_frequencies'
        ] = g  # save the internal-frequency vector to results
        self.results['ground_truth'] = G  # save the ground-truth network to results
        self.results['TS'] = TS  # save the timeseries data to results

        return TS

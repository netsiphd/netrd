"""
kuramoto.py
-----------
Kuramoto model of oscillators.

author: Harrison Hartle
"""

from .base import BaseDynamics
import networkx as nx
import numpy as np
import scipy.integrate as it
from ..utilities import unweighted


class Kuramoto(BaseDynamics):
    """Kuramoto model of oscillators."""

    @unweighted
    def simulate(self, G, L, dt=0.01, strength=1, phases=None, freqs=None):
        r"""Simulate Kuramoto model on a ground truth network.

        Kuramoto oscillators model synchronization processes. At each time
        step, each node adjusts its phase :math:`\theta_i` according to the
        equation

        .. math::
            \theta_i = \omega_i + \frac{\lambda}{N}\sum_{j=1}^{N}\sin\left(\theta_j - \theta_i\right),


        where :math:`\lambda`, is a coupling `strength` parameter and each node
        has an internal frequency :math:`\omega_i`; the `freqs` function
        parameter provides the option to initialize these frequencies with
        user-defined values (or leave as `None` to randomly initialize). Each
        node's initial phase :math:`\theta_{i0}` can be randomly initialized
        (the default behavior) or set by specifying the `phases` parameter.

        The results dictionary also stores the ground truth network as
        `'ground_truth'` and the internal frequencies of the process as
        `'internal_frequencies'`.

        For more information on the Kuramoto model, see the review essay
        included below.

        Parameters
        ----------

        G (nx.Graph)
            the input (ground-truth) graph with :math:`N` nodes.

        L (int)
            the length of the desired time series.

        dt (float)
            size of timestep for numerical integration.

        strength (float)
            coupling strength (prefactor for interaction terms).

        phases (np.ndarray)
            an :math:`N \times 1` array of initial phases.

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
            N = G.number_of_nodes()
            L = int(1e4)
            omega = np.random.uniform(0.95, 1.05, N)
            dynamics = Kuramoto()
            TS = dynamics.simulate(G, L, dt=0.01, strength=0.3, freqs=omega)

        References
        ----------
        .. [1] F. Rodrigues, T. Peron, P. Ji, J. Kurths.
               The Kuramoto model in complex networks.
               https://arxiv.org/abs/1511.07139

        """
        A = nx.to_numpy_array(G)
        N = G.number_of_nodes()

        try:
            if phases is not None:
                assert len(phases) == N
                theta_0 = phases
            else:
                theta_0 = 2 * np.pi * np.random.rand(N)

            if freqs is not None:
                assert len(freqs) == N
                omega = freqs
            else:
                omega = np.random.uniform(0.9, 1.1, N)

        except AssertionError:
            raise ValueError("Initial conditions must be None or lists of length N.")

        t = np.linspace(dt, L * dt, L)  # time-vector
        one = np.ones(N)  # define a rate of change function

        def ddt_theta(theta, t, g, strength, A):
            prefactor = strength / N
            first = np.outer(one, theta)
            second = np.outer(theta, one)

            return g + prefactor * (A * np.sin(first - second)).dot(one)

        # integrate the equations of motion numerically
        args = (omega, strength, A)
        TS_T = it.odeint(ddt_theta, theta_0, t, args=args)

        # odeint returns LxN result
        # transposing yields reversed-order nodes => apply flipud.
        TS = np.flipud(TS_T.T)

        # adjust phases
        TS = TS % (2 * np.pi)

        self.results["internal_frequencies"] = omega
        self.results["ground_truth"] = G
        self.results["TS"] = TS

        return TS

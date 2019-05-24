"""
lotka_volterra.py
----------------

Implementation to simulate a Lotka-Volterra model on a network.

author: Chia-Hung Yang
Submitted as part of the 2019 NetSI Collabathon.
"""

from netrd.dynamics import BaseDynamics
import numpy as np
import networkx as nx
from numpy.random import uniform, normal
from scipy.integrate import ode


class LotkaVolterra(BaseDynamics):
    """Lotka-Volterra dynamics of species abundance."""

    def simulate(
        self,
        G,
        L,
        init=None,
        gr=None,
        cap=None,
        inter=None,
        dt=1e-2,
        stochastic=True,
        pertb=None,
    ):
        r"""Simulate time series on a network from the Lotka-Volterra model.

        The Lotka-Volterra model was designed to describe dynamics of
        species abundances in an ecosystem. Species :math:`i`'s abundance
        change per time is :math:`\frac{d X_i}{d t} = r_i X_i \left(1 -
        \frac{X_i}{K_i} + \sum_{j \neq i} W_{ij} \frac{X_j}{K_i}\right)`
        where :math:`r_i` and :math:`K_i` are the growth rate and the
        carrying capacity of species :math:`i` respectively, and
        :math:`W_{ij}` are the relative interaction strength of species
        :math:`j` on :math:`i`.

        The results dictionary also stores the ground truth network as
        `'ground_truth'` and the intermediate time steps as `'time_steps'`.

        Parameters
        ----------

        G (nx.Graph)
            Underlying ground-truth network of simulated time series which
            has :math:`N` nodes.

        L (int)
            Length of time series.

        init (np.ndarray)
            Length-:math:`N` 1D array of nodes' initial condition. If not
            specified an initial condition is unifromly generated from 0 to
            the nodes' carrying capacity.

        gr (np.ndarray)
            Length-:math:`N` 1D array of nodes' growth rate. If not
            specified, default to 1 for all nodes.

        cap (np.ndarray)
            Length-:math:`N` 1D array of nodes' carrying capacity. If not
            specified, default to 1 for all nodes.

        inter (np.ndarray)
            :math:`N \times N` array of interaction weights between
            nodes. If not specified, default to a zero-diagonal matrix
            whose [i, j] entry is :math:`\frac{sign(j - i)}{N - 1}`.

        dt (float or np.ndarray)
            Sizes of time steps when simulating the continuous-time
            dynamics.

        stochastic (bool)
            Whether to simulate the stochastic or deterministic dynamics.

        pertb (np.ndarray)
            Length-:math:`N` 1D array of perturbation magnitude of nodes'
            growth. If not specified, default to 0.01 for all nodes.

        Returns
        -------

        TS (np.ndarray)
            :math:`N \times L` array of `L` observations on :math:`N` nodes.

        Notes
        -----

        The deterministic dynamics is simulated through the forth-order
        Runge-Kutta method, and the stochastic one is simulated through
        mutliplicative noise with the Euler-Maruyama method.

        The ground-truth network, time steps and the time series can be
        found in results['ground-truth'], reuslts['time_steps'] and
        results['time_series'] respectively.

        """

        N = G.number_of_nodes()
        adjmat = nx.to_numpy_array(G)

        # Initialize the model's parameters if not specified
        if gr is None:
            gr = np.ones(N, dtype=float)
        if cap is None:
            cap = np.ones(N, dtype=float)
        if inter is None:
            wei = 1 / (N - 1)
            full = np.full((N, N), wei, dtype=float)
            inter = np.zeros((N, N), dtype=float)
            inter += np.triu(full) - np.tril(full)

        if stochastic and pertb is None:
            pertb = 1e-2 * np.ones(N, dtype=float)

        # Randomly initialize an initial condition if not speciefied
        TS = np.zeros((N, L), dtype=float)
        if init is None:
            init = uniform(low=0, high=cap)
        TS[:, 0] = init

        # Define the function of dynamics
        mat = np.where(adjmat == 1, inter, 0.0) + np.diag(-np.ones(N))
        mat /= cap[:, np.newaxis]

        def dyn(t, state):
            return state * (gr + np.dot(mat, state))

        # Simulate the time series
        if isinstance(dt, float):
            dt = dt * np.ones(L - 1)

        # Deterministic dynamics
        if not stochastic:
            integrator = ode(dyn).set_integrator('dopri5')
            integrator.set_initial_value(init, 0.0)
            for t in range(L - 1):
                if integrator.succesful():
                    TS[:, t + 1] = integrator.integrate(integrator.t + dt[t])
                else:
                    message = 'Integration not succesful. '
                    message += 'Change sizes of time steps or the parameters.'
                    raise RuntimeError(message)

        # Stochastic dynamics
        else:
            for t in range(L - 1):
                state = TS[:, t].copy()
                _next = state + dyn(t, state) * dt[t]
                _next += state * normal(scale=pertb) * np.sqrt(dt[t])
                TS[:, t + 1] = _next

        # Store the results
        self.results['ground_truth'] = G
        self.results['time_steps'] = np.cumsum(dt)
        self.results['TS'] = TS

        return TS

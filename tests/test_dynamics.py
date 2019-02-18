"""
test_dynamics.py
----------------

Test dynamics algorithms.

"""

import networkx as nx
import numpy as np
from netrd import dynamics
from netrd.dynamics import BaseDynamics


def test_dynamics_valid_dimensions():
    """Dynamics models should return N x L arrays."""

    G = nx.karate_club_graph()
    N = G.number_of_nodes()

    for L in [25, 100]:
        for obj in dynamics.__dict__.values():
            if isinstance(obj, type) and BaseDynamics in obj.__bases__:
                TS = obj().simulate(G, L)
                assert TS.shape == (N, L)

    assert BaseDynamics().simulate(G, 25).shape == (N, 25)
    assert BaseDynamics().simulate(G, 100).shape == (N, 100)

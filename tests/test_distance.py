"""
test_distance.py
----------------

Test distance algorithms.

"""

import networkx as nx
from netrd import distance


def test_same_graph_frobenius():
    """Frobenius distance between two equal graphs must be zero."""
    G = nx.erdos_renyi_graph(100, 0.1)
    dist = distance.Frobenius().dist(G, G)
    assert dist == 0.0

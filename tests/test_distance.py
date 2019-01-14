"""
test_distance.py
----------------

Test distance algorithms.

"""

import networkx as nx
from netrd import distance
from netrd.distance import BaseDistance


def test_same_graph():
    """The distance between two equal graphs must be zero."""
    G = nx.karate_club_graph()

    for obj in distance.__dict__.values():
        if isinstance(obj, type) and BaseDistance in obj.__bases__:
            dist = obj().dist(G, G)
            assert dist == 0.0

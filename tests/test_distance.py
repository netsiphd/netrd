"""
test_distance.py
----------------

Test distance algorithms.

"""

import warnings
import numpy as np
import networkx as nx
from netrd import distance
from netrd.distance import BaseDistance


def test_same_graph():
    """The distance between two equal graphs must be zero."""
    G = nx.karate_club_graph()

    for label, obj in distance.__dict__.items():
        if isinstance(obj, type) and BaseDistance in obj.__bases__:
            dist = obj().dist(G, G)
            assert np.isclose(dist, 0.0)


def test_different_graphs():
    """ The distance between two different graphs must be nonzero."""
    ## NOTE: This test is not totally rigorous. For example, two different
    ## networks may have the same eigenvalues, thus a method that compares
    ## their eigenvalues would result in distance 0. However, this is very
    ## unlikely in the constructed case, so we rely on it for now.
    G1 = nx.fast_gnp_random_graph(100, 0.3)
    G2 = nx.barabasi_albert_graph(100, 5)

    for obj in distance.__dict__.values():
        if isinstance(obj, type) and BaseDistance in obj.__bases__:
            dist = obj().dist(G1, G2)
            assert dist > 0.0


def test_symmetry():
    """The distance between two graphs must be symmetric."""
    G1 = nx.barabasi_albert_graph(100, 4)
    G2 = nx.fast_gnp_random_graph(100, 0.3)

    for label, obj in distance.__dict__.items():
        if isinstance(obj, type) and BaseDistance in obj.__bases__:
            dist1 = obj().dist(G1, G2)
            dist2 = obj().dist(G2, G1)
            assert np.isclose(dist1, dist2)


def test_quantum_jsd():
    """Run the above tests again using the collision entropy instead of the
    Von Neumann entropy to ensure that all the logic of the JSD implementation
    is tested.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="JSD is only a metric for 0 ≤ q < 2.")
        JSD = distance.QuantumJSD()
        G = nx.karate_club_graph()
        dist = JSD.dist(G, G, beta=0.1, q=2)
        assert np.isclose(dist, 0.0)

        G1 = nx.fast_gnp_random_graph(100, 0.3)
        G2 = nx.barabasi_albert_graph(100, 5)
        dist = JSD.dist(G1, G2, beta=0.1, q=2)
        assert dist > 0.0

        G1 = nx.barabasi_albert_graph(100, 4)
        G2 = nx.fast_gnp_random_graph(100, 0.3)
        dist1 = JSD.dist(G1, G2, beta=0.1, q=2)
        dist2 = JSD.dist(G2, G1, beta=0.1, q=2)
        assert np.isclose(dist1, dist2)


def test_directed_input():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Coercing directed graph to undirected."
        )
        G = nx.fast_gnp_random_graph(100, 0.3, directed=True)

        for label, obj in distance.__dict__.items():
            if isinstance(obj, type) and BaseDistance in obj.__bases__:
                dist = obj().dist(G, G)
                assert np.isclose(dist, 0.0)

        G1 = nx.fast_gnp_random_graph(100, 0.3, directed=True)
        G2 = nx.fast_gnp_random_graph(100, 0.3, directed=True)

        for label, obj in distance.__dict__.items():
            if isinstance(obj, type) and BaseDistance in obj.__bases__:
                dist1 = obj().dist(G1, G2)
                dist2 = obj().dist(G2, G1)
                assert np.isclose(dist1, dist2)

        for obj in distance.__dict__.values():
            if isinstance(obj, type) and BaseDistance in obj.__bases__:
                dist = obj().dist(G1, G2)
                assert dist > 0.0


def test_weighted_input():
    G1 = nx.karate_club_graph()
    G2 = nx.karate_club_graph()
    rand = np.random.RandomState(seed=42)
    edge_weights = {e: rand.randint(0, 1000) for e in G2.edges}
    nx.set_edge_attributes(G2, edge_weights, "weight")
    assert nx.is_isomorphic(G1, G2)

    for label, obj in distance.__dict__.items():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if isinstance(obj, type) and BaseDistance in obj.__bases__:
                dist = obj().dist(G1, G2)
                warning_triggered = False
                for warning in w:
                    if "weighted" in str(warning.message):
                        warning_triggered = True
                if not warning_triggered:
                    assert not np.isclose(dist, 0.0)
                else:
                    assert np.isclose(dist, 0.0)

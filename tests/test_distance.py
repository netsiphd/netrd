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
    G = nx.barbell_graph(10, 5)

    for label, obj in distance.__dict__.items():
        if isinstance(obj, type) and BaseDistance in obj.__bases__:
            dist = obj().dist(G, G)
            assert np.isclose(dist, 0.0), f"{label} not deterministic"


def test_different_graphs():
    """The distance between two different graphs must be nonzero."""
    ## NOTE: This test is not totally rigorous. For example, two different
    ## networks may have the same eigenvalues, thus a method that compares
    ## their eigenvalues would result in distance 0. However, this is very
    ## unlikely in the constructed case, so we rely on it for now.
    G1 = nx.fast_gnp_random_graph(100, 0.3)
    G2 = nx.barabasi_albert_graph(100, 5)

    for obj in distance.__dict__.values():
        if isinstance(obj, type) and BaseDistance in obj.__bases__:
            dist = obj().dist(G1, G2)
            assert dist > 0.0, f"{label} not nonzero"


def test_symmetry():
    """The distance between two graphs must be symmetric."""
    G1 = nx.barabasi_albert_graph(100, 4)
    G2 = nx.fast_gnp_random_graph(100, 0.3)

    for label, obj in distance.__dict__.items():
        if isinstance(obj, type) and BaseDistance in obj.__bases__:
            dist1 = obj().dist(G1, G2)
            dist2 = obj().dist(G2, G1)
            assert np.isclose(dist1, dist2), f"{label} not symmetric"


def test_quantum_jsd():
    """Run the above tests again using the collision entropy instead of the
    Von Neumann entropy to ensure that all the logic of the JSD implementation
    is tested.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="JSD is only a metric for 0 ≤ q < 2.")
        JSD = distance.QuantumJSD()
        G = nx.barbell_graph(10, 5)
        dist = JSD.dist(G, G, beta=0.1, q=2)
        assert np.isclose(dist, 0.0), "collision entropy not deterministic"

        G1 = nx.fast_gnp_random_graph(100, 0.3)
        G2 = nx.barabasi_albert_graph(100, 5)
        dist = JSD.dist(G1, G2, beta=0.1, q=2)
        assert dist > 0.0, "collision entropy not nonzero"

        G1 = nx.barabasi_albert_graph(100, 4)
        G2 = nx.fast_gnp_random_graph(100, 0.3)
        dist1 = JSD.dist(G1, G2, beta=0.1, q=2)
        dist2 = JSD.dist(G2, G1, beta=0.1, q=2)
        assert np.isclose(dist1, dist2), "collision entropy not symmetric"


def test_directed_input():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Coercing directed graph to undirected."
        )
        G = nx.fast_gnp_random_graph(100, 0.3, directed=True)

        for label, obj in distance.__dict__.items():
            if isinstance(obj, type) and BaseDistance in obj.__bases__:
                dist = obj().dist(G, G)
                assert np.isclose(dist, 0.0), f"{label} not deterministic"

        G1 = nx.fast_gnp_random_graph(100, 0.3, directed=True)
        G2 = nx.fast_gnp_random_graph(100, 0.3, directed=True)

        for label, obj in distance.__dict__.items():
            if isinstance(obj, type) and BaseDistance in obj.__bases__:
                dist1 = obj().dist(G1, G2)
                dist2 = obj().dist(G2, G1)
                assert np.isclose(dist1, dist2), f"{label} not symmetric"

        for obj in distance.__dict__.values():
            if isinstance(obj, type) and BaseDistance in obj.__bases__:
                dist = obj().dist(G1, G2)
                assert dist > 0.0, f"{label} not nonzero"


def test_weighted_input():
    G1 = nx.barbell_graph(10, 5)
    G2 = nx.barbell_graph(10, 5)
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
                    assert not np.isclose(dist, 0.0), f"{label} = 0"
                else:
                    assert np.isclose(dist, 0.0), f"{label} != 0"


def test_isomorphic_input():
    G1 = nx.fast_gnp_random_graph(150, 0.10)

    N = G1.order()
    new_nodes = [(i + 5) % N for i in G1.nodes]

    # create G1 by permuting the adjacency matrix
    new_adj_mat = nx.to_numpy_array(G1, nodelist=new_nodes)
    G2 = nx.from_numpy_array(new_adj_mat)

    assert nx.is_isomorphic(G1, G2)

    # not all distances should be invariant under isomorphism
    # document those here
    EXCLUDED_DISTANCES = [
        "Hamming",
        "Frobenius",
        "JaccardDistance",
        "HammingIpsenMikhailov",
        "ResistancePerturbation",
        "LaplacianSpectral",
        "PolynomialDissimilarity",
        "DeltaCon",
        "QuantumJSD",
        "DistributionalNBD",
        "NonBacktrackingSpectral",
        "GraphDiffusion",
    ]

    for label, obj in distance.__dict__.items():
        print(label)
        if (
            isinstance(obj, type)
            and BaseDistance in obj.__bases__
            and label not in EXCLUDED_DISTANCES
        ):
            dist = obj().dist(G1, G2)
            assert np.isclose(
                dist, 0.0, atol=1e-3
            ), f"{label} not invariant under isomorphism"

"""
graph.py
--------

Utilities for creating and interacting with graph objects.

author: Stefan McCabe (stefanmccabe at gmail dot com)

Submitted as part of the 2019 NetSI Collabathon.

"""
from functools import wraps
import warnings
import numpy as np
import networkx as nx


def create_graph(A, create_using=None, remove_self_loops=True):
    """Flexibly creating a networkx graph from a numpy array.

    Parameters
    ----------
    A (np.ndarray)
        A numpy array.

    create_using (nx.Graph or None)
        Create the graph using a specific networkx graph. Can be used for
        forcing an asymmetric matrix to create an undirected graph, for
        example.

    remove_self_loops (bool)
        If True, remove the diagonal of the matrix before creating the
        graph object.

    Returns
    -------
    G
        A graph, typically a nx.Graph or nx.DiGraph.

    """
    if remove_self_loops:
        np.fill_diagonal(A, 0)

    if create_using is None:
        if np.allclose(A, A.T):
            G = nx.from_numpy_array(A, create_using=nx.Graph())
        else:
            G = nx.from_numpy_array(A, create_using=nx.DiGraph())
    else:
        G = nx.from_numpy_array(A, create_using=create_using)

    return G


def ensure_undirected(G):
    """Ensure the graph G is undirected.

    If it is not, coerce it to undirected and warn the user.

    Parameters
    ----------
    G (networkx graph)
        The graph to be checked

    Returns
    -------

    G (nx.Graph)
        Undirected version of the input graph

    """
    if nx.is_directed(G):
        G = G.to_undirected(as_view=False)
        warnings.warn("Coercing directed graph to undirected.", RuntimeWarning)
    return G


def undirected(func):
    """
    Decorator applying ``ensure_undirected()`` to all ``nx.Graph``-subclassed
    arguments of ``func``.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [
            ensure_undirected(arg) if issubclass(arg.__class__, nx.Graph) else arg
            for arg in args
        ]
        return func(*args, **kwargs)

    return wrapper


def ensure_unweighted(G):
    """Ensure the graph G is unweighted.

    If it is not, coerce it to unweighted and warn the user.

    Parameters
    ----------
    G (networkx graph)
        The graph to be checked

    Returns
    -------

    G (nx.Graph)
        Unweighted version of the input graph

    """

    for _, _, attr in G.edges(data=True):
        if not np.isclose(attr.get("weight", 1.0), 1.0):
            H = G.__class__()
            H.add_nodes_from(G)
            H.add_edges_from(G.edges)
            warnings.warn("Coercing weighted graph to unweighted.", RuntimeWarning)
            return H

    return G


def unweighted(func):
    """
    Decorator applying ``ensure_unweighted()`` to all ``nx.Graph``-subclassed
    arguments of ``func``.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [
            ensure_unweighted(arg) if issubclass(arg.__class__, nx.Graph) else arg
            for arg in args
        ]
        return func(*args, **kwargs)

    return wrapper

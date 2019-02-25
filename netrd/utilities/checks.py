"""
checks.py
------------

Utilities for "type checking"

author: Tim LaRock
email: timothylarock at gmail dot com
Submitted as part of the 2019 NetSI Collabathon

"""

import warnings
import networkx as nx

def ensure_undirected(G):
    '''
    Ensure the graph G is undirected. If it is not, coerce it to undirected
    and warn the user.

    Params
    ------

    G (networkx graph): The graph to be checked

    Returns
    -------

    G (networkx graph): Undirected version of the input graph

    '''

    if nx.is_directed(G):
        G = nx.to_undirected(G)
        warnings.warn("Coercing directed graph to undirected.", RuntimeWarning)

    return G


"""
frobenius.py
------------

Frobenius norm between two adjacency matrices.

"""

import numpy as np
import networkx as nx
from .base import BaseDistance


class Frobenius(BaseDistance):
    """The distance between two graphs is the Frobenius distance between their
    adjacency matrices. More specifically, if $a_ij$ and $b_ij$ are the two
    adjacency matrices we define

        $d(G1, G2) = \sqrt{\sum_{i,j} |a_{ij} - b_{ij}|**2}$

    The graphs must have the same number of nodes.

    """

    def dist(self, G1, G2):
        """Frobenius distance between two graphs."""
        adj1 = nx.adjacency_matrix(G1)
        adj2 = nx.adjacency_matrix(G2)
        dist = np.linalg.norm((adj1 - adj2).A)
        self.results['dist'] = dist
        self.results['adj'] = np.array([adj1, adj2])
        return dist

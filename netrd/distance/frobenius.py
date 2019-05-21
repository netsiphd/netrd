"""
frobenius.py
------------

Frobenius norm between two adjacency matrices.

"""

import numpy as np
import networkx as nx
from .base import BaseDistance


class Frobenius(BaseDistance):
    """The Frobenius distance between their adjacency matrices.

    More specifically, if :math:`a_{ij}` and :math:`b_{ij}` are the two
    adjacency matrices we define

    .. math::
        d(G1, G2) = \sqrt{\sum_{i,j} |a_{ij} - b_{ij}|^2}


    The graphs must have the same number of nodes.

    The results dictionary also stores a 2-tuple of the underlying
    adjacency matrices in the key `'adjacency_matrices'`.

    """

    def dist(self, G1, G2):
        """Frobenius distance between two graphs."""
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)
        dist = np.linalg.norm((adj1 - adj2))
        self.results['dist'] = dist
        self.results['adjacency_matrices'] = adj1, adj2
        return dist

"""
hamming.py
--------------

Hamming distance, wrapper for scipy function:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html#scipy.spatial.distance.hamming

"""

import scipy
import numpy as np
import networkx as nx
from .base import BaseDistance


class Hamming(BaseDistance):
    """Entry-wise disagreement between adjacency matrices."""

    def dist(self, G1, G2):
        r"""The proportion of disagreeing nodes between the flattened adjacency
        matrices.

        If :math:`u` and :math:`v` are boolean vectors, then Hamming
        distance is:

        .. math::

            \frac{c_{01} + c_{10}}{n}

        where :math:`c_{ij}` is the number of occurrences of where
        :math:`u[k] = i` and :math:`v[k] = j` for :math:`k < n`.

        The graphs must have the same number of nodes. A small modification
        to this code could allow weights can be applied, but only one set
        of weights that apply to both graphs.

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        Returns
        -------

        dist (float)
            the distance between `G1` and `G2`.

        References
        ----------

        .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html#scipy.spatial.distance.hamming

        """
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)
        dist = scipy.spatial.distance.hamming(adj1.flatten(), adj2.flatten())
        self.results['dist'] = dist
        self.results['adjacency_matrices'] = adj1, adj2
        return dist

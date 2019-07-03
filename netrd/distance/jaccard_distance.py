"""
jaccard_distance.py
--------------

Graph distance based on the Jaccard index between edge sets.

author: David Saffo
email: saffo.d@husky.neu.edu
Submitted as part of the 2019 NetSI Collabathon.

"""

from .base import BaseDistance
import networkx as nx
import numpy as np


class JaccardDistance(BaseDistance):
    """Jaccard distance between edge sets."""

    def dist(self, G1, G2):
        r"""Compute the Jaccard index between two graphs.

        The Jaccard index between two sets

        .. math::
            J(A, B) = \frac{|A \cap B|}{|A \cup B|}

        provides a measure of similarity between sets. Here, we use the edge
        sets of two graphs. The index, a measure of similarity, is converted to
        a distance

        .. math::
            d_J(A, B) = 1 - J(A, B)

        for consistency with other graph distances.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two graphs to be compared.

        Returns
        -------

        dist (float)
            the distance between G1 and G2.

        """

        e1 = set(G1.edges)
        e2 = set(G2.edges)
        cup = set.union(e1, e2)
        cap = set.intersection(e1, e2)

        dist = 1 - len(cap) / len(cup)

        self.results["dist"] = dist
        return dist

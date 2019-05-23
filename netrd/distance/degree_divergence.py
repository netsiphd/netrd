"""
degree_divergence.py
--------------------------

Baseline distance measure: the K-L divergence
 between the two degree distributions.

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon.

"""

from collections import Counter
import numpy as np
import networkx as nx
from .base import BaseDistance
from ..utilities import entropy


class DegreeDivergence(BaseDistance):
    """Compare two degree distributions."""

    def dist(self, G1, G2):
        """Jenson-Shannon divergence between degree distributions.

        Assumes undirected networks.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        Returns
        -------

        dist (float)
            the distance between `G1` and `G2`.

        """

        def degree_vector_histogram(graph):
            """Return the degrees in both formats.

            max_deg is the length of the histogram, to be padded with
            zeros.

            """
            vec = np.array(list(dict(graph.degree()).values()))
            max_deg = max(vec)
            counter = Counter(vec)
            hist = np.array([counter[v] for v in range(max_deg)])
            return vec, hist

        deg1, hist1 = degree_vector_histogram(G1)
        deg2, hist2 = degree_vector_histogram(G2)
        self.results['degree_vectors'] = deg1, deg2
        self.results['degree_histograms'] = hist1, hist2

        max_len = max(len(hist1), len(hist2))
        p1 = np.pad(hist1, (0, max_len - len(hist1)), 'constant', constant_values=0)
        p2 = np.pad(hist2, (0, max_len - len(hist2)), 'constant', constant_values=0)
        self.results['dist'] = entropy.js_divergence(p1, p2)
        return self.results['dist']

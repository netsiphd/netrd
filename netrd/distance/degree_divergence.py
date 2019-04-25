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
from scipy.stats import entropy
from .base import BaseDistance

class DegreeDivergence(BaseDistance):
    def dist(self, G1, G2):
        """
        Return the Jenson-Shannon divergence between two graphs.

        Note: The method assumes undirected networks.

        Params
        ------

        G1, G2 (nx.Graph): two networkx graphs to be compared.

        Returns
        -------

        dist (float): the distance between G1 and G2.

        """

        # get the degrees
        deg1 = np.array(list(dict(G1.degree()).values()))
        deg2 = np.array(list(dict(G2.degree()).values()))

        self.results['degree_vectors'] = deg1, deg2

        N = G1.number_of_nodes()

        # from degree sequences to degree histograms
        p1 = np.zeros(N)
        p2 = np.zeros(N)
        for k, v in Counter(deg1).items():
            p1[k] = v
        for k, v in Counter(deg2).items():
            p2[k] = v

        self.results['degree_histograms'] = p1, p2

        def js_divergence(P, Q):
            """Jenson-Shannon divergence between P and Q."""
            M = 0.5*(P+Q)

            KLDpm = entropy(P, M, base=2)
            KLDqm = entropy(Q, M, base=2)
            JSDpq = 0.5*(KLDpm + KLDqm)

            return JSDpq

        dist = js_divergence(p1, p2)

        self.results['dist'] = dist
        return dist

"""
degree_divergence.py
--------------------------

Baseline distance measure: the K-L divergence
 between the two degree distributions.

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon.

"""

import numpy as np
import networkx as nx
from scipy.stats import entropy
from .base import BaseDistance

class DegreeDivergence(BaseDistance):
    def dist(self, G1, G2):
        """
        Return the K-L divergence between two graphs. Note that this distance
        is not symmetric.

        Note : The method assumes undirected networks.

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

        def js_divergence(P, Q):
            """Jenson-Shannon divergence between P and Q."""
            M = 0.5*(P+Q)

            KLDpm = entropy(P, M, base=2)
            KLDqm = entropy(Q, M, base=2)
            JSDpq = 0.5*(KLDpm + KLDqm)

            return JSDpq

        # K-L divergence
        dist = js_divergence(deg1, deg2)

        self.results['dist'] = dist
        return dist

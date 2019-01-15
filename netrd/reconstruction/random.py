"""
random.py
---------

Reconstruct a random network from time series.

"""

import networkx as nx
from .base import BaseReconstructor


class RandomReconstructor(BaseReconstructor):
    def fit(self, TS):
        """Reconstruct a random graph."""
        G = nx.erdos_renyi_graph(TS.shape[0], 0.1)
        self.results['graph'] = G
        return G

"""
random.py
---------

Reconstruct a random network from time series.

"""

from .base import BaseReconstructor


class RandomReconstructor(BaseReconstructor):

    def fit(self, T):
        """Reconstruct a random graph."""
        G = nx.erdos_renyi_graph(T.shape[0], 0.1)
        self.results['graph'] = G
        return G

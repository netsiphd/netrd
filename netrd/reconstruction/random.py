"""
random.py
---------

Reconstruct a random network from time series.

"""

from netrd import BaseReconstructor

class RandomReconstructor(RandomReconstructor):

    def fit(self, T):
        """Reconstruct a random graph."""
        G = nx.erdos_renyi_graph(T.shape[0], 0.1)
        self.results['graph'] = G
        return G

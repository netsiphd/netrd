"""
dk_series.py
--------------------------

Graph distance based on the dk-series.

author: Brennan Klein & Stefan McCabe
email: brennanjamesklein@gmail.com
Submitted as part of the 2019 NetSI Collabathon.

"""


import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
import itertools as it
from collections import defaultdict
from .base import BaseDistance
from ..utilities import entropy, ensure_undirected


class dkSeries(BaseDistance):
    def dist(self, G1, G2, d=2):
        r"""Compute the distance between two graphs by using the Jensen-Shannon
        divergence between the :math:`dk`-series of the graphs.

        The :math:`dk`-series of a graph is the collection of distributions of
        size :math:`d` subgraphs, where nodes are labelled by degrees. For
        simplicity, we currently consider only the :math:`1k`-series, i.e., the
        degree distribution, or the :math:`2k`-series, i.e., the
        distribution of edges between nodes of degree :math:`(k_i, k_j)`. The
        distance between these :math:`dk`-series is calculated using the
        Jensen-Shannon divergence.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared

        d (int)
            the size of the subgraph to consider

        Returns
        -------

        dist (float)
            the distance between `G1` and `G2`.

        References
        ----------

        .. [1] Orsini, Chiara, Marija M. Dankulov, Pol Colomer-de-Simón,
               Almerima Jamakovic, Priya Mahadevan, Amin Vahdat, Kevin E.
               Bassler, et al. 2015. “Quantifying Randomness in Real Networks.”
               Nature Communications 6 (1). https://doi.org/10.1038/ncomms9627.

        """

        G1 = ensure_undirected(G1)
        G2 = ensure_undirected(G2)
        N = max(len(G1), len(G2))

        if d == 1:
            from .degree_divergence import DegreeDivergence

            degdiv = DegreeDivergence()
            dist = degdiv.dist()

            # the 2k-distance stores the distribution in a sparse matrix,
            # so here we take the output of DegreeDivergence and
            # produce a comparable object
            hist1, hist2 = degdiv.results['degree_histograms']
            hist1 /= len(G1)
            hist2 /= len(G2)
            hist1 = coo_matrix(hist1)
            hist2 = coo_matrix(hist2)

            self.results["dk_distributions"] = hist1, hist2

        elif d == 2:
            D1 = dk2_series(G1, N)
            D2 = dk2_series(G2, N)

            # store the 2K-distributions
            self.results["dk_distributions"] = D1, D2

            # flatten matrices. this is safe because we've padded to the same size
            G1_dk_normed = D1.toarray()[np.triu_indices(N)].flatten()
            G2_dk_normed = D2.toarray()[np.triu_indices(N)].flatten()

            assert np.isclose(G1_dk_normed.sum(), 1)
            assert np.isclose(G2_dk_normed.sum(), 1)

            dist = entropy.js_divergence(G1_dk_normed, G2_dk_normed)
        else:
            raise NotImplementedError()

        self.results["dist"] = dist
        return dist


def dk2_series(G, N=None):
    """
    Calculate the 2k-series (i.e. the number of edges between
    degree-labelled nodes) for G.
    """

    if N is None:
        N = len(G)

    k_dict = dict(nx.degree(G))
    dk2 = defaultdict(int)

    for (i, j) in G.edges:
        k_i = k_dict[i]
        k_j = k_dict[j]

        # We're enforcing order here because at the end we're going to
        # leverage that all the information can be stored in the upper
        # triangular for convenience.
        if k_i <= k_j:
            dk2[(k_i, k_j)] += 1
        else:
            dk2[(k_j, k_i)] += 1

    # every edge should be counted once
    assert sum(list(dk2.values())) == G.size()

    # convert from dict to sparse matrix
    row = [i for (i, j) in dk2.keys()]
    col = [j for (i, j) in dk2.keys()]
    data = [x for x in dk2.values()]

    D = coo_matrix((data, (row, col)), shape=(N, N))

    # this should be normalized by the number of edges
    D = D / G.size()

    return D

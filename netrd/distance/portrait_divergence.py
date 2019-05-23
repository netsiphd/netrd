"""
portrait_divergence.py
---------------------
Adapted from "An information-theoretic, all-scales approach to comparing networks" by James P. Bagrow and Erik M. Bollt, 2018 arXiv:1804.03665
and [this repository](https://github.com/bagrow/portrait-divergence)

author: Brennan Klein
email: brennanjamesklein at gmail dot com
submitted as part of the 2019 NetSI Collabathon
"""
from .base import BaseDistance
from collections import Counter
import numpy as np
import networkx as nx
from ..utilities import entropy


class PortraitDivergence(BaseDistance):
    """Compares graph portraits."""

    def dist(self, G1, G2, bins=None, binedges=None):
        """Distance measure based on the two graphs' "portraits".

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'` and the
        portrait matrices in `'portrait_matrices'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two graphs

        bins (int)
            width of bins in percentiles

        binedges (list)
            vector of bin edges (mutually exclusive from bins)


        Returns
        -------
        dist (float)
            the portrait divergence between two graphs.


        References
        ----------
        [1] An information-theoretic, all-scales approach to comparing networks
        James P. Bagrow and Erik M. Bollt, 2018 arXiv:1804.03665

        [2] https://github.com/bagrow/portrait-divergence

        """
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)

        ## NOTE dijkstra cannot handle negative weights
        if (adj1 < 0).any() or (adj2 < 0).any():
            adj1 = np.abs(adj1)
            adj2 = np.abs(adj2)
            G1 = nx.from_numpy_array(adj1)
            G2 = nx.from_numpy_array(adj2)

        paths_G1 = list(nx.all_pairs_dijkstra_path_length(G1))
        paths_G2 = list(nx.all_pairs_dijkstra_path_length(G2))

        # get bin_edges in common for G and H:
        if binedges is None:
            if bins is None:
                bins = 1

            UPL_G1 = set(_get_unique_path_lengths(G1, paths=paths_G1))
            UPL_G2 = set(_get_unique_path_lengths(G2, paths=paths_G2))

            unique_path_lengths = sorted(list(UPL_G1 | UPL_G2))
            binedges = np.percentile(unique_path_lengths, np.arange(0, 101, bins))

        # get weighted portraits:
        BG1 = weighted_portrait(G1, paths=paths_G1, binedges=binedges)
        BG2 = weighted_portrait(G2, paths=paths_G2, binedges=binedges)

        dist = portrait_divergence(
            BG1, BG2, N1=G1.number_of_nodes(), N2=G2.number_of_nodes()
        )

        self.results['dist'] = dist
        self.results['adjacency_matrices'] = adj1, adj2
        self.results['portrait_matrices'] = BG1, BG2

        return dist


def portrait(G):
    """
    Parameters
    ----------
    G (nx.Graph or nx.DiGraph): a graph.

    Returns
    -------
    B (np.ndarray): a matrix B such that B[i,j] is the number of starting
    nodes in graph with j nodes in shell i.
    """

    dia = nx.diameter(G)
    N = G.number_of_nodes()

    # B indices are 0...dia x 0...N-1:
    B = np.zeros((dia + 1, N))

    max_path = 1
    adj = G.adj

    for starting_node in G.nodes():
        nodes_visited = {starting_node: 0}
        search_queue = [starting_node]
        d = 1

        while search_queue:
            next_depth = []
            extend = next_depth.extend

            for n in search_queue:
                l = [i for i in adj[n] if i not in nodes_visited]
                extend(l)

                for j in l:
                    nodes_visited[j] = d

            search_queue = next_depth
            d += 1

        node_distances = nodes_visited.values()
        max_node_distances = max(node_distances)

        curr_max_path = max_node_distances
        if curr_max_path > max_path:
            max_path = curr_max_path

        # build individual distribution:
        dict_distribution = dict.fromkeys(node_distances, 0)
        for d in node_distances:
            dict_distribution[d] += 1

        # add individual distribution to matrix:
        for shell, count in dict_distribution.items():
            B[shell][count] += 1

        # HACK: count starting nodes that have zero nodes in farther shells
        max_shell = dia
        while max_shell > max_node_distances:
            B[max_shell][0] += 1
            max_shell -= 1

    return B[: max_path + 1, :]


def weighted_portrait(G, paths=None, binedges=None):
    """
    Compute weighted portrait, using Dijkstra's algorithm for finding
    shortest paths.

    Parameters
    ----------
    G (nx.Graph or nx.DiGraph): a graph.
    paths (list): a list of all pairs of pahts
    binedges (list): sampled path lengths

    Returns
    -------
    B (np.ndarray): a matrix B where B[i,j] is the number of starting nodes in graph with j nodes at distance d_i <  d < d_{i+1}.
    """

    # all pairs path lengths
    if paths is None:
        paths = list(nx.all_pairs_dijkstra_path_length(G))

    if binedges is None:
        unique_path_lengths = _get_unique_path_lengths(G, paths=paths)
        sampled_path_lengths = np.percentile(unique_path_lengths, np.arange(0, 101, 1))
    else:
        sampled_path_lengths = binedges

    UPL = np.array(sampled_path_lengths)

    l_s_v = []
    for i, (s, dist_dict) in enumerate(paths):
        distances = np.array(list(dist_dict.values()))
        s_v, e = np.histogram(distances, bins=UPL)
        l_s_v.append(s_v)

    M = np.array(l_s_v)

    B = np.zeros((len(UPL) - 1, G.number_of_nodes() + 1))

    for i in range(len(UPL) - 1):
        col = M[:, i]  # ith col = numbers of nodes at d_i <= distance < d_i+1

        for n, c in Counter(col).items():
            B[i, n] += c

    return B


def _get_unique_path_lengths(G, paths=None):
    """
    Compute the unique path lengths.

    Parameters
    ----------
    G (nx.Graph or DiGraph): a graph.
    paths (list): list of paths.

    Returns
    -------
    unique_path_lengths (list): sorted unique path lengths.
    """

    if paths is None:
        paths = list(nx.all_pairs_dijkstra_path_length(G))

    unique_path_lengths = set()

    for starting_node, dist_dict in paths:
        unique_path_lengths |= set(dist_dict.values())

    unique_path_lengths = sorted(list(unique_path_lengths))
    return unique_path_lengths


def pad_portraits_to_same_size(B1, B2):
    """
    Make sure that two matrices are padded with zeros and/or trimmed of
    zeros to be the same dimensions.

    Parameters
    ----------
    B1 (np.ndarray): Portrait matrix of a graph (k x N)
    B2 (np.ndarray):

    Returns
    -------
    BigB1, BigB2 (np.ndarray): padded versions of B1 and B2 so they align
    """
    ns, ms = B1.shape
    nl, ml = B2.shape

    # Bmats have N columns, find last *occupied* column and trim both down:
    lastcol1 = max(np.nonzero(B1)[1])
    lastcol2 = max(np.nonzero(B2)[1])
    lastcol = max(lastcol1, lastcol2)

    B1 = B1[:, : lastcol + 1]
    B2 = B2[:, : lastcol + 1]

    BigB1 = np.zeros((max(ns, nl), lastcol + 1))
    BigB2 = np.zeros((max(ns, nl), lastcol + 1))

    BigB1[: B1.shape[0], : B1.shape[1]] = B1
    BigB2[: B2.shape[0], : B2.shape[1]] = B2

    return BigB1, BigB2


def _graph_or_portrait(X):
    """
    Check if X is a nx.(Di)Graph. If it is, get its portrait. Otherwise assume
    it's a portrait and just return it.
    """
    if isinstance(X, (nx.Graph, nx.DiGraph)):
        return portrait(X)

    return X


def _get_prob_distance(B):
    """
    Helper function.
    """
    d, K = B.shape

    v = np.arange(0, K)
    f = (B * v).sum(axis=1)
    return f / f.sum()


def _get_prob_k_given_L(B, N=None):
    """
    Helper function.
    """
    if N is None:
        N = int(B[0, 1])
    return B / N


def portrait_divergence(G1, G2, N1=None, N2=None):
    """
    Compute the portrait divergence between graphs G1 and G2.

    Parameters
    ----------
    G1 (nx.Graph or nx.DiGraph): a graph.
    G2 (nx.Graph or nx.DiGraph): a graph.

    Returns
    -------
    JSDpq (float): the Jensen-Shannon divergence between the portraits of G1 and G2

    """
    BG1 = _graph_or_portrait(G1)
    BG2 = _graph_or_portrait(G2)
    BG1, BG2 = pad_portraits_to_same_size(BG1, BG2)

    # build joint distribution for G:
    P_L = _get_prob_distance(BG1)
    P_KgL = _get_prob_k_given_L(BG1, N=N1)
    P_KaL = P_KgL * P_L[:, None]

    # build joint distribution for H:
    Q_L = _get_prob_distance(BG2)
    Q_KgL = _get_prob_k_given_L(BG2, N=N2)
    Q_KaL = Q_KgL * Q_L[:, None]

    # flatten distribution matrices as arrays:
    P = P_KaL.ravel()
    Q = Q_KaL.ravel()

    return entropy.js_divergence(P, Q)

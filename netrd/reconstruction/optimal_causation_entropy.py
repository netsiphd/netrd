"""
optimal_causal_entropy.py
-------------------------

Graph reconstruction algorithm based on Sun et al., SIAM (2015)
https://doi.org/10.1137/140956166

author: Chia-Hung Yang
Submitted as part of the 2019 NetSI Collabathon.
"""

from .base import BaseReconstructor
from netrd.utilities.entropy import categorized_data, conditional_entropy
import networkx as nx
import numpy as np
from ..utilities import create_graph


class OptimalCausationEntropy(BaseReconstructor):
    """Optimizes causation entropy."""

    def fit(self, TS, n_bins=40, atol=1e-6, **kwargs):
        r"""Reconstruct causal parents of nodes by optimizing causation entropy.

        Optimal causation entropy method reconstructs parents of nodes in a
        causal diagram for systems that rest on three Markov assumptions:
        Let :math:`X_t` be the system state at time :math:`t`, denote node
        :math:`i`'s causal parents as :math:`N_i` and its state as
        :math:`X_t^{(i)}`. The following three statements hold for every
        node :math:`i`:

        1. :math:`P(X_t | X_{t-1}, X_{t-2}, ...) = P(X_t | X_{t-1}) = P(X_{t'} | X_{t'-1})`

        2. :math:`P(X_t^{(i)} | X_{t-1}) = P(X_t^{(i)} | X_{t-1}^{(N_i)})`

        3. :math:`P(X_t^{(i)} | X_{t-1}^{(J)}) \neq P(X_t^{(i) | X_{t-1}^{(K)}})`
           whenever :math:`J, K` are sets of nodes such that :math:`J \cap N_i \neq K \cap N_i`

        Sun et al. proved that for any set of nodes :math:`I` in systems
        satisfying the above three conditions, its causal parents
        :math:`N_I` is the minimal set of nodes :math:`K` that maximizes
        the causation entropy :math:`C_{K \rightarrow I}`.  The more
        general form of causation entropy is defined as :math:`C_{J
        \rightarrow I | K} = H(X_{t+1}^{(I)} | X_t^{(K)}) - H(X_{t+1}^{(I)}
        | X_t^{(K)}, X_t^{(J)})` where :math:`H(X|Y)` is the conditional
        entropy of :math:`X` conditioned on :math:`Y`.  Sun et al. also
        showed that the causal parents :math:`N_I` can be efficiently found
        by first building a superset :math:`S \supset N_I` via heuristic
        and then removing noncausal nodes in :math:`S`. The causal diagram
        can hence be reconstructed from time series data by applying the
        proposed algorithm to every node.

        The results dictionary stores the causal parents of individual
        nodes in `'parents'` and the raw adjacency matrix in
        `'adjacency_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            :math:`N \times L` array consisting of :math:`L` observations
            from :math:`N` sensors.

        data (np.ndarray)
            Array of data with nodes as columns and observations of
            quantity on nodes as rows.

        n_bins (int)
            Number of bins when transforming continuous data into its
            binned categorical version (universal for all nodes).

        atol (float)
            Absolute tolerance to determine whether causalentropy is closed
            to zero.

        Returns
        -------

        G (nx.Graph)
            A reconstructed graph with :math:`N` nodes.

        Notes
        -----
        1. Nodes' causal parents can be found in ``results['parents']``.

        2. Current implementation naively thresholds the causation entropy to
           determine whether it's closed to zero or not. This can potentially
           lead to sensitivity to the tolerance hyperparameter. Sun et al.
           suggested to perform a permutation test for every causation entropy
           computed to determine its siginificance, which is more costly on
           computations.

        References
        ----------

        .. [1] Sun et al., SIAM (2015) https://doi.org/10.1137/140956166

        """
        data = TS.T  # Transpose the time series to make observations the rows
        _, N = data.shape

        # Transform the data into its binned categorical version,
        # which is a pre-processing before computing entropy
        data = categorized_data(data, n_bins)

        # Find causal parents of each node
        adjlist = dict()
        for node in range(N):
            parents = causal_superset({node}, data, atol)
            remove_noncausal(parents, {node}, data, atol)
            adjlist[node] = parents

        # Build the reconstructed graph
        A = nx.to_numpy_array(nx.DiGraph(adjlist).reverse())
        G = create_graph(A, create_using=nx.DiGraph(), remove_self_loops=False)
        self.results['adjacency_matrix'] = A
        self.results['graph'] = G
        self.results['parents'] = adjlist

        return G


def causal_superset(nodes_I, data, atol):
    """
    Return a superset of causal parents for a set of nodes :math:`I` by a heuristic
    (adding node that maximizes causation entropy aggregatively).

    Parameters
    ----------
    nodes_I (set): Set of node indices.

    data (np.ndarray): Array of categorical data with nodes as columns and
                       observations of quantity on nodes as rows, which will
                       be involved in computing causation entropy.

    atol (float): Absolute tolerance to determine whether causation entropy is
                  closed to zero.

    """
    _, N = data.shape
    superset = set()
    nodes = set(range(N))  # Set of nodes that are not in the superset
    max_entrp = np.inf
    candidate = set()  # Candidate node to be added to the superset

    # Aggregatively add the node which leads to maximum causal entropy to the
    # superset, until the maximum causation entropy is closed to zero within a
    # given absolute tolerance
    while max_entrp > atol and len(nodes) > 0:
        superset |= candidate
        nodes -= candidate
        # print(f'Node {candidate} is added to the superset.')
        # Find the node outside of the superset with maximum causation entropy
        _max = -np.inf
        for n in nodes:
            cand = {n}
            entrp = causation_entropy(data, nodes_I, cand, superset)
            if entrp > _max:
                _max = entrp
                candidate = cand
        max_entrp = _max

    return superset


def remove_noncausal(superset, nodes_I, data, atol):
    """
    Remove noncausal nodes in the superset of nodes :math:`I`'s causal parents,
    where noncausal nodes are identified via zero causation entropy.

    Parameters
    ----------
    superset (set): Set of node indices, which contains the causal parents.

    nodes_I (set): Set of node indices.

    data (np.ndarray): Array of categorical data with nodes as columns and
                       observations of quantity on nodes as rows, which will
                       only be involved in computing causation entropy.

    atol (float): Absolute tolerance to determine whether causation entropy is
                  closed to zero.

    """
    nodes = superset.copy()  # Set of nodes in the superset
    for n in nodes:
        cand = {n}  # Candidate node to be removed
        entrp = causation_entropy(data, nodes_I, cand, superset - cand)
        # print(f'CSE from node {cand} = {entrp}')
        if entrp <= atol:
            superset -= cand
            # print(f'Node {cand} is removed from the superset.')


def causation_entropy(data, nodes_I, nodes_J, nodes_K):
    """
    Return the causation entropy from a set of nodes :math:`J` to nodes :math:`I`
    conditioning on nodes :math:`K`.

    Parameters
    ----------
    data (np.ndarray): Array of categorical data with nodes as columns and
                       observations of quantity on nodes as rows.

    nodes_I, nodes_J, nodes_K (set): Sets of node indices.

    Returns
    -------
    entrp (float): Causation entropy defined as
                   :math:`C_{J \rightarrow I | K} = H(X_{t+1}^{(I)} | X_t^{(K)}) -
                    H(X_{t+1}^{(I)} | X_t^{(K)}, X_t^{(J)})`
                   where :math:`H(X|Y)` is the conditional entropy of :math:`X`
                   conditioned on :math:`Y`.

    """
    # If nodes_J is a subset of nodes_K, return 0 to avoid redundancy
    if nodes_J <= nodes_K:
        return 0

    _vars = tuple(nodes_I)
    conds = tuple(nodes_K)
    joint = tuple(nodes_K | nodes_J)
    entrp = conditional_entropy(data[1:, _vars], data[:-1, conds])
    entrp -= conditional_entropy(data[1:, _vars], data[:-1, joint])

    return entrp

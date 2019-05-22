"""
onion_divergence.py
--------------------------

The Jensen-Shannon divergence between the two graphs onion decomposition.

Graph distance based on :
https://www.nature.com/articles/srep31708
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.011023

authors: Laurent Hébert-Dufresne and Guillaume St-Onge
email: guillaume.st-onge.4@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.

"""

import numpy as np
import networkx as nx
from .base import BaseDistance
from functools import reduce
from netrd.utilities import ensure_undirected


class OnionDivergence(BaseDistance):
    """Compares various types of feature distributions."""

    def dist(self, G1, G2, dist='lccm'):
        """Jenson-Shannon divergence between the feature distributions fixed by dist.

        Assumes simple graphs.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        dist (str)
            type of distribution divergence to output. Choices are 'cm',
            'ccm', 'lccm_node' and 'lccm'. The type stand for the
            associated random graph ensemble. 'cm' compares only the degree
            distribution. 'ccm' compares the networks according to the
            edges degree-degree distribution.  'lccm_node' compares the
            distribution of nodes according to their onion centrality
            (degree, coreness, and layer within core). Finally, 'lccm'
            compares the networks according to the edges joint
            degree, coreness and layer distribution for both endpoints.

        Returns
        -------
        dist (float)
            the distance between `G1` and `G2`.

        References
        ----------
        .. [1] https://www.nature.com/articles/srep31708

        .. [2] https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.011023

        """
        # take the simple graph version
        G1_simple = ensure_undirected(G1)
        G2_simple = ensure_undirected(G2)
        G1_simple.remove_edges_from(nx.selfloop_edges(G1_simple))
        G2_simple.remove_edges_from(nx.selfloop_edges(G2_simple))

        # get sparse matrices values for each graph
        matrices_G1 = _create_sparse_matrices_for_graph(G1_simple)
        matrices_G2 = _create_sparse_matrices_for_graph(G2_simple)

        # get the different distances
        cm_dist = _divergence_of_sparse_matrices(*matrices_G1['cm'], *matrices_G2['cm'])
        ccm_dist = _divergence_of_sparse_matrices(
            *matrices_G1['ccm'], *matrices_G2['ccm']
        )
        lccm_node_dist = _divergence_of_sparse_matrices(
            *matrices_G1['lccm_node'], *matrices_G2['lccm_node']
        )
        lccm_dist = _divergence_of_sparse_matrices(
            *matrices_G1['lccm'], *matrices_G2['lccm']
        )
        # store the distances
        self.results['cm_dist'] = cm_dist
        self.results['ccm_dist'] = ccm_dist
        self.results['lccm_node_dist'] = lccm_node_dist
        self.results['lccm_dist'] = lccm_dist
        dist_id = '{}_dist'.format(dist)
        self.results['dist'] = self.results[dist_id]

        return self.results[dist_id]


def _onion_decomposition(G):
    # Creates a copy of the graph (to be able to remove vertices and edges)
    G_copy = G.copy()
    # Dictionaries to register the k-core/onion decompositions.
    coreness_map = {}
    layer_map = {}
    local_layer_map = {}
    # Performs the onion decomposition.
    current_core = 0
    current_layer = 1
    local_layer = 1
    while G_copy.number_of_nodes() > 0:
        # Sets properly the current core.
        degree_sequence = [d for n, d in G_copy.degree()]
        min_degree = min(degree_sequence)
        if min_degree >= (current_core + 1):
            current_core = min_degree
            local_layer = 1
        # Identifies vertices in the current layer.
        this_layer_ = []
        for v in G_copy.nodes():
            if G_copy.degree()[v] <= current_core:
                this_layer_.append(v)
        # Identifies the core/layer of the vertices in the current layer.
        for v in this_layer_:
            coreness_map[v] = current_core
            layer_map[v] = current_layer
            local_layer_map[v] = local_layer
            G_copy.remove_node(v)
        # Updates the layer count.
        current_layer = current_layer + 1
        local_layer = local_layer + 1
    # Returns the dictionaries containing the k-shell and onion layer of
    # each vertices.
    return (layer_map, local_layer_map, coreness_map)


def _update_sparse_matrix(dictionary, values, key, index):
    # Creates entry if it doesn't exist or else update the corresponding value
    if not key in dictionary:
        index_i = index
        dictionary[key] = index
        values.append(1)
        index += 1
    else:
        index_i = dictionary[key]
        values[index_i] += 1
    # Returns new index (i.e. the nb of nonzero entry in sparse matrix)
    return index


def _create_sparse_matrices_for_graph(G):
    onion, local_layer, kcore = _onion_decomposition(G)
    # creates of copy of the graph
    G_copy = G.copy()
    cm_sparse_stub_matrix = {}
    ccm_sparse_stub_matrix = {}
    lccm_sparse_stub_matrix = {}
    lccm_sparse_node_matrix = {}
    cm_values = []
    cm_index = 0
    ccm_values = []
    ccm_index = 0
    lccm_node_values = []
    lccm_node_index = 0
    lccm_values = []
    lccm_index = 0
    for n in G_copy.nodes():
        d1 = G_copy.degree(n)
        c1 = kcore[n]
        l1 = local_layer[n]
        cm_index = _update_sparse_matrix(cm_sparse_stub_matrix, cm_values, d1, cm_index)
        lccm_node_index = _update_sparse_matrix(
            lccm_sparse_node_matrix, lccm_node_values, (d1, c1, l1), lccm_node_index
        )
    for e in G_copy.edges():
        d1 = G_copy.degree(e[0])
        d2 = G_copy.degree(e[1])
        c1 = kcore[e[0]]
        c2 = kcore[e[1]]
        l1 = local_layer[e[0]]
        l2 = local_layer[e[1]]
        ccm_index = _update_sparse_matrix(
            ccm_sparse_stub_matrix, ccm_values, (d1, d2), ccm_index
        )
        lccm_index = _update_sparse_matrix(
            lccm_sparse_stub_matrix, lccm_values, (d1, c1, l1, d2, c2, l2), lccm_index
        )
        ccm_index = _update_sparse_matrix(
            ccm_sparse_stub_matrix, ccm_values, (d2, d1), ccm_index
        )
        lccm_index = _update_sparse_matrix(
            lccm_sparse_stub_matrix, lccm_values, (d2, c2, l2, d1, c1, l1), lccm_index
        )
    cm_values = [x / G_copy.number_of_nodes() for x in cm_values]
    ccm_values = [x / (2 * G_copy.number_of_edges()) for x in ccm_values]
    lccm_values = [x / (2 * G_copy.number_of_edges()) for x in lccm_values]
    lccm_node_values = [x / G_copy.number_of_nodes() for x in lccm_node_values]

    # output the results in a dictionary
    results = dict()
    results['cm'] = (cm_sparse_stub_matrix, cm_values)
    results['ccm'] = (ccm_sparse_stub_matrix, ccm_values)
    results['lccm'] = (lccm_sparse_stub_matrix, lccm_values)
    results['lccm_node'] = (lccm_sparse_node_matrix, lccm_node_values)
    return results


def _divergence_of_sparse_matrices(keys1, values1, keys2, values2):
    keys = reduce(set.union, map(set, map(dict.keys, [keys1, keys2])))
    JSD = 0
    for key in keys:
        if key in keys1:
            value1 = values1[keys1[key]]
        else:
            value1 = 0.0
        if key in keys2:
            value2 = values2[keys2[key]]
        else:
            value2 = 0.0
        avg = (value1 + value2) / 2
        if value1 > 0:
            JSD += 0.5 * value1 * np.log2(value1 / avg)
        if value2 > 0:
            JSD += 0.5 * value2 * np.log2(value2 / avg)
    return JSD

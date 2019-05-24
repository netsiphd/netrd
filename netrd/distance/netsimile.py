"""
netsimile.py
--------------

Graph distance based on:
Berlingerio, M., Koutra, D., Eliassi-Rad, T. & Faloutsos, C. NetSimile: A Scalable Approach to Size-Independent Network Similarity. arXiv (2012)

author: Alex Gates
email: ajgates42@gmail.com (optional)
Submitted as part of the 2019 NetSI Collabathon.

"""
import networkx as nx
import numpy as np
import warnings
from scipy.spatial.distance import canberra
from scipy.stats import skew, kurtosis

from .base import BaseDistance


class NetSimile(BaseDistance):
    """Compares node signature distributions."""

    def dist(self, G1, G2):
        """A scalable approach to network similarity.

        A network similarity measure based on node signature distrubtions.

        The results dictionary includes the underlying feature matrices in
        `'feature_matrices'` and the underlying signature vectors in
        `'signature_vectors'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two undirected networkx graphs to be compared.

        Returns
        -------

        dist (float)
            the distance between `G1` and `G2`.

        References
        ----------

        .. [1] Michele Berlingerio, Danai Koutra, Tina Eliassi-Rad,
               Christos Faloutsos: NetSimile: A Scalable Approach to
               Size-Independent Network Similarity. CoRR abs/1209.2684
               (2012)

        """
        # NOTE: the measure only works for undirected
        # graphs. For now we will silently convert a
        # directed graph to be undirected.
        directed_flag = False
        if nx.is_directed(G1):
            G1 = nx.to_undirected(G1)
            directed_flag = True
        if nx.is_directed(G2):
            G2 = nx.to_undirected(G2)
            directed_flag = True

        if directed_flag:
            warnings.warn("Coercing directed graph to undirected.", RuntimeWarning)

        # find the graph node feature matrices
        G1_node_features = feature_extraction(G1)
        G2_node_features = feature_extraction(G2)

        # get the graph signature vectors
        G1_signature = graph_signature(G1_node_features)
        G2_signature = graph_signature(G2_node_features)

        # the final distance is the absolute canberra distance
        dist = abs(canberra(G1_signature, G2_signature))

        self.results['feature_matrices'] = G1_node_features, G2_node_features
        self.results['signature_vectors'] = G1_signature, G2_signature
        self.results['dist'] = dist

        return dist


def feature_extraction(G):
    """Node feature extraction.

        Parameters
        ----------

        G (nx.Graph): a networkx graph.

        Returns
        -------

        node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = [nx.ego_graph(G, n) for n in node_list]
    egonet_dict = {node: egonet for node, egonet in zip(node_list, egonets)}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # number of edges in the neighborhood
    neighbor_edges = [
        egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
        for n in node_list
    ]

    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    # node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    neighbor_outgoing_edges = [
        len(
            [
                edge
                for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                if not egonets[i].has_edge(*edge)
            ]
        )
        for i in node_list
    ]

    # number of neighbors of neighbors (not in neighborhood)
    neighbors_of_neighbors = [
        len(
            set([p for m in G.neighbors(n) for p in G.neighbors(m)])
            - set(G.neighbors(n))
            - set([n])
        )
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts
    node_features[:, 4] = neighbor_edges
    node_features[:, 5] = neighbor_outgoing_edges
    node_features[:, 6] = neighbors_of_neighbors

    return np.nan_to_num(node_features)


def graph_signature(node_features):
    signature_vec = np.zeros(7 * 5)

    # for each of the 7 features
    for k in range(7):
        # find the mean
        signature_vec[k * 5] = node_features[:, k].mean()
        # find the median
        signature_vec[k * 5 + 1] = np.median(node_features[:, k])
        # find the std
        signature_vec[k * 5 + 2] = node_features[:, k].std()
        # find the skew
        signature_vec[k * 5 + 3] = skew(node_features[:, k])
        # find the kurtosis
        signature_vec[k * 5 + 4] = kurtosis(node_features[:, k])

    return signature_vec


"""
# sample usage
>>>from netrd.distance import NetSimile
>>>G1 = nx.karate_club_graph()
>>>G2 = nx.krackhardt_kite_graph()

>>>test = NetSimile()
>>>print(test.dist(G1, G2))
20.180783067167326
"""

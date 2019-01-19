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
from scipy.spatial.distance import canberra
from scipy.stats import skew, kurtosis

from .base import BaseDistance


class NetSimile(BaseDistance):
    def dist(self, G1, G2):
        """A scalable approach to network similarity.

        A network similarity measure based on node signature distrubtions.

        Params
        ------

        G1, G2 (nx.Graph): two undirected networkx graphs to be compared.

        Returns
        -------

        dist (float): the distance between G1 and G2.

        """

        # the measure only works for undirected Graphs

        # TODO: replace with netrd error messages
        assert not nx.is_directed(G1), "G1 must be an undirected networkx graph"
        assert not nx.is_directed(G2), "G2 must be an undirected networkx graph"
        
        # find the graph node feature matrices
        G1_node_features = feature_extraction(G1)
        G2_node_features = feature_extraction(G2)
        
        # get the graph signature vectors
        G1_signature = graph_signature(G1_node_features)
        G2_signature = graph_signature(G2_node_features)
        
        # the final distance is the absolute canberra distance
        dist = abs(canberra(G1_signature, G2_signature))
        
        self.results['dist'] = dist

        return dist


def feature_extraction(G):
    """Node feature extraction.

        Params
        ------

        G (nx.Graph): a networkx graph.

        Returns
        -------

        node_features (float): the Nx7 matrix of node features."""

    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    
    node_list = sorted(G.nodes())
    
    # node degrees
    node_degree_dict = dict(G.degree())
    node_features[:,0] = [node_degree_dict[n] for n in node_list]
    
    # clustering coefficient
    node_clustering_dict = dict(nx.clustering(G))
    node_features[:,1] = [node_clustering_dict[n] for n in node_list]
    
    # average degree of neighborhood
    node_features[:,2] = [np.mean([node_degree_dict[m] for m in G.neighbors(n)]) if node_degree_dict[n] > 0 else 0 for n in node_list]
    
    # average clustering coefficient of neighborhood
    node_features[:,3] = [np.mean([node_clustering_dict[m] for m in G.neighbors(n)]) if node_degree_dict[n] > 0 else 0 for n in node_list]
    
    # number of edges in the neighborhood
    node_features[:,4] = [G.subgraph(list(G.neighbors(n)) + [n]).number_of_edges() if node_degree_dict[n] > 0 else 0 for n in node_list]
    
    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    
    # number of neighbors of neighbors (not in neighborhood)
    node_features[:,6] = [len(set([p for m in G.neighbors(n) for p in G.neighbors(m)]) - set(G.neighbors(n)) - set([n])) if node_degree_dict[n] > 0 else 0 for n in node_list]
    
    return np.nan_to_num(node_features)

def graph_signature(node_features):
    signature_vec = np.zeros(7*5)
    
    # for each of the 7 features
    for k in range(7):
        # find the mean
        signature_vec[k*5] = node_features[:,k].mean()
        # find the median
        signature_vec[k*5 + 1] = np.median(node_features[:,k])
        # find the std
        signature_vec[k*5 + 2] = node_features[:,k].std()
        # find the skew
        signature_vec[k*5 + 3] = skew(node_features[:,k])
        # find the kurtosis
        signature_vec[k*5 + 4] = kurtosis(node_features[:,k])
        
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

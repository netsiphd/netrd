"""
cluster.py
--------

Utilities for creating a seriated/ordered adjacency matrix with hierarchical clustering.

author: David Saffo
email: saffo.d@husky.neu.edu
Submitted as part of the 2019 NetSI Collabathon

"""
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage


def clusterGraph(G, method='single', metric='euclidean', optimal_ordering=False):
    """
    Function for creating seriated adjacency matrix 

    Parameters
    ----------
    G: a networkx graph
    
    method: the clustering algorithm to use for options see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    
    metric: distance method to use
    
    optimal_ordering: if true tries to minimize the distance of succesive indexes 

    Returns
    -------
    adjClustered: a numpy matrix with rows and columns reordered based on clustering
    order: a list with the new index order for rows and columns
    dend: a dictionary with the hierarchy for the dendogram
    link: a linkage matrix with results from clustering
    

    """
    adj = nx.to_numpy_matrix(G)
    link = linkage(adj, method, metric, optimal_ordering)
    dend = dendrogram(link, no_plot=True)
    order = dend['leaves']
    adjClustered = adj[order, :]
    adjClustered = adjClustered[:, order]
    return adjClustered, order, dend, link

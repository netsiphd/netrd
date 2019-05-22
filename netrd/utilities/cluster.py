"""
cluster.py
----------

Utilities for creating a seriated/ordered adjacency matrix with
hierarchical clustering.

author: David Saffo (saffo.d@husky.neu.edu)

Submitted as part of the 2019 NetSI Collabathon.

"""
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage


def clusterGraph(G, method='single', metric='euclidean', optimal_ordering=False):
    """Create seriated adjacency matrix.

    Parameters
    ----------

    G (nx.Graph)
        a networkx graph

    method
        the clustering algorithm to use for options see [1].

    metric (str)
        linkage method to use

    optimal_ordering (bool)
        if true tries to minimize the distance of succesive indexes

    Returns
    -------

    adjClustered (np.ndarray)
        a numpy matrix with rows and columns reordered based on clustering

    order (list)
        a list with the new index order for rows and columns

    dend (dict)
        a dictionary with the hierarchy for the dendogram

    link (np.ndarray)
        a linkage matrix with results from clustering

    References
    ----------

    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    """
    adj = nx.to_numpy_matrix(G)
    link = linkage(adj, method, metric, optimal_ordering)
    dend = dendrogram(link, no_plot=True)
    order = dend['leaves']
    adjClustered = adj[order, :]
    adjClustered = adjClustered[:, order]
    return adjClustered, order, dend, link

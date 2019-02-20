"""
standardize.py
------------

Utilities for computing standardization values for distance measures.

author: Harrison Hartle/Tim LaRock
email: timothylarock at gmail dot com
Submitted as part of the 2019 NetSI Collabathon

"""

import numpy as np
import networkx as nx

def mean_GNP_distance(n, edge_probability, samples=10, distance=None, **kwargs):
    '''
    Compute the mean distance between _samples_ GNP graphs with
    parameters N=n,p=edge_probability using distance function _distance_,
    whose parameters are passed with **kwargs.


    Params
    ------

    n (int): Number of nodes in ER graphs to be generated
    edge_probability (float): Probability of edge in ER graphs to be generated.
    samples (int): Number of samples to average distance over.
    distance (function): Function from netrd.distances.<distance>.dist
    kwargs (dict): Keyword arguments to pass to the distance function.


    Returns
    -------

    mean (float): The average distance between the sampled ER networks.
    std (float): The standard deviation of the distances.
    dist (np.ndarray): Array storing the actual distances.

    '''
    # generate sample graphs
    a=[]
    for _ in range(samples):
       a.append(nx.fast_gnp_random_graph(n, edge_probability))

    # get distances
    dis=[]
    for i in range(samples):
       for j in range(i):
           dis.append(distance(a[i], a[j], **kwargs))

    # get mean distances \ std distances
    mean_dist=np.mean(dis)
    std_dist=np.std(dis)

    return mean_dist,std_dist,dis


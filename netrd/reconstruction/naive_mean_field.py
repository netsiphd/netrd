"""
naive_mean_field.py
---------------------
Reconstruction of graphs using a naive mean field approximation
author: Brennan Klein
email: brennanjamesklein at gmail dot com
submitted as part of the 2019 NetSI Collabathon
"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx
import scipy as sp
from scipy import linalg
from ..utilities.graph import create_graph


class NaiveMeanFieldReconstructor(BaseReconstructor):
    def fit(self, TS):
        """
        Given a (N,L) time series, infer inter-node coupling weights using a 
        naive mean field approximation. After [this tutorial]
        (https://github.com/nihcompmed/network-inference/blob/master/sphinx/codesource/inference.py) 
        in python.
        
        Params
        ------
        TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.
        
        Returns
        -------
        G (nx.Graph or nx.DiGraph): a reconstructed graph.

        """

        N, L = np.shape(TS)             # N nodes, length L
        m = np.mean(TS, axis=1)         # empirical value

        # A matrix
        A = 1 - m**2
        A_inv = np.diag(1 / A)
        A = np.diag(A)

        ds = TS.T - m                   # equal time correlation
        C = np.cov(ds, rowvar=False, bias=True)
        C_inv = linalg.inv(C)
        
        s1 = TS[:,1:]                   # one-step-delayed correlation

        ds1 = s1.T - np.mean(s1, axis=1)
        D = cross_cov(ds1, ds[:-1])

        # predict W:
        B = np.dot(D, C_inv)
        W = np.dot(A_inv, B)

        # construct the network
        self.results['graph'] = create_graph(W)
        self.results['matrix'] = W
        G = self.results['graph']

        return G

def cross_cov(a, b):
    """ 
    cross_covariance
    a,b -->  <(a - <a>)(b - <b>)>  (axis=0) 
    """    
    da = a - np.mean(a, axis=0)
    db = b - np.mean(b, axis=0)

    return np.matmul(da.T, db) / a.shape[0]
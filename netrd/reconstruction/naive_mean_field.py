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

def cross_cov(a, b):
    """ 
    cross_covariance
    a,b -->  <(a - <a>)(b - <b>)>  (axis=0) 
    """    
    da = a - np.mean(a, axis=0)
    db = b - np.mean(b, axis=0)

    return np.matmul(da.T, db) / a.shape[0]


class NaiveMeanFieldReconstructor(BaseReconstructor):
    def fit(self, ts):
        """
        Given a (N,t) time series, infer inter-node coupling weights using a 
        naive mean field approximation. After [this tutorial]
        (https://github.com/nihcompmed/network-inference/blob/master/sphinx/codesource/inference.py) 
        in python.
        
        Params
        ------
        ts (np.ndarray): Array consisting of $T$ observations from $N$ sensors.
        
        Returns
        -------
        G (nx.Graph or nx.DiGraph): a reconstructed graph.

        """
        
        N, t = np.shape(ts)             # N nodes, length t
        m = np.mean(ts, axis=1)         # empirical value

        # A matrix
        A = 1 - m**2 
        A_inv = np.diag(1/A)
        A = np.diag(A)

        ds = ts.T - m                   # equal time correlation
        C = np.cov(ds, rowvar=False, bias=True)
        C_inv = linalg.inv(C)
        
        s1 = ts[:,1:]                   # one-step-delayed correlation
        ds1 = s1.T - np.mean(s1, axis=1)
        D = cross_cov(ds1,ds[:-1])    
        
        # predict W:
        B = np.dot(D, C_inv)
        W = np.dot(A_inv, B)

        # construct the network
        self.results['graph'] = nx.from_numpy_array(W)
        self.results['matrix'] = W
        G = self.results['graph']

        return G
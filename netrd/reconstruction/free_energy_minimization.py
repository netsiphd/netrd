"""
free_energy_minimization.py
---------------------
Reconstruction of graphs by minimizing a free energy of your data
author: Brennan Klein
email: brennanjamesklein at gmail dot com
submitted as part of the 2019 NetSI Collabathon
"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx
import scipy as sp
from scipy import linalg

class FreeEnergyMinimizationReconstructor(BaseReconstructor):
    def fit(self, ts):
        """
        Given a (N,t) time series, infer inter-node coupling weights by 
        minimizing a free energy over the data structure. After [this tutorial]
        (https://github.com/nihcompmed/network-inference/blob/master/sphinx/codesource/inference.py) in python.
        
        Params
        ------
        ts (np.ndarray): Array consisting of $T$ observations from $N$ sensors.
        
        Returns
        -------
        G (nx.Graph or nx.DiGraph): a reconstructed graph.

        """
    
        N, t = np.shape(ts)             # N nodes, length t
        m  = np.mean(ts[:,:-1], axis=1) # model average
        ds = ts[:,:-1].T - m            # discrepancy
        t1 = t - 1                      # time limit
        

        # covariance of the discrepeancy 
        c = np.cov(ds, rowvar=False, bias=True) 
        
        c_inv = linalg.inv(c)           # inverse
        dst = ds.T                      # discrepancy at t

        # empty matrix to populate w/ inferred couplings
        W = np.empty((N,N)) 
        
        nloop = 10000       # failsafe

        for i0 in range(N): # for each node
            
            ts1 = ts[i0,1:] # take its entire time series
            h = ts1         # calculate the the local field
            
            cost = np.full(nloop, 100.)
            
            for iloop in range(nloop):
                
                h_av  = np.mean(h)                # average local field
                hs_av = np.dot(dst, h-h_av) / t1  # deltaE_i delta\sigma_k
                w = np.dot(hs_av, c_inv)          # expectation under model 

                h = np.dot(ts[:,:-1].T, w[:])     # estimate of local field
                ts_model    = np.tanh(h)          # under kinetic Ising model
                
                # discrepancy cost
                cost[iloop] = np.mean( (ts1[:]-ts_model[:])**2 ) 
                
                if cost[iloop] >= cost[iloop-1]:                 
                    break # if it increases, break
                           
                # complicated, but this seems to be the estimate of W_i
                h *= np.divide(ts1, ts_model, out=np.ones_like(ts1), 
                               where=ts_model!=0)

            W[i0,:] = w[:]

        # construct the network
        self.results['graph'] = nx.from_numpy_array(W)
        self.results['matrix'] = W
        G = self.results['graph']

        return G
"""
communicability_jsd.py
--------------------------

Distance measure based on the Jensen-Shannon Divergence 
between the communicability sequence of two graphs as 
defined in:

Chen, D., Shi, D. D., Qin, M., Xu, S. M., & Pan, G. J. (2018). 
Complex network comparison based on communicability 
sequence entropy. Physical Review E, 98(1), 012319.

https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.012319

author: Brennan Klein
email: brennanjamesklein@gmail.com
Submitted as part of the 2019 NetSI Collabathon.

"""

import networkx as nx
import numpy as np
import scipy as sp
from .base import BaseDistance

class CommunicabilityJSD(BaseDistance):
    def dist(self, G1, G2):
        """        
        Params
        ------
        G1 (nx.Graph): the first graph
        G2 (nx.Graph): the second graph
        
        Returns
        -------
        dist (float): between zero and one, this is the communicability 
                      sequence distance bewtween G1 and G2.
        
        """

        N1 = G1.number_of_nodes()
        N2 = G2.number_of_nodes()

        C1 = nx.communicability_exp(G1)
        C2 = nx.communicability_exp(G2)

        Ca1 = np.zeros((N1, N1))
        Ca2 = np.zeros((N2, N2))

        for i in range(Ca1.shape[0]):
            Ca1[i] = np.array(list(C1[i].values()))
        for i in range(Ca2.shape[0]):
            Ca2[i] = np.array(list(C2[i].values()))

        lil_sigma1 = np.triu(Ca1).flatten()
        lil_sigma2 = np.triu(Ca2).flatten()

        big_sigma1 = sum(lil_sigma1[np.nonzero(lil_sigma1)[0]])
        big_sigma2 = sum(lil_sigma2[np.nonzero(lil_sigma2)[0]])

        P1 = lil_sigma1 / big_sigma1
        P2 = lil_sigma2 / big_sigma2
        P0 = (P1 + P2) / 2

        H1 = sp.stats.entropy(P1)
        H2 = sp.stats.entropy(P2)
        H0 = sp.stats.entropy(P0)
        dist = np.sqrt(H0 - 0.5 * (H1 + H2))

        self.results['P1'] = P1
        self.results['P2'] = P2
        self.results['P0'] = P0

        self.results['entropy_1']       = H1
        self.results['entropy_2']       = H2
        self.results['entropy_mixture'] = H0
        self.results['dist'] = dist

        return dist

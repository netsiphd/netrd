"""
<file_name>.py
--------------

Graph distance based on <link to paper, or website, or repository>.

author: <your name here>
email: <name at server dot com> (optional)
Submitted as part of the 2019 NetSI Collabathon.

"""

import scipy
import numpy as np
import networkx as nx
from .base import BaseDistance


class Hamming(BaseDistance):
    def dist(self, G1, G2):
        """The Hamming distance between two graphs is the proportion of
        disagreeing nodes between two flattened adjacency matricies. If
        $u$ and $v$ are boolean vectors, then Hamming distance is:

            $\\frac{c_{01} + c_{10}}{n}$

        where $c_{ij}$ is the number of occurrences of where $u[k] = i$ and 
        $v[k] = j$ for $k < n$.

        The graphs must have the same number of nodes. A small modification to 
        this code could allow weights can be applied, but only one set of 
        weights that apply to both graphs.

        Params
        ------

        G1, G2 (nx.Graph): two networkx graphs to be compared.

        Returns
        -------

        dist (float): the distance between G1 and G2.

        """

        adj1 = nx.adjacency_matrix(G1)
        adj2 = nx.adjacency_matrix(G2)
        dist = scipy.spatial.distance.hamming(
            adj1.toarray().flatten(), 
            adj2.toarray().flatten()
        )
        self.results['dist'] = dist
        self.results['adj'] = np.array([adj1, adj2])

        return dist


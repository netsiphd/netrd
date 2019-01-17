"""
jaccard_distance.py
--------------

Graph distance based on https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html

author: David Saffo
email: saffo.d@husky.neu.edu
Submitted as part of the 2019 NetSI Collabathon.

"""

from .base import BaseDistance
from sklearn.metrics import jaccard_similarity_score
import networkx as nx



class JaccardDistance(BaseDistance):
    def dist(self, G1, G2):
        """Computes the average jaccard index between two sparse matrices
        
        implementation details here:
        https://scikit-learn.org/stable/modules/model_evaluation.html#jaccard-similarity-score
        and here:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html

        Params
        ------

        G1, G2 (nx.Graph): two networkx graphs to be compared.

        Returns
        -------

        dist (float): the distance between G1 and G2.

        """
        adj1 = nx.to_numpy_matrix(G1)
        adj2 = nx.to_numpy_matrix(G2)
        dist = jaccard_similarity_score(adj1,adj2)
        self.results['dist'] = float(dist)
        self.results['adj'] = np.array([adj1, adj2])
        return dist



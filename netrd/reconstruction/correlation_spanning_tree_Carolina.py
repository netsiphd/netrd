"""
correlation_spanning_tree.py
---------------------

Reconstruction of graphs using a minimum spanning tree from the correlation matrix.

author: Carolina Mattsson
email: mattsson dot c at husky dot neu dot edu
Submitted as part of the 2019 NetSI Collabathon

"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx


class CorrelationMSTReconstructor(BaseReconstructor):
    def fit(self, TS, distance='root_inv', matrix='corr', matrix_args=None):
        """
        Reconstruct a network from time series data using a minimum spanning tree
        of the correlation matrix. The raw correlations are first converted into
        a notion of distance. After: Mantegna, R. N. Hierarchical Structure in
        Financial Markets. The European Physical Journal B 11, 193–197 (1999).

        Params
        ------
        TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors
        distance:
            'inv_square': calculates distance as 1-corr_ij^2 (Mantegna 1999)
            'root_inv': calculates distance as sqrt(2*(1-corr_ij)) (Bonanno et al. 2003)
        matrix:
            'corr': uses the matrix of correlation coefficients
            'partial': uses the partial correlation coefficients
        matrix_args: takes a dictionary of keyword arguments to pass to the correlation calculation

        Returns
        -------
        G: a reconstructed graph.

        """
        import math

        # get the correlation matrix
        if matrix == 'corr':
            from .correlation_matrix import CorrelationMatrixReconstructor
            cor =  CorrelationMatrixReconstructor.fit(TS) if not matrix_args else CorrelationMatrixReconstructor.fit(TS,**matrix_args)
        if matrix == 'partial':
            from .partial_correlation_matrix import PartialCorrelationMatrixReconstructor
            cor =  PartialCorrelationMatrixReconstructor.fit(TS) if not matrix_args else PartialCorrelationMatrixReconstructor.fit(TS,**matrix_args)
        self.results['matrix'] = cor

        # convert the correlations into distances
        if distance == 'root_inv':
            distance_function = np.vectorize(lambda x: math.sqrt(2*(1-x)))
        if distance == 'inv_square':
            distance_function = np.vectorize(lambda x: 1-x**2)
        dist = distance_function(cor)
        self.results['distance'] = dist

        # construct the network
        self.results['full_graph'] = nx.from_numpy_array(dist)

        # get the minimum spanning tree
        self.results['graph'] = nx.minimum_spanning_tree(self.results['full_graph'], weight='weight')
        G = self.results['graph']

        return G

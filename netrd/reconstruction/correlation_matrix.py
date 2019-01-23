"""
correlation_matrix.py
---------------------

Reconstruction of graphs using the correlation matrix.

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon

"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx
from ..utilities.graph import create_graph


class CorrelationMatrixReconstructor(BaseReconstructor):
    def fit(self, TS, cutoffs=[(-1, 1)]):
        """
        Reconstruct a network from time series data using an unregularized form of
        the precision matrix. After [this tutorial](
        https://github.com/valeria-io/visualising_stocks_correlations/blob/master/corr_matrix_viz.ipynb).

        Params
        ------
        TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors
        cutoffs (list of tuples): When thresholding, include only edges whose
        correlations fall within a given range or set of ranges. The lower
        value must come first.

        Returns
        -------
        G: a reconstructed graph.

        """

        # get the correlation matrix
        cor = np.corrcoef(TS)
        self.results['matrix'] = cor

        # get the mask using the cutoffs
        mask_function = np.vectorize(lambda x: any([x>=cutoff[0] and x<=cutoff[1] for cutoff in cutoffs]))
        mask = mask_function(cor)

        # use the mask to threshold the correlation matrix
        A = cor * mask

        # construct the network
        self.results['graph'] = create_graph(A)
        G = self.results['graph']

        return G

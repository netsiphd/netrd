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
from ..utilities import create_graph, threshold



class CorrelationMatrixReconstructor(BaseReconstructor):
    def fit(self, TS, threshold_type='range', **kwargs):
        """
        Reconstruct a network from time series data using an unregularized form of
        the precision matrix. After [this tutorial](
        https://github.com/valeria-io/visualising_stocks_correlations/blob/master/corr_matrix_viz.ipynb).

        Params
        ------
        TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors
        threshold_type (str): Which thresholding function to use on the matrix of
        weights. See `netrd.utilities.threshold.py` for documentation. Pass additional
        arguments to the thresholder using `**kwargs`.

        Returns
        -------
        G: a reconstructed graph.

        """

        # get the correlation matrix
        cor = np.corrcoef(TS)
        self.results['weights_matrix'] = cor

        # threshold the correlation matrix
        A = threshold(cor, threshold_type, **kwargs)

        # construct the network
        self.results['graph'] = create_graph(A)
        G = self.results['graph']

        return G

"""
<u_inference.py
--------------

Graph reconstruction algorithm based on [1, 2].

[1] P. Barucca, "Localization in covariance matrices of coupled heterogeneous
Ornstein-Uhlenbeck processes", Phys. Rev. E 90, 062129 (2014).
[2] https://github.com/paolobarucca/OUinference.

author: Charles Murphy
email: charles.murphy.1@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.
"""

from .base import BaseReconstructor
import networkx as nx
import numpy as np
from scipy.linalg import eig, inv
from ..utilities import create_graph, threshold


class OUInference(BaseReconstructor):
    """Assumes a Orstein-Uhlenbeck generative model."""

    def fit(self, TS, threshold_type='range', **kwargs):
        """Infers the coupling coefficients assuming a Orstein-Uhlenbeck process
        generative model.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'`, the covariance matrix in `covariance_matrix`
        and the thresholded version of the weight matrix as
        `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N`
            sensors.

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        -------

        G (nx.Graph)
            A reconstructed graph with :math:`N` nodes.

        """
        N, T = np.shape(TS)

        temperatures = np.mean((TS[:, 1:] - TS[:, :-1]) ** 2, 1) / 2
        index = np.where(temperatures > 0)
        Y = TS[index, :][0]

        yCovariance = np.cov(Y)
        index_pair = np.array([(i, j) for i in index for j in index])
        weights = inverse_method(-yCovariance, temperatures)
        self.results['covariance_matrix'] = np.zeros([N, N])
        self.results['covariance_matrix'][index_pair] = yCovariance

        self.results['weights_matrix'] = np.zeros([N, N])
        self.results['weights_matrix'][index_pair] = weights

        # threshold the network
        W_thresh = threshold(self.results['weights_matrix'], threshold_type, **kwargs)
        self.results['thresholded_matrix'] = W_thresh

        # construct the network
        self.results['graph'] = create_graph(W_thresh)
        G = self.results['graph']

        return G


def inverse_method(covariance, temperatures):
    """This function finds the weights of an heterogenous Ornstein-Uhlenbeck
    process
    covariance  = covariance matrix of the zero-mean signal

    Parameters
    ----------

    covariance (np.ndarray): Covariance matrix of the zero-mean signal.

    temperatures (np.ndarray): Diffusion coefficient of each of the signals.

    Returns
    -------

    weights (np.ndarray): Coupling between nodes under the OU process asumption.

    """

    if len(np.shape(temperatures)) == 1:
        T = np.diag(temperatures)
    elif len(np.shape(temperatures)) == 2:
        T = temperatures
    else:
        raise ValueError("temperature must either be a vector or a matrix.")

    n, m = np.shape(covariance)

    eig_val, eig_vec = eig(-covariance)
    eig_val = np.diag(eig_val)

    e_mat = np.matmul(eig_vec.T, np.matmul(T, eig_vec))

    eig_val = np.matmul(np.ones([n, n]), eig_val)
    eig_val = (eig_val + eig_val.T) ** (-1)
    eig_val = eig_val.real
    weights = -np.matmul(eig_vec, np.matmul(2 * eig_val * e_mat, eig_vec.T))

    return weights

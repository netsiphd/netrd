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

class OUInferenceReconstructor(BaseReconstructor):
    def fit(self, TS):
        """A brief one-line description of the algorithm goes here.

        A short paragraph may follow. The paragraph may include $latex$ by
        enclosing it in dollar signs $\textbf{like this}$.

        Params
        ------
        TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.
        N x T

        Returns
        -------
        G (nx.Graph): A reconstructed graph with $N$ nodes.

        """
        N, T = np.shape(TS)

        temperatures = np.mean((TS[:,1:] - TS[:,:-1])**2, 1) / 2
        index = np.where(temperatures>0)
        Y = TS[index, :][0]

        yCovariance = np.cov(Y)
        index_pair = np.array([(i, j) for i in index for j in index])
        couplings = inverse_method(-yCovariance, temperatures)
        self.results['covariance'] = np.zeros([N, N]);
        self.results['covariance'][index_pair] = yCovariance;

        self.results['couplings'] = np.zeros([N, N]);
        self.results['couplings'][index_pair] = couplings;

        G = nx.from_numpy_array(self.results['couplings'])
        self.results['graph'] = G

        return G


def inverse_method(covariance, temperatures):
    """This function finds the couplings of an heterogenous Ornstein-Uhlenbeck 
    process 
    covariance  = covariance matrix of the zero-mean signal

    Params
    ------

    covariance (np.ndarray): Covariance matrix of the zero-mean signal.

    temperatures (np.ndarray): Diffusion coefficient of each of the signals.

    Returns
    -------

    couplings (np.ndarray): Coupling between nodes under the OU process asumption.

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
    eig_val = (eig_val + eig_val.T)**(-1)
    eig_val = eig_val.real
    couplings = -np.matmul(eig_vec, np.matmul(2 * eig_val * e_mat, eig_vec.T))

    return couplings

"""
quantum_jsd.py
--------------------------

Graph distance based on the quantum $q$-Jenson-Shannon divergence.

De Domenico, Manlio, and Jacob Biamonte. 2016. “Spectral Entropies as
Information-Theoretic Tools for Complex Network Comparison.” Physical Review X
6 (4). https://doi.org/10.1103/PhysRevX.6.041062.


author: Stefan McCabe & Brennan Klein
email: 
Submitted as part of the 2019 NetSI Collabathon.

"""

import warnings
import networkx as nx
import numpy as np
from scipy.linalg import expm
from .base import BaseDistance


class QuantumJSD(BaseDistance):
    def dist(self, G1, G2, beta=0.1, q=None):
        """
        Return the square root of the quantum $q$-Jenson-Shannon divergence
        between two graphs.

        The generalized Jensen-Shannon divergence compares two graphs by the
        spectral entropies of their quantum-statistical-mechanical density
        matrices. It can be written as
        $$
        \mathcal{J}_q(\mathbf{\rho} || \mathbf{\sigma}) =
        S_q\left( \frac{\mathbf{\rho} + \mathbf{\sigma}}{2} \right) -
        \frac{1}{2} [S_q(\mathbf{\rho}) + S_q(\mathbf{\sigma})],
        $$
        where $\mathbf{\rho}$ and $\mathbf{\sigma}$ are density matrices and $q$
        is the order parameter.

        The density matrix
        $$
        \mathbf{\rho} = \frac{e^{-\beta\mathbf{L}}}{Z},
        $$
        where
        $$
        Z = \sum_{i=1}^{N}e^{-\beta\lambda_i(\mathbf{L})}
        $$
        and $\lambda_i(\mathbf{L})$ is the $i$th eigenvalue of the Laplacian
        matrix $L$, represents an imaginary diffusion process over the network
        with time parameter $\beta > 0$.
        
        For these density matrices and the mixture matrix, we calculate the
        Rényi entropy of order $q$
        $$
        S_q = \frac{1}{1-q} \log_2 \sum_{i=1}^{N}\lambda_i(\mathbf{\rho})^q,
        $$
        or, if $q=1$, the Von Neumann entropy
        $$
        S_1 = - \sum_{i=1}^{N}\lambda_i(\mathbf{\rho})\log_2\lambda_i(\mathbf{\rho}).
        $$

        Note that this implementation is not exact because the matrix
        exponentiation is performed using the Padé approximation.

        Params
        ------

        G1, G2 (nx.Graph): two networkx graphs to be compared.
        beta (float): time parameter for diffusion propagator
        q (float): order parameter for Rényi entropy. If None or 1, use the
        Von Neumann entropy (i.e., Shannon entropy) instead.

        Returns
        -------

        dist (float): the distance between G1 and G2.

        """

        if beta <= 0:
            raise ValueError("beta must be positive.")

        if q and q >= 2:
            warnings.warn("JSD is only a metric for 0 ≤ q < 2.",
                          RuntimeWarning)

        def density_matrix(A, beta):
            """
            Create the density matrix encoding probabilities for entropies.
            This is done using a fictive diffusion process with time parameter
            `beta`.
            """
            L = np.diag(np.sum(A, axis=1)) - A
            rho = expm(-1 * beta * L)
            rho = rho / np.trace(rho)

            return rho

        def renyi_entropy(X, q=None):
            """
            Calculate the Rényi entropy with order `q`, or the Von Neumann
            entropy if `q` is `None` or 1.
            """
            eigs = np.linalg.eigh(X)[0]
            eigs = eigs[eigs > 0]

            if q is None or q == 1:
                # plain Von Neumann entropy
                H = -1 * np.sum(eigs * np.log2(eigs))
            else:
                prefactor = 1 / (1 - q)
                H = prefactor * np.log2((eigs**q).sum())
            return H

        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)

        rho1 = density_matrix(A1, beta)
        rho2 = density_matrix(A2, beta)
        mix = (rho1 + rho2) / 2

        H0 = renyi_entropy(mix, q)
        H1 = renyi_entropy(rho1, q)
        H2 = renyi_entropy(rho2, q)

        dist = np.sqrt(H0 - 0.5 * (H1 + H2))

        self.results['density_matrix_1'] = rho1
        self.results['density_matrix_2'] = rho2
        self.results['mixture_matrix'] = mix
        self.results['entropy_1'] = H1
        self.results['entropy_2'] = H2
        self.results['entropy_mixture'] = H0
        self.results['dist'] = dist
        return dist

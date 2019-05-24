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
    """Compares the spectral entropies of the density matrices."""

    def dist(self, G1, G2, beta=0.1, q=None):
        r"""Square root of the quantum :math:`q`-Jenson-Shannon divergence between two
        graphs.

        The generalized Jensen-Shannon divergence compares two graphs by the
        spectral entropies of their quantum-statistical-mechanical density
        matrices. It can be written as

        .. math::
            \mathcal{J}_q(\mathbf{\rho} || \mathbf{\sigma}) =
            S_q\left( \frac{\mathbf{\rho} + \mathbf{\sigma}}{2} \right) -
            \frac{1}{2} [S_q(\mathbf{\rho}) + S_q(\mathbf{\sigma})],


        where :math:`\mathbf{\rho}` and :math:`\mathbf{\sigma}` are density
        matrices and :math:`q` is the order parameter.

        The density matrix

        .. math::
            \mathbf{\rho} = \frac{e^{-\beta\mathbf{L}}}{Z},


        where

        .. math::
            Z = \sum_{i=1}^{N}e^{-\beta\lambda_i(\mathbf{L})}


        and :math:`\lambda_i(\mathbf{L})` is the :math:`i`th eigenvalue of the Laplacian
        matrix :math:`L`, represents an imaginary diffusion process over the network
        with time parameter :math:`\beta > 0`.

        For these density matrices and the mixture matrix, we calculate the
        Rényi entropy of order :math:`q`

        .. math::
            S_q = \frac{1}{1-q} \log_2 \sum_{i=1}^{N}\lambda_i(\mathbf{\rho})^q,


        or, if :math:`q=1`, the Von Neumann entropy

        .. math::
            S_1 = - \sum_{i=1}^{N}\lambda_i(\mathbf{\rho})\log_2\lambda_i(\mathbf{\rho}).


        Note that this implementation is not exact because the matrix
        exponentiation is performed using the Padé approximation and
        because of imprecision in the calculation of the eigenvalues of the
        density matrix.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared

        beta (float)
            time parameter for diffusion propagator

        q (float)
            order parameter for Rényi entropy. If None or 1, use the Von
            Neumann entropy (i.e., Shannon entropy) instead.

        Returns
        -------

        dist (float)
            the distance between `G1` and `G2`.

        References
        ----------

        .. [1] De Domenico, Manlio, and Jacob Biamonte. 2016. "Spectral
               Entropies as Information-Theoretic Tools for Complex Network
               Comparison." Physical Review X 6
               (4). https://doi.org/10.1103/PhysRevX.6.041062.

        """
        if beta <= 0:
            raise ValueError("beta must be positive.")

        if q and q >= 2:
            warnings.warn("JSD is only a metric for 0 ≤ q < 2.", RuntimeWarning)

        def density_matrix(A, beta):
            """
            Create the density matrix encoding probabilities for entropies.
            This is done using a fictive diffusion process with time parameter
            :math:`beta`.
            """
            L = np.diag(np.sum(A, axis=1)) - A
            rho = expm(-1 * beta * L)
            rho = rho / np.trace(rho)

            return rho

        def renyi_entropy(X, q=None):
            """
            Calculate the Rényi entropy with order :math:`q`, or the Von Neumann
            entropy if :math:`q` is `None` or 1.
            """
            # Note that where there are many zero eigenvalues (i.e., large
            # values of beta) in the density matrix, floating-point precision
            # issues mean that there will be negative eigenvalues and the
            # eigenvalues will not sum to precisely one. To avoid encountering
            # `nan`s in `np.log2`, we remove all eigenvalues that are close
            # to zero within 1e-6 tolerance. As for the eigenvalues not summing
            # to exactly one, this is a small source of error in the
            # calculation.
            eigs = np.linalg.eigvalsh(X)
            zero_eigenvalues = np.isclose(np.abs(eigs), 0, atol=1e-6)
            eigs = eigs[np.logical_not(zero_eigenvalues)]

            if q is None or q == 1:
                # plain Von Neumann entropy
                H = -1 * np.sum(eigs * np.log2(eigs))
            else:
                prefactor = 1 / (1 - q)
                H = prefactor * np.log2((eigs ** q).sum())
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

"""
polynomial_dissimilarity.py
--------------

From
----
Donnat, Claire, and Susan Holmes. "Tracking
network dynamics: A survey of distances
and similarity metrics." arXiv
preprint arXiv:1801.07351 (2018).

author: Jessica T. Davis
email:
Submitted as part of the 2019 NetSI Collabathon.

"""
import numpy as np
import networkx as nx
import scipy.sparse as ss
from .base import BaseDistance


class PolynomialDissimilarity(BaseDistance):
    """Compares polynomials relating to the eigenvalues of the adjacency matrices."""

    def dist(self, G1, G2, k=5, alpha=1):
        r"""Compares the polynomials of the eigenvalue decomposition of
        two adjacency matrices.

        Note that the :math:`ij`-th element of where :math:`A^k`
        corresponds to the number of paths of length :math:`k` between
        nodes :math:`i` and :math:`j`.

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        k (float)
            maximum degree of the polynomial

        alpha (float)
            weighting factor

        Returns
        -------
        dist (float)
            Polynomial Dissimilarity between `G1`, `G2`

        References
        ----------
        .. [1] Donnat, Claire, and Susan Holmes. "Tracking network
               dynamics: A survey of distances and similarity metrics."
               arXiv preprint arXiv:1801.07351 (2018).

        """
        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)

        P_A1 = calculate_polynomial(A1, k, alpha)
        P_A2 = calculate_polynomial(A2, k, alpha)

        dist = np.linalg.norm(P_A1 - P_A2, ord='fro') / A1.shape[0] ** 2

        self.results['adjacency_matrices'] = A1, A2
        self.results['dist'] = dist
        return dist


def calculate_polynomial(A, k, alpha):
    eig_vals, Q = np.linalg.eig(A)

    n = A.shape[0]
    W = np.diag(
        sum(
            list(
                map(
                    lambda kp: eig_vals ** kp / (n - 1) ** (alpha * kp - 1),
                    range(1, k + 1),
                )
            )
        )
    )

    P_A = np.dot(np.dot(Q, W), Q.T)

    return P_A

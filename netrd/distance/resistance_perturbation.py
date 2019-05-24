"""
resistance_perturbation.py
--------------------------

Graph distance based on resistance perturbation (https://arxiv.org/abs/1605.01091v2)

author: Ryan J. Gallagher & Jessica T. Davis

Submitted as part of the 2019 NetSI Collabathon.

"""
import warnings
import numpy as np
import networkx as nx
import scipy.sparse as ss
from .base import BaseDistance
from ..utilities import ensure_undirected


class ResistancePerturbation(BaseDistance):
    """Compares the resistance matrices."""

    def dist(self, G1, G2, p=2):
        r"""The p-norm of the difference between two graph resistance matrices.

        The resistance perturbation distance changes if either graph is
        relabeled (it is not invariant under graph isomorphism), so node
        labels should be consistent between the two graphs being
        compared. The distance is not normalized.

        The resistance matrix of a graph :math:`G` is calculated as
        :math:`R = \text{diag}(L_i) 1^T + 1 \text{diag}(L_i)^T - 2L_i`,
        where :math:`L_i` is the Moore-Penrose pseudoinverse of the
        Laplacian of :math:`G`.

        The resistance perturbation distance between :math:`G_1` and
        :math:`G_2` is calculated as the :math:`p`-norm of the difference
        in their resitance matrices,

        .. math::
            d_{r(p)} = | R^{(1)} - R^{(2)} | = ( \sum_{i,j \in V} | R^{(1)}_{i,j} - R^{(2)}_{i,j} |^p )^{1/p},

        where :math:`R^{(1)}` and :math:`R^{(2)}` are the resistance
        matrices of :math:`G_1` and :math:`G_2` respectively. When :math:`p
        = \infty`, we have

        .. math::
            d_{r(\infty)} = \max_{i,j \in V} |R^{(1)}_{i,j} - R^{(2)}_{i,j}|.


        This method assumes that the input graphs are undirected; if
        directed graphs are used, it will coerce them to undirected graphs
        and emit a RuntimeWarning.

        The results dictionary also stores a 2-tuple of the underlying
        resistance matrices in the key `'resistance_matrices'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        p (float or str, optional)
            :math:`p`-norm to take of the difference between the resistance
            matrices. Specify ``np.inf`` to take :math:`\infty`-norm.

        Returns
        -------
        dist (float)
            the distance between G1 and G2.

        References
        ----------

        .. [1] https://arxiv.org/abs/1605.01091v2

        """
        # Coerce to undirected, if needed.
        G1 = ensure_undirected(G1)
        G2 = ensure_undirected(G2)

        # Check for connected graphs
        if not nx.is_connected(G1) or not nx.is_connected(G2):
            raise ValueError(
                "Resistance perturbation is undefined for disconnected graphs."
            )

        # Get resistance matrices
        R1 = get_resistance_matrix(G1)
        R2 = get_resistance_matrix(G2)
        self.results['resistance_matrices'] = R1, R2

        # Get resistance perturbation distance
        if not np.isinf(p):
            dist = np.power(np.sum(np.power(np.abs(R1 - R2), p)), 1 / p)
        else:
            dist = np.amax(np.abs(R1 - R2))
        self.results['dist'] = dist

        return dist


def get_resistance_matrix(G):
    """Get the resistance matrix of a networkx graph.

    The resistance matrix of a graph :math:`G` is calculated as
    :math:`R = \text{diag}(L_i) 1^T + 1 \text{diag}(L_i)^T - 2L_i`,
    where L_i is the Moore-Penrose pseudoinverse of the Laplacian of :math:`G`.

    Parameters
    ----------
    G (nx.Graph): networkx graph from which to get its resistance matrix

    Returns
    -------
    R (np.array): resistance matrix of G

    """
    # Get adjacency matrix
    n = len(G.nodes())
    A = nx.to_numpy_array(G)
    # Get Laplacian
    D = np.diag(A.sum(axis=0))
    L = D - A
    # Get Moore-Penrose pseudoinverses of Laplacian
    # Note: converts to dense matrix and introduces n^2 operation here
    I = np.eye(n)
    J = (1 / n) * np.ones((n, n))
    L_i = np.linalg.solve(L + J, I) - J
    # Get resistance matrix
    ones = np.ones(n)
    ones = ones.reshape((1, n))
    L_i_diag = np.diag(L_i)
    L_i_diag = L_i_diag.reshape((n, 1))
    R = np.dot(L_i_diag, ones) + np.dot(ones.T, L_i_diag.T) - 2 * L_i
    return R

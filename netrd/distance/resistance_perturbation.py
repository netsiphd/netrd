"""
resistance_perturbation.py
--------------
Graph distance based on resistance perturbation (https://arxiv.org/abs/1605.01091v2)

author: Ryan J. Gallagher & Jessica T. Davis
email:
Submitted as part of the 2019 NetSI Collabathon.

"""
import numpy as np
import networkx as nx
import scipy.sparse as ss
from .base import BaseDistance

class ResistancePerturbation(BaseDistance):
    def dist(self, G1, G2, p=2):
        """The resistance perturbation graph distance is the p-norm of the
        difference between two graph resistance matrices.

        The resistance perturbation distance changes if either graph is relabeled
        (it is not invariant under graph isomorphism), so node labels should be
        consistent between the two graphs being compared. The distance is not
        normalized.

        The resistance matrix of a graph $G$ is calculated as
        $R = \text{diag}(L_i) 1^T + 1 \text{diag}(L_i)^T - 2L_i$,
        where L_i is the Moore-Penrose pseudoinverse of the Laplacian of $G$.

        The resistance perturbation graph distance of $G_1$ and $G_2$ is
        calculated as the $p$-norm of the differenc in their resitance matrices,
        $d_{r(p)} = ||R^{(1)} - R_^{(2)}|| = [\sum_{i,j \in V} |R^{(1)}_{i,j} - R^{(2)}_{i,j}|^p]^{1/p}$,
        where R^{(1)} and R^{(2)} are the resistance matrices of $G_1$ and $G_2$
        respectively. When $p = \infty$,
        $d_{r(\infty)} = \max_{i,j \in V} |R^{(1)}_{i,j} - R^{(2)}_{i,j}|$.

        For details, see https://arxiv.org/abs/1605.01091v2

        Params
        ------
        G1, G2 (nx.Graph): two networkx graphs to be compared.
        p (float or str, optional): $p$-norm to take of the difference between
            the resistance matrices. Specify 'infinity' to take $\infty$-norm.

        Returns
        -------
        dist (float): the distance between G1 and G2.

        """
        # Get resistance matrices
        R1 = get_resistance_matrix(G1)
        R2 = get_resistance_matrix(G2)
        self.results['resist1'] = R1
        self.results['resist2'] = R2

        # Get resistance perturbation distance
        if p != 'infinity':
            dist = np.power(np.sum(np.power(np.abs(R1 - R2), p)), 1/p)
        else:
            dist = np.amax(np.abs(R1 - R2))
        self.results['dist'] = dist

        return dist

def get_resistance_matrix(G):
    """Get the resistance matrix of a networkx graph.

    The resistance matrix of a graph $G$ is calculated as
    $R = \text{diag}(L_i) 1^T + 1 \text{diag}(L_i)^T - 2L_i$,
    where L_i is the Moore-Penrose pseudoinverse of the Laplacian of $G$.

    Params
    ------
    G (nx.Graph): networkx graph from which to get its resistance matrix

    Returns
    -------
    R (np.array): resistance matrix of G

    """
    # Get adjacency matrix
    n = len(G.nodes())
    A = nx.adjacency_matrix(G)
    # Get Laplacian
    D = ss.diags(np.squeeze(np.asarray(A.sum(axis=0))))
    L = D - A
    # Get Moore-Penrose pseudoinverses of Laplacian
    # Note: converts to dense matrix and introduces n^2 operation here
    I = np.eye(n)
    J = (1/n)*np.ones((n,n))
    L_i = np.linalg.solve(L+J, I) - J
    # Get resistance matrix
    ones = np.ones(n)
    ones = ones.reshape((1,n))
    L_i_diag = np.diag(L_i)
    L_i_diag = L_i_diag.reshape((n,1))
    R = np.dot(L_i_diag, ones) + np.dot(ones.T, L_i_diag.T) - 2*L_i
    return R

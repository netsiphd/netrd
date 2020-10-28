"""
distributional_nbd.py
------

Distributional Non-backtracking Spectral Distance.

"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import euclidean, chebyshev
from ..utilities.graph import unweighted

from .base import BaseDistance


class DistributionalNBD(BaseDistance):
    """
    Distributional Non-backtracking Spectral Distance.

    Computes the distance between two graphs using the empirical spectral density
    of the non-backtracking operator.

    See:
    "Graph Comparison via the Non-backtracking Spectrum"
    A. Mellor & A. Grusovin
    arXiv:1812.05457 / 10.1103/PhysRevE.99.052309

    """

    VECTOR_DISTANCES = {'euclidean': euclidean, 'chebyshev': chebyshev}

    @unweighted
    def dist(
        self,
        G1,
        G2,
        sparse=False,
        shave=True,
        keep_evals=True,
        k=None,
        vector_distance='euclidean',
        **kwargs
    ):
        """
        Distributional Non-backtracking Spectral Distance.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            The two graphs to compare.

        sparse (bool)
            If sparse, matrices and eigenvalues found using sparse methods.
            If sparse, parameter 'k' should also be specified.
            Default: False

        k (int)
            The number of largest eigenvalues to be calculated for the
            spectral density.

        vector_distance (str)
            The distance measure used to compare two empirical distributions.
            Currently available are 'euclidean' and 'chebyshev', implemented
            using SciPy.
            Default: 'euclidean'

        keep_evals (bool)
            If True, stores the eigenvalues of the reduced non-backtracking
            matrix in self.results['evals']
            Default: False


        Returns
        -------
        float
            The distance between `G1` and `G2`

        """
        B1 = reduced_hashimoto(G1, shave=shave, sparse=sparse, **kwargs)
        B2 = reduced_hashimoto(G2, shave=shave, sparse=sparse, **kwargs)

        # Find spectrum
        evals1 = nb_eigenvalues(B1, k=k)
        evals2 = nb_eigenvalues(B2, k=k)

        # Save spectrum
        if keep_evals:
            self.results['eigenvalues'] = (evals1, evals2)

        # Find rescaled spectral density
        distribution_1 = spectral_distribution(evals1)
        distribution_2 = spectral_distribution(evals2)

        # Compute distance
        distance_metric = self.__class__.VECTOR_DISTANCES[vector_distance]

        return distance_metric(distribution_1, distribution_2)


def shave_graph(graph):
    """
    Returns the two-core of a graph.

    Iteratively remove the nodes of degree 0 or 1, until all nodes have
    degree at least 2.

    NOTE: duplicated from "nbd.py" to avoid excessive imports.

    """
    core = graph.copy()
    while True:
        to_remove = [node for node, neighbors in core.adj.items() if len(neighbors) < 2]
        core.remove_nodes_from(to_remove)
        if len(to_remove) == 0:
            break
    return core


def pseudo_hashimoto(graph):
    """
    Return the pseudo-Hashimoto matrix.

    The pseudo Hashimoto matrix of a graph is the block matrix defined as
    B' = [0  D-I]
         [-I  A ]

    Where D is the degree-diagonal matrix, I is the identity matrix and A
    is the adjacency matrix.  The eigenvalues of B' are always eigenvalues
    of B, the non-backtracking or Hashimoto matrix.

    Parameters
    ----------

    graph (nx.Graph): A NetworkX graph object.

    Returns
    -------

    A sparse matrix in csr format.

    NOTE: duplicated from "nbd.py" to avoid excessive imports.

    """
    # Note: the rows of nx.adjacency_matrix(graph) are in the same order as
    # the list returned by graph.nodes().
    degrees = graph.degree()
    degrees = sp.diags([degrees[n] for n in graph.nodes()])
    adj = nx.adjacency_matrix(graph)
    ident = sp.eye(graph.order())
    pseudo = sp.bmat([[None, degrees - ident], [-ident, adj]])
    return pseudo.asformat('csr')


def reduced_hashimoto(graph, shave=True, sparse=True):
    """


    Parameters
    ----------

    shave (bool)
        If True, first reduce the graph to its two-core.
        Else graph processed in its entirety.

    sparse (bool)
        If True, returned matrix will be sparse,
        else it will be dense.

    Returns
    -------

    np.ndarray/sp.csr_matrix
        The reduced Hashimoto Matrix.

    """

    if shave:
        graph = shave_graph(graph)
        if len(graph) == 0:
            # We can provide a workaround for this case, however it is best
            # that it is brought to the attention of the user.
            raise NotImplementedError(
                "Graph two-core is empty: non-backtracking methods unsuitable."
            )

    B = pseudo_hashimoto(graph)

    if not sparse:
        B = B.todense()

    return B


def nb_eigenvalues(B, k=None, **kwargs):
    """
    Calculates the eigenvalues of a matrix B.

    Detects whether B is sparse/dense and uses the appropriate method.
    If B is sparse then parameter 'k' should be provided.
    """

    if isinstance(B, np.ndarray):
        return np.linalg.eigvals(B)

    elif isinstance(B, sp.csr_matrix):
        random_state = np.random.RandomState(
            1
        )  # Ensures that eigenvalue calculation is deterministic.
        return sp.linalg.eigs(
            B, k=k, v0=random_state.random(B.shape[0]), return_eigenvectors=False
        )
    else:
        raise Exception("Matrix must be of type np.ndarray or scipy.sparse.csr")


def logr(r, rmax):
    """
    Logarithm to the base r.

    NOTE:Maps zero to zero as a special case.
    """

    if r == 0:
        return 0
    return np.log(r) / np.log(rmax)


def spectral_distribution(points, cumulative=True):
    """
    Returns the distribution of complex values (in r,theta-space).
    """

    points = np.array([(np.abs(z), np.angle(z)) for z in points])
    r, theta = np.split(points, 2, axis=1)

    r = np.array([logr(x, r.max()) for x in r])

    Z, R, THETA = np.histogram2d(
        x=r[:, 0],
        y=theta[:, 0],
        bins=(np.linspace(0, 1, 101), np.linspace(0, np.pi, 101)),
    )

    if cumulative:
        Z = Z.cumsum(axis=0).cumsum(axis=1)
        Z = Z / Z.max()

    return Z.flatten()

"""
distributional_nbd.py
------

Non-backtracking spectral distance between two graphs.

"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import euclidean, chebyshev

from .base import BaseDistance


class DistributionalNBD(BaseDistance):
    """


    """
    VECTOR_DISTANCES = {'euclidean': euclidean,
                        'chebyshev': chebyshev}

    def dist(self, G1, G2, sparse=False, shave=True, keep_evals=False, k=None, vector_distance='euclidean', **kwargs):
        """Non-Backtracking Distance between two graphs.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            The two graphs to compare.

        topk (int or 'automatic')
            The number of eigenvalues to compute. If `'automatic'` (default),
            use only the eigenvalues that are larger than the square root
            of the largest eigenvalue.  Note this may yield different
            number of eigenvalues for each graph.

        batch (int)
            If topk is `'automatic'`, this is the number of eigenvalues to
            compute each time until the condition is met. Default
            :math:`100`.

        tol (float)
            Numerical tolerance when computing eigenvalues.

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
            self.results['evals'] = (evals1, evals2)

        # Find rescaled spectral density
        distribution_1 = spectral_distribution(evals1)
        distribution_2 = spectral_distribution(evals2)

        # Compute distance
        distance_metric = self.__class__.VECTOR_DISTANCES[vector_distance]

        return distance_metric(distribution_1, distribution_2)

def shave_graph(graph):
    """Return the 2-core of a graph.

    Iteratively remove the nodes of degree 0 or 1, until all nodes have
    degree at least 2.

    Note: duplicated from "nbd.py" to avoid excessive imports.

    """
    core = graph.copy()
    while True:
        to_remove = [node for node, neighbors in core.adj.items() if len(neighbors) < 2]
        core.remove_nodes_from(to_remove)
        if len(to_remove) == 0:
            break
    return core

def pseudo_hashimoto(graph):
    """Return the pseudo-Hashimoto matrix.

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

    Note: duplicated from "nbd.py" to avoid excessive imports.

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
    """

    if shave:
        graph = shave_graph(graph)
        if len(graph) == 0:
            # We can provide a workaround for this case, however it is best
            # that it is brought to the attention of the user.
            raise Exception("Graph two-core is empty: non-backtracking methods unsuitable.")

    B = pseudo_hashimoto(graph)

    if not sparse:
        B = B.todense()

    return B

def nb_eigenvalues(B, k=None, **kwargs):
    """
    """

    if isinstance(B, np.ndarray):
        return np.linalg.eigvals(B)

    elif isinstance(B, sp.csr_matrix):
        np.random.seed(1) # Ensures that eigenvalue calculation is deterministic.
        return sp.linalg.eigs(B, 
                              k=k,
                              v0=np.random.random(B.shape[0]),
                              return_eigenvectors=False)
    else:
        raise Exception("Matrix must be of type np.ndarray or scipy.sparse.csr")

def logr(r,rmax):
    """Logarithm to the base r. Maps zero to zero."""
    
    if r == 0:
        return 0
    return np.log(r)/np.log(rmax)
        
def spectral_distribution(points, cumulative=True):
    """ Returns the distribution of complex values (in r,theta-space) """
    
    points = np.array([(np.abs(z), np.angle(z)) for z in points])
    r, theta = np.split(points, 2, axis=1)
    
    r = np.array([logr(x, r.max()) for x in r])
       
    Z,R,THETA = np.histogram2d(x=r[:,0], 
                               y=theta[:,0], 
                               bins=(np.linspace(0,1,101),
                                     np.linspace(0,np.pi,101))
                              )
    
    if cumulative:
        Z = Z.cumsum(axis=0).cumsum(axis=1)
        Z = Z/Z.max()
        
    return Z.flatten()
"""
nbd.py
------

Non-backtracking spectral distance between two graphs.

"""

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import scipy.sparse as sparse
from ot import emd2
from .base import BaseDistance


class NonBacktrackingSpectral(BaseDistance):
    """Compares the empirical spectral distribution of the non-backtracking
matrices.

    The eigenvalues are stored in the results dictionary.

    """

    def dist(self, G1, G2, topk='automatic', batch=100, tol=1e-5):
        """Non-Backtracking Distance between two graphs.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            The graphs to compare.

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
        vals1 = nbvals(G1, topk, batch, tol)
        vals2 = nbvals(G2, topk, batch, tol)
        mass = lambda num: np.ones(num) / num
        vals_dist = distance_matrix(vals1, vals2)
        dist = emd2(mass(vals1.shape[0]), mass(vals2.shape[0]), vals_dist)
        self.results['vals'] = (vals1, vals2)
        return dist


def nbvals(graph, topk='automatic', batch=100, tol=1e-5):
    """Compute the largest-magnitude non-backtracking eigenvalues.

    Parameters
    ----------

    graph (nx.Graph): The graph.

    topk (int or 'automatic'): The number of eigenvalues to compute.  The
    maximum number of eigenvalues that can be computed is 2*n - 4, where n
    is the number of nodes in graph.  All the other eigenvalues are equal
    to +-1. If 'automatic', return all eigenvalues whose magnitude is
    larger than the square root of the largest eigenvalue.

    batch (int): If topk is 'automatic', compute this many eigenvalues at a
    time until the condition is met.  Must be at most 2*n - 4; default 100.

    tol (float): Numerical tolerance.  Default 1e-5.

    Returns
    -------

    An array with the eigenvalues.

    """
    if not isinstance(topk, str) and topk < 1:
        return np.array([[], []])

    # The eigenvalues are left untouched by removing the nodes of degree 1.
    # Moreover, removing them makes the computations faster.  This
    # 'shaving' leaves us with the 2-core of the graph.
    core = shave(graph)
    matrix = pseudo_hashimoto(core)
    if not isinstance(topk, str) and topk > matrix.shape[0] - 1:
        topk = matrix.shape[0] - 2
        print('Computing only {} eigenvalues'.format(topk))

    if topk == 'automatic':
        batch = min(batch, 2 * graph.order() - 4)
        if 2 * graph.order() - 4 < batch:
            print('Using batch size {}'.format(batch))
        topk = batch
    eigs = lambda k: sparse.linalg.eigs(matrix, k=k, return_eigenvectors=False, tol=tol)
    count = 1
    while True:
        vals = eigs(topk * count)
        largest = np.sqrt(abs(max(vals, key=abs)))
        if abs(vals[0]) <= largest or topk != 'automatic':
            break
        count += 1
    if topk == 'automatic':
        vals = vals[abs(vals) > largest]

    # The eigenvalues are returned in no particular order, which may yield
    # different feature vectors for the same graph.  For example, if a
    # graph has a + ib and a - ib as eigenvalues, the eigenvalue solver may
    # return [..., a + ib, a - ib, ...] in one call and [..., a - ib, a +
    # ib, ...] in another call.  To avoid this, we sort the eigenvalues
    # first by absolute value, then by real part, then by imaginary part.
    vals = sorted(vals, key=lambda x: x.imag)
    vals = sorted(vals, key=lambda x: x.real)
    vals = np.array(sorted(vals, key=np.linalg.norm))

    # Return eigenvalues as a 2D array, with one row per eigenvalue, and
    # each row containing the real and imaginary parts separately.
    vals = np.array([(z.real, z.imag) for z in vals])
    return vals


def shave(graph):
    """Return the 2-core of a graph.

    Iteratively remove the nodes of degree 0 or 1, until all nodes have
    degree at least 2.

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

    """
    # Note: the rows of nx.adjacency_matrix(graph) are in the same order as
    # the list returned by graph.nodes().
    degrees = graph.degree()
    degrees = sparse.diags([degrees[n] for n in graph.nodes()])
    adj = nx.adjacency_matrix(graph)
    ident = sparse.eye(graph.order())
    pseudo = sparse.bmat([[None, degrees - ident], [-ident, adj]])
    return pseudo.asformat('csr')


def half_incidence(graph, ordering='blocks', return_ordering=False):
    """Return the 'half-incidence' matrices of the graph.

    If the graph has n nodes and m *undirected* edges, then the
    half-incidence matrices are two matrices, P and Q, with n rows and 2m
    columns.  That is, there is one row for each node, and one column for
    each *directed* edge.  For P, the entry at (n, e) is equal to 1 if node
    n is the source (or tail) of edge e, and 0 otherwise.  For Q, the entry
    at (n, e) is equal to 1 if node n is the target (or head) of edge e,
    and 0 otherwise.

    Parameters
    ----------

    graph (nx.Graph): The graph.

    ordering (str): If 'blocks' (default), the two columns corresponding to
    the i'th edge are placed at i and i+m.  That is, choose an arbitarry
    direction for each edge in the graph.  The first m columns correspond
    to this orientation, while the latter m columns correspond to the
    reversed orientation.  Columns are sorted following graph.edges().  If
    'consecutive', the first two columns correspond to the two orientations
    of the first edge, the third and fourth row are the two orientations of
    the second edge, and so on.  In general, the two columns for the i'th
    edge are placed at 2i and 2i+1.

    return_ordering (bool): if True, return a function that maps an edge id
    to the column placement.  That is, if ordering=='blocks', return the
    function lambda x: (x, m+x), if ordering=='consecutive', return the
    function lambda x: (2*x, 2*x + 1).  If False, return None.


    Returns
    -------

    P (sparse matrix), Q (sparse matrix), ordering (function or None).


    Notes
    -----

    The nodes in graph must be labeled by consecutive integers starting at
    0.  This function always returns three values, regardless of the value
    of return_ordering.

    """
    numnodes = graph.order()
    numedges = graph.size()

    if ordering == 'blocks':
        src_pairs = lambda i, u, v: [(u, i), (v, numedges + i)]
        tgt_pairs = lambda i, u, v: [(v, i), (u, numedges + i)]
    if ordering == 'consecutive':
        src_pairs = lambda i, u, v: [(u, 2 * i), (v, 2 * i + 1)]
        tgt_pairs = lambda i, u, v: [(v, 2 * i), (u, 2 * i + 1)]

    def make_coo(make_pairs):
        """Make a sparse 0-1 matrix.

        The returned matrix has a positive entry at each coordinate pair
        returned by make_pairs, for all (idx, node1, node2) edge triples.

        """
        coords = list(
            zip(
                *(
                    pair
                    for idx, (node1, node2) in enumerate(graph.edges())
                    for pair in make_pairs(idx, node1, node2)
                )
            )
        )
        data = np.ones(2 * graph.size())
        return sparse.coo_matrix((data, coords), shape=(numnodes, 2 * numedges))

    src = make_coo(src_pairs).asformat('csr')
    tgt = make_coo(tgt_pairs).asformat('csr')

    if return_ordering:
        if ordering == 'blocks':
            func = lambda x: (x, numedges + x)
        else:
            func = lambda x: (2 * x, 2 * x + 1)
        return src, tgt, func
    else:
        return src, tgt

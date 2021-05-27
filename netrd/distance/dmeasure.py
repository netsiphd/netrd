"""
d_measure.py
--------------------------

Distance measure based on the Jensen-Shannon Divergence
between the network node dispersion distributions of two graphs.

Schieber, T. A. et al.
Quantification of network structural dissimilarities.
Nat. Commun. 8, 13928 (2017).

https://www.nature.com/articles/ncomms13928

author: Brennan Klein
email: brennanjamesklein@gmail.com
Submitted as part of the 2019 NetSI Collabathon.

"""

from collections import Counter
import networkx as nx
import numpy as np
from scipy.stats import entropy
from .base import BaseDistance
from ..utilities.entropy import js_divergence
from ..utilities import undirected


class DMeasure(BaseDistance):
    """Compare two graphs by their network node dispersion."""

    @undirected
    def dist(self, G1, G2, w1=0.45, w2=0.45, w3=0.10, niter=50):
        r"""The D-Measure is a comparison of structural dissimilarities between graphs.

        The key concept is the network node dispersion

        .. math::
            NND(G) = \frac{\mathcal{J}(\mathbf{P}_1,\ldots,\mathbf{P}_N)}{\log(d+1)},

        where :math:`\mathcal{J}` is the Jenson-Shannon divergence between
        :math:`N` node-distance distributions

        .. math::
            \mathbf{P}_i = \{p_i(j)\},

        and :math:`p_i(j)` is the fraction of nodes at distance :math:`i` from
        node :math:`j`.

        The D-measure itself is a weighted sum of three components: the square
        root of the Jensen-Shannon divergence between the average node-distance
        probabilities of the two graphs

        .. math::
           \mu_j = \frac{1}{N}\sum_{i=1}^N p_i(j),

        the second term is the absolute value of the differences in the square
        roots of the network node dispersions of the two graphs, and the third
        term is the sum of the square roots of the Jensen-Shannon divergences
        between the probability distributions of the alpha centralities of two
        graph and of their complements.


        Parameters
        ----------

        G1 (nx.Graph):
            the first graph to be compared.
        G2 (nx.Graph):
            the second graph to be compared.
        w1 (float):
            weight of the first term in the calculation;
            with w2 and w3, must sum to 1.0.
        w2 (float):
            weight of the second term in the calculation;
            with w1 and w3, must sum to 1.0.
        w3 (float):
            weight of the third term in the calculation;
            with w1 d w2, must sum to 1.0.
        niter (int):
            the alpha centralities are calculated using power iteration, with
            this many iterations

        Returns
        -------

        dist (float):
            between 0 and 1, the D-measure distance between G1 and G2


        Notes
        -----
        The default values for w1, w2, and w3 are from the original paper.


        References
        ----------

        .. [1] Schieber, T. A. et al. Quantification of network structural
               dissimilarities. Nat. Commun. 8, 13928 (2017).
               https://www.nature.com/articles/ncomms13928

        """

        if sum([w1, w2, w3]) != 1:
            raise ValueError("Weights must sum to one.")

        first_term = 0
        second_term = 0
        third_term = 0

        if w1 + w2 > 0:
            g1_nnd, g1_pdfs = network_node_dispersion(G1)
            g2_nnd, g2_pdfs = network_node_dispersion(G2)

            first_term = np.sqrt(js_divergence(g1_pdfs, g2_pdfs))
            second_term = np.abs(np.sqrt(g1_nnd) - np.sqrt(g2_nnd))

        if w3 > 0:

            def alpha_jsd(G1, G2):
                """
                Compute the Jensen-Shannon divergence between the
                alpha-centrality probability distributions of two graphs.
                """
                p1 = alpha_centrality_prob(G1, niter=niter)
                p2 = alpha_centrality_prob(G2, niter=niter)

                m = max([len(p1), len(p2)])

                P1 = np.zeros(m)
                P2 = np.zeros(m)

                P1[(m - len(p1)) : m] = p1
                P2[(m - len(p2)) : m] = p2

                return js_divergence(P1, P2)

            G1c = nx.complement(G1)
            G2c = nx.complement(G2)

            first_jsd = alpha_jsd(G1, G2)
            second_jsd = alpha_jsd(G1c, G2c)
            third_term = 0.5 * (np.sqrt(first_jsd) + np.sqrt(second_jsd))

        dist = w1 * first_term + w2 * second_term + w3 * third_term

        self.results["components"] = (first_term, second_term, third_term)
        self.results["weights"] = (w1, w2, w3)
        self.results["dist"] = dist

        return dist


def shortest_path_matrix(G):
    """
    Return a matrix of pairwise shortest path lengths between nodes.

    Parameters
    ----------
    G (nx.Graph): the graph in question

    Returns
    -------
    pmat (np.ndarray): a matrix of shortest paths between nodes in G

    """

    N = G.number_of_nodes()
    pmat = np.zeros((N, N)) + N

    paths = nx.all_pairs_shortest_path_length(G)
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            pmat[node_i, node_j] = length_ij

    pmat[pmat == np.inf] = N

    return pmat


def node_distance(G):
    """
    Return an NxN matrix that consists of histograms of shortest path
    lengths between nodes i and j. This is useful for eventually taking
    information theoretic distances between the nodes.

    Parameters
    ----------
    G (nx.Graph): the graph in question.

    Returns
    -------
    out (np.ndarray): a matrix of binned node distance values.

    """

    N = G.number_of_nodes()
    a = np.zeros((N, N))

    dists = nx.shortest_path_length(G)
    for idx, row in enumerate(dists):
        counts = Counter(row[1].values())
        a[idx] = [counts[l] for l in range(1, N + 1)]

    return a / (N - 1)


def network_node_dispersion(G):
    """
    This function calculates the network node dispersion of a graph G. This
    function also returns the average of the each node-distance distribution.

    Parameters
    ----------
    G (nx.Graph): the graph in question.

    Returns
    -------
    nnd (float): the nearest node dispersion
    nd_vec (np.ndarray): a vector of averages of the
                         node-distance distributions

    """

    N = G.number_of_nodes()
    nd = node_distance(G)
    pdfm = np.mean(nd, axis=0)

    # NOTE: the paper says that the normalization is the diameter plus one,
    # but the previous implementation uses the number of nonzero entries in the
    # node-distance matrix. This number should typically be the diameter plus
    # one anyway.
    norm = np.log(nx.diameter(G) + 1)

    ndf = nd.flatten()
    # calculate the entropy, with the convention that 0/0 = 0
    entr = -1 * sum(ndf * np.log(ndf, out=np.zeros_like(ndf), where=(ndf != 0)))

    nnd = max([0, entropy(pdfm) - entr / N]) / norm

    return nnd, pdfm


def alpha_centrality_prob(G, niter):
    """
    Returns a probability distribution over alpha centralities for the network.

    Parameters
    ----------
    G (nx.Graph): the graph in question.
    niter (int): the number of iterations needed to converge properly.

    Returns:
    alpha_prob (np.ndarray): a vector of probabilities for each node in G.
    """

    # calculate the alpha centrality for each node
    N = G.number_of_nodes()
    alpha = 1 / N

    A = nx.to_numpy_array(G)

    s = A.sum(axis=1)
    cr = s.copy()

    for _ in range(niter):
        cr = s + alpha * A.dot(cr)

    # turn the alpha centralities into a probability distribution
    cr = cr / (N - 1)
    r = sorted(cr / (N ** 2))
    alpha_prob = list(r) + [max([0, 1 - sum(r)])]

    return np.array(alpha_prob)

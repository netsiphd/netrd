"""
mutual_information_matrix.py
--------------

Graph reconstruction algorithm based on
Donges, Zou, Marwan, & Kurths. "The backbone of the climate network."
EPL (Europhysics Letters) 87.4 (2009): 48007
http://iopscience.iop.org/article/10.1209/0295-5075/87/48007.

author: Harrison Hartle
email: hartle.h@husky.neu.edu
Submitted as part of the 2019 NetSI Collabathon.
"""

from .base import BaseReconstructor
import numpy as np
import networkx as nx
from ..utilities import create_graph, threshold


class MutualInformationMatrix(BaseReconstructor):
    """Uses the mutual information between nodes."""

    def fit(self, TS, nbins=10, threshold_type='degree', **kwargs):
        """Calculates the mutual information between the probability distributions
        of the (binned) values of the time series of pairs of nodes.

        First, the mutual information is computed between each pair of
        vertices.  Then, a thresholding condition is applied to obtain
        edges.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N`
            sensors.

        nbins (int)
            number of bins for the pre-processing step (to yield a discrete
            probability distribution)

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        -------

        G (nx.Graph)
            A reconstructed graph with :math:`N` nodes.

        """
        N = TS.shape[0]
        rang = [np.min(TS), np.max(TS)]

        # saving these lines because there is a chance the "binned_edges" data would be useful
        #         _,bin_edges = np.histogram(TS[0], range=np.array(rang), bins=nbins)
        #         self.results['bin_edges'] = bin_edges

        # mutual information requires a joint probability and a "product probability" distribution
        IndivP = find_individual_probability_distribution(TS, rang, nbins)
        ProduP = find_product_probability_distribution(IndivP, N)
        JointP = find_joint_probability_distribution(TS, rang, nbins)

        # calculate the mutual information between each pair of nodes--this is the
        # mutual information matrix
        I = mutual_info_all_pairs(JointP, ProduP, N)
        self.results['weights_matrix'] = I

        # the adjacency matrix is the binarized thresholded mutual information matrix
        # tau=threshold_from_degree(deg,I)
        # A = np.array(I>tau, dtype=int)
        A = threshold(I, threshold_type, **kwargs)
        self.results['thresholded_matrix'] = A

        G = create_graph(A)
        self.results['graph'] = G

        return G


def find_individual_probability_distribution(TS, rang, nbins):
    """
    Assign each node to a vector of length nbins where each element is the probability of the
    node in the time series being in that binned "state"

    Parameters
    ----------
    TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.
    rang (list): list of the minimum and maximum value in the time series
    nbins (int): number of bins for the pre-processing step (to yield a discrete probability distribution)

    Returns
    -------
    IndivP (dict): a dictionary where the keys are nodes in the graph and the values are vectors
                   of empirical probabilities of that node's time series being in that binned state.
    """

    N, L = TS.shape  # N nodes and L length
    IndivP = (
        dict()
    )  # create a dict to put the individual binned probability vectors into

    for j in range(N):
        P, _ = np.histogram(
            TS[j], bins=nbins, range=rang
        )  # bin node j's time series data into nbins
        IndivP[j] = (
            P / L
        )  # normalize that by the length of time series data to make it a vector of probs

    return IndivP


def find_product_probability_distribution(IndivP, N):
    """
    Assign each node j to a vector of length nbins where each element is the product of its own
    individual_probability_distribution and its neighbors'. P(x) * P(y) <-- as opposed to P(x,y)

    Parameters
    ----------
    IndivP (dict): dictionary that gets output by find_individual_probability_distribution()
    N (int): number of nodes in the graph

    Returns
    -------
    ProduP (dict): a dictionary where the keys are nodes in the graph and the values
                         are nbins x nbins arrays corresponding to products of two probability vectors
    """

    ProduP = dict()  # create a dict to put the product prob distributions into

    for l in range(N):  # for each node
        for j in range(l):  # for each possible edge (this method is symmetric)
            ProduP[(j, l)] = np.outer(
                IndivP[j], IndivP[l]
            )  # outer product between two vectors

    return ProduP


def find_joint_probability_distribution(TS, rang, nbins):
    """
    Assign each node j to a vector of length nbins where each element is the product of its own
    individual_probability_distribution and its neighbors'. P(x) * P(y) <-- as opposed to P(x,y)

    Parameters
    ----------
    TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.
    rang (list): list of the minimum and maximum value in the time series
    nbins (int): number of bins for the pre-processing step (to yield a discrete probability distribution)

    Returns
    -------
    JointP (dict): a dictionary where the keys are nodes in the graph and the
                         are nbins x nbins arrays corresponding to joint probability vectors
    """

    N, L = TS.shape  # N nodes and L length
    JointP = dict()  # create a dict to put the joint prob distributions into

    for l in range(N):  # for each node
        for j in range(l):  # for each possible edge
            P, _, _ = np.histogram2d(
                TS[j], TS[l], bins=nbins, range=np.array([rang, rang])
            )
            JointP[(j, l)] = P / L

    return JointP


def mutual_info_node_pair(JointP_jl, ProduP_jl):
    """
    Calculate the mutual information between two nodes.

    Parameters
    ----------
    JointP_jl (np.ndarray): nbins x nbins array of two nodes' joint probability distributions
    ProduP_jl (np.ndarray): nbins x nbins array of two nodes' product probability distributions

    Returns
    -------
    I_jl (float): the mutual information between j and l, or
                  the (j,l)'th entry of the mutual information matrix
                  Note: np.log returns an information value in nats
    """

    I_jl = 0  # initialize the mutual information to zero

    for q, p in zip(JointP_jl.flatten(), ProduP_jl.flatten()):
        if q > 0 and p > 0:  # avoid log(0) or dividing by zero
            I_jl += q * np.log(q / p)  # an instance of the KL divergence

    return I_jl


def mutual_info_all_pairs(JointP, ProduP, N):
    """
    Calculate the mutual information between all pairs of nodes.

    Parameters
    ----------
    JointP (dict): a dictionary where the keys are pairs of nodes in the graph and the
                   are nbins x nbins arrays corresponding to joint probability vectors
    ProduP (dict): a dictionary where the keys are pairs of nodes in the graph and the values
                   are nbins x nbins arrays corresponding to products of two probability vectors
    N (int): list of the minimum and maximum value in the time series

    Returns
    -------
    I (np.ndarray): the NxN mutual information matrix from a time series
    """

    I = np.zeros((N, N))  # initialize an empty matrix

    for l in range(N):
        for j in range(l):

            JointP_jl = JointP[(j, l)]
            ProduP_jl = ProduP[(j, l)]

            I[j, l] = mutual_info_node_pair(JointP_jl, ProduP_jl)  # fill in the matrix
            I[l, j] = I[j, l]  # this method is symmetric

    return I


def threshold_from_degree(deg, M):
    """
    Compute the required threshold (tau) in order to yield a reconstructed graph of mean degree deg.
    Parameters
    ----------
    deg (int): Target degree for which the appropriate threshold will be computed
    M (np.ndarray): Pre-thresholded NxN array
    Returns
    ------
    tau (float): Required threshold for A=np.array(I<tau,dtype=int) to have an average of deg ones per row/column

    """
    N = len(M)
    A = np.ones((N, N))  # start with a complete graph (lowest possible threshold)
    for tau in sorted(M.flatten()):  # consider increasingly large entry-values
        A[M == tau] = 0  # remove edges of weight less than the current entry
        if (
            np.mean(np.sum(A, 1)) < deg
        ):  # stop once the matrix is trimmed down to mean degree deg
            break
    return tau  # return this critical threshold value

"""
branching_process.py
--------------------

Adapted from:
Levina, Anna, and Viola Priesemann. "Subsampling scaling."
Nature communications 8 (2017): 15140.
at [this link](https://www.nature.com/articles/ncomms15140)

author: Brennan Klein
email: brennanjamesklein at gmail dot com
submitted as part of the 2019 NetSI Collabathon
"""

from .base import BaseDynamics
import networkx as nx
import numpy as np


class BranchingModel(BaseDynamics):
    """A sand-pile-like brancing process."""

    def __init__(self):
        self.results = {}

    def simulate(
        self,
        G,
        L,
        initial_fraction=0.1,
        m=0.9975,
        target_Ahat=0.2,
        distribution_type='unif',
        scale=0.95,
        noise=True,
    ):
        r"""Simulate a (sand-pile-like) branching processs dynamics .

        The results dictionary also stores the ground truth network as
        `'ground_truth'`.

        Parameters
        ----------

        G (nx.Graph)
            directed or undirected ground truth graph

        L (int)
            desired length of time series

        initial_fraction (float)
            fraction of nodes that start as active

        m (float)
            branching ratio of the dynamical process. :math:`m=1.0` means
            the system will be at criticality

        target_Ahat (float)
            desired average activity. This will ensure the process does not
            reach a stationary state and will always have some external
            drive.

        num_edges (int)
            the length of the cache, which should correspond to the
            combination of all possible activity over the simulation.

        distribution_type (str)
            string describing which type of random numbers

        scale (float)
            scale for how likely nodes are to topple

        noise (bool)
            add nonzero values to the time series

        Returns
        -------
        TS (np.ndarray)
            an :math:`N \times L` time series

        References
        ----------

        .. [1] Levina, Anna, and Viola Priesemann. "Subsampling scaling."
           Nature communications 8 (2017) 15140.
           https://www.nature.com/articles/ncomms15140

        """
        N = G.number_of_nodes()  # number of nodes
        M = G.number_of_edges()  # number of edges
        A = nx.to_numpy_array(G)  # adjacency matrix
        W = np.zeros(A.shape)  # transition probability matrix (for weights)
        for i in range(A.shape[0]):
            if A[i].sum() > 0:
                W[i] = A[i] / A[i].sum()
        Gw = nx.from_numpy_array(W)
        G = nx.to_directed(Gw)  # convert back into a graph object

        TS = initialize_history(N, L, initial_fraction, m, target_Ahat, noise)

        # because there's noise added, dont want to get false positives
        new_activity_times = np.nonzero(np.round(TS[:, 1:].sum(axis=0), 3))[0]

        # store
        cache = initialize_threshold_cache(M * L, distribution_type, scale)

        # now run dynamics

        for t in range(L - 1):
            if t not in list(new_activity_times):
                current_state = TS[:, t]

                # because there's noise added, dont want to get false positives
                active_nodes = list(np.nonzero(np.round(current_state, 3))[0])
                active_edges = G.out_edges(nbunch=active_nodes, data=True)

                if len(active_edges) != 0:
                    current_sources = list(list(zip(*active_edges))[0])
                    current_targets = list(list(zip(*active_edges))[1])
                    weights_array = np.array([j[2]['weight'] for j in active_edges])

                    if len(cache) <= len(weights_array):
                        cache = initialize_threshold_cache(
                            M * L, distribution_type, scale
                        )

                    # find edges with edges that will exceed the weights cache
                    # and thus will successfully propagate the information
                    over_the_threshold = weights_array > cache[: len(weights_array)]
                    cache = cache[(len(weights_array) + 1) :]  # update the cache

                    next_active_units = np.unique(
                        np.array(current_targets)[over_the_threshold]
                    )
                    TS[next_active_units, t + 1] = 1

        # save the ground-truth network to results
        self.results['ground_truth'] = G
        # save the timeseries data to results
        self.results['TS'] = TS

        return TS


def initialize_history(N, L, initial_fraction, m, target_Ahat, noise):
    """
    Initializes the TS of this simulation based on a configuration of
    parameters corresponding to the initial_fraction of active nodes,
    the branching ratio, m, and the target number of avalanches.

    Parameters
    ----------
    N (int): number of nodes
    L (int): desired length of time series
    initial_fraction (float): fraction of nodes that start as active
    m (float): branching ratio of the branching process
    target_Ahat (float): desired average activity. This will ensure the
                         process does not reach a stationary state and
                         will always have some external drive.
    noise (bool): add nonzero values to the time series


    Returns
    -------
    TS_init (np.ndarray): an N x L time series with nonzero entries in
                          the first column

    """

    TS_init = np.zeros((N, L))
    num_init = np.round(initial_fraction * N).astype('int')

    TS_init[np.random.permutation(N)[0:num_init], 0] = 1

    # maybe also here initialize TS_init with external drives?
    if m != 1.0:
        N_nodes = 1000
        if N > 1000:
            N_nodes = N
        h_vals = np.random.poisson(target_Ahat * N_nodes * np.abs(1 - m), L)
    else:
        N_nodes = 100
        if N > 100:
            N_nodes = N
        h_vals = np.random.poisson(0.01, L)
    #         h_vals = np.random.poisson(target_Ahat*N_nodes * 0.01, L)

    sum_h_vals = sum(h_vals)
    external_drive_timestamps = sorted(list(np.nonzero(h_vals)[0]))
    external_drive_activenodes = list(np.random.choice(N, sum_h_vals))

    for timestamp in external_drive_timestamps:
        num_pops = h_vals[timestamp]
        active_nodes = [external_drive_activenodes.pop() for i in range(num_pops)]
        TS_init[active_nodes, timestamp] = 1  # or maybe equals 1?

    if noise:
        TS_init += np.random.uniform(-np.exp(-12), np.exp(-12), TS_init.shape)

    return TS_init


def initialize_threshold_cache(num_edges, distribution_type='unif', scale=1.0):
    """
    A cache of random numbers. This is useful for speed, as calling the numpy
    random number generator can get costly with large networks and time series.

    Parameters
    ----------
    num_edges (int): the length of the cache, which should correspond to the
                     combination of all possible activity over the simulation.
    distribution_type (str): string that describes which type of random numbers.
    scale (float): scale for how likely nodes are to topple

    Returns
    -------
    edges (np.ndarray): a vector of probability thresholds, above which the node
                        will topple and send information to the following node.

    """

    if distribution_type == 'unif':
        edges = scale * np.random.rand(num_edges)
        return edges

    elif distribution_type == 'normal':
        edges = scale * np.random.randn(num_edges)
        return edges

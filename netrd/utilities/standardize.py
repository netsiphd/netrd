"""
standardize.py
--------------

Utilities for computing standardization values for distance measures.

author: Harrison Hartle/Tim LaRock (timothylarock at gmail dot com)

Submitted as part of the 2019 NetSI Collabathon.

"""

import numpy as np
import networkx as nx


def mean_GNP_distance(n, prob, distance, samples=10, **kwargs):
    r"""Mean distance between :math:`G(n, p)` graphs.

    Compute the mean distance between `samples` :math:`G(n, p)` graphs with
    parameters using distance function `distance`, whose parameters are
    passed with ``**kwargs``.


    Parameters
    ----------

    n (int)
        Number of nodes in ER graphs to be generated

    prob (float)
        Probability of edge in ER graphs to be generated.

    samples (int)
        Number of samples to average distance over.

    distance (function)
        Function from `netrd.distances.<distance>.dist`

    **kwargs (dict)
        Keyword arguments to pass to the distance function.

    Returns
    -------
    mean (float)
        The average distance between the sampled ER networks.

    std (float)
        The standard deviation of the distances.

    dist (np.ndarray)
        Array storing the actual distances.

    Examples
    --------
    .. code:: python

        dist_obj = netrd.distance.ResistancePerturbation()
        kwargs = {'p':2}
        mean, std, dists = netrd.utilities.mean_GNP_distance(100, 0.1, dist_obj.dist, **kwargs)


    Notes
    -----
    Ideally, each sample would involve generating two :math:`G(n, p)`
    graphs, computing the distance between them, then throwing them both
    away.  However, this will be computationally expensive, so for now we
    are reusing samples. The diagonal of the distance matrix is excluded,
    i.e., do not compute the distance between a sample graph and itself.

    """
    graphs = [nx.fast_gnp_random_graph(n, prob) for _ in range(samples)]
    dis_mat = np.full((samples, samples), np.nan)
    for i in range(samples):
        for j in range(samples):
            if i == j:
                continue
            dis_mat[i, j] = distance(graphs[i], graphs[j], **kwargs)

    # the nan* versions below ignore NaNs and normalize appropriately
    return np.nanmean(dis_mat), np.nanstd(dis_mat), dis_mat

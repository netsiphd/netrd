"""
ipsen_mikhailov.py
--------------------------

Graph distance based on paper:
Evolutionary reconstruction of network
Available here:
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.046109

author: Guillaume St-Onge
email: guillaume.st-onge.4@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.

"""

import numpy as np
import networkx as nx
from .base import BaseDistance
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.integrate import quad


class IpsenMikhailov(BaseDistance):
    """Compares the spectrum of the Laplacian matrices."""

    def dist(self, G1, G2, hwhm=0.08):
        """Compare the spectrum ot the associated Laplacian matrices.

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        hwhm (float)
            half with at half maximum of the lorentzian kernel.

        Returns
        -------

        dist (float)
            the distance between G1 and G2.

        Notes
        -----

        Requires undirected networks.

        References
        ----------

        .. [1] https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.046109

        """
        N = len(G1)

        # get the adjacency matrices
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)
        self.results['adjacency_matrices'] = adj1, adj2

        # get the IM distance
        dist = _im_distance(adj1, adj2, hwhm)

        self.results['dist'] = dist

        return dist


def _im_distance(adj1, adj2, hwhm):
    """Computes the Ipsen-Mikhailov distance for two symmetric adjacency
    matrices

    Base on this paper :
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.046109

    Note : this is also used by the file hamming_ipsen_mikhailov.py

    Parameters
    ----------

    adj1, adj2 (array): adjacency matrices.

    hwhm (float) : hwhm of the lorentzian distribution.

    Returns
    -------

    dist (float) : Ipsen-Mikhailov distance.

    """
    N = len(adj1)
    # get laplacian matrix
    L1 = laplacian(adj1, normed=False)
    L2 = laplacian(adj2, normed=False)

    # get the modes for the positive-semidefinite laplacian
    w1 = np.sqrt(np.abs(eigh(L1)[0][1:]))
    w2 = np.sqrt(np.abs(eigh(L2)[0][1:]))

    # we calculate the norm for both spectrum
    norm1 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w1 / hwhm))
    norm2 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w2 / hwhm))

    # define both spectral densities
    density1 = lambda w: np.sum(hwhm / ((w - w1) ** 2 + hwhm ** 2)) / norm1
    density2 = lambda w: np.sum(hwhm / ((w - w2) ** 2 + hwhm ** 2)) / norm2

    func = lambda w: (density1(w) - density2(w)) ** 2

    return np.sqrt(quad(func, 0, np.inf, limit=100)[0])

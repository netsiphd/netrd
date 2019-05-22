"""
hamming_ipsen_mikhailov.py
--------------------------

Graph distance based on paper:
The HIM glocal metric and kernel for network comparison and classification
Available here:
https://ieeexplore.ieee.org/abstract/document/7344816

author: Guillaume St-Onge
email: guillaume.st-onge.4@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.

"""

import numpy as np
import networkx as nx
from .base import BaseDistance
from scipy.optimize import fsolve
from .ipsen_mikhailov import _im_distance


class HammingIpsenMikhailov(BaseDistance):
    """Combination of Hamming and Ipsen-Mikhailov distances."""

    def dist(self, G1, G2, combination_factor=1):
        """Graph distance combining local and global distances.

        The local metric H is the Hamming distance, corresponding to the
        difference for the edges in both networks.

        The global (spectral) metric IM is the Ipsen-Mikailov distance,
        corresponding to the square-root of the squared difference of the
        Laplacian spectrum for each network.

        The Hamming-Ipsen-Mikhailov (HIM) distance is an Euclidean metric
        on the space created by the Cartesian product of the metric space
        associated with H and IM. The trade-off between global and local
        information is governed by a combination factor: when this is
        one, local and global information are balanced; when it is zero,
        it reduces to the (local) Hamming distance; and as it approaches
        infinity it becomes the (global) Ipsen-Mikhailov distance.

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'`, the Hamming
        distance in `'hamming_dist'`, the Ipsen-Mikhailov distance in
        `'ipsen_mikhailov_dist'`, and the Lorentzian half-width at
        half-maximum in `'hwhm'`. If the networks being compared are
        directed, the augmented adjacency matrices are calculated and
        stored in `'augmented_adjacency_matrices'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        combination_factor (float)
            positive factor in front of the IM metric.

        Returns
        -------

        dist (float)
            the distance between G1 and G2.

        Notes
        -----
        Requires networks with the same number of nodes.  The networks can
        be directed and weighted (with weights in the range
        :math:`[0,1]`). Both (H and IM) are also saved in the results
        dictionary.

        References
        ----------

        [1] https://ieeexplore.ieee.org/abstract/document/7344816

        """
        N = len(G1)

        # get the adjacency matrices
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)
        self.results['adjacency_matrices'] = adj1, adj2

        # verify if the graphs are directed
        directed = nx.is_directed(G1) or nx.is_directed(G2)

        if directed:
            null_mat = np.zeros((N, N))
            # create augmented adjacency matrices
            adj1_aug = np.block([[null_mat, adj1.T], [adj1, null_mat]])
            adj2_aug = np.block([[null_mat, adj2.T], [adj2, null_mat]])
            self.results['augmented_adjacency_matrices'] = adj1_aug, adj2_aug

            # get the normalized Hamming distance
            H = np.sum(np.abs(adj1_aug - adj2_aug)) / (2 * N * (N - 1))
            self.results['hamming_dist'] = H

            # get the appropriate hwhm for the network size
            hwhm = _get_hwhm_directed(N)
            self.results['hwhm'] = hwhm

            # get the IM distance
            IM = _im_distance(adj1_aug, adj2_aug, hwhm)
            self.results['ipsen_mikhailov_dist'] = IM

        else:
            # get the normalized Hamming distance
            H = np.sum(np.abs(adj1 - adj2)) / (N * (N - 1))
            self.results['hamming_dist'] = H

            # get the appropriate hwhm for the network size
            hwhm = _get_hwhm_undirected(N)
            self.results['hwhm'] = hwhm

            # get the IM distance
            IM = _im_distance(adj1, adj2, hwhm)
            self.results['ipsen_mikhailov_dist'] = IM

        # determine the glocal distance from the combination
        HIM = np.sqrt(H ** 2 + combination_factor * IM ** 2) / np.sqrt(
            1 + combination_factor
        )
        self.results['dist'] = HIM

        return HIM


def _get_hwhm_undirected(N):
    """Obtain the lorentzian half-width at half-maximum (hwhm)
    to get a normalized HIM distance

    For undirected networks.

    Parameters
    ----------

    N (int): Number of nodes.

    Returns
    -------

    hwhm (float) : hwhm of the lorentzian distribution.

    """

    def func(g):
        sN = np.sqrt(N)
        v = np.arctan(sN / g)
        return (
            -1
            + 1 / (np.pi * g)
            + (np.pi / 2 + g * sN / (g ** 2 + N) + v) / (2 * g * (np.pi / 2 + v) ** 2)
            - 4
            * g
            * (np.pi - g * np.log(g ** 2 / (g ** 2 + N)) / sN + v)
            / ((np.pi / 2 + v) * np.pi * (4 * g ** 2 + N))
        )

    return fsolve(func, 0.5)[0]


def _get_hwhm_directed(N):
    """Obtain the lorentzian half-width at half-maximum (hwhm)
    to get a normalized HIM distance

    For directed networks.

    Parameters
    ----------

    N (int): Number of nodes.

    Returns
    -------

    hwhm (float) : hwhm of the lorentzian distribution.

    """

    def func(g):
        Nm2 = N - 2
        sN = np.sqrt(N)
        sNm2 = np.sqrt(N - 2)
        s2Nm2 = np.sqrt(2 * N - 2)
        atN = np.arctan(sN / g)
        atNm2 = np.arctan(sNm2 / g)
        at2Nm2 = np.arctan(s2Nm2 / g)
        K = 1 / ((2 * N - 1) * np.pi / 2 + (N - 1) * (atNm2 + atN) + at2Nm2)
        Z = 2 * g / np.pi
        W = g * (N - 1) * K
        Wp = W / (N - 1)
        M0 = np.pi / (4 * g ** 3)
        MN = (g ** 2 * atN + N * atN + g * sN) / (2 * (g ** 5 + N * g ** 3)) + np.pi / (
            4 * g ** 3
        )
        MNm2 = (g ** 2 * atNm2 + Nm2 * atNm2 + g * sNm2) / (
            2 * (g ** 5 + Nm2 * g ** 3)
        ) + np.pi / (4 * g ** 3)
        M2Nm2 = (g ** 2 * at2Nm2 + (2 * N - 2) * at2Nm2 + g * s2Nm2) / (
            2 * (g ** 5 + (2 * N - 2) * g ** 3)
        ) + np.pi / (4 * g ** 3)
        L = lambda T, U: (-np.log(g ** 2 + U) + np.log(g ** 2 + T)) / (
            (4 * g ** 2 + T + 3 * U) * np.sqrt(T)
            - (4 * g ** 2 + 3 * T + U) * np.sqrt(U)
        ) + (np.pi + np.arctan(np.sqrt(T) / g) + np.arctan(np.sqrt(U) / g)) / (
            4 * g ** 3 + g * T - 2 * g * np.sqrt(U * T) + g * U
        )

        return (
            -1
            + Z ** 2 * M0
            + W ** 2 * (MNm2 + MN)
            + Wp ** 2 * M2Nm2
            - 2 * Z * W * L(0, Nm2)
            - 2 * Z * W * L(0, N)
            - 2 * Z * Wp * L(0, 2 * N - 2)
            + 2 * W ** 2 * L(Nm2, N)
            + 2 * W * Wp * L(Nm2, 2 * N - 2)
            + 2 * W * Wp * L(N, 2 * N - 2)
        )

    return fsolve(func, 0.5)[0]

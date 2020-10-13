"""
graph_diffusion.py
--------------------------

Graph diffusion distance, from

Hammond, D. K., Gur, Y., & Johnson, C. R. (2013, December). Graph diffusion
distance: A difference measure for weighted graphs based on the graph Laplacian
exponential kernel. In Global Conference on Signal and Information Processing,
2013 IEEE (pp 419-422). IEEE. https://doi.org/10.1109/GlobalSIP.2013.6736904

This implementation is adapted from the authors' MATLAB code, available at
https://rb.gy/txbfrh, and available under an MIT license with the authors'
permission.

author: Brennan Klein
email: brennanjamesklein at gmail dot com
Submitted as part of the 2019 NetSI Collabathon.

"""

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import laplacian
from .base import BaseDistance
from ..utilities import undirected


class GraphDiffusion(BaseDistance):
    """Find the maximally dissimilar diffusion kernels between two graphs."""

    @undirected
    def dist(self, G1, G2, thresh=1e-08, resolution=1000):
        r"""The graph diffusion distance between two graphs, :math:`G` and :math:`G'`,
        is a distance measure based on the notion of flow within each graph. As
        such, this measure uses the unnormalized Laplacian matrices of both
        graphs, :math:`\mathcal{L}` and :math:`\mathcal{L}'`, and uses them to
        construct time-varying Laplacian exponential diffusion kernels,
        :math:`e^{-t\mathcal{L}}` and :math:`e^{-t\mathcal{L}'}`, by
        effectively simulating a diffusion process for :math:`t` timesteps,
        creating a column vector of node-level activity at each timestep. The
        distance :math:`d_\texttt{GDD}(G, G')` is defined as the Frobenius norm
        between the two diffusion kernels at the timestep :math:`t^{*}` where
        the two kernels are maximally different. That is, we compute the
        Frobenius norms and their differences for each timestep, and return the
        maximum difference.

        .. math::
            D_{GDD}(G,G') = \sqrt{||e^{-t^{*}\mathcal{L}}-e^{-t^{*}\mathcal{L}'}||}

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in `adjacency_matrices`, the Laplacian matrices in
        `laplacian_matrices`, and the output of the optimization process
        (`peak_diffusion_time` and `peak_deviation`).

        Adapted from the authors' MATLAB code, available at: https://rb.gy/txbfrh


        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        thresh (float)
            minimum value above which the eigenvalues will be considered.

        resolution (int)
            number of :math:`t` values to span through.

        Returns
        -------
        dist (float)
            the distance between `G1` and `G2`.

        References
        ----------

        .. [1] Hammond, D. K., Gur, Y., & Johnson, C. R. (2013, December).
               Graph diffusion distance: A difference measure for weighted graphs based on the
               graph Laplacian exponential kernel. In Global Conference on Signal and
               Information Processing, 2013 IEEE (pp 419-422). IEEE.
               https://doi.org/10.1109/GlobalSIP.2013.6736904

        """

        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)

        L1 = laplacian(A1)
        L2 = laplacian(A2)

        def sort_eigs(eigs):
            vals, vecs = eigs
            idx = np.argsort(abs(vals))
            return vals[idx], vecs[:, idx]

        vals1, vecs1 = sort_eigs(np.linalg.eig(L1))
        vals2, vecs2 = sort_eigs(np.linalg.eig(L2))

        eigs = np.hstack((np.diag(vals1), np.diag(vals2)))
        eigs = eigs[np.where(eigs > thresh)]
        eigs = np.sort(eigs)

        if len(eigs) == 0:
            dist = 0
            self.results["dist"] = dist
            return dist

        t_upperbound = np.real(1.0 / eigs[0])
        ts = np.linspace(0, t_upperbound, resolution)

        # Find the Frobenius norms between all the diffusion kernels at
        # different times. Return the value and where this vector is minimized.
        E = -exponential_diffusion_diff(vecs1, vals1, vecs2, vals2, ts)
        f_val, t_star = (np.nanmin(E), np.argmin(E))

        dist = np.sqrt(-f_val)

        self.results["adjacency_matrices"] = A1, A2
        self.results["laplacian_matrices"] = L1, L2
        self.results["peak_diffusion_time"] = t_star
        self.results["peak_deviation"] = f_val

        self.results["dist"] = dist

        return dist


def exponential_diffusion_diff(vecs1, vals1, vecs2, vals2, ts):
    """
    Computes Frobenius norm of difference of Laplacian exponential diffusion
    kernels, at specified timepoints.

    Parameters
    ----------

    vecs1, vecs2 (np.array)
        eigenvectors of the Laplacians of `G1` and `G2`

    vals1, vals2 (np.array)
        eigenvalues of the Laplacians of `G1` and `G2`

    ts (np.array)
        times at which to compute the difference in Frobenius norms

    Returns
    -------

    diffs (np.array)
        same shape as :math:`t`, contains differences of Frobenius norms

    """

    diffs = np.zeros(len(ts))

    for kt, t in enumerate(ts):
        exp_diag_1 = np.diag(np.exp(-t * np.diag(vals1)))
        exp_diag_2 = np.diag(np.exp(-t * np.diag(vals2)))

        # multiply the eigenvectors element-wise by the appropriate diffusion value
        # before left-multiplying the eigenvectors again.
        norm1 = vecs1.dot(np.multiply(exp_diag_1, vecs1).T)
        norm2 = vecs2.dot(np.multiply(exp_diag_2, vecs2).T)
        diff = norm1 - norm2

        diffs[kt] = (diff ** 2).sum()

    return diffs

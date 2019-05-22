"""
laplacian_spectral_method.py
----------------------------

Graph distance based on :
https://www.sciencedirect.com/science/article/pii/S0303264711001869
https://arxiv.org/pdf/1005.0103.pdf
https://www.nature.com/articles/s41598-018-37534-2

author: Guillaume St-Onge
email: guillaume.st-onge.4@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.

"""
import numpy as np
import networkx as nx
from .base import BaseDistance
from scipy.special import erf
from scipy.integrate import quad
from scipy.linalg import eigvalsh
from scipy.sparse.csgraph import csgraph_from_dense
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


class LaplacianSpectral(BaseDistance):
    """Flexible distance able to compare the spectrum of the Laplacian in many ways."""

    def dist(
        self,
        G1,
        G2,
        normed=True,
        kernel='normal',
        hwhm=0.011775,
        measure='jensen-shannon',
        k=None,
        which='LM',
    ):
        """Graph distances using different measure between the Laplacian
        spectra of the two graphs

        The spectra of both Laplacian matrices (normalized or not) is
        computed. Then, the discrete spectra are convolved with a kernel to
        produce continuous ones. Finally, these distribution are compared
        using a metric.

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'`, the Laplacian
        matrices in `'laplacian_matrices'`, the eigenvalues of the
        Laplacians in `'eigenvalues'`. If the networks being compared are
        directed, the augmented adjacency matrices are calculated and
        stored in `'augmented_adjacency_matrices'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        normed (bool)
            If True, uses the normalized laplacian matrix, otherwise the
            raw laplacian matrix is used.

        kernel (str)
            kernel to obtain a continuous spectrum. Choices available are
            'normal', 'lorentzian', or None. If None is chosen, the
            discrete spectrum is used instead, and the measure is simply
            the euclidean distance between the vector of eigenvalues for
            each graph.

        hwhm (float)
            half-width at half-maximum for the kernel. The default value is
            chosen such that the standard deviation for the normal
            distribution is :math:`0.01`, as in reference [1]_. This option
            is relevant only if kernel is not None.

        measure (str)
            metric between the two continuous spectra. Choices available
            are 'jensen-shannon' or 'euclidean'. This option is relevant
            only if kernel is not None.

        k (int)
            number of eigenvalues kept for the (discrete) spectrum, also
            used to create the continuous spectrum. If None, all the
            eigenvalues are used. k must be smaller (strictly) than the
            size of both graphs.

        which (str)
            if k is not None, this option specifies the eigenvalues that
            are kept. See the choices offered by
            `scipy.sparse.linalg.eigsh`.  The largest eigenvalues in
            magnitude are kept by default.

        Returns
        -------

        dist (float)
            the distance between G1 and G2.

        Notes
        -----
        The methods are usually applied to undirected (unweighted)
        networks. We however relax this assumption using the same method
        proposed for the Hamming-Ipsen-Mikhailov. See [2]_.

        References
        ----------

        .. [1] https://www.sciencedirect.com/science/article/pii/S0303264711001869.

        .. [2] https://ieeexplore.ieee.org/abstract/document/7344816.

        """
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)
        self.results['adjacency_matrices'] = adj1, adj2
        directed = nx.is_directed(G1) or nx.is_directed(G2)

        if directed:
            # create augmented adjacency matrices
            N1 = len(G1)
            N2 = len(G2)
            null_mat1 = np.zeros((N1, N1))
            null_mat2 = np.zeros((N2, N2))
            adj1 = np.block([[null_mat1, adj1.T], [adj1, null_mat1]])
            adj2 = np.block([[null_mat2, adj2.T], [adj2, null_mat2]])
            self.results['augmented_adjacency_matrices'] = adj1, adj2

        # get the laplacian matrices
        lap1 = laplacian(adj1, normed=normed)
        lap2 = laplacian(adj2, normed=normed)
        self.results['laplacian_matrices'] = lap1, lap2

        # get the eigenvalues of the laplacian matrices
        if k is None:
            ev1 = np.abs(eigvalsh(lap1))
            ev2 = np.abs(eigvalsh(lap2))
        else:
            # transform the dense laplacian matrices to sparse representations
            lap1 = csgraph_from_dense(lap1)
            lap2 = csgraph_from_dense(lap2)
            ev1 = np.abs(eigsh(lap1, k=k, which=which)[0])
            ev2 = np.abs(eigsh(lap2, k=k, which=which)[0])
        self.results['eigenvalues'] = ev1, ev2

        if kernel is not None:
            # define the proper support
            a = 0
            if normed:
                b = 2
            else:
                b = np.inf

            # create continuous spectra
            density1 = _create_continuous_spectrum(ev1, kernel, hwhm, a, b)
            density2 = _create_continuous_spectrum(ev2, kernel, hwhm, a, b)

            # compare the spectra
            dist = _spectra_comparison(density1, density2, a, b, measure)
            self.results['dist'] = dist
        else:
            # euclidean distance between the two discrete spectra
            dist = np.linalg.norm(ev1 - ev2)
            self.results['dist'] = dist

        return dist


def _create_continuous_spectrum(eigenvalues, kernel, hwhm, a, b):
    """Convert a set of eigenvalues into a normalized density function

    The discret spectrum (sum of dirac delta) is convolved with a kernel and
    renormalized.

    Parameters
    ----------

    eigenvalues (array): list of eigenvalues.

    kernel (str): kernel to be used for the convolution with the discrete
    spectrum.

    hwhm (float): half-width at half-maximum for the kernel.

    a,b (float): lower and upper bounds of the support for the eigenvalues.

    Returns
    -------

    density (function): one argument function for the continuous spectral
    density.

    """
    # define density and repartition function for each eigenvalue
    if kernel == "normal":
        std = hwhm / 1.1775
        f = lambda x, xp: np.exp(-(x - xp) ** 2 / (2 * std ** 2)) / np.sqrt(
            2 * np.pi * std ** 2
        )
        F = lambda x, xp: (1 + erf((x - xp) / (np.sqrt(2) * std))) / 2
    elif kernel == "lorentzian":
        f = lambda x, xp: hwhm / (np.pi * (hwhm ** 2 + (x - xp) ** 2))
        F = lambda x, xp: np.arctan((x - xp) / hwhm) / np.pi + 1 / 2

    # compute normalization factor and define density function
    Z = np.sum(F(b, eigenvalues) - F(a, eigenvalues))
    density = lambda x: np.sum(f(x, eigenvalues)) / Z

    return density


def _spectra_comparison(density1, density2, a, b, measure):
    """Apply a metric to compare the continuous spectra

    Parameters
    ----------

    density1, density2 (function): one argument functions for the continuous
    spectral densities.

    a,b (float): lower and upper bounds of the support for the eigenvalues.

    measure (str): metric between the two continuous spectra.

    Returns
    -------

    dist (float): distance between the spectra.

    """
    if measure == "jensen-shannon":
        M = lambda x: (density1(x) + density2(x)) / 2
        jensen_shannon = (
            _kullback_leiber(density1, M, a, b) + _kullback_leiber(density2, M, a, b)
        ) / 2
        dist = np.sqrt(jensen_shannon)

    elif measure == "euclidean":
        integrand = lambda x: (density1(x) - density2(x)) ** 2
        dist = np.sqrt(quad(integrand, a, b)[0])

    return dist


def _kullback_leiber(f1, f2, a, b):
    def integrand(x):
        if f1(x) > 0 and f2(x) > 0:
            result = f1(x) * np.log(f1(x) / f2(x))
        else:
            result = 0
        return result

    return quad(integrand, a, b)[0]

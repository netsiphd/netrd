"""
laplacian_spectral_method.py
----------------------------

Graph distance based on :
https://www.sciencedirect.com/science/article/pii/S0303264711001869
https://arxiv.org/pdf/1005.0103.pdf

author: Guillaume St-Onge
email: guillaume.st-onge.4@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.

"""
import numpy as np
from .base import BaseDistance
from scipy.special import erf
from scipy.integrate import quad
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh



class LaplacianSpectralMethod(BaseDistance):
    def dist(self, G1, G2, normed=True, kernel='normal', hwhm=0.011775,
             measure='jensen-shannon'):
        """Graph distances using different measure between the Laplacian
        spectra of the two graphs

        The spectra of both Laplacian matrices (normalized or not) is
        computed. Then, the discrete spectra are convolved with a kernel
        to produce continuous ones. Finally, these distribution are
        compared using a metric.

        Note : The methods are usually applied to undirected (unweighted)
        networks. We however relax this assumption using the same method
        proposed for the Hamming-Ipsen-Mikhailov. See paper :
        https://ieeexplore.ieee.org/abstract/document/7344816.

        Params
        ------

        G1, G2 (nx.Graph): two networkx graphs to be compared.

        normed (bool): If true, uses the normalized laplacian matrix,
        otherwise the raw laplacian matrix is used.

        kernel (str): kernel to obtain a continuous spectrum. Choices
        available are
            -normal
            -lorentzian

        hwhm (float): half-width at half-maximum for the kernel. The default
        value is chosen such that the standard deviation for the normal
        distribution is 0.01, as in the paper
        https://www.sciencedirect.com/science/article/pii/S0303264711001869.

        measure (str): metric between the two continuous spectra. Choices
        available are
            -jensen-shannon
            -euclidean

        Returns
        -------

        dist (float): the distance between G1 and G2.

        """

        #get the adjacency matrices
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)
        self.results['adj1'] = adj1
        self.results['adj2'] = adj2

        #verify if the graphs are directed (at least one)
        directed = nx.is_directed(G1) or nx.is_directed(G2)

        if directed:
            #create augmented adjacency matrices
            N1 = len(G1)
            N2 = len(G2)
            null_mat1 = np.zeros((N1,N1))
            null_mat2 = np.zeros((N2,N2))
            adj1 = np.block([[null_mat1, adj1.T],[adj1, null_mat1]])
            adj2 = np.block([[null_mat2, adj2.T],[adj2, null_mat2]])
            self.results['adj1_aug'] = adj1
            self.results['adj2_aug'] = adj2

        #get the laplacian and their eigenvalues
        L1 = laplacian(adj1, normed=normed)
        L2 = laplacian(adj2, normed=normed)
        ev1 = np.abs(eigh(L1)[0])
        ev2 = np.abs(eigh(L2)[0])

        #define the proper support
        a = 0
        if normed:
            b = 2
        else:
            b = np.inf

        #create continuous spectra
        density1 = _create_continuous_spectrum(ev1, kernel, hwhm, a, b)
        density2 = _create_continuous_spectrum(ev2, kernel, hwhm, a, b)

        #compare the spectra
        dist = _spectra_comparizon(density1, density2, a, b, measure)

        return dist


def _create_continuous_spectrum(eigenvalues, kernel, hwhm, a, b):
    """Convert a set of eigenvalues into a normalized density function

    The discret spectrum (sum of dirac delta) is convolved with a kernel and
    renormalized.

    Params
    ------

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
    #define density and repartition function for each eigenvalue
    if kernel == "normal":
        std = hwhm/1.1775
        f = lambda x, xp: np.exp(-(x-xp)/(2*std**2))/np.sqrt(2*np.pi*std**2)
        F = lambda x, xp: (1 + erf((x-xp)/(np.sqrt(2)*std)))/2
    elif kernel == "lorentzian":
        f = lambda x, xp: hwhm/(np.pi*(hwhm**2 + (x-xp)**2))
        F = lambda x, xp: np.arctan((x-xp)/hwhm)/np.pi + 1/2

    #compute normalization factor and define density function
    Z = np.sum(F(b, eigenvalues) - F(a, eigenvalues))
    density = lambda x: np.sum(f(x, xp))/Z

    return density


def _spectra_comparizon(density1, density2, a, b, measure):
    """Apply a metric to compare the spectra

    Params
    ------

    density1, density2 (function): one argument functions for the continuous
    spectral densities.

    a,b (float): lower and upper bounds of the support for the eigenvalues.

    measure (str): metric between the two continuous spectra.

    Returns
    -------

    dist (float): distance between the spectra.

    """
    if measure == "jensen-shannon":
        integrand1 = lambda x: density1(x)*np.log(density1(x)/density2(x))
        integrand2 = lambda x: density2(x)*np.log(density2(x)/density1(x))
        dist = np.sqrt(
            quad(integrand1, a, b)[0]/2 + quad(integrand2, a, b)[0]/2)
    elif measure == "euclidean":
        integrand = lambda x: (density1(x) - density2(x))**2
        dist = np.sqrt(quad(integrand, a, b)[0])

    return dist

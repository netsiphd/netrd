"""
netlsd.py
--------------

Graph distance based on:
A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein & E. Müller. NetLSD: Hearing the Shape of a Graph. KDD 2018

author: Anton Tsitsulin

"""
import numpy as np
import networkx as nx
import scipy.linalg as spl

from .base import BaseDistance


class NetLSD(BaseDistance):
    """Compares spectral node signature distributions."""

    def dist(self, G1, G2, normalization=None, timescales=None):
        """NetLSD: Hearing the Shape of a Graph.

        A network similarity measure based on spectral node signature
        distributions.

        The results dictionary includes the underlying signature vectors in
        `'signatures'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two undirected networkx graphs to be compared.

        normalization (str)
            type of normalization of the heat kernel vectors. either
            `'complete'`, `'empty'` or `'none'`

        timescales (np.ndarray)
            timescales for the comparison. None yields default.

        Returns
        -------

        dist (float)
            the distance between `G1` and `G2`.

        References
        ----------

        .. [1] A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein &
               E. Müller. NetLSD: Hearing the Shape of a Graph. KDD 2018

        """
        if normalization is None:
            normalization = 'none'
        if timescales is None:
            timescales = np.logspace(-2, 2, 256)
        assert isinstance(
            normalization, str
        ), 'Normalization parameter must be of string type'

        lap1 = nx.normalized_laplacian_matrix(G1)
        lap2 = nx.normalized_laplacian_matrix(G2)

        # Note: this is O(n^3) worst-case.
        eigs1 = spl.eigvalsh(lap1.todense())
        eigs2 = spl.eigvalsh(lap2.todense())

        hkt1 = _lsd_signature(eigs1, timescales, normalization)
        hkt2 = _lsd_signature(eigs2, timescales, normalization)

        self.results['signatures'] = (hkt1, hkt2)
        self.results['dist'] = np.linalg.norm(hkt1 - hkt2)

        return self.results['dist']


def _lsd_signature(eigenvalues, timescales, normalization):
    """
    Computes heat kernel trace from given eigenvalues, timescales, and normalization.

    Parameters
    --------------
    eigenvalues (numpy.ndarray): Eigenvalue vector
    timescales (numpy.ndarray): Vector of discrete timesteps for the kernel computation
    normalization (str):
        Either 'empty', 'complete' or 'none'.
        If 'none' or any other value, return unnormalized heat kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
    Returns
    -------
    numpy.ndarray
        Heat kernel trace signature
    """
    nv = eigenvalues.shape[0]
    hkt = np.zeros(timescales.shape)
    for idx, t in enumerate(timescales):
        hkt[idx] = np.sum(np.exp(-t * eigenvalues))
    if normalization == 'empty':
        return hkt / nv
    if normalization == 'complete':
        return hkt / (1 + (nv - 1) * np.exp(-(1 + 1 / (nv - 1)) * timescales))
    return hkt

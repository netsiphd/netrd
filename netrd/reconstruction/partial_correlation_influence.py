"""
partial_correlation_influence.py
--------------------------------

Reconstruction of graphs using the partial correlation influence, as defined in:

Kenett, D. Y. et al. Dominating clasp of the financial sector revealed by
partial correlation analysis of the stock market. PLoS ONE 5, e15032 (2010).

The index variable option as in:

Kenett, D. Y., Huang, X., Vodenska, I., Havlin, S. & Stanley, H. E. Partial correlation
analysis: applications for financial markets. Quantitative Finance 15, 569–578 (2015).


author: Carolina Mattsson
email: mattsson dot c at northeastern dot edu
Submitted as part of the 2019 NetSI Collabathon
"""
from .base import BaseReconstructor
import numpy as np
import networkx as nx
from scipy import stats, linalg
from ..utilities import create_graph, threshold


class PartialCorrelationInfluence(BaseReconstructor):
    """Uses average effect from a sensor to all others."""

    def fit(self, TS, index=None, threshold_type='range', **kwargs):
        r"""Uses the average effect of a series :math:`Z` on the correlation between
        a series :math:`X` and all other series.

        The partial correlation influence:

        .. math::

            d(X:Z) = <d(X,Y:Z)>_Y \neq X,

        where :math:`d(X,Y:Z) = \rho(X,Y) - \rho(X,Y:Z)`


        If an index is given, both terms become partial correlations:

        .. math::

            d(X,Y:Z) ≡ ρ(X,Y:M) − ρ(X,Y:M,Z)


        The results dictionary also stores the matrix of partial
        correlations as `'weights_matrix'` and the thresholded version of
        the partial correlation matrix as `'thresholded_matrix'`.

        Parameters
        ----------

        index (int, array of ints, or None)
            Take the partial correlations of each pair of elements holding
            constant an index variable or set of index variables. If None,
            take the partial correlations of the variables holding constant
            all other variables.

        threshold_type (str):
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        -------

        G (nx.Graph)
            a reconstructed graph.

        References
        -----------

        .. [1] Kenett, D. Y. et al. Dominating clasp of the financial
               sector revealed by partial correlation analysis of the stock
               market. PLoS ONE 5, e15032 (2010).

        .. [2] Kenett, D. Y., Huang, X., Vodenska, I., Havlin, S. &
               Stanley, H. E. Partial correlation analysis: applications
               for financial markets. Quantitative Finance 15, 569–578
               (2015).

        """
        if index:
            p_cor = partial_corr(TS, index=index)
            n_TS = p_cor.shape[0]
            p_cor = np.delete(p_cor, index, axis=0)
            p_cor = np.delete(p_cor, index, axis=1)
        else:
            p_cor = partial_corr(TS)

        np.fill_diagonal(p_cor, float("nan"))

        n = p_cor.shape[0]

        p_cor_zs = np.zeros((n, n, n))

        if index:
            for k, z in enumerate(np.delete(range(n_TS), index)):
                index_z = np.append(index, z)
                p_cor_z = partial_corr(TS, index=index_z)
                p_cor_z = np.delete(p_cor_z, index, axis=0)
                p_cor_z = np.delete(p_cor_z, index, axis=1)
                p_cor_z = p_cor - p_cor_z
                p_cor_z[:, k] = float("nan")
                p_cor_z[k, :] = -np.inf
                p_cor_zs[z] = p_cor_z
        else:
            index = np.array([], dtype=int)
            for z in range(n):
                index_z = z
                p_cor_z = partial_corr(TS, index=index_z)
                p_cor_z = p_cor - p_cor_z
                p_cor_z[:, z] = float("nan")
                p_cor_z[z, :] = -np.inf
                p_cor_zs[z] = p_cor_z

        p_cor_inf = np.nanmean(p_cor_zs, axis=2)  # mean over the Y axis

        self.results['weights_matrix'] = p_cor_inf

        # threshold the network
        W_thresh = threshold(p_cor_inf, threshold_type, **kwargs)

        # construct the network
        self.results['graph'] = create_graph(W_thresh)
        self.results['thresholded_matrix'] = W_thresh

        G = self.results['graph']

        return G


# This partial correlation function is adapted from Fabian Pedregosa-Izquierdo's
# implementation of partial correlation in Python, found at [this gist](
# https://gist.github.com/fabianp/9396204419c7b638d38f)
"""
Partial Correlation in Python (clone of Matlab's partialcorr)

This uses the linear regression approach to compute the partial
correlation (might be slow for a huge number of variables). The
algorithm is detailed here:

    http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression

Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
the algorithm can be summarized as

    1) perform a normal linear least-squares regression with X as the target and Z as the predictor
    2) calculate the residuals in Step #1
    3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
    4) calculate the residuals in Step #3
    5) calculate the correlation coefficient between the residuals from Steps #2 and #4;

    The result is the partial correlation between X and Y while controlling for the effect of Z


Date: Nov 2014
Author: Fabian Pedregosa-Izquierdo, f@bianp.net
Testing: Valentina Borghesani, valentinaborghesani@gmail.com
"""


def partial_corr(C, index=None):
    """Returns the sample linear partial correlation coefficients between pairs of
    variables in C, controlling for the remaining variables in C.


    Parameters
    ----------
    C : array-like, shape (p, n)
        Array with the different variables. Each row of C is taken as a variable


    Returns -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j]
        controlling for the remaining variables in C.

    """

    C = np.asarray(C).T
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)

    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            if index is None:
                idx = np.ones(p, dtype=np.bool)
                idx[i] = False
                idx[j] = False
            elif type(index) is int or (
                isinstance(index, np.ndarray)
                and issubclass(index.dtype.type, np.integer)
            ):
                idx = np.zeros(p, dtype=np.bool)
                idx[index] = True
            else:
                raise ValueError(
                    "Index must be an integer, an array of " "integers, or None."
                )

            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr

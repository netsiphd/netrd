"""
partial_correlation_influence.py
--------------------------------

Reconstruction of graphs using the partial correlation influence, as defined in:

Kenett, D. Y. et al. Dominating clasp of the financial sector revealed by
partial correlation analysis of the stock market. PLoS ONE 5, e15032 (2010).

The index variable option as in:

Kenett, D. Y., Huang, X., Vodenska, I., Havlin, S. & Stanley, H. E. Partial correlation
analysis: applications for financial markets. Quantitative Finance 15, 569–578 (2015).


author: Carolina Mattsson and Chia-Hung Yang
email: mattsson dot c at northeastern dot edu
Submitted as part of the 2019 NetSI Collabathon
"""
from .base import BaseReconstructor
import numpy as np
from scipy import linalg
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
        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N` sensors.

        index (int, array of ints, or None)
            An index variable or set of index variables, which are assumed to
            be confounders of all other variables. They are held constant when
            calculating the partial correlations. Default to None.

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
        data = TS.T
        N = data.shape[1]

        # Create masks to separate variables of interests from the pre-included
        # index variables
        mask = np.ones(N, dtype=bool)
        if index is not None:
            mask[index] = False

        # Compute partial correlations with the index variables held constant
        p_corr = np.full((N, N), np.nan)
        p_corr[np.ix_(mask, mask)] = partial_corr(data[:, mask], data[:, ~mask])

        # For every non-index variable Z, compute partial correlation influence
        # between other variables when Z is also held constant
        p_corr_inf = np.full((N, N, N), np.nan)
        for z in np.arange(N)[mask]:
            m_new = mask.copy()  # New mask including variable Z
            m_new[z] = False

            diff = p_corr[np.ix_(m_new, m_new)]
            diff -= partial_corr(data[:, m_new], data[:, ~m_new])
            p_corr_inf[np.ix_(m_new, m_new, [z])] = diff[:, :, np.newaxis]

            # Exclude the cases of Y = X
            np.fill_diagonal(p_corr_inf[:, :, z], np.nan)
            # Set PCI for X = Z to 0 for consistency after averaging
            p_corr_inf[z, :, z] = 0

        # Obtain the average partial correlation influence
        influence = np.zeros((N, N))  # Default self-influence by zero
        influence[mask, mask] = np.nanmean(p_corr_inf[mask, mask], axis=1)

        influence[~mask, :] = np.inf  # Index variables influence all others
        influence[:, ~mask] = 0  # but no one influences the index variables

        self.results['weights_matrix'] = influence

        # threshold the network
        W_thresh = threshold(influence, threshold_type, **kwargs)

        # construct the network
        self.results['graph'] = create_graph(W_thresh)
        self.results['thresholded_matrix'] = W_thresh

        G = self.results['graph']

        return G


def partial_corr(_vars, idx_vars):
    """
    Return the partial correlations between pairs of variables, given a set of
    index variables held constant.

    Parameters
    ----------
    _vars (numpy.ndarray)
        Variables of interests (which are columns of the array).

    idx_vars (numpy.ndarray)
        Index variables to be held constant (which are columns of the array).
        If the array has zero size, namely no index variable, return the
        Pearson correlations between variables.

    Return
    ------
    p_corr (numpy.ndarray)
         Square array of pairwise partial correlations between variables.

    Note
    ----
    Precondition: The index variables should not contain or synchronize with
                  a variable of interests.

    """
    if idx_vars.size == 0:
        return np.corrcoef(_vars, rowvar=False)
    else:
        coef = linalg.lstsq(idx_vars, _vars)[0]  # Coefficients of regression
        resid = _vars - idx_vars.dot(coef)  # Residuals
        return np.corrcoef(resid, rowvar=False)

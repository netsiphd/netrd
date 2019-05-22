"""
graphical_lasso.py
--------------

Graph reconstruction algorithm based on [1, 2].

[1] J. Friedman, T. Hastie, R. Tibshirani, "Sparse inverse covariance estimation with
the graphical lasso", Biostatistics 9, pp. 432–441 (2008).
[2] https://github.com/CamDavidsonPilon/Graphical-Lasso-in-Finance

author: Chales Murphy
email: charles.murphy.1@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.
"""

import networkx as nx
import numpy as np
from sklearn.linear_model import lars_path
from .base import BaseReconstructor
from ..utilities import create_graph, threshold


class GraphicalLasso(BaseReconstructor):
    """Performs graphical lasso."""

    def fit(
        self,
        TS,
        alpha=0.01,
        max_iter=100,
        convg_threshold=0.001,
        threshold_type='degree',
        **kwargs
    ):
        """Performs a graphical lasso.

        For details see [1, 2].

        The results dictionary also stores the covariance matrix as
        `'weights_matrix'`, the precision matrix as `'precision_matrix'`,
        and the thresholded version of the covariance matrix as
        `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            Array consisting of :math:`L` observations from :math:`N`
            sensors.

        alpha (float, default=0.01)
            Coefficient of penalization, higher values means more
            sparseness

        max_iter (int, default=100)
            Maximum number of iterations.

        convg_threshold (float, default=0.001)
            Stop the algorithm when the duality gap is below a certain
            threshold.

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        -------

        G (nx.Graph)
            A reconstructed graph with :math:`N` nodes.

        References
        ----------

        .. [1] J. Friedman, T. Hastie, R. Tibshirani, "Sparse inverse
               covariance estimation with the graphical lasso",
               Biostatistics 9, pp. 432–441 (2008).

        .. [2] https://github.com/CamDavidsonPilon/Graphical-Lasso-in-Finance

        """
        cov, prec = graphical_lasso(TS, alpha, max_iter, convg_threshold)
        self.results['weights_matrix'] = cov
        self.results['precision_matrix'] = prec

        # threshold the network
        self.results['thresholded_matrix'] = threshold(
            self.results['weights_matrix'], threshold_type, **kwargs
        )

        # construct the network
        G = create_graph(self.results['thresholded_matrix'])
        self.results['graph'] = G

        return G


def graphical_lasso(TS, alpha=0.01, max_iter=100, convg_threshold=0.001):
    """ This function computes the graphical lasso algorithm as outlined in [1].

    Parameters
    ----------
    TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.

    alpha (float, default=0.01): Coefficient of penalization, higher values
    means more sparseness

    convg_threshold (float, default=0.001): Stop the algorithm when the
    duality gap is below a certain threshold.

    Returns
    -------
    cov (np.ndarray): Estimator of the inverse covariance matrix with sparsity.

    """

    TS = TS.T

    if alpha < 1e-15:
        covariance_ = cov_estimator(TS)
        precision_ = np.linalg.pinv(TS)
        return covariance_, precision_
    n_features = TS.shape[1]

    mle_estimate_ = cov_estimator(TS)
    covariance_ = mle_estimate_.copy()
    precision_ = np.linalg.pinv(mle_estimate_)
    indices = np.arange(n_features)
    for i in range(max_iter):
        for n in range(n_features):
            sub_estimate = covariance_[indices != n].T[indices != n]
            row = mle_estimate_[n, indices != n]
            # solve the lasso problem
            _, _, coefs_ = lars_path(
                sub_estimate,
                row,
                Xy=row,
                Gram=sub_estimate,
                alpha_min=alpha / (n_features - 1.0),
                copy_Gram=True,
                method="lars",
            )
            coefs_ = coefs_[:, -1]  # just the last please.
            # update the precision matrix.
            precision_[n, n] = 1.0 / (
                covariance_[n, n] - np.dot(covariance_[indices != n, n], coefs_)
            )
            precision_[indices != n, n] = -precision_[n, n] * coefs_
            precision_[n, indices != n] = -precision_[n, n] * coefs_
            temp_coefs = np.dot(sub_estimate, coefs_)
            covariance_[n, indices != n] = temp_coefs
            covariance_[indices != n, n] = temp_coefs

        # if test_convergence( old_estimate_, new_estimate_, mle_estimate_, convg_threshold):
        if np.abs(_dual_gap(mle_estimate_, precision_, alpha)) < convg_threshold:
            break
    else:
        # this triggers if not break command occurs
        print(
            "The algorithm did not converge. Try increasing the max number of iterations."
        )

    return covariance_, precision_


def cov_estimator(TS):
    """
    Computes the covariance estimate for the time series.

    Parameters
    ----------
    TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.

    Returns
    -------
    cov (np.ndarray): Estimator of the inverse covariance matrix.

    """
    return np.cov(TS.T)


def _dual_gap(emp_cov, precision, alpha):
    """Expression of the dual gap convergence criterion. The specific definition
    is given in Duchi "Projected Subgradient Methods for Learning Sparse
    Gaussians".

    Parameters
    ----------
    emp_cov (np.ndarray): Empirical covariance matrix

    precision (np.ndarray): Precision matrix

    alpha (float, default=0.01): Coefficient of penalization, higher values
    means more sparseness

    """
    gap = np.sum(emp_cov * precision)
    gap -= precision.shape[0]
    gap += alpha * (np.abs(precision).sum() - np.abs(np.diag(precision)).sum())
    return gap

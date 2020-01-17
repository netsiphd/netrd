"""
graphical_lasso.py
--------------

Graph reconstruction algorithm based on [1, 2].

[1] J. Friedman, T. Hastie, R. Tibshirani, "Sparse inverse covariance estimation with
the graphical lasso", Biostatistics 9, pp. 432–441 (2008).
[2] https://github.com/CamDavidsonPilon/Graphical-Lasso-in-Finance

author: Charles Murphy
email: charles.murphy.1@ulaval.ca
Submitted as part of the 2019 NetSI Collabathon.
"""

import numpy as np
from sklearn.covariance import graphical_lasso
from .base import BaseReconstructor
from ..utilities import create_graph, threshold


class GraphicalLasso(BaseReconstructor):
    """Performs graphical lasso."""

    def fit(
        self,
        TS,
        alpha=0.01,
        max_iter=100,
        tol=0.0001,
        threshold_type='degree',
        **kwargs
    ):
        """Performs a graphical lasso.

        For details see [1, 2].

        The results dictionary also stores the covariance matrix as
        `'weights_matrix'`, the precision matrix as `'precision_matrix'`,
        and the thresholded version of the covariance matrix as
        `'thresholded_matrix'`.

        This implementation uses `scikit-learn`'s implementation of the
        graphical lasso; for convenience two control parameters `tol` and
        `max_iter` are available to interface with their method.

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

        tol (float, default=0.0001)
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
        emp_cov = np.cov(TS)

        cov, prec = graphical_lasso(emp_cov, alpha, max_iter=max_iter, tol=tol)
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

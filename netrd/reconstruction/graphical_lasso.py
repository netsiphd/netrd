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

import numpy as np
from sklearn.linear_model import lars_path
from .base import BaseReconstructor


class GraphicalLassoReconstructor(BaseReconstructor):
    def fit(self, TS, alpha=0.01, max_iter=100, convg_threshold=0.001):
        """A brief one-line description of the algorithm goes here.

        A short paragraph may follow. The paragraph may include $latex$ by
        enclosing it in dollar signs $\textbf{like this}$.

        Params
        ------
        TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.

        alpha (float, default=0.01): Coefficient of penalization, higher values
        means more sparseness

        max_iter (int, default=100): Maximum number of iterations.

        convg_threshold (float, default=0.001): Stop the algorithm when the
        duality gap is below a certain threshold.


        Returns
        -------
        G (nx.Graph): A reconstructed graph with $N$ nodes.

        """

        cov, prec = graphical_lasso(TS, alpha, max_iter, convg_threshold)
        G = nx.from_numpy_array(cov)
        self.results['covariance'] = cov
        self.results['precision'] = prec
        self.results['graph'] = G

        return G


def graphical_lasso(TS, alpha=0.01, max_iter=100, convg_threshold=0.001):
    """ This function computes the graphical lasso algorithm as outlined in [1].
        
    Params
    ------
    TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.

    alpha (float, default=0.01): Coefficient of penalization, higher values
    means more sparseness

    convg_threshold (float, default=0.001): Stop the algorithm when the
    duality gap is below a certain threshold.

    Returns
    -------
    cov (np.ndarray): Estimator of the inverse covariance matrix with sparsity.
    
    """
    
    if alpha == 0:
        return cov_estimator(TS)
    n_features = TS.shape[1]

    mle_estimate_ = cov_estimator(TS)
    covariance_ = mle_estimate_.copy()
    precision_ = np.linalg.pinv( mle_estimate_ )
    indices = np.arange( n_features)
    for i in xrange( max_iter):
        for n in range( n_features ):
            sub_estimate = covariance_[ indices != n ].T[ indices != n ]
            row = mle_estimate_[ n, indices != n]
            #solve the lasso problem
            _, _, coefs_ = lars_path( sub_estimate, row, Xy = row, Gram = sub_estimate, 
                                        alpha_min = alpha/(n_features-1.), copy_Gram = True,
                                        method = "lars")
            coefs_ = coefs_[:,-1] #just the last please.
        #update the precision matrix.
            precision_[n,n] = 1./( covariance_[n,n] 
                                    - np.dot( covariance_[ indices != n, n ], coefs_  ))
            precision_[indices != n, n] = - precision_[n, n] * coefs_
            precision_[n, indices != n] = - precision_[n, n] * coefs_
            temp_coefs = np.dot( sub_estimate, coefs_)
            covariance_[ n, indices != n] = temp_coefs
            covariance_[ indices!=n, n ] = temp_coefs
        
        #if test_convergence( old_estimate_, new_estimate_, mle_estimate_, convg_threshold):
        if np.abs( _dual_gap( mle_estimate_, precision_, alpha ) )< convg_threshold:
                break
    else:
        #this triggers if not break command occurs
        print "The algorithm did not coverge. Try increasing the max number of iterations."
    
    return covariance_, precision_        
        
        
def cov_estimator(TS):
    """
    Computes the covariance estimate for the time series.

    Params
    ------
    TS (np.ndarray): Array consisting of $L$ observations from $N$ sensors.

    Returns
    -------
    cov (np.ndarray): Estimator of the inverse covariance matrix.

    """
    return np.cov( TS.T)


def _dual_gap(emp_cov, precision, alpha):
    """Expression of the dual gap convergence criterion. The specific definition
    is given in Duchi "Projected Subgradient Methods for Learning Sparse
    Gaussians".

    Params
    ------
    emp_cov (np.ndarray): Empirical covariance matrix

    precision (np.ndarray): Precision matrix

    alpha (float, default=0.01): Coefficient of penalization, higher values
    means more sparseness

    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum()
                    - np.abs(np.diag(precision_)).sum())
    return gap


J. Friedman, T. Hastie, R. Tibshirani, Biostatistics 9, pp. 432–441 (2008).
"""
naive_transfer_entropy.py
--------------
Graph reconstruction algorithm based on
Schreiber, T. (2000).  Measuring information transfer.
Physical Review Letters, 85(2):461–464
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.85.461

author: Chia-Hung Yang and Brennan Klein
email: yang.chi[at]husky[dot]neu[dot]edu and klein.br@husky.neu.edu
Submitted as part of the 2019 NetSI Collabathon.
"""

from .base import BaseReconstructor
import numpy as np
from itertools import permutations
from ..utilities import create_graph, threshold
from ..utilities.entropy import conditional_entropy, categorized_data


class NaiveTransferEntropy(BaseReconstructor):
    """Uses transfer entropy between sensors."""

    def fit(self, TS, delay_max=1, n_bins=2, threshold_type='range', **kwargs):
        r"""Calculates the transfer entropy from i --> j.

        The resulting network is asymmetric, and each element
        :math:`TE_{ij}` represents the amount of information contained
        about the future states of :math:`i` by knowing the past states of
        :math:`i` and past states of :math:`j`. Presumably, if one time
        series :math:`i` does not depend on the other :math:`j`, knowing
        all of i does not increase your certainty about the next state of
        :math:`i`.

        The reason that this method is referred to as "naive" transfer
        entropy is because it appears there are much more complicated
        conditional mutual informations that need to be calculated in order
        for this method to be true to the notion of information
        transfer. These are implemented in state of the art algorighms, as
        in the Java Information Dynamics Toolkit [1]_.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            array consisting of :math:`L` observations from :math:`N`
            sensors.

        delay_max (int)
            the number of timesteps in the past to aggregate and average in
            order to get :math:`TE_{ij}`

        n_bins (int)
            the number of bins to turn values in the time series to categorical
            data, which is a pre-processing step to compute entropy.

        threshold_type (str)
            Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        -------

        G (nx.Graph)
            a reconstructed graph with :math:`N` nodes.

        References
        ----------

        .. [1] https://github.com/jlizier/jidt

        """
        N, L = TS.shape  # Get the shape and length of the time series
        data = TS.T  # Transpose the time series to make observations the rows
        if delay_max >= L:
            raise ValueError('Max steps of delay exceeds time series length.')

        # Transform the data into its binned categorical version,
        # which is a pre-processing before computing entropy
        data = categorized_data(data, n_bins)

        # Compute the transfer entropy of every tuple of nodes
        TE = np.zeros((N, N))  # Initialize an matrix for transfer entropy
        for i, j in permutations(range(N), 2):
            # Check several delay values and average them together
            # This average is naive, but appears to be sufficient in
            # some circumstances
            te_list = [
                transfer_entropy(data[:, i], data[:, j], delay)
                for delay in range(1, delay_max + 1)
            ]
            TE[i, j] = np.mean(te_list)

        self.results['weights_matrix'] = TE

        # threshold the network
        TE_thresh = threshold(TE, threshold_type, **kwargs)
        self.results['thresholded_matrix'] = TE_thresh

        # construct the network
        self.results['graph'] = create_graph(TE_thresh)
        G = self.results['graph']

        return G


def transfer_entropy(X, Y, delay):
    """
    This is a TE implementation: asymmetric statistic measuring the reduction
    in uncertainty for the dynamics of Y given the history of X. Or the
    amount of information from X to Y. The calculation is done via conditional
    mutual information.

    Parameters
    ----------
    X (np.ndarray): time series of categorical values from node :math:`i`
    Y (np.ndarray): time series of categorical values from node :math:`j`
    delay (int): steps with which node :math:`i` past state is accounted

    Returns
    -------
    te (float): the transfer entropy from nodes i to j

    """
    X_past = X[:-delay, np.newaxis]
    Y_past = Y[:-delay, np.newaxis]
    joint_past = np.hstack((Y_past, X_past))
    Y_future = Y[delay:, np.newaxis]

    te = conditional_entropy(Y_future, Y_past)
    te -= conditional_entropy(Y_future, joint_past)

    return te

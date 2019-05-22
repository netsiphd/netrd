"""
entropy.py
----------

Utility functions computing entropy of variables in time series data.

author: Chia-Hung Yang

Submitted as part of the 2019 NetSI Collabathon.
"""

import numpy as np
from scipy.stats import entropy as sp_entropy


def js_divergence(P, Q):
    """Jenson-Shannon divergence between `P` and `Q`.

    Parameters
    ----------

    P, Q (np.ndarray)
        Two discrete distributions represented as 1D arrays. They are
        assumed to have the same support

    Returns
    -------

    float
        The Jensen-Shannon divergence between `P` and `Q`.

    """
    M = 0.5 * (P + Q)
    return 0.5 * (sp_entropy(P, M, base=2) + sp_entropy(Q, M, base=2))


def entropy(var):
    """Return the Shannon entropy of a variable.

    Parameters
    ----------

    var (ndarray)
        1D array of observations of the variable.

    Notes
    -----

    1. :math:`H(X) = - \sum p(X) \log_2(p(X))`
    2. Data of the variable must be categorical.

    """
    return joint_entropy(var[:, np.newaxis])


def joint_entropy(data):
    r"""Joint entropy of all variables in the data.

    Parameters
    ----------
    data (np.ndarray)
        Array of data with variables as columns and observations as rows.

    Returns
    -------
    float
        Joint entrpoy of the variables of interests.

    Notes
    -----
    1. :math:`H(\{X_i\}) = - \sum p(\{X_i\}) \log_2(p(\{X_i\}))`
    2. The data of variables must be categorical.

    """
    # Entropy is computed through summing contribution of states with
    # non-zero empirical probability in the data
    count = dict()
    for state in data:
        key = tuple(state)
        count.setdefault(key, 0)
        count[key] += 1

    return sp_entropy(list(count.values()), base=2)


def conditional_entropy(data, given):
    r"""Conditional entropy of variables in the data conditioned on
    a given set of variables.

    Parameters
    ----------
    data (np.ndarray)
        Array of data with variables of interests as columns and
        observations as rows.

    given (np.ndarray)
        Array of data with the conditioned variables as columns and
        observations as rows.

    Returns
    -------
    float
        Conditional entrpoy of the variables :math:`\{X_i\}` of interest
        conditioned on variables :math:`\{Y_j\}`.

    Notes
    -----
    1. :math:`H(\{X_i\}|\{Y_j\}) = - \sum p(\{X_i\}\cup\{Y_j\}) \log_2(p(\{X_i\}|\{Y_j\}))`
    2. The data of vairiables must be categorical.

    """
    joint = np.hstack((data, given))
    entrp = joint_entropy(joint) - joint_entropy(given)

    return entrp


def categorized_data(raw, n_bins):
    """Categorize data.

    An entry in the returned array is the index of the bin of the
    linearly-binned raw continuous data.

    Parameters
    ----------
    raw (np.ndarray)
        Array of raw continuous data.
    n_bins (int)
        A universal number of bins for all the variables.

    Returns
    -------
    np.ndarray
        Array of bin indices after categorizing the raw data.

    """
    bins = linear_bins(raw, n_bins)
    data = np.ones(raw.shape, dtype=int)

    # Find the index of bins each element in the raw data array belongs to
    for (i, j), val in np.ndenumerate(raw):
        data[i, j] = np.argmax(bins[1:, j] >= val)

    return data


def linear_bins(raw, n_bins):
    r"""Separators of linear bins for each variable in the raw data.

    Parameters
    ----------
    raw (np.ndarray)
        Array of raw continuous data.

    n_bins (int)
        A universal number of bins for all the variables.

    Returns
    -------
    np.ndarray
        Array where a column is the separators of bins for a variable.

    Notes
    -----
    The bins are :math:`B_0 = [b_0, b_1]`, :math:`B_i = (b_i, b_{i+1}]`,
    where :math:`b_i` s are the separators of bins.

    """
    _min = raw.min(axis=0)
    _max = raw.max(axis=0)
    bins = np.array(
        [np.linspace(start, end, num=n_bins + 1) for start, end in zip(_min, _max)]
    )
    return bins.T

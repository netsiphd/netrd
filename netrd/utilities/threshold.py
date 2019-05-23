"""
threshold.py
------------

Utilities for thresholding matrices based on different criteria

author: Stefan McCabe (stefanmccabe at gmail dot com)

Submitted as part of the 2019 NetSI Collabathon.

"""
import numpy as np
import warnings


def threshold_in_range(mat, **kwargs):
    r"""Threshold by setting values not within a list of ranges to zero.

    Parameters
    ----------
    mat (np.ndarray)
        A numpy array.

    cutoffs (list of tuples)
        When thresholding, include only edges whose correlations fall
        within a given range or set of ranges. The lower value must come
        first in each tuple. For example, to keep those values whose
        absolute value is between :math:`0.5` and :math:`1`, pass
        ``cutoffs=[(-1, -0.5), (0.5, 1)]``.

    Returns
    -------
    thresholded_mat (np.ndarray)
        the thresholded numpy array

    """
    if 'cutoffs' in kwargs:
        cutoffs = kwargs['cutoffs']
    else:
        warnings.warn(
            "Setting 'cutoffs' argument is strongly encouraged. Using cutoff range of (-1, 1).",
            RuntimeWarning,
        )
        cutoffs = [(-1, 1)]

    mask_function = np.vectorize(
        lambda x: any([x >= cutoff[0] and x <= cutoff[1] for cutoff in cutoffs])
    )
    mask = mask_function(mat)

    thresholded_mat = mat * mask

    if kwargs.get('binary', False):
        thresholded_mat = np.abs(np.sign(thresholded_mat))

    if kwargs.get('remove_self_loops', True):
        np.fill_diagonal(thresholded_mat, 0)

    return thresholded_mat


def threshold_on_quantile(mat, **kwargs):
    """Threshold by setting values below a given quantile to zero.

    Parameters
    ----------

    mat (np.ndarray)
        A numpy array.

    quantile (float)
        The threshold above which to keep an element of the array, e.g.,
        set to zero elements below the 90th quantile of the array.

    Returns
    -------
    thresholded_mat
        the thresholded numpy array

    """
    if 'quantile' in kwargs:
        quantile = kwargs['quantile']
    else:
        warnings.warn(
            "Setting 'quantile' argument is strongly recommended. Using target quantile of 0.9 for thresholding.",
            RuntimeWarning,
        )
        quantile = 0.9

    if kwargs.get('remove_self_loops', True):
        np.fill_diagonal(mat, 0)

    if quantile != 0:
        thresholded_mat = mat * (mat > np.percentile(mat, quantile * 100))
    else:
        thresholded_mat = mat

    if kwargs.get('binary', False):
        thresholded_mat = np.abs(np.sign(thresholded_mat))

    return thresholded_mat


def threshold_on_degree(mat, **kwargs):
    """Threshold by setting values below a given quantile to zero.

    Parameters
    ----------

    mat (np.ndarray)
        A numpy array.

    avg_k (float)
        The average degree to target when thresholding the matrix.

    Returns
    -------
    thresholded_mat
        the thresholded numpy array

    """

    if 'avg_k' in kwargs:
        avg_k = kwargs['avg_k']
    else:
        warnings.warn(
            "Setting 'avg_k' argument is strongly encouraged. Using average "
            "degree of 1 for thresholding.",
            RuntimeWarning,
        )
        avg_k = 1

    n = len(mat)
    A = np.ones((n, n))

    if kwargs.get('remove_self_loops', True):
        np.fill_diagonal(A, 0)
        np.fill_diagonal(mat, 0)

    if np.mean(np.sum(A, 1)) <= avg_k:
        # degenerate case: threshold the whole matrix
        thresholded_mat = mat
    else:
        for m in sorted(mat.flatten()):
            A[mat == m] = 0
            if np.mean(np.sum(A, 1)) <= avg_k:
                break
        thresholded_mat = mat * (mat > m)

    if kwargs.get('binary', False):
        thresholded_mat = np.abs(np.sign(thresholded_mat))

    return thresholded_mat


def threshold(mat, rule, **kwargs):
    """A flexible interface to other thresholding functions.

    Parameters
    ----------

    mat (np.ndarray)
        A numpy array.

    rule (str)
        A string indicating which thresholding function to invoke.

    kwargs (dict)
        Named arguments to pass to the underlying threshold function.

    Returns
    -------
    thresholded_mat
        the thresholded numpy array

    """
    try:
        if rule == 'degree':
            return threshold_on_degree(mat, **kwargs)
        elif rule == 'range':
            return threshold_in_range(mat, **kwargs)
        elif rule == 'quantile':
            return threshold_on_quantile(mat, **kwargs)
        elif rule == 'custom':
            return kwargs['custom_thresholder'](mat)
    except KeyError:
        raise ValueError("missing threshold parameter")

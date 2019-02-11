"""
threshold.py
------------

Utilities for thresholding matrices based on different criteria

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon

"""
import numpy as np


def threshold_in_range(mat, **kwargs):
    """
    Threshold a numpy array by setting values not within a list of ranges to zero.

    Params
    ------
    mat: (np.ndarray): A numpy array.
    cutoffs (list of tuples): When thresholding, include only edges whose
    correlations fall within a given range or set of ranges. The lower
    value must come first in each tuple. For example, to keep those values whose absolute
    value is between 0.5 and 1, pass `cutoffs=[(-1, -0.5), (0.5, 1)]`

    Returns
    -------
    thresholded_mat: the thresholded numpy array

    """

    if 'cutoffs' in kwargs:
        cutoffs = kwargs['cutoffs']
    else:
        cutoffs = [(-1, 1)]

    mask_function = np.vectorize(lambda x: any([x>=cutoff[0] and x<=cutoff[1] for cutoff in cutoffs]))
    mask = mask_function(mat)

    thresholded_mat = mat * mask

    if 'binary' in kwargs and kwargs['binary']:
        thresholded_mat = np.abs(np.sign(thresholded_mat))

    return thresholded_mat


def threshold_on_quantile(mat, **kwargs):
    """
    Threshold a numpy array by setting values below a given quantile to zero.

    Params
    ------
    mat: (np.ndarray): A numpy array.
    quantile (float): The threshold above which to keep an element of the array,
    e.g., set to zero elements below the 90th quantile of the array.

    Returns
    -------
    thresholded_mat: the thresholded numpy array

    """
    if 'quantile' in kwargs:
        quantile = kwargs['quantile']
    else:
        quantile = 0.9

    if quantile != 0:
        thresholded_mat = mat * (mat > np.percentile(mat, quantile * 100))
    else:
        thresholded_mat = mat

    if 'binary' in kwargs and kwargs['binary']:
        thresholded_mat = np.abs(np.sign(thresholded_mat))

    return thresholded_mat


def threshold_on_degree(mat, **kwargs):
    """
    Threshold a numpy array by setting values below a given quantile to zero.

    Params
    ------
    mat: (np.ndarray): A numpy array.
    avg_k (float): The average degree to target when thresholding the matrix.

    Returns
    -------
    thresholded_mat: the thresholded numpy array

    """

    if 'avg_k' in kwargs:
        avg_k = kwargs['avg_k']
    else:
        avg_k = 1

    n = len(mat)
    A = np.ones((n, n))

    if np.mean(np.sum(A, 1)) <= avg_k:
        # degenerate case: threshold the whole matrix
        thresholded_mat = mat
    else:
        for m in sorted(mat.flatten()):
            A[mat == m] = 0
            if np.mean(np.sum(A, 1)) <= avg_k:
                break
        thresholded_mat = mat * (mat > m)

    if 'binary' in kwargs and kwargs['binary']:
        thresholded_mat = np.abs(np.sign(thresholded_mat))

    return thresholded_mat

def threshold(mat, rule, **kwargs):
    """
    A flexible interface to other thresholding functions.

    Params
    ------
    mat: (np.ndarray): A numpy array.
    rule (str): A string indicating which thresholding function to invoke.
    kwargs (dict): Named arguments to pass to the underlying threshold function.

    Returns
    -------
    thresholded_mat: the thresholded numpy array

    ---
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

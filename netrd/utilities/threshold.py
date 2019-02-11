"""
threshold.py
------------

Utilities for thresholding matrices based on different criteria

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon

"""
import numpy as np


def threshold_in_range(mat, cutoffs=[(-1, 1)]):
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

    mask_function = np.vectorize(lambda x: any([x>=cutoff[0] and x<=cutoff[1] for cutoff in cutoffs]))
    mask = mask_function(mat)

    thresholded_mat = mat * mask
    return thresholded_mat


def threshold_on_quantile(mat, quantile=0.9):
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

    return mat * (mat > np.percentile(mat, quantile * 100))


def threshold_on_degree(mat, avg_k=1):
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

    n = len(mat)
    A = np.ones((n, n))

    for m in sorted(mat.flatten()):
        A[mat == m] = 0
        if np.mean(np.sum(A, 1)) <= avg_k:
            break

    return mat * (mat > m)

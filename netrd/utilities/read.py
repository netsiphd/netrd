"""
read.py
-------

Utilities for reading data.

author: Tim LaRock (timothylarock at gmail dot com)

Submitted as part of the 2019 NetSI Collabathon.

"""
import numpy as np
import networkx as nx


def read_time_series(filename, delimiter=','):
    r"""Read a time series from a file into an array.

    This function expects `filename` to be a comma separated text file with
    only data (no headers).

    Parameters
    ----------
    filename (str)
        path to a file that will be read

    delimiter (str)
        delimiter in the file

    Returns
    -------

    arr
        the array read from filename

    """
    return np.loadtxt(filename, delimiter=delimiter)

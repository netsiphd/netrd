"""
read.py
------------

Utilities for reading data

author: Tim LaRock
email: timothylarock at gmail dot com
Submitted as part of the 2019 NetSI Collabathon

"""
import numpy as np
import networkx as nx

def read_time_series(filename, delimiter=','):
    """
    Read an NxL time series from a file into a numpy.ndarray.

    This function expects filename to be a comma separated
    text file with _only_ data.

    Params
    ------
    filename: (str) path to a file that will be read

    Returns
    -------
    arr: the array read from filename

    """
    return np.loadtxt(filename, delimiter=delimiter)

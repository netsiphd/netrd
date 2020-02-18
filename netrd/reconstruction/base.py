import networkx as nx
import numpy as np
import warnings
from scipy.sparse.csgraph import minimum_spanning_tree


class BaseReconstructor:
    """Base class for graph reconstruction algorithms.

    The basic usage of a graph reconstruction algorithm is as follows:

    >>> R = ReconstructionAlgorithm()
    >>> G = R.fit(TS, <some_params>).to_graph()

    However, this is probably not the desired behavior, because (depending on
    the method used) it will return a complete graph with varying weights
    between the edges. The network should typically be thresholded in some way
    to produce a more useful structure.

    >>> R = ReconstructionAlgorithm()
    >>> R = R.fit(TS, <some_params>)
    >>> R = R.remove_self_loops().threshold('quantile', quantile=0.8)
    >>> G = R.to_graph()

    Note that these can all be combined into a single call using method
    chaining.

    All algorithms subclass BaseReconstructor and override the fit() method;
    see the documentation for each subclass's fit() method for documentation of
    the algorithm.

    """

    def __init__(self):
        """Constructor for reconstructor classes. These take no arguments and define
        three attributes:

        1. `self.graph`: A representation of the reconstructed network as a
        NetworkX graph (or subclass).

        2. `self.matrix`: A representation of the reconstructed network as a
        (dense) NumPy array.

        3. `self.results`: A dictionary for storing any intermediate data
        objects or diagnostics generated by a reconstruction method.

        `self.graph` and `self.matrix` should not be accessed directly by
        users; instead, use the `to_matrix()` and `to_graph()` methods.

        NOTE/TODO: should these be renamed `self._graph` and `self._matrix` to
        make this more explicit?
        """

        self.graph = None
        self.matrix = None
        self.results = {}

    def fit(self, TS):
        """The key method of the class. This takes an NxL numpy array representing a
    time series and reconstructs a network from it.

        Any new reconstruction method should subclass from BaseReconstructor
        and override fit(). This method should reconstruct the network and
        assign it to either self.graph or self.matrix, then return self.

        """
        self.matrix = np.zeros((TS.shape[0], TS.shape[0]))
        return self

    def update_matrix(self, new_mat):
        """
        Update the contents of `self.matrix`. This also empties out
        `self.graph` to avoid inconsistent state between the graph and matrix
        representations of the networks.
        """
        self.matrix = new_mat.copy()
        self.graph = None

    def update_graph(self, new_graph):
        """
        Update the contents of `self.graph`. This also empties out
        `self.matrix` to avoid inconsistent state between the graph and matrix
        representations of the networks.
        """
        self.graph = new_graph.copy()
        self.matrix = None

    def to_matrix(self):
        """Return the matrix representation of the reconstructed network."""
        if self.matrix is not None:
            return self.matrix
        elif self.graph:
            self.matrix = nx.to_numpy_array(self.graph)
            return self.matrix
        else:
            raise ValueError("Matrix and graph representations both missing. "
                             "Have you fit the data yet?")

    def to_graph(self, create_using=None):
        """Return the graph representation of the reconstructed network."""
        if self.graph:
            return self.graph
        elif self.matrix is not None:
            A = self.matrix.copy()

            if not create_using:
                if np.allclose(A, A.T):
                    G = nx.from_numpy_array(A, create_using=nx.Graph())
                else:
                    G = nx.from_numpy_array(A, create_using=nx.DiGraph())
            else:
                G = nx.from_numpy_array(A, create_using=create_using)

            self.graph = G
            return self.graph
        else:
            raise ValueError("Matrix and graph representations both missing. "
                             "Have you fit the data yet?")

    def threshold_in_range(self, c=None, **kwargs):
        """Threshold by setting values not within a list of ranges to zero.

        Parameters
        ----------
        cutoffs (list of tuples)
            When thresholding, include only edges whose correlations fall
            within a given range or set of ranges. The lower value must come
            first in each tuple. For example, to keep those values whose
            absolute value is between :math:`0.5` and :math:`1`, pass
            ``cutoffs=[(-1, -0.5), (0.5, 1)]``.
        """
        if 'cutoffs' in kwargs and not c:
            cutoffs = kwargs['cutoffs']
        elif not c:
            warnings.warn(
                "Setting 'cutoffs' argument is strongly encouraged. "
                "Using cutoff range of (-1, 1).",
                RuntimeWarning,
            )
            cutoffs = [(-1, 1)]
        else:
            cutoffs = c

        mat = self.to_matrix()
        mask_function = np.vectorize(
            lambda x: any([x >= cutoff[0] and x <= cutoff[1] for cutoff in cutoffs])
        )
        mask = mask_function(mat)

        thresholded_mat = mat * mask

        self.update_matrix(thresholded_mat)
        return self

    def threshold_on_quantile(self, q=None, **kwargs):
        """Threshold by setting values below a given quantile to zero.

        Parameters
        ----------
        quantile (float)
            The threshold above which to keep an element of the array, e.g.,
            set to zero elements below the 90th quantile of the array.

        """
        if 'quantile' in kwargs and not q:
            quantile = kwargs['quantile']
        elif not q:
            warnings.warn(
                "Setting 'quantile' argument is strongly recommended."
                "Using target quantile of 0.9 for thresholding.",
                RuntimeWarning,
            )
            quantile = 0.9
        else:
            quantile = q
        mat = self.matrix.copy()

        if quantile != 0:
            thresholded_mat = mat * (mat > np.percentile(mat, quantile * 100))
        else:
            thresholded_mat = mat

        self.update_matrix(thresholded_mat)
        return self

    def threshold_on_degree(self, k=None, **kwargs):
        """Threshold by iteratively setting the smallest entries in the weights
        matrix to zero until the average degree falls below a given value.

        Parameters
        ----------
        avg_k (float)
            The average degree to target when thresholding the matrix.

        """
        if 'avg_k' in kwargs and not k:
            avg_k = kwargs['avg_k']
        elif not k:
            warnings.warn(
                "Setting 'avg_k' argument is strongly encouraged. Using average "
                "degree of 1 for thresholding.",
                RuntimeWarning,
            )
            avg_k = 1
        else:
            avg_k = k
        mat = self.matrix.copy()

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

        self.update_matrix(thresholded_mat)
        return self

    def threshold(self, rule, **kwargs):
        """A flexible interface to other thresholding functions.

        Parameters
        ----------
        rule (str)
            A string indicating which thresholding function to invoke.

        kwargs (dict)
            Named arguments to pass to the underlying threshold function.
        """
        try:
            if rule == 'degree':
                return self.threshold_on_degree(**kwargs)
            elif rule == 'range':
                return self.threshold_in_range(**kwargs)
            elif rule == 'quantile':
                return self.threshold_on_quantile(**kwargs)
            elif rule == 'custom':
                mat = self.to_matrix()
                self.update_matrix(kwargs['custom_thresholder'](mat))
                return self

        except KeyError:
            raise ValueError("missing threshold parameter")

    def minimum_spanning_tree(self):
        MST = minimum_spanning_tree(self.to_matrix()).todense()
        self.update_matrix(MST)
        return self

    def binarize(self):
        self.update_matrix(np.abs(np.sign(self.to_matrix())))
        return self

    def remove_self_loops(self):
        mask = np.diag_indices(self.matrix.shape[0])
        new_mat = self.to_matrix().copy()
        new_mat[mask] = 0
        self.update_matrix(new_mat)
        return self

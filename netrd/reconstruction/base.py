import networkx as nx
import numpy as np
import warnings
from scipy.sparse.csgraph import minimum_spanning_tree


class BaseReconstructor:
    def __init__(self):
        self.graph = None
        self.matrix = None
        self.results = {}

    def fit(self, TS):
        self.matrix = np.zeros((TS.shape[0], TS.shape[0]))
        return self

    def update_matrix(self, new_mat):
        self.matrix = new_mat.copy()
        self.graph = None

    def update_graph(self, new_graph):
        self.graph = new_graph.copy()
        self.matrix = None

    def to_matrix(self):
        if self.matrix is not None:
            return self.matrix
        elif self.graph:
            self.matrix = nx.to_numpy_array(self.graph)
            return self.matrix
        else:
            raise ValueError("")

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
            raise ValueError("")

    def threshold_in_range(self, c=None, **kwargs):
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
        self.update_matrix(np.abs(np.sign(self.matrix)))
        return self

    def remove_self_loops(self):
        mask = np.diag_indices(self.matrix.shape[0])
        new_mat = self.matrix.copy()
        new_mat[mask] = 0
        self.update_matrix(new_mat)
        return self

import numpy as np


class BaseDynamics:
    """Base class for all dynamics processes.

    The basic usage is as follows:

    >>> ground_truth = nx.read_edgelist("ground_truth.txt")
    >>> dynamics_model = Dynamics()
    >>> synthetic_TS = dynamics_model.simulate(ground_truth, <some_params>)
    >>> # G = Reconstructor().fit(synthetic_TS)

    This produces a numpy array of time series data.

    """

    def __init__(self):
        self.results = {}

    def simulate(self, G, L):
        r"""Simulate dynamics on a ground truth network.

        The results dictionary stores the ground truth network as
        `'ground_truth'`.

        Parameters
        ----------

        G (nx.Graph)
            the input (ground-truth) graph with :math:`N` nodes.

        L (int)
            the length of the desired time series.

        Returns
        -------

        TS (np.ndarray)
            an :math`N \times L` array of synthetic time series data.

        """
        N = G.number_of_nodes()
        self.results['ground_truth'] = G
        self.results['TS'] = np.ones((N, L))
        return self.results['TS']

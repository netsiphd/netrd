class BaseReconstructionAlgorithm():
    """Base class for graph reconstruction algorithms.

    The basic usage of a graph reconstruction algorithm is as follows:

    >>> reconstructor = ReconstructionAlgorithm(<some_params>)
    >>> G = reconstructor.fit(ts_data)
    >>> # or alternately, G = reconstructor.results['graph']

    Here, `ts_data` is an $N \times T$ numpy array consisting of $T$
    observations for each of $N$ sensors. This constrains the graphs to have
    integer-valued nodes.

    The `results` dict object, in addition to containing the graph object,
    may also contain objects created as a side effect of reconstructing
    the network, which may be useful for debugging or considering goodness
    of fit. What is returned will vary between reconstruction algorithms.

    """

    def __init__(self):
        self.results = dict()
        pass

    def fit(self, ts_data):
        """

        Params
        ------
        ts_data (np.ndarray): A numpy array consisting of $T$ observations
                              from $N$ sensors.

        Returns
        -------
        G (nx.Graph): A reconstructed graph with $N$ nodes.

        """

        pass

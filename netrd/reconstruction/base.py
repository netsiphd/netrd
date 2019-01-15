class BaseReconstructor:
    """Base class for graph reconstruction algorithms.

    The basic usage of a graph reconstruction algorithm is as follows:

    >>> reconstructor = ReconstructionAlgorithm(<some_params>)
    >>> G = reconstructor.fit(T)
    >>> # or alternately, G = reconstructor.results['graph']

    Here, `TS` is an $N \times L$ numpy array consisting of $L$
    observations for each of $N$ sensors. This constrains the graphs to have
    integer-valued nodes.

    The `results` dict object, in addition to containing the graph object,
    may also contain objects created as a side effect of reconstructing
    the network, which may be useful for debugging or considering goodness
    of fit. What is returned will vary between reconstruction algorithms.

    """

    def __init__(self):
        self.results = {}

    def fit(self, TS):
        """Reconstruct a graph from time series TS.

        Params
        ------
        L (np.ndarray): Array consisting of $L$ observations from $N$ sensors.

        Returns
        -------
        G (nx.Graph): A reconstructed graph with $N$ nodes.

        """
        G = nx.Graph()  # reconstruct the graph
        self.results['graph'] = G  # and store it in self.results
        # self.results[..] = ..   # also store other values if needed
        return G

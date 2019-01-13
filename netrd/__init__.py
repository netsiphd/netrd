"""
netrd
-----

netrd stands for Network Reconstruction and Distances. It is a repository
of different algorithms for constructing a network from time series data,
as well as for comparing two networks. It is the product of the Network
Science Insitute 2019 Collabathon.

"""


class BaseReconstructor:
    """Base class for graph reconstruction algorithms.

    The basic usage of a graph reconstruction algorithm is as follows:

    >>> reconstructor = ReconstructionAlgorithm(<some_params>)
    >>> G = reconstructor.fit(T)
    >>> # or alternately, G = reconstructor.results['graph']

    Here, `T` is an $N \times T$ numpy array consisting of $T$
    observations for each of $N$ sensors. This constrains the graphs to have
    integer-valued nodes.

    The `results` dict object, in addition to containing the graph object,
    may also contain objects created as a side effect of reconstructing
    the network, which may be useful for debugging or considering goodness
    of fit. What is returned will vary between reconstruction algorithms.

    """

    def __init__(self):
        self.results = {}

    def fit(self, T):
        """Reconstruct a graph from time series T.

        Params
        ------
        T (np.ndarray): Array consisting of $T$ observations from $N$ sensors.

        Returns
        -------
        G (nx.Graph): A reconstructed graph with $N$ nodes.

        """
        G = nx.Graph()            # reconstruct the graph
        self.results['graph'] = G # and store it in self.results
        # self.results[..] = ..   # also store other values if needed
        return G


class BaseDistance:
    """Base class for all distance algorithms.

    The basic usage of a distance algorithm is as follows:

    >>> dist_obj = DistanceAlgorithm(<some_params>)
    >>> distance = dist_obj.dist(G1, G2)
    >>> # or alternatively: distance = dist_obj.results['dist']

    Here, G1 and G2 are nx.Graph objects (or subclasses such as
    nx.DiGraph). The results dictionary holds the distance value, as well
    as any other values that were computed as a side effect.

    """


    def __init__(self):
        self.results = {}

    def dist(self, G1, G2):
        """Compute distance between two graphs.

        Values computed as side effects of the distance method can be foun
        in self.results.

        Params
        ------

        G1, G2 (nx.Graph): two graphs.

        Returns
        -------

        distance (float).

        """
        dist = -1                   # compute the distance
        self.results['dist'] = dist # store dist in self.results
        # self.results[..] = ..     # also store other values if needed
        return dist                 # return only one value!

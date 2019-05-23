class BaseDistance:
    """Base class for all distance algorithms.

    The basic usage of a distance algorithm is as follows:

    >>> dist_obj = DistanceAlgorithm()
    >>> distance = dist_obj.dist(G1, G2, <some_params>)
    >>> # or alternatively: distance = dist_obj.results['dist']

    Here, `G1` and `G2` are ``nx.Graph`` objects (or subclasses such as
    ``nx.DiGraph``). The results dictionary holds the distance value, as
    well as any other values that were computed as a side effect.

    """

    def __init__(self):
        self.results = {}

    def dist(self, G1, G2):
        """Compute distance between two graphs.

        Values computed as side effects of the distance method can be foun
        in self.results.

        Parameters
        ----------

        G1, G2 (nx.Graph): two graphs.

        Returns
        -----------

        distance (float).

        """
        dist = -1  # compute the distance
        self.results['dist'] = dist  # store dist in self.results
        # self.results[..] = ..     # also store other values if needed
        return dist  # return only one value!

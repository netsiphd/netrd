"""
deltacon.py
--------------------------

Deltacon measure for graph distance, after:

Koutra, Danai, Joshua T. Vogelstein, and Christos Faloutsos. 2013. “Deltacon: A
Principled Massive-Graph Similarity Function.” In Proceedings of the 2013 SIAM
International Conference on Data Mining, 162–70. Society for Industrial and
Applied Mathematics. https://doi.org/10.1137/1.9781611972832.18.

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon.

"""

import numpy as np
import networkx as nx
from .base import BaseDistance


class DeltaCon(BaseDistance):
    """Compare matrices related to Fast Belief Propagation."""

    def dist(self, G1, G2, exact=True, g=None):
        """DeltaCon is based on the Matsusita between matrices created from fast
        belief propagation (FBP) on graphs G1 and G2.

        Because the FBP algorithm requires a costly matrix inversion, there
        is a faster, roughly linear, algorithm that gives approximate
        results.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        exact (bool)
            if True, use the slower but exact algorithm (DeltaCon_0)

        g (int)
            the number of groups to use in the efficient algorithm. If
            exact is set to False but g is not set, the efficient algorithm
            will still behave like the exact algorithm, since each node is
            put in its own group.

        Returns
        -------

        dist (float)
            the distance between G1 and G2.

        References
        ----------

        .. [1] Koutra, Danai, Joshua T. Vogelstein, and Christos
               Faloutsos. 2013. "Deltacon: A Principled Massive-Graph
               Similarity Function." In Proceedings of the 2013 SIAM
               International Conference on Data Mining, 162–70. Society for
               Industrial and Applied
               Mathematics. https://doi.org/10.1137/1.9781611972832.18.

        """
        assert G1.number_of_nodes() == G2.number_of_nodes()
        N = G1.number_of_nodes()

        if not exact and g is None:
            g = N

        A1 = nx.to_numpy_array(G1)
        L1 = nx.laplacian_matrix(G1).toarray()
        D1 = L1 + A1

        A2 = nx.to_numpy_array(G2)
        L2 = nx.laplacian_matrix(G2).toarray()
        D2 = L2 + A2

        eps_1 = 1 / (1 + np.max(D1))
        eps_2 = 1 / (1 + np.max(D2))

        if exact:
            S1 = np.linalg.inv(np.eye(N) + (eps_1 ** 2) * D1 - eps_1 * A1)
            S2 = np.linalg.inv(np.eye(N) + (eps_2 ** 2) * D2 - eps_2 * A2)
        else:
            raise NotImplementedError(
                "The efficient algorithm is not "
                "implemented. Please use the exact "
                "algorithm."
            )

        def matusita_dist(X, Y):
            r""" Return the Matusita distance

            .. math::

                \sqrt{\sum_i \sum_j \left( \sqrt{X_{ij}} - \sqrt{Y_{ij}} \right)^{2}}


            between X and Y.
            """
            return np.sqrt(np.sum(np.square(np.sqrt(X) - np.sqrt(Y))))

        dist = matusita_dist(S1, S2)

        self.results['belief_matrix_1'] = S1
        self.results['belief_matrix_2'] = S2

        self.results['dist'] = dist
        return dist

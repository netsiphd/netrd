"""
marchenko_pastur.py
--------------

Graph reconstruction algorithm based on Marchenko, V. A., & Pastur, L. A. (1967).
Distribution of eigenvalues for some sets of random matrices. Matematicheskii
Sbornik, 114(4), 507-536.

author: Matteo Chinazzi
Submitted as part of the 2019 NetSI Collabathon.
"""

from .base import BaseReconstructor
import numpy as np
import networkx as nx
from ..utilities import create_graph, threshold


class MarchenkoPastur(BaseReconstructor):
    """Uses Marchenko-Pastur law to remove noise."""

    def fit(
        self,
        TS,
        remove_largest=False,
        metric_distance=False,
        threshold_type='range',
        **kwargs
    ):
        r"""Create a correlation-based graph using Marchenko-Pastur law to remove noise.

        A signed graph is built by constructing a projection of the
        empirical correlation matrix generated from the time series data
        after having removed noisy components.  This method combines the
        results presented in [1]_, [2]_, and [3]_.

        The results dictionary also stores the weight matrix as
        `'weights_matrix'` and the thresholded version of the weight matrix
        as `'thresholded_matrix'`.

        Parameters
        ----------

        TS (np.ndarray)
            :math:`N \times L` array consisting of :math:`L` observations
            from :math:`N` sensors.

        remove_largest (bool), optional
            If ``False``, all the eigenvectors associated to the
            significant eigenvalues will be used to reconstruct the
            de-noised empirical correlation matrix. If ``True``, the
            eigenvector associated to the largest eigenvalue (normally
            known as the ``market`` mode, [2]) is going to be excluded from
            the recontruction step.  metric_distance (bool), optional: If
            ``False``, a signed graph is obtained.  The weights associated
            to the edges represent the de-noised correlation coefficient
            :math:`\rho_{i,j}` between time series :math:`i` and :math:`j`.
            If ``True``, the correlation is transformed by defining a
            metric distance between each pair of nodes where :math:`d_{i,j}
            = \sqrt{2(1-\rho_{i,j})}` as proposed in [3].  threshold_type
            (str): Which thresholding function to use on the matrix of
            weights. See `netrd.utilities.threshold.py` for
            documentation. Pass additional arguments to the thresholder
            using ``**kwargs``.

        Returns
        -------

        G (nx.Graph)
            A reconstructed graph with :math:`N` nodes.

        Examples
        --------
        .. code:: python

            import numpy as np
            import networkx as nx
            from matplotlib import pyplot as plt
            from netrd.reconstruction import MarchenkoPastur

            N = 250
            T = 300
            M = np.random.normal(size=(N,T))

            print('Create correlated time series')
            market_mode = 0.4*np.random.normal(size=(1,T))
            M += market_mode

            sector_modes = {d: 0.5*np.random.normal(size=(1,T)) for d in range(5)}
            for sector_mode, vals in sector_modes.items():
                M[sector_mode*50:(sector_mode+1)*50,:] += vals

            print('Network reconstruction step')
            mp_net = MarchenkoPastur()
            G = mp_net.fit(M, only_positive=True)
            G_no_market = mp_net.fit(M, only_positive=True, remove_largest=True)

            print('Observed noisy correlation')
            C = np.corrcoef(M)
            C[C<0] = 0 # remove negative values
            np.fill_diagonal(C,0) # remove self-loops
            G_noisy = nx.from_numpy_array(C) # create graph

            print('Plot observed noisy correlation graph')
            fig, ax = plt.subplots()
            nx.draw(G_noisy, ax=ax)

            print('Plot reconstructed correlation graph')
            fig, ax = plt.subplots()
            nx.draw(G, ax=ax)

            print('Plot reconstructed correlation graph without market mode')
            fig, ax = plt.subplots()
            nx.draw(G_no_market, ax=ax)


        References
        ----------
        .. [1] Marchenko, V. A., & Pastur, L. A. (1967). Distribution of
               eigenvalues for some sets of random
               matrices. Matematicheskii Sbornik, 114(4), 507-536.
               http://www.mathnet.ru/links/a8d2a49dec161f50c944d9a96298c35a/sm4101.pdf

        .. [2] Laloux, L., Cizeau, P., Bouchaud, J. P., & Potters,
               M. (1999). Noise dressing of financial correlation
               matrices. Physical review letters, 83(7), 1467.
               https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.83.1467

        .. [3] Bonanno, G., Caldarelli, G., Lillo, F., Micciche, S.,
               Vandewalle, N., & Mantegna, R. N. (2004). Networks of
               equities in financial markets. The European Physical Journal
               B, 38(2), 363-371.
               https://link.springer.com/article/10.1140/epjb/e2004-00129-6

        """
        N, L = TS.shape
        if N > L:
            raise ValueError("L must be greater or equal than N.")

        Q = L / N
        C = np.corrcoef(TS)  # Empirical correlation matrix

        w, v = np.linalg.eigh(C)  # Spectral decomposition of C

        w_min = 1 + 1 / Q - 2 * np.sqrt(1 / Q)
        w_max = 1 + 1 / Q + 2 * np.sqrt(1 / Q)

        selected = (w < w_min) | (w > w_max)

        if selected.sum() == 0:
            G = nx.empty_graph(n=N)
            self.results['graph'] = G
            return G

        if remove_largest:
            selected[-1] = False

        w_signal = w[selected]
        v_signal = v[:, selected]

        C_signal = v_signal.dot(np.diag(w_signal)).dot(v_signal.T)

        if metric_distance:
            C_signal = np.sqrt(2 * (1 - C_signal))

        self.results['weights_matrix'] = C_signal

        # threshold signal matrix

        self.results['thresholded_matrix'] = threshold(
            C_signal, threshold_type, **kwargs
        )

        G = create_graph(self.results['thresholded_matrix'])

        self.results['graph'] = G
        return G

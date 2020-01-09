"""
SIS.py
------

Implementation of Susceptible-Infected-Susceptible models dynamics on a
network.

author: Stefan McCabe

Submitted as part of the 2019 NetSI Collabathon.

"""

from netrd.dynamics import BaseDynamics
import numpy as np
import networkx as nx


class SISModel(BaseDynamics):
    """Susceptible-Infected-Susceptible dynamical process."""

    def simulate(self, G, L, num_seeds=1, beta=None, mu=None):
        r"""Simulate SIS model dynamics on a network.

        The results dictionary also stores the ground truth network as
        `'ground_truth'`.

        Parameters
        ----------
        G (nx.Graph)
            the input (ground-truth) graph with :math:`N` nodes.

        L (int)
            the length of the desired time series.

        num_seeds (int)
            the number of initially infected nodes.

        beta (float)
            the infection rate for the SIS process.

        mu (float)
            the recovery rate for the SIS process.

        Returns
        -------
        TS (np.ndarray)
            an :math:`N \times L` array of synthetic time series data.

        """
        H = G.copy()
        N = H.number_of_nodes()
        TS = np.zeros((N, L))
        index_to_node = dict(zip(range(G.order()), list(G.nodes())))

        # sensible defaults for beta and mu
        if not beta:
            avg_k = np.mean(list(dict(H.degree()).values()))
            beta = 1 / avg_k
        if not mu:
            mu = 1 / H.number_of_nodes()

        seeds = np.random.permutation(
            np.concatenate([np.repeat(1, num_seeds), np.repeat(0, N - num_seeds)])
        )
        TS[:, 0] = seeds
        infected_attr = {index_to_node[i]: s for i, s in enumerate(seeds)}
        nx.set_node_attributes(H, infected_attr, 'infected')
        nx.set_node_attributes(H, 0, 'next_infected')

        # SIS dynamics
        for t in range(1, L):
            nodes = np.random.permutation(H.nodes)
            for i in nodes:
                if H.nodes[i]['infected']:
                    neigh = H.neighbors(i)
                    for j in neigh:
                        if np.random.random() < beta:
                            H.nodes[j]['next_infected'] = 1
                    if np.random.random() < mu:
                        H.nodes[i]['infected'] = 0
            infections = nx.get_node_attributes(H, 'infected')
            next_infections = nx.get_node_attributes(H, 'next_infected')

            # store SIS dynamics for time t
            TS[:, t] = np.array(list(infections.values()))
            nx.set_node_attributes(H, next_infections, 'infected')
            nx.set_node_attributes(H, 0, 'next_infected')

            # if the epidemic dies off, stop
            if TS[:, t].sum() < 1:
                break

        # if the epidemic died off, pad the time series to the right shape
        if TS.shape[1] < L:
            TS = np.hstack([TS, np.zeros((N, L - TS.shape[1]))])

        self.results['ground_truth'] = H
        self.results['TS'] = TS
        self.results['index_to_node'] = index_to_node

        return TS

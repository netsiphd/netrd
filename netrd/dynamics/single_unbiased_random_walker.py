import networkx as nx
import numpy as np

class SingleUnbiasedRandomWalker(BaseDynamics):

    def __init__(self):
        self.results={}

    def simulate(self,G,L,initial_node=None):
    """
    Simulate single random-walker dynamics on a ground truth network.
    Generates an N x L time series TS; TS[j,t]==1 if the walker is at
    node j at time t, and TS[j,t]==0 otherwise.

    Example Usage:
    #######
    G = nx.ring_of_cliques(4,16)
    L = 2001
    dynamics = SingleUnbiasedRandomWalker()
    TS = dynamics.simulate(G, L)
    #######

    Params
    ------
    G (nx.Graph): the input (ground-truth) graph with $N$ nodes.
    L (int): the length of the desired time series.

    Returns
    -------
    TS (np.ndarray): an $N \times L$ array of synthetic time series data.
    """
        # get adjacency matrix and set up vector of indices
        A=nx.adjacency_matrix(G)
        N=len(A)
        W=np.zeros(L,dtype=int)
        # place walker at initial location
        if initial_node:
            W[0]=initial_node
        else:
            W[0]=np.random.randint(N)
        # run dynamical process
        for t in range(L-1):
            W[t+1]=ra.choice(np.where(A[W[t],:])[0])
        self.results['node_index_sequence']=W
        # turn into a binary-valued
        TS=np.zeros((N,L))
        for t,w in enumerate(W):
            TS[w,t]=1
        self.results['TS']=TS
        self.results['ground_truth']=G
        return TS

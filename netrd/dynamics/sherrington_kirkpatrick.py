import networkx as nx
import numpy as np

class SherringtonKirkpatrickIsing:
    """
    Kinetic Ising model from 
    D. Sherrington and S. Kirkpatrick, Phys. Rev. Lett. 35, 1792 (1975)

    Generates an N x L time series 

    """

    def __init__(self):
        self.results = {}

    def simulate(self, G, L, noisy=False):
        """
        Simulate dynamics on a ground truth network.

        Params
        ------
        G (nx.Graph): the input (ground-truth) graph with $N$ nodes.
        L (int): the length of the desired time series.

        Returns
        -------
        TS (np.ndarray): an $N \times L$ array of synthetic time series data.
        """

        N = G.number_of_nodes()
        
        # get transition probability matrix of G
        A = nx.to_numpy_array(G) 
        W = np.zeros(A.shape)
        for i in range(A.shape[0]):
            if A[i].sum()>0:
                W[i] = A[i]/A[i].sum()
        
        # initialize a time series of ones
        ts = np.ones((L, N))
        for t in range(1, L-1):
            h = np.sum(W[:,:] * ts[t,:], axis=1) # Wij from j to i
            p = 1 / ( 1 + np.exp(-2*h) )
            if noisy:
                ts[t+1,:]= p - np.random.rand(N)
            else:
                ts[t+1,:]= sign_vec(p - np.random.rand(N))


        self.results['ground_truth'] = G
        self.results['TS'] = ts.T

        return self.results['TS']

def sign(x):
    """
    np.sign(0) = 0 but here to avoid value 0, 
    we redefine it as def sign(0) = 1
    """
    return 1. if x >= 0 else -1.

def sign_vec(x):
    """
    Binarize an array
    """
    x_vec = np.vectorize(sign)
    return x_vec(x)


'''
# Example Usage

G = nx.ring_of_cliques(4,16)
L = 2001

dynamics = SherringtonKirkpatrickIsing()
TS = dynamics.simulate(G, L)

recon = netrd.reconstruction.FreeEnergyMinimizationReconstructor()
G = recon.fit(TS)

fig, (ax0,ax1) = plt.subplots(1,2,figsize=(21,10))
ax0.imshow(TS, aspect='auto', cmap='bone_r')
ax1.imshow(recon.results['matrix'],aspect='auto', cmap='bone_r')
ax0.set_title("Time Series", fontsize=20)
ax1.set_title("Reconstruction", fontsize=20)
plt.savefig('example.png', dpi=425, bbox_inches='tight')
plt.show()
'''
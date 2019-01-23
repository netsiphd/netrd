"""
convergent_cross_mapping.py
---------------------------

Graph reconstruction algorithm from time series data based on
Sugihara et al., Detecting Causality in Complex Ecosystems, Science (2012)
DOI: 10.1126/science.1227079

author: Chia-Hung Yang and Dina Mistry
Submitted as part of the 2019 NetSI Collabathon.
"""

from .base import BaseReconstructor
import numpy as np
from itertools import permutations
from scipy.stats import pearsonr
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from ..utilities.graph import create_graph


class ConvergentCrossMappingReconstructor(BaseReconstructor):
    def fit(self, TS, tau=1, alpha=0.05):
        """Infer causal relation applying Takens Theorem of dynamical systems.

        Convergent cross-mapping infers dynamical causal relation between
        vairiables from time series data. Time series data portray an attractor
        manifold of the dynamical system of interests. Existing approaches of
        attractor reconstruction involved building the shadow manifold for a
        single variable $X$, which is defined by the time-lagged vectors
        $(X(t), X(t-\tau), X(t-2\tau), ..., X(t-(N-1)\tau))$ where $N$ denotes
        number of variables in the system and $\tau$ is an arbitrary time-
        lagged interval. Takens theorem and its generalization indicated that
        if a variable $X$ is causally influencing another variable $Y$ in a
        dynamical system, then there exists a one-to-one and local-structure-
        preserving mapping from $X$'s shadow manifold to $Y$'s.

        The convergent cross-mapping algorithm first constructs the shadow data
        cloud (a portray of the shadow manifold) for every variable, and it
        then examines the mutual predictability for all pairs of them.
        Specifically, for a point $u(t)$ in $X$'s shadow data cloud, the time
        indices $\{t_1, t_2, ..., t_{N+1}\}$ of its $N+1$ nearest neighbors are
        obtained, which are mapped to a neighborhood in $Y$'s shadow data cloud
        $\{v(t_1), v(t_2), ..., v(t_{N+1})\}$. The estimate $\hat{Y}(t)$ is
        computed as an average over this neighborhood, with weights decaying
        exponentially with corresponding distance in $X$'s shadow data clould.
        The algorithm concludes a causal link from $X$ to $Y$ if correlation
        between $Y$ and $\hat{Y}$ is significant.

        Furthermore, Sugihara et al.'s simulation suggested that the
        correlation converging to 1 as the length of time series data grows is
        a necessary condition for $X$ causally affects $Y$ in a deterministic
        dynamical system. The convergent cross-mapping approach is thus
        numerically validated to infer causal relation from time series data.

        Params
        ------
        TS (np.ndarray): $N \times L$ array consisting of $L$ observations
                         from $N$ sensors.

        tau (int): Number of time steps for a single time-lag.

        alpha (float): Significance level of statistical test for correlation.

        Returns
        -------
        G (nx.Graph): A reconstructed graph with $N$ nodes.

        Notes
        -----
        1. The length of time series data must be long enough such that
           $L \geq 3 + (N-1)(1+\tau)$.

        2. The matrix of correlation between estimates and the observed values
           and the matrix of p-value for correlation test can be found in
           results['correlation'] and resutls['pvalue'] respectively.

        3. The $\[i, j\]$ of the correlation matrix entails the correlation
           between the $j$th variable and its estimate from the $i$th variable.
           A similar rule applies to the p-value matrix.

        4. The computation complexity of this implementation is $O(N^3 L)$.

        """
        data = TS.T  # Transpose the time series to make observations the rows
        L, N = data.shape

        # Raise error if there is not enough data to run the implementation
        if L < 3 + (N-1) * (1+tau):
            message = 'Need more data.'
            message += ' L must be not less than 3+(N-1)*(1+tau).'
            raise ValueError(message)

        # Create shadow data cloud for each variable
        shadows = [shadow_data_cloud(data[:, i], N, tau) for i in range(N)]

        # Obtain nearest neighbors of points in the shadow data clould and
        # their weights for time series estimates
        neighbors, distances = zip(*[nearest_neighbors(shad, L)
                                     for shad in shadows])
        weights = [neighbor_weights(dist) for dist in distances]

        # For every variable X and every other variable Y,
        # construct the estimates of Y from X's shadow data cloud and
        # compute the Pearson correlation between Y and its estimates
        # along with the p-value
        correlation = np.ones((N, N), dtype=float)
        pvalue = np.zeros((N, N), dtype=float)
        for i, j in permutations(range(N), 2):
            estimates = time_series_estimates(data[:, j],
                                              neighbors[i], weights[i])
            M, = estimates.shape
            correlation[i, j], pvalue[i, j] = pearsonr(estimates, data[-M:, j])

        # Build the reconstructed graph by finding significantly correlated
        # variables
        G = create_graph(pvalue.T < alpha, create_using=nx.DiGraph)

        # Save the graph object, matrices of correlation and p-values into the
        # "results" field (dictionary)
        self.results['graph'] = G
        self.results['correlation'] = correlation
        self.results['pvalue'] = pvalue

        return G


def shadow_data_cloud(data, N, tau):
    """Return the lagged-vector data cloud of a given variable's time series.

    Params
    ------
    data (np.ndarray): Length-$L$ 1D array of a single variable's times series.

    N (int): Number of variables.

    tau (int): Number of time steps for a single time-lag.

    Returns
    -------
    shadow (np.ndarray): $M \times N$ array of the lagged-vector data cloud,
                         where $M = L - (N-1) * \tau$ is the number of points
                         in the data cloud.

    Notes
    -----
    Given the data $(x_1(t), x_2(t), ..., x_N(t)), t = 1, 2, ..., L$, the
    $M$ lagged vectors for the $i$th variable are defined as
    $(x_i(t), x_i(t-tau), x_i(t-2*tau), ..., x_i(t-(N-1)*tau))$
    for $t = (N-1)*tau + 1, (N-1)*tau + 2, ..., L$.

    """
    L, = data.shape
    M = L - (N - 1) * tau  # Number of points in the shadow data cloud
    shadow = np.zeros((M, N), dtype=data.dtype)

    for j in reversed(range(N)):  # Fill in column values from the right
        delta = (N - 1 - j) * tau  # Amount of time-lag for this column
        shadow[:, j] = data[delta:delta+M]

    return shadow


def nearest_neighbors(shadow, L):
    """
    Return time indices of the N+1 nearest neighbors for every point in the
    shadow data cloud and their corresponding Euclidean distances.

    Params
    ------
    shadow (np.ndarray): Array of the shadow data cloud.

    L (int): Number of observations in the time series.

    Returns
    -------
    nei (np.ndarray): $M \times (N+1)$ array of time indices of nearest
                      neighbors where $M$ is the number of points in the
                      shadow data cloud.

    dist (np.ndarray): $M \times (N+1)$ array of corresponding Euclidean
                       distance between the data point to the neighbors.

    Notes
    -----
    1. The nearest neighbors are ordered in increasing order of distances.

    2. If the length of time series is at a larger scale than the number of
       variables, specifically N + 2 < M/2, the implementation finds nearest
       neighbors using the Ball Tree data structure; otherwise the brute force
       method is applied.

    """
    M, N = shadow.shape
    k = N + 2  # Number of nearest neighbors to be found, including itself
    method = 'ball_tree' if k < M/2 else 'brute'
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=method).fit(shadow)
    dist, nei = nbrs.kneighbors(shadow)

    # Remove the first column for both arrays, which stands for the data points
    # themselves
    nei = np.delete(nei, 0, axis=1)
    dist = np.delete(dist, 0, axis=1)

    # Modify the neighbors' indices to their time indices
    nei += (L - M)

    return nei, dist


def neighbor_weights(dist):
    """Return the weights of neighbors in time seires estimates.

    Params
    ------
    dist (np.ndarray): $M \times (N+1)$ array of Euclidean distances between a
                       point to its nearest neighbors in the shadow data cloud
                       (sorted by increasing order of distances), where $M$ is
                       the number of points in the shadow data cloud.

    Returns
    -------
    wei (np.ndarray): $M \times (N+1)$ array of exponentially decaying weights
                      of the nearest neighbors.

    Notes
    -----
    For every point $u(t)$ in the shadow data cloud, based on its Euclidean
    distance to the nearest neighbors $d(u(t), u(t_k))$, $l = 1, 2, ..., N+1$
    (sorted by increasing order of distances), the weight of neighbor $k$ is
    $w_k = f_k / \sum_{l=1}^{N+1} f_l$ where
    $f_k = e^{-\[ d(u(t), u(t_k)) / d(u(t), u(t_1)) \]}$.

    """
    expn = dist / dist[:, 0, np.newaxis]
    wei = np.exp(-expn)
    wei = wei / wei.sum(axis=1, keepdims=True)

    return wei


def time_series_estimates(data_y, nei_x, wei_x):
    """Return estimates of variable $Y$ from variable $X$'s shadow data cloud.

    Params
    ------
    data_y (np.ndarray): 1D array of variable $Y$'s time series data.

    nei_x (np.ndarray): $M \times (N+1)$ array of time indices of nearest
                        neighbors in $X$'s shadow data cloud, where $M$ is the
                        number of points in the shadow data cloud.

    wei_x (np.ndarray): Array of corresponding weights of the nearest
                        neighbors in $X$'s shadow data cloud.

    Returns
    -------
    ests (np.ndarray): Length-$M$ 1D array of estimates of $Y$'s time series.

    Notes
    -----
    Let $t_1, t_2, ..., t_{N+1}$ be the time indices of nearest neighbor of
    point $t$ in $X$'s shadow data cloud. Its corresponding estimates of $Y$
    is $\hat{Y}(t) = \sum_{k=1}^{N+1} w(t_k) Y(t_k)$, where $w$s are weights of
    nearest neighbors.

    """
    ests = (data_y[nei_x] * wei_x).sum(axis=1)

    return ests

[![ReadTheDocs](https://img.shields.io/readthedocs/netrd.svg)](
    https://netrd.readthedocs.io)
[![Travis](https://img.shields.io/travis/netsiphd/netrd.svg)](
    https://travis-ci.org/netsiphd/netrd)

# `netrd`: A library for network {reconstruction, distances, dynamics}

NOTE: This library is pre-alpha. **Use at your own risk.**

This library provides a consistent, NetworkX-based interface to various
utilities for graph distances, graph reconstruction from time series data,
and simulated dynamics on networks. For the API reference visit
[this link](https://netrd.readthedocs.io/en/latest/).

To see the library in action, visit the [netrd
explorer](https://netrdexplorer.herokuapp.com/).

# Installation

```
git clone https://github.com/netsiphd/netrd
cd netrd
pip install .
```

Aside from NetworkX and the Python scientific computing stack, this library also
has dependencies on Cython and [POT](https://github.com/rflamary/POT).

# Usage

## Reconstructing a graph

All reconstruction algorithms provide a simple interface. First, initialize the
reconstructor object by calling its constructor with no arguments. Then, use the
`fit()` method to obtain the reconstructed network.

```python
TS = np.loadtxt('data/synth_4clique_N64_simple.csv',
                delimiter=',',
                encoding='utf8')
# TS is a NumPy array of shape N (number of nodes) x L (observations).

recon = netrd.reconstruction.RandomReconstructor()
G = recon.fit(TS)
```

Many reconstruction algorithms store additional metadata in a `results`
dictionary. 

```python
# Another way to obtain the reconstructed graph
G = recon.results['graph']

# A dense matrix of weights
W = recon.results['weights_matrix']

# The binarized matrix from which the graph is created
A = recon.results['thresholded_matrix']
```

Many, though not all, reconstruction algorithms work by assigning each potential
edge a weight and then thresholding the matrix to obtain a sparse
representation. This thresholding can be controlled by setting the
`threshold_type` argument to one of four values:

* `range`: Consider only weights whose values fall within a range.
* `degree`: Consider only the largest weights, targeting a specific average
  degree.
* `quantile`: Consider only weights in, e.g., the 0.90 quantile and above.
* `custom`: Pass a custom function for thresholding the matrix yourself.

Each of these has a specific argument to pass to tune the thresholding:

* `cutoffs`: A list of 2-tuples specifying the values to keep. For example, to
  keep only values whose absolute values are above 0.5, use `cutoffs=[(-np.inf,
  -0.5), (0.5, np.inf)]`
* `avg_k`: The desired average degree of the network.
* `quantile`: The appropriate quantile (not percentile).
* `custom_thresholder`: A user-defined function that returns an N x N NumPy
  array.

```python
H = recon.fit(TS, threshold_type='degree', avg_k = 15.125)


print(nx.info(G))
# This network is a complete graph.

print(nx.info(H))
# This network is not.
```

## Distances between graphs

Distances behave similarly to reconstructors. All distance objects have a
`dist()` method that takes two NetworkX graphs.

```python
G1 = nx.fast_gnp_random_graph(1000, 0.1)
G2 = nx.fast_gnp_random_graph(1000, 0.1)

dist = netrd.distance.NetSimile()
D = dist.dist(G1, G2)
```

Some distances also store metadata in `results` dictionaries.

```python
# Another way to get the distance
D = dist.results['dist']

# The underlying features used in NetSimile
vecs = dist.results['signature_vectors']
```

## Dynamics on graphs

As a utility, we also implement various ways to simulate dynamics on a network.
These have a similar interface to reconstructors and distances. Their
`simulate()` method takes an input graph and the desired length of the dynamics,
returning the same N x L array used in the graph reconstruction methods.

```python
model = netrd.dynamics.VoterModel()
TS = model.simulate(G, 1000, noise=.001)

# Another way to get the dynamics
TS = model.results['TS']

# The original graph is stored in results
H = model.results['ground_truth']
```

# Contributing

Contributing guidelines can be found in
[CONTRIBUTING.md](CONTRIBUTING.md).

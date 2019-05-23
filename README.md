[![ReadTheDocs](https://img.shields.io/readthedocs/netrd.svg)](
    https://netrd.readthedocs.io)
[![Travis](https://img.shields.io/travis/netsiphd/netrd.svg)](
    https://travis-ci.org/netsiphd/netrd)

# `netrd`: A library for network {reconstruction, distances, dynamics}

This library provides a consistent, NetworkX-based interface to various
utilities for graph distances, graph reconstruction from time series data, and
simulated dynamics on networks. 

Some resources that maybe of interest:

* An interactive demonstration: [netrd
  explorer](https://netrdexplorer.herokuapp.com)
* A [tutorial](https://netrd.readthedocs.io/en/latest/tutorial.html) on how to use the library
* The API [reference](https://netrd.readthedocs.io/en/latest/) 
* A [notebook](https://nbviewer.jupyter.org/github/netsiphd/netrd/blob/master/notebooks/00%20-%20netrd_introduction.ipynb) showing advanced usage

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

The basic usage of a graph reconstruction algorithm is as follows:

```
>>> reconstructor = ReconstructionAlgorithm()
>>> G = reconstructor.fit(TS, <some_params>)
>>> # or alternately, G = reconstructor.results['graph']
```

Here, `TS` is an N x L numpy array consisting of L
observations for each of N sensors. This constrains the graphs
to have integer-valued nodes.

The `results` dict object, in addition to containing the graph
object, may also contain objects created as a side effect of
reconstructing the network, which may be useful for debugging or
considering goodness of fit. What is returned will vary between
reconstruction algorithms.

## Distances between graphs

The basic usage of a distance algorithm is as follows:

```
>>> dist_obj = DistanceAlgorithm()
>>> distance = dist_obj.dist(G1, G2, <some_params>)
>>> # or alternatively: distance = dist_obj.results['dist']
```

Here, `G1` and `G2` are `nx.Graph` objects (or subclasses such as
`nx.DiGraph`). The results dictionary holds the distance value, as
well as any other values that were computed as a side effect.

## Dynamics on graphs

The basic usage of a dynamics algorithm is as follows:

```
>>> ground_truth = nx.read_edgelist("ground_truth.txt")
>>> dynamics_model = Dynamics()
>>> synthetic_TS = dynamics_model.simulate(ground_truth, <some_params>)
>>> # G = Reconstructor().fit(synthetic_TS)
```

This produces a numpy array of time series data.


# Contributing

Contributing guidelines can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

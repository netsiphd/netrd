[![PyPI version](https://badge.fury.io/py/netrd.svg)](https://badge.fury.io/py/netrd)
[![ReadTheDocs](https://img.shields.io/readthedocs/netrd.svg)](
    https://netrd.readthedocs.io)
![CI](https://github.com/netsiphd/netrd/workflows/build/badge.svg)

# netrd: A library for network {reconstruction, distances, dynamics}

This library provides a consistent, NetworkX-based interface to various
utilities for graph distances, graph reconstruction from time series data, and
simulated dynamics on networks. 

Some resources that maybe of interest:

* A [tutorial](https://netrd.readthedocs.io/en/latest/tutorial.html) on how to use the library
* The API [reference](https://netrd.readthedocs.io/en/latest/) 
* A [notebook](https://nbviewer.jupyter.org/github/netsiphd/netrd/blob/master/notebooks/example.ipynb) showing advanced usage

# Installation

`netrd` is easy to install through pip:

```
pip install netrd
```

If you are thinking about contributing to `netrd`, you can install a
development version by executing

```
git clone https://github.com/netsiphd/netrd
cd netrd
pip install .
```

# Usage

## Reconstructing a graph

<p align="center">
<img src="netrd_reconstruction_example.png" alt="example reconstruction" width="95%"/>
</p>

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

<p align="center">
<img src="netrd_distance_example.png" alt="example distance" width="95%"/>
</p>

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

<p align="center">
<img src="netrd_dynamics_example.png" alt="example distance" width="95%"/>
</p>

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


# Publications

* Hartle H., Klein B., McCabe S., Daniels A., St-Onge G., Murphy C., and
Hébert-Dufresne L. (2020). Network comparison and the within-ensemble graph
distance. *Proc. R. Soc. A* 20190744.
doi: [10.1098/rspa.2019.0744](http://dx.doi.org/10.1098/rspa.2019.0744).
    + recent work introducing a baseline measure for comparing graph distances


---
title: 'netrd: A library for network reconstruction and graph distances'
tags:
  - Python
  - network science
  - network reconstruction
  - graph distance
  - network dynamics
authors:
  - name: Stefan McCabe
    orcid: 0000-0002-7180-145X
    affiliation: 1
  - name: Leo Torres
    orcid: 0000-0002-2675-2775
    affiliation: 1
  - name: Timothy LaRock
    orcid: 0000-0003-0801-3917
    affiliation: 1
  - name: Syed Arefinul Haque
    orcid: 0000-0002-8371-2366
    affiliation: 1
  - name: Chia-Hung Yang
    orcid: 0000-0002-4936-808X
    affiliation: 1
  - name: Harrison Hartle
    orcid: 0000-0002-0917-6112
    affiliation: 1
  - name: Brennan Klein
    orcid: 0000-0001-8326-5044
    affiliation: "1, 2"
affiliations:
 - name: Network Science Institute, Northeastern University, Boston, MA, USA
   index: 1
 - name: Laboratory for the Modeling of Biological and Socio-Technical Systems, Northeastern University, Boston, USA
   index: 2
date: 29 October 2020
bibliography: paper.bib
---

# Statement of need

Complex systems throughout nature and society are often best represented as *networks*. Over the last two decades, alongside the increased availability of large network datasets, we have witnessed the rapid rise of network science [@Amaral2004; @Vespignani2008; @Newman2010; @Barabasi2016]. This field is built around the idea that an increased understanding of the complex structural properties of a variety systems will allow us to better observe, predict, and even control the behavior of these systems.

However, for many systems, the data we have access to is not a direct description of the underlying network. More and more, we see the drive to study networks that have been inferred or reconstructed from non-network data---in particular, using *time series* data from the nodes in a system to infer likely connections between them [@Brugere2018; @Runge2018]. Selecting the most appropriate technique for this task is a challenging problem in network science. Different reconstruction techniques usually have different assumptions, and their performance varies from system to system in the real world. One way around this problem could be to use several different reconstruction techniques and compare the resulting networks. However, network comparison is also not an easy problem, as it is not obvious how best to quantify the differences between two networks, in part because of the diversity of tools for doing so.

The `netrd` Python package seeks to address these two parallel problems in network science.

# Summary

`netrd` offers, to our knowledge, the most extensive collection of both network reconstruction techniques and network comparison techniques (often referred to as *graph distances*) in a single library. Below, we expand on these two main functionalities of the `netrd` package.

The first core use of `netrd` is to reconstruct networks from time series data. Given time series data, $TS$, of the behavior of $N$ nodes / components / sensors of a system over the course of $L$ timesteps, and given the assumption that the behavior of every node, $v_i$, may have been influenced by the past behavior of other nodes, $v_j$, there are dozens of techniques that can be used to infer which connections, $e_{ij}$, are likely to exist between the nodes. That is, we can use one of many *network reconstruction* techniques to create a network representation, $G_r$, that attempts to best capture the relationships between the time series of every node in $TS$. `netrd` lets users perform this network reconstruction task using 17 different techniques. This means that up to 17 different networks can formed created from a single time series dataset. For example, in \autoref{fig:ground} we show the outputs of 15 different reconstruction techniques applied to time series data generated from an example network [@Sugihara2012; @Mishchenko2011; @Hoang2019; @Sheikhattar2018; @Friedman2008; @Edelman2005; @Zeng2013; @Donges2009; @Barucca2014; @Ledoit2003; @Stetter2012; @Peixoto2019].

Practitioners often apply these network reconstruction algorithms to real time series data. For example, in neuroscience, researchers often try to reconstruct functional networks from time series readouts of neural activity [@Mishchenko2011]. In economics, researchers can infer networks of influence between companies based on time series of changes in companies' stock prices [@Squartini2018]. At the same time, it is often quite helpful having the freedom to *simulate* arbitrary time series dynamics on randomly generated networks. This provides a controlled setting to assess the performance of network reconstruction algorithms. For this reason, the `netrd` package also includes a number of different techniques for simulating dynamics on networks.

The second core use of `netrd` addresses a common goal when studying networks: describing and quantifying the differences between two networks. This is a challenging problem, as there are countless axes upon which two networks can differ; as such, a number of *graph distance* measures have emerged over the years attempting to address this problem. As is the case for many hard problems in network science, it can be difficult to know which (of many) measures are suited for a given setting. In `netrd`, we consolidate over 20 different graph distance measures into a single package [@Jaccard1901; @Hamming1950; @Jurman2015; @Golub2013; @Donnat2018; @Carpi2011; @Bagrow2019; @DeDomenico2016; @Chen2018; @Hammond2013; @Monnig2018; @Tsitsulin2018; @Jurman2011; @Ipsen2002; @Torres2019; @Mellor2019; @Schieber2017; @Koutra2016; @Berlingerio2012]. \autoref{fig:dists} shows an example of just how different these measures can be when comparing two networks, $G_1$ and $G_2$. This submodule in `netrd` has already been used in recent work with a novel characterization of the graph distance literature [@Hartle2020].

This package builds on commonly used Python packages (e.g. `networkx` [@SciPyProceedings11], `numpy` [@Harris2020], `scipy` [@SciPy2020]) and is already a widely used resource for network scientists and other multidisciplinary researchers. With ongoing open-source development, we see this as a tool that will continue to be used by all sorts of researchers to come.

# Related software packages

In the network reconstruction literature, there are often software repositories that detail a single technique or a few related ones. For example Lizier (2014) implemented a Java package (portable to Python, octave, R, Julia, Clojure, MATLAB) that uses information-theoretic approaches for inferring network structure from time-series data [@Lizier2014]; Runge et al. (2019) created a Python package that combines linear or nonlinear conditional independence tests with a causal discovery algorithm to reconstruct causal networks from large-scale time series datasets [@Runge2019]. These are two examples of powerful and widely used packages though neither includes as wide-ranging techniques as `netrd` (nor were they explicitly designed to). In the graph distance literature, the same trend is broadly true: many one-off software repositories exist for specific measures. However, there are some packages that do include multiple graph distances; for example, Wills (2017) created a `NetComp` package that includes several variants of a few distance measures included here [@Wills2017].


# Figures
![**Example of the network reconstruction pipeline.** (Top row) A sample network, its adjacency matrix, and an example time series, $TS$, of node-level activity simulated on the network. (Bottom rows) The outputs of 15 different network reconstruction algorithms, each using $TS$ to create a new adjacency matrix that captures key structural properties of the original network.\label{fig:ground}](allRecons_withGroundtruth_SherringtonKirkpatrick.pdf)

![**Example of the graph distance measures in `netrd`.** Here, we measure the graph distance between two networks using 20 different distance measures from `netrd`.\label{fig:dists}](netrd_distance_example.pdf)

# Acknowledgements

The authors thank Kathryn Coronges, Mark Giannini, and Alessandro Vespignani for contributing to the coordination of the 2019 Network Science Institute "Collabathon", where much of the development of this package began. The authors acknowledge the support of ten other contributors to this package: Guillaume St-Onge, Andrew Mellor, Charles Murphy, David Saffo, Carolina Mattsson, Ryan Gallagher, Matteo Chinazzi, Jessica Davis, Alexander J. Gates, and Anton Tsitulin. **Funding:** This research was supported by the Network Science Institute at Northeastern University.

# References
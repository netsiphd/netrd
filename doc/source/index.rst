``netrd``: A library for network {reconstruction, distances, dynamics}
======================================================================

This library provides a consistent, NetworkX-based interface to various
utilities for graph distances, graph reconstruction from time series
data, and simulated dynamics on networks.

To see the library in action, visit the `netrd
explorer <https://netrdexplorer.herokuapp.com/>`__.

Installation
============

::

   git clone https://github.com/netsiphd/netrd
   cd netrd
   pip install .

Aside from NetworkX and the Python scientific computing stack, this
library also has dependencies on Cython and
`POT <https://github.com/rflamary/POT>`__.

Tutorial
========

A tutorial on using the library can be found `here <tutorial.html>`__. To see
more advanced usage of the library, refer to `this
notebook <https://nbviewer.jupyter.org/github/netsiphd/netrd/blob/master/notebooks/00%20-%20netrd_introduction.ipynb>`__.

Contributing
============

Contributing guidelines can be found in
`CONTRIBUTING.md <CONTRIBUTING.md>`__.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   tutorial
   dynamics
   distance
   reconstruction
   utilities


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

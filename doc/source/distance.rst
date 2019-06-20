Distance
========

Graph distance methods to compare two networks.


Base class
----------
.. autoclass:: netrd.distance.BaseDistance


Available distances
-------------------

All of the following algorithms inherit from ``BaseDistance`` and have the
same general usage as above.

.. autosummary::
   :nosignatures:

    netrd.distance.CommunicabilityJSD
    netrd.distance.DegreeDivergence
    netrd.distance.DeltaCon
    netrd.distance.DistributionalNBD
    netrd.distance.Frobenius
    netrd.distance.Hamming
    netrd.distance.HammingIpsenMikhailov
    netrd.distance.IpsenMikhailov
    netrd.distance.JaccardDistance
    netrd.distance.LaplacianSpectral
    netrd.distance.NonBacktrackingSpectral
    netrd.distance.NetLSD
    netrd.distance.NetSimile
    netrd.distance.OnionDivergence
    netrd.distance.PolynomialDissimilarity
    netrd.distance.PortraitDivergence
    netrd.distance.QuantumJSD
    netrd.distance.ResistancePerturbation


Reference
---------

.. automodule:: netrd.distance
    :members:
    :undoc-members:

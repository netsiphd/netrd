Dynamics
========

Dynamics classes allow the user to run simulations over a network.


Base class
----------
.. autoclass:: netrd.dynamics.BaseDynamics


Available dynamics
------------------

All of the following dynamics inherit from ``BaseDynamics`` and have the
same general usage as above.

.. autosummary::
   :nosignatures:

    netrd.dynamics.BranchingModel
    netrd.dynamics.IsingGlauber
    netrd.dynamics.Kuramoto
    netrd.dynamics.LotkaVolterra
    netrd.dynamics.SISModel
    netrd.dynamics.SherringtonKirkpatrickIsing
    netrd.dynamics.SingleUnbiasedRandomWalker
    netrd.dynamics.VoterModel


Reference
---------

.. automodule:: netrd.dynamics
    :members:
    :undoc-members:

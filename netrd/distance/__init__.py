from .base import BaseDistance
from .hamming import Hamming
from .frobenius import Frobenius
from .portrait_divergence import PortraitDivergence
from .jaccard_distance import JaccardDistance
from .ipsen_mikhailov import IpsenMikhailov
from .hamming_ipsen_mikhailov import HammingIpsenMikhailov
from .resistance_perturbation import ResistancePerturbation
from .netsimile import NetSimile
from .netlsd import NetLSD
from .laplacian_spectral_method import LaplacianSpectral
from .polynomial_dissimilarity import PolynomialDissimilarity
from .degree_divergence import DegreeDivergence
from .onion_divergence import OnionDivergence
from .deltacon import DeltaCon
from .quantum_jsd import QuantumJSD
from .communicability_jsd import CommunicabilityJSD
from .distributional_nbd import DistributionalNBD
from .dk_series import dkSeries
from .dmeasure import DMeasure

nbd = False
try:
    from .nbd import NonBacktrackingSpectral

    nbd = True
except ImportError as e:
    pass


__all__ = [
    'Hamming',
    'Frobenius',
    'PortraitDivergence',
    'JaccardDistance',
    'IpsenMikhailov',
    'HammingIpsenMikhailov',
    'ResistancePerturbation',
    'NetSimile',
    'NetLSD',
    'LaplacianSpectral',
    'PolynomialDissimilarity',
    'DegreeDivergence',
    'OnionDivergence',
    'DeltaCon',
    'QuantumJSD',
    'CommunicabilityJSD',
    'DistributionalNBD',
    'dkSeries',
    'DMeasure',
]

if nbd:
    __all__ += ['NonBacktrackingSpectral']

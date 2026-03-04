"""CFIFS — Choquet Fuzzy-Integral Feature Selection for Multi-Label Data.

Public API
----------
- ``CFIFSParams``  — frozen dataclass with all embedded-solver hyper-parameters.
- ``fit_cfifs(X, Y, params)`` — compute a feature ranking via the embedded stage.

For the spectral channel, import from ``mlfs.spectral_mlfs``.
For ML-kNN evaluation, import from ``mlfs.ml_knn_gpu``.
"""

from .cfifs_embedded import CFIFSParams, fit_cfifs

__all__ = [
    "CFIFSParams",
    "fit_cfifs",
]

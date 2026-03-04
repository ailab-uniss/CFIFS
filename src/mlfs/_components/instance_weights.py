"""Rarity-aware instance weighting for multi-label learning.

Instances whose active labels are rare receive higher weight,
so that the embedded solver does not neglect tail-label patterns.

Public API
----------
- ``RarityInstanceWeightParams`` — frozen dataclass with weighting knobs.
- ``rarity_instance_weights(Y, ...)`` — compute per-instance weights.
- ``rarity_prior_from_Y(Y, ...)`` — compute the label rarity prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RarityInstanceWeightParams:
    """
    Rarity-aware instance weighting.

    Weight of instance i:
      s_i = 1 + kappa * <y_i, p>,
    where p is a rarity prior over labels:
      p_l ∝ (f_l + 1)^(-gamma),
    and f_l is the number of positives for label l in the training fold.
    """

    gamma: float = 2.0
    kappa: float = 1.5
    s_max: float = 3.0


def rarity_prior_from_Y(Y01: np.ndarray, *, gamma: float, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Compute label-frequency vector and rarity prior from a binary label matrix.

    Scope
    -----
    Measures how rare each label is and builds a normalised prior
    vector used downstream by ``rarity_instance_weights``.

    Parameters
    ----------
    Y01 : ndarray of shape (n, L)
        Binary label matrix {0, 1}.
    gamma : float
        Inverse-frequency exponent.  Larger γ → stronger emphasis on rare labels.
    eps : float
        Numerical stabiliser for the normalisation denominator.

    Preconditions
    -------------
    * *Y01* is non-empty with at least one positive per column (soft).

    Postconditions
    --------------
    * ``freq`` is shape (L,), each entry ≥ 1 (Laplace-smoothed).
    * ``prior`` is shape (L,), sums to ≈ 1.

    Returns
    -------
    freq  : ndarray (L,)  — positive counts + 1.
    prior : ndarray (L,)  — normalised rarity prior.
    """
    Y01 = (np.asarray(Y01) > 0).astype(np.float64, copy=False)
    freq = np.sum(Y01, axis=0).astype(np.float64, copy=False) + 1.0
    prior = freq ** (-float(gamma))
    prior = prior / (float(np.sum(prior)) + float(eps))
    return freq, prior


def rarity_instance_weights(
    Y01: np.ndarray,
    *,
    gamma: float,
    kappa: float,
    s_max: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-instance rarity-aware weights.

    Scope
    -----
    Each training instance *i* is assigned a weight
        s_i = 1 + κ · ⟨y_i, p⟩
    where *p* is the rarity prior from ``rarity_prior_from_Y``.
    Weights are clipped to *s_max* and mean-normalised to 1.

    Parameters
    ----------
    Y01 : ndarray of shape (n, L)
        Binary label matrix {0, 1}.
    gamma : float
        Rarity exponent (forwarded to ``rarity_prior_from_Y``).
    kappa : float
        Scaling factor for the rarity contribution.
    s_max : float
        Upper clamp on individual weights (prevents outlier
        dominance).
    eps : float
        Numerical stabiliser.

    Preconditions
    -------------
    * *Y01* is binary and non-empty.
    * γ ≥ 0, κ ≥ 0, s_max > 0.

    Postconditions
    --------------
    * ``s``     : shape (n,), mean ≈ 1, all entries > 0.
    * ``sw``    : shape (n,), element-wise √s.
    * ``prior`` : shape (L,), same as ``rarity_prior_from_Y``.

    Returns
    -------
    s     : ndarray (n,)  — instance weights.
    sw    : ndarray (n,)  — √s.
    prior : ndarray (L,)  — label rarity prior.
    """
    _freq, prior = rarity_prior_from_Y(Y01, gamma=float(gamma), eps=float(eps))
    Y01 = (np.asarray(Y01) > 0).astype(np.float64, copy=False)
    s = 1.0 + float(kappa) * (Y01 @ prior)
    s = np.minimum(s, float(s_max))
    s = s / (float(np.mean(s)) + float(eps))
    sw = np.sqrt(s)
    return s.astype(np.float64, copy=False), sw.astype(np.float64, copy=False), prior.astype(np.float64, copy=False)


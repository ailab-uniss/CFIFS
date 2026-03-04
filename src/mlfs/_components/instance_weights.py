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
    """
    Compute (freq, prior) from a binary multi-label matrix Y in {0,1}.

    freq: (L,) positive counts + 1 (Laplace smoothing, to avoid zeros)
    prior: (L,) normalized rarity prior
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
    """
    Compute per-instance weights s >= 0 and return (s, sqrt(s), prior).
    """
    _freq, prior = rarity_prior_from_Y(Y01, gamma=float(gamma), eps=float(eps))
    Y01 = (np.asarray(Y01) > 0).astype(np.float64, copy=False)
    s = 1.0 + float(kappa) * (Y01 @ prior)
    s = np.minimum(s, float(s_max))
    s = s / (float(np.mean(s)) + float(eps))
    sw = np.sqrt(s)
    return s.astype(np.float64, copy=False), sw.astype(np.float64, copy=False), prior.astype(np.float64, copy=False)


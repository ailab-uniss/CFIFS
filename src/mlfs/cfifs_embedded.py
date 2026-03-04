"""
CFIFS — Embedded scoring stage
===============================

This module implements the **embedded** scoring channel of CFIFS (Choquet
Fuzzy-Integral Feature Selection).  Given a training fold (X, Y) it learns a
group-sparse projection matrix W ∈ ℝ^{d×r} via accelerated proximal gradient
descent (FISTA) on a convex objective that combines:

    (1−α)/2 ‖D_s(XW − T)‖²_F          — squared-reconstruction loss
    + α  Σ_i s_i Σ_l CE(y_il, σ(…))    — per-label logistic loss
    + β ‖W‖_{2,1}                       — group-lasso (feature selection)
    + ρ ‖W‖²_F                          — ridge stabiliser

where T = (Y ⊙ w_lab) V is a PLST-style label embedding target and s_i are
rarity-aware instance weights that up-weight minority patterns.

Feature importance is measured by the row norms ‖W_{j:}‖₂; features with
larger norms are ranked higher.

Both a **NumPy** (CPU) and a **PyTorch** (GPU) backend are provided.  The
PyTorch backend is recommended for datasets with d > 500 or n > 2000.

Public API
----------
- ``CFIFSParams``  — frozen dataclass with all solver hyper-parameters.
- ``fit_cfifs(X, Y, params)`` — run the solver; returns (ranking, info).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from ._components.instance_weights import (
    RarityInstanceWeightParams,
    rarity_instance_weights,
)


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _power_iteration_xtx_eigmax(
    X: np.ndarray, *, n_iter: int = 25, seed: int = 0,
) -> float:
    """Estimate largest eigenvalue of X^T X via power iteration."""
    Xop = np.asarray(X, dtype=np.float64)
    d = int(Xop.shape[1])
    rng = np.random.default_rng(int(seed))
    v = rng.standard_normal((d,), dtype=np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)
    for _ in range(int(n_iter)):
        u = Xop @ v
        nu = np.linalg.norm(u)
        if nu <= 1e-12:
            return 0.0
        u = u / nu
        v = Xop.T @ u
        nv = np.linalg.norm(v)
        if nv <= 1e-12:
            return 0.0
        v = v / nv
    Xv = Xop @ v
    lam = float(np.dot(Xv, Xv)) / float(np.dot(v, v) + 1e-12)
    return float(max(lam, 0.0))


def _prox_group_lasso_rows(
    W: np.ndarray, lam: float, *, eps: float = 1e-12,
) -> np.ndarray:
    """Row-wise block soft-thresholding (group-lasso proximal operator)."""
    W = np.asarray(W, dtype=np.float64)
    norms = np.linalg.norm(W, axis=1)
    scale = np.maximum(0.0, 1.0 - float(lam) / (norms + float(eps)))
    return W * scale[:, None]


# ---------------------------------------------------------------------------
# Label embedding (PLST-style)
# ---------------------------------------------------------------------------

def _label_tail_weights(
    Y01: np.ndarray, *, label_gamma: float, eps: float,
) -> np.ndarray:
    r"""Inverse-frequency tail weighting: w_l \propto (freq_l + 1)^{-\gamma/2}."""
    Y01 = (np.asarray(Y01) > 0).astype(np.float64, copy=False)
    freq = np.sum(Y01, axis=0).astype(np.float64) + 1.0
    w_lab = (freq ** (-0.5 * float(label_gamma))).astype(np.float64, copy=False)
    w_lab = w_lab / (float(np.mean(w_lab)) + float(eps))
    return w_lab


def _label_embedding_from_Y(
    Y01: np.ndarray,
    *,
    rank: int,
    label_gamma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PLST-style label embedding via SVD of the tail-weighted label matrix.

    Returns
    -------
    V      : (L, r) orthonormal columns.
    w_lab  : (L,)   tail weights.
    svals  : (r,)   singular values.
    """
    Y01 = (np.asarray(Y01) > 0).astype(np.float64, copy=False)
    n, L = int(Y01.shape[0]), int(Y01.shape[1])
    r = int(max(1, min(int(rank), L, n)))
    eps = np.finfo(np.float64).eps

    w_lab = _label_tail_weights(
        Y01, label_gamma=float(label_gamma), eps=float(eps),
    )

    Yw = Y01 * w_lab[None, :]
    try:
        _U, S, Vt = np.linalg.svd(Yw, full_matrices=False)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(int(seed))
        Yw = Yw + 1e-10 * rng.standard_normal(Yw.shape)
        _U, S, Vt = np.linalg.svd(Yw, full_matrices=False)

    V = Vt[:r, :].T  # (L x r)
    V, _ = np.linalg.qr(V)
    svals = np.asarray(S[:r], dtype=np.float64)
    return (
        V[:, :r].astype(np.float64, copy=False),
        w_lab.astype(np.float64, copy=False),
        svals.astype(np.float64, copy=False),
    )


# ---------------------------------------------------------------------------
# Parameters dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CFIFSParams:
    r"""Configuration for the CFIFS embedded scoring stage.

    The solver minimises a convex objective over a group-sparse feature matrix
    W \in R^{d x r} with a PLST-style label embedding V \in R^{L x r}:

        min_{W, b}  (1-alpha)/2 ||D_s(XW - T)||^2_F
                   + alpha  sum_i s_i sum_l CE(y_il, sigma(x_i^T W v_l + b_l))
                   + beta ||W||_{2,1}  +  rho ||W||^2_F

    where T = (Y . w_lab) V and s_i are rarity-aware instance weights.
    Feature importance is measured by the row norms ||W_{j:}||_2.
    """
    rank: int = 50
    label_gamma: float = 0.0

    alpha: float = 0.35
    max_iter: int = 120
    tol: float = 1e-6
    beta: float = 0.01
    rho: float = 1e-4

    use_instance_weights: bool = True
    inst: RarityInstanceWeightParams = RarityInstanceWeightParams()

    seed: int = 0
    power_iter: int = 25

    # Compute backend
    backend: str = "numpy"       # "numpy" | "torch"
    device: str = "cpu"          # "cpu" | "cuda" | "cuda:0" ...
    torch_dtype: str = "float32"
    allow_tf32: bool = True


# ---------------------------------------------------------------------------
# NumPy solver
# ---------------------------------------------------------------------------

def fit_cfifs(
    X: np.ndarray,
    Y: np.ndarray,
    params: CFIFSParams | Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run the CFIFS embedded stage and return ``(ranking, info)``.

    Parameters
    ----------
    X : (n, d) training features.
    Y : (n, L) training labels in {0, 1} or {-1, +1}.
    params : solver configuration (defaults are fine for paper reproduction).

    Returns
    -------
    ranking : (d,) 1-based feature indices sorted best -> worst.
    info    : diagnostic dict including ``"scores"`` (per-feature importances).
    """
    if params is None:
        p = CFIFSParams()
    elif isinstance(params, CFIFSParams):
        p = params
    else:
        cfg = dict(params)
        if "inst" in cfg and isinstance(cfg["inst"], dict):
            cfg["inst"] = RarityInstanceWeightParams(**cfg["inst"])
        p = CFIFSParams(**cfg)

    if str(p.backend).lower().strip() == "torch":
        return _fit_cfifs_torch(X, Y, p)

    # ---- NumPy path -------------------------------------------------------
    Xd = np.asarray(X, dtype=np.float64)
    Y01 = (np.asarray(Y) > 0).astype(np.int8, copy=False)
    Yf = Y01.astype(np.float64, copy=False)
    n, d = int(Xd.shape[0]), int(Xd.shape[1])
    L = int(Yf.shape[1])
    eps = np.finfo(np.float64).eps

    # Instance weights
    if bool(p.use_instance_weights):
        s, _sw, _prior = rarity_instance_weights(
            Y01,
            gamma=float(p.inst.gamma),
            kappa=float(p.inst.kappa),
            s_max=float(p.inst.s_max),
            eps=float(eps),
        )
        s = np.asarray(s, dtype=np.float64)
    else:
        s = np.ones((n,), dtype=np.float64)
    s_col = s.reshape(-1, 1)

    # Label embedding
    V, w_lab, svals = _label_embedding_from_Y(
        Y01, rank=int(p.rank),
        label_gamma=float(p.label_gamma), seed=int(p.seed),
    )
    r = int(V.shape[1])
    Yw = Yf * w_lab[None, :]
    T = Yw @ V  # (n, r)

    alpha = float(np.clip(float(p.alpha), 0.0, 1.0))
    w_reg = float(max(0.0, 1.0 - alpha))
    beta = float(p.beta)
    rho = float(p.rho)

    # Lipschitz constants
    Xs = Xd * np.sqrt(s_col)
    xtx_eig_s = _power_iteration_xtx_eigmax(
        Xs, n_iter=int(p.power_iter), seed=int(p.seed),
    )
    Lw = (w_reg + 0.25 * alpha) * float(xtx_eig_s) + 2.0 * rho
    Lw = max(Lw, 1e-12)
    step_w = 0.9 / Lw
    Lb = 0.25 * alpha * float(np.sum(s)) + 1e-12
    step_b = 0.9 / Lb

    # Initialise
    rng = np.random.default_rng(int(p.seed))
    W = 0.01 * rng.standard_normal((d, r), dtype=np.float64)
    b = np.zeros((L,), dtype=np.float64)
    ZW = W.copy()
    Zb = b.copy()
    t = 1.0
    losses: list[float] = []

    for it in range(int(p.max_iter)):
        XW = Xd @ ZW
        # Regression gradient
        gW_reg = Xd.T @ ((XW - T) * s_col)
        # Logistic gradient
        z = XW @ V.T + Zb[None, :]
        P = _sigmoid(z)
        E = (P - Yf) * s_col
        EV = E @ V
        gW_log = Xd.T @ EV
        gb = np.sum(E, axis=0)

        gW = w_reg * gW_reg + alpha * gW_log
        if rho > 0.0:
            gW = gW + 2.0 * rho * ZW

        Wn = _prox_group_lasso_rows(
            ZW - step_w * gW, step_w * beta, eps=float(eps),
        )
        bn = Zb - step_b * alpha * gb

        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        ZW = Wn + ((t - 1.0) / tn) * (Wn - W)
        Zb = bn + ((t - 1.0) / tn) * (bn - b)
        W, b, t = Wn, bn, tn

        # Convergence check
        if (it + 1) % 10 == 0 or it == 0:
            XW2 = Xd @ W
            z2 = XW2 @ V.T + b[None, :]
            reg = 0.5 * float(np.sum(((XW2 - T) ** 2) * s_col)) \
                / max(float(np.sum(s) * r), 1.0)
            ce = np.logaddexp(0.0, z2) - Yf * z2
            logi = float(np.sum(ce * s_col)) / max(float(np.sum(s) * L), 1.0)
            gl = beta * float(np.sum(np.linalg.norm(W, axis=1)))
            ridge = rho * float(np.sum(W * W))
            obj = (1.0 - alpha) * reg + alpha * logi + gl + ridge
            losses.append(float(obj))
            if len(losses) >= 2:
                rel = abs(losses[-1] - losses[-2]) \
                    / max(abs(losses[-2]), 1e-12)
                if rel < float(p.tol):
                    break

    # Feature ranking by row norms
    scores = np.linalg.norm(W, axis=1).astype(np.float64)
    ranking0 = np.argsort(-scores, kind="mergesort")
    ranking = (ranking0 + 1).astype(np.int64)

    info: Dict[str, Any] = {
        "n": int(n), "d": int(d), "L": int(L), "rank": int(r),
        "label_gamma": float(p.label_gamma),
        "backend": "numpy",
        "alpha": float(alpha), "beta": float(beta), "rho": float(rho),
        "use_instance_weights": bool(p.use_instance_weights),
        "inst_gamma": float(p.inst.gamma),
        "inst_kappa": float(p.inst.kappa),
        "inst_s_max": float(p.inst.s_max),
        "xtx_eig_s": float(xtx_eig_s),
        "step_w": float(step_w), "step_b": float(step_b),
        "iterations": int(it + 1),
        "score_minmax": [float(np.min(scores)), float(np.max(scores))],
        "scores": scores,
    }
    if losses:
        info["obj_start"] = float(losses[0])
        info["obj_end"] = float(losses[-1])
    return ranking, info


# ---------------------------------------------------------------------------
# PyTorch solver (GPU-accelerated)
# ---------------------------------------------------------------------------

def _torch_dtype_from_str(name: str):
    import torch
    name0 = str(name).lower().strip()
    if name0 in ("float32", "fp32", "f32"):
        return torch.float32
    if name0 in ("float64", "fp64", "f64"):
        return torch.float64
    raise ValueError("torch_dtype must be 'float32' or 'float64'.")


def _power_iteration_xtx_eigmax_torch(X, *, n_iter: int, seed: int) -> float:
    import torch
    d = int(X.shape[1])
    gen = torch.Generator(device=X.device)
    gen.manual_seed(int(seed))
    v = torch.randn((d,), generator=gen, device=X.device, dtype=X.dtype)
    v = v / (torch.linalg.norm(v) + 1e-12)
    for _ in range(int(n_iter)):
        u = X @ v
        nu = torch.linalg.norm(u)
        if float(nu) <= 1e-12:
            return 0.0
        u = u / (nu + 1e-12)
        v = X.T @ u
        nv = torch.linalg.norm(v)
        if float(nv) <= 1e-12:
            return 0.0
        v = v / (nv + 1e-12)
    Xv = X @ v
    lam = float((Xv @ Xv) / (v @ v + 1e-12))
    return float(max(lam, 0.0))


def _prox_group_lasso_rows_torch(W, lam, *, eps: float = 1e-12):
    import torch
    norms = torch.linalg.norm(W, dim=1)
    scale = torch.clamp(1.0 - float(lam) / (norms + float(eps)), min=0.0)
    return W * scale[:, None]


def _fit_cfifs_torch(
    X: np.ndarray,
    Y: np.ndarray,
    p: CFIFSParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dense PyTorch backend (GPU-friendly) for the CFIFS embedded stage."""
    import torch
    import torch.nn.functional as F

    device = torch.device(str(p.device))
    dtype = _torch_dtype_from_str(str(p.torch_dtype))

    if device.type == "cuda" and dtype == torch.float32 and bool(p.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    Xd0 = np.asarray(X, dtype=np_dtype, order="C")
    Y01 = (np.asarray(Y) > 0).astype(np.int8, copy=False)
    Yf_np = Y01.astype(np_dtype, copy=False)
    n, d = int(Xd0.shape[0]), int(Xd0.shape[1])
    L = int(Yf_np.shape[1])
    eps = 1e-12

    X_t = torch.from_numpy(np.asarray(Xd0, order="C")).to(
        device=device, dtype=dtype,
    )
    Y_t = torch.from_numpy(Yf_np).to(device=device, dtype=dtype)

    # Instance weights
    if bool(p.use_instance_weights):
        s_np, _sw, _prior = rarity_instance_weights(
            Y01,
            gamma=float(p.inst.gamma),
            kappa=float(p.inst.kappa),
            s_max=float(p.inst.s_max),
            eps=float(eps),
        )
        s_np = np.asarray(s_np, dtype=np.float64)
    else:
        s_np = np.ones((n,), dtype=np.float64)
    s_sum = float(np.sum(s_np)) + float(eps)
    s_t = torch.from_numpy(s_np.astype(np_dtype, copy=False)).to(
        device=device, dtype=dtype,
    )
    s_col = s_t.reshape(-1, 1)

    # Label embedding
    V_np, w_lab, svals = _label_embedding_from_Y(
        Y01, rank=int(p.rank),
        label_gamma=float(p.label_gamma), seed=int(p.seed),
    )
    r = int(V_np.shape[1])
    T_np = (Yf_np * w_lab.reshape(1, -1)) @ V_np
    V_t = torch.from_numpy(V_np.astype(np_dtype, copy=False)).to(
        device=device, dtype=dtype,
    )
    T_t = torch.from_numpy(T_np.astype(np_dtype, copy=False)).to(
        device=device, dtype=dtype,
    )

    alpha = float(np.clip(float(p.alpha), 0.0, 1.0))
    w_reg = float(max(0.0, 1.0 - alpha))
    beta = float(p.beta)
    rho = float(p.rho)

    # Lipschitz constants
    Xs_t = X_t * torch.sqrt(s_col)
    xtx_eig_s = _power_iteration_xtx_eigmax_torch(
        Xs_t, n_iter=int(p.power_iter), seed=int(p.seed),
    )
    Lw = (w_reg + 0.25 * alpha) * float(xtx_eig_s) + 2.0 * rho
    Lw = max(Lw, 1e-12)
    step_w = 0.9 / Lw
    Lb = 0.25 * alpha * float(s_sum) + 1e-12
    step_b = 0.9 / Lb

    # Initialise (use NumPy RNG for cross-backend reproducibility)
    rng = np.random.default_rng(int(p.seed))
    W0_np = 0.01 * rng.standard_normal((d, r), dtype=np.float64)
    W = torch.from_numpy(
        np.asarray(W0_np.astype(np_dtype), order="C"),
    ).to(device=device, dtype=dtype)
    b = torch.zeros((L,), device=device, dtype=dtype)
    ZW = W.clone()
    Zb = b.clone()
    t = 1.0
    losses: list[float] = []

    for it in range(int(p.max_iter)):
        XW = X_t @ ZW
        # Regression gradient
        gW_reg = X_t.T @ ((XW - T_t) * s_col)
        # Logistic gradient
        z = XW @ V_t.T + Zb.reshape(1, -1)
        P = torch.sigmoid(z)
        E = (P - Y_t) * s_col
        EV = E @ V_t
        gW_log = X_t.T @ EV
        gb = torch.sum(E, dim=0)

        gW = w_reg * gW_reg + alpha * gW_log
        if rho > 0.0:
            gW = gW + 2.0 * rho * ZW

        Wn = _prox_group_lasso_rows_torch(
            ZW - float(step_w) * gW, float(step_w) * beta, eps=float(eps),
        )
        bn = Zb - float(step_b) * alpha * gb

        tn = 0.5 * (1.0 + float(np.sqrt(1.0 + 4.0 * t * t)))
        ZW = Wn + ((t - 1.0) / tn) * (Wn - W)
        Zb = bn + ((t - 1.0) / tn) * (bn - b)
        W, b, t = Wn, bn, tn

        # Convergence check
        if (it + 1) % 10 == 0 or it == 0:
            XW2 = X_t @ W
            z2 = XW2 @ V_t.T + b.reshape(1, -1)
            reg = 0.5 * torch.sum(((XW2 - T_t) ** 2) * s_col) \
                / max(float(s_sum) * float(r), 1.0)
            ce = F.softplus(z2) - Y_t * z2
            logi = torch.sum(ce * s_col) \
                / max(float(s_sum) * float(L), 1.0)
            gl = beta * torch.sum(torch.linalg.norm(W, dim=1))
            ridge = rho * torch.sum(W * W)
            obj = (1.0 - alpha) * reg + alpha * logi + gl + ridge
            losses.append(float(obj.detach().cpu().item()))
            if len(losses) >= 2:
                rel = abs(losses[-1] - losses[-2]) \
                    / max(abs(losses[-2]), 1e-12)
                if rel < float(p.tol):
                    break

    # Feature ranking by row norms (CPU)
    W_np_final = W.detach().cpu().double().numpy()
    scores = np.linalg.norm(W_np_final, axis=1).astype(np.float64)
    ranking0 = np.argsort(-scores, kind="mergesort")
    ranking = (ranking0 + 1).astype(np.int64)

    info: Dict[str, Any] = {
        "n": int(n), "d": int(d), "L": int(L), "rank": int(r),
        "label_gamma": float(p.label_gamma),
        "backend": "torch", "device": str(device),
        "torch_dtype": str(p.torch_dtype), "allow_tf32": bool(p.allow_tf32),
        "alpha": float(alpha), "beta": float(beta), "rho": float(rho),
        "use_instance_weights": bool(p.use_instance_weights),
        "inst_gamma": float(p.inst.gamma),
        "inst_kappa": float(p.inst.kappa),
        "inst_s_max": float(p.inst.s_max),
        "xtx_eig_s": float(xtx_eig_s),
        "step_w": float(step_w), "step_b": float(step_b),
        "iterations": int(it + 1),
        "score_minmax": [float(np.min(scores)), float(np.max(scores))],
        "scores": scores,
    }
    if losses:
        info["obj_start"] = float(losses[0])
        info["obj_end"] = float(losses[-1])
    return ranking, info

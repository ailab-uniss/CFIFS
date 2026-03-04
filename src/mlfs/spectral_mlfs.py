#!/usr/bin/env python3
"""
Spectral Multi-Label Feature Selection via Label-Aware Graph Diffusion (SLAGD).

Core idea
---------
1. Build a sample-sample affinity graph **from the label matrix Y** (training only).
   Each edge (i,j) is weighted by the Jaccard similarity of their label vectors.
2. Form the normalised graph Laplacian  L = I - D^{-1/2} A D^{-1/2}.
3. For every feature f, compute the *Dirichlet energy*
       E(f) = x_f^T  L  x_f  /  (x_f^T x_f + eps)
   Features with LOW energy are "smooth" on the label graph → informative.
4. Additionally, compute a *label-relevance* score per feature via the
   Hilbert-Schmidt Independence Criterion (HSIC) between X[:,f] and Y,
   using a kernel on Y derived from the same label graph.
5. The final score blends the two:
       score(f) = (1-α) · (1 - E_norm(f))  +  α · HSIC_norm(f)
   Both terms are min-max normalised to [0,1] before blending.

Computational cost
------------------
- Label graph: O(n² L)  (L = number of labels, typically < 200)
- Laplacian:   O(n²)
- Dirichlet:   O(n² d)  via a single matrix multiply  X^T L X  (diagonal only)
- HSIC:        O(n² d)  via  X^T K_Y X  (diagonal only)
- Total:       O(n²(L + d))  — no iterative solver, no hyper-parameter tuning.

For large n, we support a Nyström approximation of the label graph (TODO).

GPU acceleration
----------------
All heavy linear algebra is done in PyTorch when a CUDA device is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SLAGDParams:
    """Parameters for Spectral Label-Aware Graph Diffusion feature selection.

    Scope
    -----
    Encapsulates all hyper-parameters for the SLAGD spectral scoring
    channel.  The ``alpha`` field controls the Dirichlet-vs-HSIC blend;
    the remaining fields configure label graph construction, optional
    instance weighting, and the compute backend.

    Attributes
    ----------
    alpha : float
        Blending weight in [0, 1].  0 = pure Dirichlet smoothness,
        1 = pure HSIC relevance.  Paper default: 0.70.
    label_sim : str
        Similarity measure for label-graph edges: ``'jaccard'``,
        ``'cosine'``, or ``'hamming'``.
    label_knn : int
        If > 0, sparsify the label graph to *k*-NN.  0 = full graph.
    label_self_loops : bool
        Whether to add self-loops before computing the Laplacian.
    instance_weight_gamma : float
        Inverse-frequency exponent for optional instance weighting.
        0 = uniform.
    backend : str
        ``'auto'`` | ``'torch'`` | ``'numpy'``.
    device : str
        ``'auto'`` | ``'cpu'`` | ``'cuda'``.
    torch_dtype : str
        ``'float32'`` or ``'float64'``.
    """
    # Blending weight: 0 = pure Dirichlet smoothness, 1 = pure HSIC relevance.
    alpha: float = 0.50
    # Label graph construction
    label_sim: str = "jaccard"        # jaccard | cosine | hamming
    label_knn: int = 0                # If >0, sparsify graph to k-NN (0 = full)
    label_self_loops: bool = False    # Add self-loops before Laplacian
    # Optional: instance weighting for rare-label upweighting
    instance_weight_gamma: float = 0.0  # 0 = uniform; >0 = inverse-freq weighting
    # GPU
    backend: str = "auto"             # auto | torch | numpy
    device: str = "auto"              # auto | cpu | cuda
    torch_dtype: str = "float32"


def _to_torch(arr: np.ndarray, device, dtype):
    """Convert a NumPy array to a contiguous PyTorch tensor on *device*.

    Scope
    -----
    Convenience wrapper that ensures C-contiguous memory layout
    before calling ``torch.from_numpy``, avoiding stride-related
    errors.

    Parameters
    ----------
    arr : ndarray
        Source array.
    device : torch.device
        Target device (CPU or CUDA).
    dtype : str
        ``'float32'`` or ``'float64'``.

    Preconditions
    -------------
    * PyTorch must be importable.

    Postconditions
    --------------
    * Returns a torch.Tensor on the specified device and dtype.
    """
    import torch
    td = {"float32": torch.float32, "float64": torch.float64}[dtype]
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=td)


def fit_slagd(
    X: np.ndarray,       # (n, d) training features
    Y: np.ndarray,       # (n, L) training labels {0,1} or {-1,+1}
    params: SLAGDParams | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a training-only spectral feature ranking.

    Scope
    -----
    Public entry point for SLAGD.  Builds a label-affinity graph,
    computes Dirichlet energy and HSIC for every feature, blends
    them, and returns a 1-based feature ranking.  Automatically
    dispatches to the NumPy or PyTorch backend.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Training features (preferably scaled to [0, 1]).
    Y : ndarray of shape (n, L)
        Binary training labels {0, 1} or {-1, +1}.
    params : SLAGDParams | None
        Configuration.  ``None`` uses defaults.

    Preconditions
    -------------
    * *X* and *Y* share the same *n*.
    * *Y* contains at least one positive entry.

    Postconditions
    --------------
    * ``ranking`` is a permutation of {1, …, d}, best first.
    * ``info`` contains ``'scores_min'``, ``'scores_max'``,
      ``'scores_mean'``, ``'alpha'``, and ``'backend_used'``.

    Returns
    -------
    ranking : ndarray of shape (d,), 1-based feature indices sorted best→worst.
    info    : dict with diagnostic information.
    """
    if params is None:
        params = SLAGDParams()

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, d = X.shape
    L = Y.shape[1]

    # Ensure Y is {0,1}
    if Y.min() < 0:
        Y = (Y > 0).astype(np.float64)

    info: Dict[str, Any] = {"n": n, "d": d, "L": L, "params": {
        "alpha": params.alpha,
        "label_sim": params.label_sim,
        "backend": params.backend,
    }}

    # Select backend
    use_torch = False
    if params.backend in ("auto", "torch"):
        try:
            import torch
            if params.device == "cuda" or (params.device == "auto" and torch.cuda.is_available()):
                dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")
            use_torch = True
        except ImportError:
            use_torch = False

    if use_torch:
        ranking, scores, extra = _fit_torch(X, Y, params, dev)
    else:
        ranking, scores, extra = _fit_numpy(X, Y, params)

    info.update(extra)
    info["scores_min"] = float(np.min(scores))
    info["scores_max"] = float(np.max(scores))
    info["scores_mean"] = float(np.mean(scores))

    return ranking, info


def _build_label_affinity(Y: np.ndarray, sim: str) -> np.ndarray:
    """Build a symmetric (n, n) label-based affinity matrix.

    Scope
    -----
    Measures pairwise similarity between instances based on their
    label vectors.  The diagonal is set to zero so that the
    resulting graph has no self-loops.

    Parameters
    ----------
    Y : ndarray of shape (n, L)
        Binary label matrix {0, 1}.
    sim : str
        Similarity metric: ``'jaccard'``, ``'cosine'``, or
        ``'hamming'``.

    Preconditions
    -------------
    * *Y* is binary and non-empty.

    Postconditions
    --------------
    * Returns a symmetric float64 matrix with zero diagonal.
    * All entries are in [0, 1].
    """
    n = Y.shape[0]
    if sim == "jaccard":
        # Jaccard: |A ∩ B| / |A ∪ B|
        YY = Y @ Y.T                     # (n, n) intersection counts
        row_sums = Y.sum(axis=1)          # (n,)
        union = row_sums[:, None] + row_sums[None, :] - YY
        union = np.maximum(union, 1e-10)
        A = YY / union
    elif sim == "cosine":
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        Yn = Y / norms
        A = Yn @ Yn.T
    elif sim == "hamming":
        # 1 - hamming_distance
        agree = Y @ Y.T + (1 - Y) @ (1 - Y).T
        A = agree / float(Y.shape[1])
    else:
        raise ValueError(f"Unknown label_sim: {sim}")
    np.fill_diagonal(A, 0.0)
    return A


def _sparsify_knn(A: np.ndarray, k: int) -> np.ndarray:
    """Keep only the *k* nearest neighbours per row and symmetrise.

    Scope
    -----
    Sparsifies a dense affinity matrix by retaining only the top-*k*
    entries per row, then symmetrises via element-wise maximum so
    that the resulting graph is undirected.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Dense affinity matrix.
    k : int
        Number of neighbours to keep.  If *k* ≤ 0 or *k* ≥ n the
        matrix is returned unchanged.

    Preconditions
    -------------
    * *A* is square and non-negative.

    Postconditions
    --------------
    * Returns a symmetric matrix of the same shape.
    * Each row has at most 2*k* non-zero entries (after symmetrisation).
    """
    n = A.shape[0]
    if k <= 0 or k >= n:
        return A
    B = np.zeros_like(A)
    for i in range(n):
        idx = np.argpartition(A[i], -k)[-k:]
        B[i, idx] = A[i, idx]
    # Symmetrize
    B = np.maximum(B, B.T)
    return B


def _fit_numpy(
    X: np.ndarray, Y: np.ndarray, params: SLAGDParams
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """NumPy backend for SLAGD spectral scoring.

    Scope
    -----
    Computes the full SLAGD pipeline on CPU: label affinity →
    normalised Laplacian → Dirichlet energy → HSIC → blend → ranking.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Training features.
    Y : ndarray of shape (n, L)
        Binary training labels {0, 1}.
    params : SLAGDParams
        Configuration (label_sim, alpha, etc.).

    Preconditions
    -------------
    * *X* and *Y* satisfy the constraints of :func:`fit_slagd`.

    Postconditions
    --------------
    * ``ranking`` is 1-based, sorted by descending *scores*.
    * ``scores`` ∈ [0, 1]^d.
    * ``extra`` dict contains ``'dirichlet_mean'``, ``'hsic_mean'``,
      ``'alpha'``, ``'backend_used'``.

    Returns
    -------
    ranking : ndarray (d,)
    scores  : ndarray (d,)
    extra   : dict
    """
    n, d = X.shape
    eps = 1e-10

    # 1. Label affinity
    A = _build_label_affinity(Y, params.label_sim)
    if params.label_knn > 0:
        A = _sparsify_knn(A, params.label_knn)
    if params.label_self_loops:
        np.fill_diagonal(A, 1.0)

    # 2. Normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.where(deg > eps, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Instance weighting (optional)
    W = np.ones(n, dtype=np.float64)
    if params.instance_weight_gamma > 0:
        label_freq = Y.mean(axis=0)
        inv_freq = 1.0 / np.maximum(label_freq, eps)
        sample_weight = (Y * inv_freq[None, :]).sum(axis=1)
        sample_weight = sample_weight ** params.instance_weight_gamma
        sample_weight /= sample_weight.mean()
        W = sample_weight

    # 3. Dirichlet energy per feature: E(f) = x_f^T L x_f / (x_f^T x_f)
    # Vectorised: diag(X^T L X) / diag(X^T X)
    Xw = X * W[:, None]  # weighted X
    LX = L_norm @ Xw
    dirichlet_num = np.sum(Xw * LX, axis=0)    # (d,)
    dirichlet_den = np.sum(Xw * Xw, axis=0)    # (d,)
    dirichlet = dirichlet_num / (dirichlet_den + eps)

    # 4. HSIC score per feature
    # K_Y = A (label kernel already computed)
    # Centre: H K_Y H  where H = I - 1/n 11^T
    K_Y = A.copy()
    row_mean = K_Y.mean(axis=1, keepdims=True)
    col_mean = K_Y.mean(axis=0, keepdims=True)
    total_mean = K_Y.mean()
    K_Yc = K_Y - row_mean - col_mean + total_mean

    # HSIC(f) ∝ x_f^T K_Yc x_f  (centred kernel alignment)
    K_Yc_X = K_Yc @ Xw
    hsic_num = np.sum(Xw * K_Yc_X, axis=0)  # (d,)
    # Normalise by feature variance
    hsic = hsic_num / (dirichlet_den + eps)

    # 5. Normalise both to [0,1] and blend
    def minmax(v):
        lo, hi = v.min(), v.max()
        if hi - lo < eps:
            return np.ones_like(v) * 0.5
        return (v - lo) / (hi - lo)

    smoothness = 1.0 - minmax(dirichlet)   # low energy = high smoothness
    relevance = minmax(hsic)

    alpha = float(params.alpha)
    scores = (1.0 - alpha) * smoothness + alpha * relevance

    # Ranking: descending scores → 1-based indices
    order = np.argsort(-scores)
    ranking = order + 1  # 1-based

    extra = {
        "dirichlet_mean": float(dirichlet.mean()),
        "hsic_mean": float(hsic.mean()),
        "alpha": alpha,
        "backend_used": "numpy",
    }
    return ranking, scores, extra


def _fit_torch(
    X: np.ndarray, Y: np.ndarray, params: SLAGDParams, device
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """PyTorch backend for SLAGD spectral scoring.

    Scope
    -----
    GPU-accelerated version of :func:`_fit_numpy`.  All O(n² d)
    operations (Laplacian products, HSIC kernel products) are
    executed on *device*.  Results are moved back to CPU/NumPy
    before returning.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Training features (NumPy, converted internally).
    Y : ndarray of shape (n, L)
        Binary training labels.
    params : SLAGDParams
        Configuration.
    device : torch.device
        Target device (e.g. ``torch.device('cuda')``).

    Preconditions
    -------------
    * PyTorch is installed; *device* is available.

    Postconditions
    --------------
    * Same return contract as :func:`_fit_numpy`; all arrays on CPU.

    Returns
    -------
    ranking : ndarray (d,)
    scores  : ndarray (d,)
    extra   : dict
    """
    import torch
    n, d = X.shape
    eps = 1e-10
    dt = {"float32": torch.float32, "float64": torch.float64}[params.torch_dtype]

    Xt = torch.from_numpy(X).to(device=device, dtype=dt)
    Yt = torch.from_numpy(Y).to(device=device, dtype=dt)

    # 1. Label affinity on GPU
    if params.label_sim == "jaccard":
        YY = Yt @ Yt.T
        row_sums = Yt.sum(dim=1)
        union = row_sums.unsqueeze(1) + row_sums.unsqueeze(0) - YY
        union = torch.clamp(union, min=eps)
        A = YY / union
    elif params.label_sim == "cosine":
        Yn = torch.nn.functional.normalize(Yt, p=2, dim=1)
        A = Yn @ Yn.T
    elif params.label_sim == "hamming":
        agree = Yt @ Yt.T + (1 - Yt) @ (1 - Yt).T
        A = agree / float(Y.shape[1])
    else:
        raise ValueError(f"Unknown label_sim: {params.label_sim}")

    A.fill_diagonal_(0.0)

    # k-NN sparsification
    if params.label_knn > 0 and params.label_knn < n:
        k = params.label_knn
        _, topk_idx = torch.topk(A, k, dim=1)
        mask = torch.zeros_like(A, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        mask = mask | mask.T
        A = A * mask.to(dt)

    if params.label_self_loops:
        A.fill_diagonal_(1.0)

    # 2. Normalised Laplacian
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > eps, 1.0 / torch.sqrt(deg), torch.zeros_like(deg))
    # L = I - D^{-1/2} A D^{-1/2}
    # Efficient: (D^{-1/2} A D^{-1/2})_{ij} = d_i^{-1/2} A_{ij} d_j^{-1/2}
    A_norm = A * deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)
    # L_norm X = X - A_norm X
    # We don't form L explicitly.

    # Instance weighting
    W = torch.ones(n, device=device, dtype=dt)
    if params.instance_weight_gamma > 0:
        label_freq = Yt.mean(dim=0)
        inv_freq = 1.0 / torch.clamp(label_freq, min=eps)
        sw = (Yt * inv_freq.unsqueeze(0)).sum(dim=1)
        sw = sw ** params.instance_weight_gamma
        sw = sw / sw.mean()
        W = sw

    Xw = Xt * W.unsqueeze(1)

    # 3. Dirichlet energy: diag(Xw^T L Xw) = diag(Xw^T Xw) - diag(Xw^T A_norm Xw)
    XtX_diag = (Xw * Xw).sum(dim=0)                    # (d,)
    AX = A_norm @ Xw                                     # (n, d)
    XtAX_diag = (Xw * AX).sum(dim=0)                    # (d,)
    dirichlet_num = XtX_diag - XtAX_diag                 # (d,)
    dirichlet = dirichlet_num / (XtX_diag + eps)

    # 4. HSIC: centred label kernel
    K_Y = A.clone()
    row_mean = K_Y.mean(dim=1, keepdim=True)
    col_mean = K_Y.mean(dim=0, keepdim=True)
    total_mean = K_Y.mean()
    K_Yc = K_Y - row_mean - col_mean + total_mean

    K_Yc_X = K_Yc @ Xw                                   # (n, d)
    hsic_num = (Xw * K_Yc_X).sum(dim=0)                  # (d,)
    hsic = hsic_num / (XtX_diag + eps)

    # 5. Normalise and blend
    def minmax_t(v):
        lo = v.min()
        hi = v.max()
        rng = hi - lo
        if rng < eps:
            return torch.full_like(v, 0.5)
        return (v - lo) / rng

    smoothness = 1.0 - minmax_t(dirichlet)
    relevance = minmax_t(hsic)

    alpha = float(params.alpha)
    scores = (1.0 - alpha) * smoothness + alpha * relevance

    # Back to CPU numpy
    scores_np = scores.cpu().numpy().astype(np.float64)
    order = np.argsort(-scores_np)
    ranking = order + 1  # 1-based

    extra = {
        "dirichlet_mean": float(dirichlet.mean().item()),
        "hsic_mean": float(hsic.mean().item()),
        "alpha": alpha,
        "backend_used": "torch",
        "device_used": str(device),
    }
    return ranking, scores_np, extra

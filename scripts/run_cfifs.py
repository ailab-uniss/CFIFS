#!/usr/bin/env python3
"""
CFIFS — Choquet Fuzzy-Integral Feature Selection  (main runner)
===============================================================

This script reproduces the full CFIFS pipeline described in the paper:

    Casu, Lagorio & Trunfio,
    "Choquet-Based Fusion of Embedded and Spectral Scores for
     Multi-Label Feature Selection", Information Fusion, 2026.

Pipeline (executed independently for each training fold)
--------------------------------------------------------
1. **Embedded scoring** — solve a group-sparse convex problem that combines
   squared-reconstruction loss with a per-label logistic term, regularised by
   ℓ₂₁ (group lasso on feature rows) plus ridge.  Feature importance is the
   row-norm ‖W_{j:}‖₂.  (``src/mlfs/cfifs_embedded.py``)

2. **Spectral scoring** — SLAGD (Spectral Label-Affinity Graph Dirichlet):
   compute a label-affinity graph, derive the graph Laplacian, and score each
   feature by its Dirichlet energy combined with an HSIC term.
   (``src/mlfs/spectral_mlfs.py``)

3. **Rank normalisation** — map each score channel to [0, 1] via empirical
   CDF (rank normalisation), making the two channels commensurate.

4. **Choquet integral fusion** — fuse the two normalised score vectors via a
   2-source Choquet integral with free singleton capacities (μₑ, μₛ).  The
   capacity pair is selected by an inner K-fold CV loop that evaluates
   ML-kNN on the top-p% features, maximising the geometric mean of
   Micro-F1 and Macro-F1.

5. **Output** — write a 1-based ranking CSV and a JSON metadata file per fold.

Score fusion (Choquet integral)
-------------------------------
Let eⱼ, sⱼ ∈ [0, 1] be the rank-normalised embedded and spectral scores for
feature j.  The 2-source Choquet integral with normalised capacity
μ({emb, spec}) = 1 and singleton capacities μₑ, μₛ ∈ [0, 1] is:

    if eⱼ ≥ sⱼ:  gⱼ = sⱼ + (eⱼ − sⱼ) · μₑ
    else:         gⱼ = eⱼ + (sⱼ − eⱼ) · μₛ

The interaction index I = 1 − μₑ − μₛ captures redundancy (I > 0) or synergy
(I < 0) between the two channels.

Usage example (paper settings)
------------------------------
::

    python scripts/run_cfifs.py \\
        --data-dir  data/panorama30_matlab_minmax_cv10 \\
        --results-dir results/bench_panorama30_cv10 \\
        --method CFIFS \\
        --score-norm rank \\
        --capacity-mode free \\
        --mu-grid "0.0,0.2,0.4,0.6,0.8,1.0" \\
        --icv-criterion gm \\
        --icv-aggregate hard \\
        --folds 10 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlfs.cfifs_embedded import CFIFSParams, fit_cfifs
from mlfs.ml_knn_gpu import MLkNNConfig, MLkNNModel
from mlfs.spectral_mlfs import SLAGDParams


def _get_embedded_scores(X: np.ndarray, Y: np.ndarray, params: CFIFSParams) -> np.ndarray:
    """Run the embedded scoring stage and return per-feature importances."""
    _ranking, info = fit_cfifs(X, Y, params)
    return np.asarray(info["scores"], dtype=np.float64)


def _get_spectral_scores(X: np.ndarray, Y: np.ndarray, params: SLAGDParams) -> np.ndarray:
    """Run SLAGD spectral scoring and return per-feature importances."""
    from mlfs.spectral_mlfs import _fit_numpy, _fit_torch
    import torch

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.min() < 0:
        Y = (Y > 0).astype(np.float64)

    use_torch = False
    if params.backend in ("auto", "torch"):
        try:
            if params.device == "cuda" or (params.device == "auto" and torch.cuda.is_available()):
                dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")
            use_torch = True
        except ImportError:
            pass

    if use_torch:
        _, scores, _ = _fit_torch(X, Y, params, dev)
    else:
        _, scores, _ = _fit_numpy(X, Y, params)
    return np.asarray(scores, dtype=np.float64)


def _minmax(v: np.ndarray) -> np.ndarray:
    """Min-max normalise a score vector to [0, 1]."""
    v = np.asarray(v, dtype=np.float64)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)

def _rank_uniform(v: np.ndarray) -> np.ndarray:
    """Empirical CDF (rank) normalization to (0,1), robust to outliers."""
    from scipy.stats import rankdata

    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n <= 1:
        return np.full_like(v, 0.5)
    r = rankdata(v, method="average")  # 1..n
    u = (r - 0.5) / float(n)
    return np.clip(u, 0.0, 1.0)


def _choquet_two_sources(e: np.ndarray, s: np.ndarray, *, mu_e: float, mu_s: float) -> np.ndarray:
    """Compute the 2-source Choquet integral element-wise.

    Parameters
    ----------
    e, s : (d,) normalised embedded / spectral score vectors.
    mu_e, mu_s : singleton capacities in [0, 1].

    Returns
    -------
    g : (d,) fused scores.
    """
    e = np.asarray(e, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    mu_e = float(np.clip(mu_e, 0.0, 1.0))
    mu_s = float(np.clip(mu_s, 0.0, 1.0))
    mask = e >= s
    out = np.empty_like(e)
    out[mask] = s[mask] + (e[mask] - s[mask]) * mu_e
    out[~mask] = e[~mask] + (s[~mask] - e[~mask]) * mu_s
    return out


def _icv_mlknn_choquet(
    emb_norm: np.ndarray,
    spec_norm: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    p_frac: float,
    cv: int,
    grid: np.ndarray,
    capacity_mode: str,
    mlknn_k: int,
    device: str,
    criterion: str,
    seed: int,
    wgm_alpha: float = 0.5,
) -> tuple[float, float, float]:
    """
    Select (mu_e, mu_s) via inner CV on ML-kNN.

    Returns:
      mu_e, mu_s, best_score
    """
    from sklearn.model_selection import KFold

    d = int(emb_norm.shape[0])
    k_feat = max(1, int(round(float(p_frac) * d)))

    if Y.min() < 0:
        Y = (Y > 0).astype(np.float64)

    kf = KFold(n_splits=int(cv), shuffle=True, random_state=int(seed))
    splits = list(kf.split(X))

    cap_mode = str(capacity_mode).lower().strip()
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    grid = np.unique(np.clip(grid, 0.0, 1.0))

    best_mu_e, best_mu_s, best_score = 0.5, 0.5, -1.0

    # Pre-materialize dense Y splits to speed up metric computations.
    for mu_e in grid:
        if cap_mode == "additive":
            cand = [(float(mu_e), float(1.0 - mu_e))]
        elif cap_mode == "symmetric":
            cand = [(float(mu_e), float(mu_e))]
        elif cap_mode == "free":
            cand = [(float(mu_e), float(mu_s)) for mu_s in grid]
        else:
            raise ValueError(f"Unknown capacity_mode: {capacity_mode!r}")

        for mu_e0, mu_s0 in cand:
            fused = _choquet_two_sources(emb_norm, spec_norm, mu_e=mu_e0, mu_s=mu_s0)
            sel = np.argpartition(-fused, k_feat - 1)[:k_feat]

            fold_micro: list[float] = []
            fold_macro: list[float] = []
            for tr_idx, va_idx in splits:
                Xtr_in = sparse.csr_matrix(X[np.ix_(tr_idx, sel)].astype(np.float32))
                Xva_in = sparse.csr_matrix(X[np.ix_(va_idx, sel)].astype(np.float32))
                Ytr_in = sparse.csr_matrix(Y[tr_idx].astype(np.float32))
                Yva_in = Y[va_idx].astype(np.float64, copy=False)

                cfg = MLkNNConfig(
                    k=min(int(mlknn_k), int(len(tr_idx) - 1)),
                    s=1.0,
                    metric="cosine",
                    backend="torch",
                    device=str(device),
                )
                model = MLkNNModel(cfg)
                model.fit(Xtr_in, Ytr_in)
                probs = model.predict_proba(Xva_in)  # (n_va, L)
                pred = (probs >= 0.5).astype(np.float64)

                # Micro-F1
                tp = float(np.sum(pred * Yva_in))
                p_sum = float(np.sum(pred))
                t_sum = float(np.sum(Yva_in))
                prec = tp / max(p_sum, 1e-12)
                rec = tp / max(t_sum, 1e-12)
                mi_f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
                fold_micro.append(mi_f1)

                # Macro-F1 (vectorised)
                tp_l = np.sum(pred * Yva_in, axis=0)
                p_l = np.sum(pred, axis=0)
                t_l = np.sum(Yva_in, axis=0)
                denom = p_l + t_l
                mask = denom > 0
                if np.any(mask):
                    f1_l = np.zeros_like(tp_l, dtype=np.float64)
                    f1_l[mask] = 2.0 * tp_l[mask] / np.maximum(denom[mask], 1e-12)
                    ma_f1 = float(np.mean(f1_l[mask]))
                else:
                    ma_f1 = 0.0
                fold_macro.append(ma_f1)

            avg_mi = float(np.mean(fold_micro)) if fold_micro else 0.0
            avg_ma = float(np.mean(fold_macro)) if fold_macro else 0.0

            if criterion == "micro":
                score = avg_mi
            elif criterion == "macro":
                score = avg_ma
            elif criterion == "gm":
                score = float(np.sqrt(max(avg_mi, 0.0) * max(avg_ma, 0.0)))
            elif criterion == "wgm":
                score = float((max(avg_ma, 0.0)**wgm_alpha) * (max(avg_mi, 0.0)**(1.0 - wgm_alpha)))
            else:
                raise ValueError(f"Unknown criterion: {criterion!r}")

            if score > best_score:
                best_score = score
                best_mu_e = float(mu_e0)
                best_mu_s = float(mu_s0)

    return best_mu_e, best_mu_s, best_score


def _icv_mlknn_choquet_scores(
    emb_norm: np.ndarray,
    spec_norm: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    p_frac: float,
    cv: int,
    grid: np.ndarray,
    capacity_mode: str,
    mlknn_k: int,
    device: str,
    criterion: str,
    seed: int,
    wgm_alpha: float = 0.5,
) -> tuple[list[tuple[float, float, float]], tuple[float, float, float]]:
    """
    Like _icv_mlknn_choquet, but returns all candidate scores.

    Returns:
      cand_scores: list of (mu_e, mu_s, score)
      best: (mu_e, mu_s, best_score)
    """
    from sklearn.model_selection import KFold

    d = int(emb_norm.shape[0])
    k_feat = max(1, int(round(float(p_frac) * d)))

    if Y.min() < 0:
        Y = (Y > 0).astype(np.float64)

    kf = KFold(n_splits=int(cv), shuffle=True, random_state=int(seed))
    splits = list(kf.split(X))

    cap_mode = str(capacity_mode).lower().strip()
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    grid = np.unique(np.clip(grid, 0.0, 1.0))

    cand_scores: list[tuple[float, float, float]] = []
    best_mu_e, best_mu_s, best_score = 0.5, 0.5, -1.0

    for mu_e in grid:
        if cap_mode == "additive":
            cand = [(float(mu_e), float(1.0 - mu_e))]
        elif cap_mode == "symmetric":
            cand = [(float(mu_e), float(mu_e))]
        elif cap_mode == "free":
            cand = [(float(mu_e), float(mu_s)) for mu_s in grid]
        else:
            raise ValueError(f"Unknown capacity_mode: {capacity_mode!r}")

        for mu_e0, mu_s0 in cand:
            fused = _choquet_two_sources(emb_norm, spec_norm, mu_e=mu_e0, mu_s=mu_s0)
            sel = np.argpartition(-fused, k_feat - 1)[:k_feat]

            fold_micro: list[float] = []
            fold_macro: list[float] = []
            for tr_idx, va_idx in splits:
                Xtr_in = sparse.csr_matrix(X[np.ix_(tr_idx, sel)].astype(np.float32))
                Xva_in = sparse.csr_matrix(X[np.ix_(va_idx, sel)].astype(np.float32))
                Ytr_in = sparse.csr_matrix(Y[tr_idx].astype(np.float32))
                Yva_in = Y[va_idx].astype(np.float64, copy=False)

                cfg = MLkNNConfig(
                    k=min(int(mlknn_k), int(len(tr_idx) - 1)),
                    s=1.0,
                    metric="cosine",
                    backend="torch",
                    device=str(device),
                )
                model = MLkNNModel(cfg)
                model.fit(Xtr_in, Ytr_in)
                probs = model.predict_proba(Xva_in)  # (n_va, L)
                pred = (probs >= 0.5).astype(np.float64)

                # Micro-F1
                tp = float(np.sum(pred * Yva_in))
                p_sum = float(np.sum(pred))
                t_sum = float(np.sum(Yva_in))
                prec = tp / max(p_sum, 1e-12)
                rec = tp / max(t_sum, 1e-12)
                mi_f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
                fold_micro.append(mi_f1)

                # Macro-F1
                tp_l = np.sum(pred * Yva_in, axis=0)
                p_l = np.sum(pred, axis=0)
                t_l = np.sum(Yva_in, axis=0)
                denom = p_l + t_l
                mask = denom > 0
                if np.any(mask):
                    f1_l = np.zeros_like(tp_l, dtype=np.float64)
                    f1_l[mask] = 2.0 * tp_l[mask] / np.maximum(denom[mask], 1e-12)
                    ma_f1 = float(np.mean(f1_l[mask]))
                else:
                    ma_f1 = 0.0
                fold_macro.append(ma_f1)

            avg_mi = float(np.mean(fold_micro)) if fold_micro else 0.0
            avg_ma = float(np.mean(fold_macro)) if fold_macro else 0.0

            if criterion == "micro":
                score = avg_mi
            elif criterion == "macro":
                score = avg_ma
            elif criterion == "gm":
                score = float(np.sqrt(max(avg_mi, 0.0) * max(avg_ma, 0.0)))
            elif criterion == "wgm":
                score = float((max(avg_ma, 0.0)**wgm_alpha) * (max(avg_mi, 0.0)**(1.0 - wgm_alpha)))
            else:
                raise ValueError(f"Unknown criterion: {criterion!r}")

            score_f = float(score)
            cand_scores.append((float(mu_e0), float(mu_s0), score_f))
            if score_f > best_score:
                best_score = score_f
                best_mu_e = float(mu_e0)
                best_mu_s = float(mu_s0)

    return cand_scores, (best_mu_e, best_mu_s, float(best_score))


def load_fold(data_dir: Path, dataset: str, fold: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load (X_train, Y_train) from a .mat fold file, or (None, None) if missing."""
    fp = data_dir / dataset / f"fold{fold}.mat"
    if not fp.exists():
        return None, None
    mat = sio.loadmat(str(fp))
    Xtr = np.asarray(mat["X_train"], dtype=np.float64)
    Ytr = np.asarray(mat["Y_train"], dtype=np.float64)
    return Xtr, Ytr


def find_datasets(data_dir: Path, explicit: list[str] | None) -> list[str]:
    """Return the list of dataset names.  If *explicit* is given use it;
    otherwise auto-discover every sub-folder of *data_dir* that contains
    ``fold0.mat``."""
    if explicit:
        return list(explicit)
    out = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "fold0.mat").exists():
            out.append(p.name)
    return out


def main() -> None:
    """CLI entry point — parse arguments, loop over datasets × folds,
    compute embedded + spectral scores, select Choquet capacity via inner CV,
    fuse, rank, and save results."""
    ap = argparse.ArgumentParser(description="CFIFS: Choquet Fuzzy-Integral Feature Selection")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--method", type=str, default="CFIFS_CFI")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")

    # Inner-CV parameters
    ap.add_argument("--icv-folds", type=int, default=3)
    ap.add_argument("--icv-p", type=float, default=0.20)
    ap.add_argument("--icv-criterion", type=str, default="gm", choices=["micro", "macro", "gm", "wgm"])
    ap.add_argument("--wgm-alpha", type=float, default=0.5, help="Weight for Macro in WGM")
    ap.add_argument("--icv-aggregate", type=str, default="hard", choices=["hard", "softmax"],
                    help="How to use inner-CV results: pick best capacity (hard) or softmax-average fused scores (softmax).")
    ap.add_argument("--softmax-tau", type=float, default=0.02,
                    help="Softmax temperature for --icv-aggregate=softmax (smaller -> closer to hard selection).")
    ap.add_argument("--capacity-mode", type=str, default="free", choices=["free", "additive", "symmetric"])
    ap.add_argument("--mu-grid", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0",
                    help="Comma-separated grid for mu values in [0,1].")
    ap.add_argument("--score-norm", type=str, default="minmax", choices=["minmax", "rank"],
                    help="Score normalization before fusion: minmax or rank (empirical CDF).")
    ap.add_argument("--fix-mu-e", type=float, default=None,
                    help="If set (together with --fix-mu-s), skip inner-CV and use this fixed mu_e.")
    ap.add_argument("--fix-mu-s", type=float, default=None,
                    help="If set (together with --fix-mu-e), skip inner-CV and use this fixed mu_s.")
    ap.add_argument("--mlknn-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    datasets = find_datasets(args.data_dir, args.datasets)
    out_dir = args.results_dir / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Solver configurations (paper defaults) -------------------------
    cfifs_params = CFIFSParams(backend="torch", device=args.device)
    slagd_params = SLAGDParams(
        alpha=0.70,
        label_sim="jaccard",
        label_knn=0,
        instance_weight_gamma=0.0,
        backend="torch",
        device=args.device,
    )

    try:
        mu_grid = np.array([float(x.strip()) for x in str(args.mu_grid).split(",") if x.strip() != ""], dtype=np.float64)
    except Exception as e:
        raise SystemExit(f"Invalid --mu-grid {args.mu_grid!r}: {e}") from e

    total = len(datasets) * int(args.folds)
    done = 0
    for ds in datasets:
        for fold in range(int(args.folds)):
            done += 1
            out_rank = out_dir / f"{ds}_fold{fold}_ranking.csv"
            out_info = out_dir / f"{ds}_fold{fold}_info.json"

            if out_rank.exists() and out_info.exists() and not args.overwrite:
                print(f"[{done}/{total}] SKIP {ds} fold{fold}")
                continue

            Xtr, Ytr = load_fold(args.data_dir, ds, fold)
            if Xtr is None:
                print(f"[{done}/{total}] SKIP {ds} fold{fold} (no data)")
                continue

            n, d = Xtr.shape
            L = int(Ytr.shape[1])
            t0 = time.time()

            emb_scores = _get_embedded_scores(Xtr, Ytr, cfifs_params)
            spec_scores = _get_spectral_scores(Xtr, Ytr, slagd_params)

            if str(args.score_norm).lower().strip() == "rank":
                emb_norm = _rank_uniform(emb_scores)
                spec_norm = _rank_uniform(spec_scores)
            else:
                emb_norm = _minmax(emb_scores)
                spec_norm = _minmax(spec_scores)

            if args.fix_mu_e is not None or args.fix_mu_s is not None:
                if args.fix_mu_e is None or args.fix_mu_s is None:
                    raise SystemExit("--fix-mu-e and --fix-mu-s must be provided together.")
                mu_e = float(args.fix_mu_e)
                mu_s = float(args.fix_mu_s)
                best_icv = float("nan")
                icv_time = 0.0
                fused = _choquet_two_sources(emb_norm, spec_norm, mu_e=mu_e, mu_s=mu_s)
            else:
                t_icv = time.time()
                agg_mode = str(args.icv_aggregate).lower().strip()
                if agg_mode == "hard":
                    mu_e, mu_s, best_icv = _icv_mlknn_choquet(
                        emb_norm,
                        spec_norm,
                        Xtr,
                        Ytr,
                        p_frac=float(args.icv_p),
                        cv=int(args.icv_folds),
                        grid=mu_grid,
                        capacity_mode=str(args.capacity_mode),
                        mlknn_k=int(args.mlknn_k),
                        device=str(args.device),
                        criterion=str(args.icv_criterion),
                        wgm_alpha=float(args.wgm_alpha),
                        seed=int(args.seed),
                    )
                    fused = _choquet_two_sources(emb_norm, spec_norm, mu_e=mu_e, mu_s=mu_s)
                elif agg_mode == "softmax":
                    cand_scores, best = _icv_mlknn_choquet_scores(
                        emb_norm,
                        spec_norm,
                        Xtr,
                        Ytr,
                        p_frac=float(args.icv_p),
                        cv=int(args.icv_folds),
                        grid=mu_grid,
                        capacity_mode=str(args.capacity_mode),
                        mlknn_k=int(args.mlknn_k),
                        device=str(args.device),
                        criterion=str(args.icv_criterion),
                        wgm_alpha=float(args.wgm_alpha),
                        seed=int(args.seed),
                    )
                    mu_e, mu_s, best_icv = best
                    tau = float(max(args.softmax_tau, 1e-6))
                    scores = np.asarray([s for _me, _ms, s in cand_scores], dtype=np.float64)
                    if len(scores) > 0 and float(np.max(scores)) > float(np.min(scores)):
                        scores = (scores - float(np.min(scores))) / (float(np.max(scores)) - float(np.min(scores)))
                    z = (scores - np.max(scores)) / tau
                    w = np.exp(z)
                    w = w / max(float(np.sum(w)), 1e-12)
                    fused = np.zeros_like(emb_norm, dtype=np.float64)
                    for wi, (me, ms, _sc) in zip(w.tolist(), cand_scores):
                        if wi <= 0.0:
                            continue
                        fused += float(wi) * _choquet_two_sources(emb_norm, spec_norm, mu_e=float(me), mu_s=float(ms))
                else:
                    raise SystemExit("--icv-aggregate must be one of: hard, softmax")
                icv_time = time.time() - t_icv

            order = np.argsort(-fused, kind="mergesort")
            ranking = order + 1

            wall = time.time() - t0
            np.savetxt(out_rank, ranking, delimiter=",", fmt="%d")
            info = {
                "dataset": ds,
                "fold": int(fold),
                "n": int(n),
                "d": int(d),
                "L": int(L),
                "fusion": "choquet_2src",
                "capacity_mode": str(args.capacity_mode),
                "mu_e": float(mu_e),
                "mu_s": float(mu_s),
                "interaction_I": float(1.0 - float(mu_e) - float(mu_s)),
                "icv_score": float(best_icv),
                "icv_folds": int(args.icv_folds),
                "icv_p_frac": float(args.icv_p),
                "icv_criterion": str(args.icv_criterion),
                "icv_aggregate": str(args.icv_aggregate),
                "softmax_tau": float(args.softmax_tau) if str(args.icv_aggregate).lower().strip() == "softmax" else None,
                "mu_grid": [float(x) for x in mu_grid.reshape(-1).tolist()],
                "fix_mu_e": float(args.fix_mu_e) if args.fix_mu_e is not None else None,
                "fix_mu_s": float(args.fix_mu_s) if args.fix_mu_s is not None else None,
                "wall_time_s": float(wall),
                "icv_time_s": float(icv_time),
            }
            out_info.write_text(json.dumps(info, indent=2))
            print(
                f"[{done}/{total}] ✓ {ds} fold{fold}  d={d}  "
                f"(mu_e,mu_s)=({mu_e:.2f},{mu_s:.2f})  I={info['interaction_I']:+.2f}  "
                f"t={wall:.1f}s (icv={icv_time:.1f}s)"
            )

    print(f"\n✓ Done. Rankings saved to: {out_dir}")


if __name__ == "__main__":
    main()

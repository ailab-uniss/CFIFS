#!/usr/bin/env python3
"""
CFIFS demo — run the full pipeline on an example dataset.

Usage
-----
    # Embedded scoring only (fast, CPU-friendly)
    python cfifs_demo.py

    # Full pipeline: embedded + spectral + Choquet fusion (paper default)
    python cfifs_demo.py --full

    # Use GPU acceleration
    python cfifs_demo.py --full --device cuda

    # Custom data
    python cfifs_demo.py --mat path/to/fold.mat --full
    # Paper ablation variants (embedded-only, different α / weight settings)
    python cfifs_demo.py --variant CFIFS_EMB    # full embedded solver, no fusion
    python cfifs_demo.py --variant ACSF_REG     # regression-only (α=0)
    python cfifs_demo.py --variant ACSF_LOG     # logistic-only  (α=1)
    python cfifs_demo.py --variant ACSF_NOWT    # no instance weights
    python cfifs_demo.py --variant ACSF_SIMPLE  # simplest ℓ₂₁ regression

    # Or set α and weights manually
    python cfifs_demo.py --alpha 0.5 --no-weights"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


# ── Helpers ─────────────────────────────────────────────────────────────

def _rank_uniform(v: np.ndarray) -> np.ndarray:
    """Empirical CDF (rank) normalisation to (0, 1)."""
    from scipy.stats import rankdata
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.size
    if n <= 1:
        return np.full_like(v, 0.5)
    return np.clip((rankdata(v, method="average") - 0.5) / n, 0.0, 1.0)


def _choquet_2src(e: np.ndarray, s: np.ndarray,
                  *, mu_e: float, mu_s: float) -> np.ndarray:
    """2-source Choquet integral (element-wise)."""
    e, s = np.asarray(e, np.float64), np.asarray(s, np.float64)
    out = np.empty_like(e)
    mask = e >= s
    out[mask] = s[mask] + (e[mask] - s[mask]) * mu_e
    out[~mask] = e[~mask] + (s[~mask] - e[~mask]) * mu_s
    return out


def _icv_select(emb_n, spec_n, X, Y, *, p_frac, cv, grid, mlknn_k,
                device, seed):
    """Select (μₑ, μₛ) via inner K-fold CV on ML-kNN (GM of micro/macro F1)."""
    from sklearn.model_selection import KFold
    from scipy import sparse as sp
    from mlfs.ml_knn_gpu import MLkNNConfig, MLkNNModel

    d = emb_n.shape[0]
    k_feat = max(1, int(round(p_frac * d)))
    if Y.min() < 0:
        Y = (Y > 0).astype(np.float64)

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    splits = list(kf.split(X))

    best_mu_e, best_mu_s, best_score = 0.5, 0.5, -1.0
    for mu_e in grid:
        for mu_s in grid:
            fused = _choquet_2src(emb_n, spec_n, mu_e=mu_e, mu_s=mu_s)
            sel = np.argpartition(-fused, k_feat - 1)[:k_feat]

            fold_mi, fold_ma = [], []
            for tr, va in splits:
                Xtr = sp.csr_matrix(X[np.ix_(tr, sel)].astype(np.float32))
                Xva = sp.csr_matrix(X[np.ix_(va, sel)].astype(np.float32))
                Ytr = sp.csr_matrix(Y[tr].astype(np.float32))
                Yva = Y[va].astype(np.float64)

                cfg = MLkNNConfig(k=min(mlknn_k, len(tr) - 1), s=1.0,
                                  metric="cosine", backend="torch",
                                  device=device)
                mdl = MLkNNModel(cfg)
                mdl.fit(Xtr, Ytr)
                pred = (mdl.predict_proba(Xva) >= 0.5).astype(np.float64)

                tp = float(np.sum(pred * Yva))
                prec = tp / max(float(np.sum(pred)), 1e-12)
                rec = tp / max(float(np.sum(Yva)), 1e-12)
                mi = 2 * prec * rec / max(prec + rec, 1e-12)
                fold_mi.append(mi)

                tp_l = np.sum(pred * Yva, axis=0)
                denom = np.sum(pred, axis=0) + np.sum(Yva, axis=0)
                m = denom > 0
                f1_l = np.where(m, 2 * tp_l / np.maximum(denom, 1e-12), 0.0)
                fold_ma.append(float(np.mean(f1_l[m])) if m.any() else 0.0)

            gm = float(np.sqrt(max(np.mean(fold_mi), 0) *
                                max(np.mean(fold_ma), 0)))
            if gm > best_score:
                best_score, best_mu_e, best_mu_s = gm, mu_e, mu_s

    return best_mu_e, best_mu_s, best_score


# ── Ablation variants (Table 5 of the paper) ───────────────────────────

ABLATION_VARIANTS = {
    "CFIFS":        dict(alpha=0.35, use_instance_weights=True,  full=True),
    "CFIFS_EMB":    dict(alpha=0.35, use_instance_weights=True,  full=False),
    "ACSF_REG":     dict(alpha=0.0,  use_instance_weights=True,  full=False),
    "ACSF_LOG":     dict(alpha=1.0,  use_instance_weights=True,  full=False),
    "ACSF_NOWT":    dict(alpha=0.35, use_instance_weights=False, full=False),
    "ACSF_SIMPLE":  dict(alpha=0.0,  use_instance_weights=False, full=False),
}


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    variant_names = list(ABLATION_VARIANTS.keys())
    ap = argparse.ArgumentParser(
        description="CFIFS demo — Choquet Fuzzy-Integral Feature Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ablation variants (--variant):\n"
               "  CFIFS        full pipeline (embedded + spectral + Choquet)\n"
               "  CFIFS_EMB    embedded solver only (α=0.35, weights ON)\n"
               "  ACSF_REG    regression-only (α=0, weights ON)\n"
               "  ACSF_LOG    logistic-only  (α=1, weights ON)\n"
               "  ACSF_NOWT   full solver, no instance weights\n"
               "  ACSF_SIMPLE simplest ℓ₂₁ regression (α=0, no weights)")
    ap.add_argument(
        "--mat", type=Path,
        default=ROOT / "example" / "emotions_fold0.mat",
        help="Path to a .mat file with X_train, Y_train (and optionally "
             "X_test, Y_test).  Default: bundled emotions dataset.")
    ap.add_argument(
        "--full", action="store_true",
        help="Run the full pipeline (embedded + spectral + Choquet fusion).  "
             "Without this flag only the embedded stage is executed.")
    ap.add_argument(
        "--variant", type=str, default=None, choices=variant_names,
        metavar="NAME",
        help="Run a named ablation variant (overrides --full/--alpha/--no-weights).")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Trade-off α ∈ [0,1]: 0 = regression-only, "
                         "1 = logistic-only (default: 0.35).")
    ap.add_argument("--no-weights", action="store_true",
                    help="Disable rarity-aware instance weighting.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Compute device (default: cpu).")
    ap.add_argument("--top-k", type=int, default=15,
                    help="Number of top features to display (default: 15).")
    args = ap.parse_args()

    # Resolve ablation configuration
    if args.variant is not None:
        vcfg = ABLATION_VARIANTS[args.variant]
        alpha = vcfg["alpha"]
        use_weights = vcfg["use_instance_weights"]
        run_full = vcfg["full"]
        print(f"Variant:  {args.variant}  "
              f"(α={alpha}, weights={'ON' if use_weights else 'OFF'}, "
              f"fusion={'YES' if run_full else 'NO'})\n")
    else:
        alpha = args.alpha if args.alpha is not None else 0.35
        use_weights = not args.no_weights
        run_full = args.full

    # ── Load data ───────────────────────────────────────────────────────
    mat_path: Path = args.mat
    if not mat_path.exists():
        sys.exit(f"Error: file not found: {mat_path}")
    mat = sio.loadmat(str(mat_path))
    X = np.asarray(mat["X_train"], dtype=np.float64)
    Y = np.asarray(mat["Y_train"], dtype=np.float64)
    if Y.min() < 0:
        Y = (Y > 0).astype(np.float64)
    n, d = X.shape
    L = int(Y.shape[1])
    print(f"Dataset:  {mat_path.stem}")
    print(f"  samples = {n},  features = {d},  labels = {L}\n")

    backend = "torch" if args.device == "cuda" else "numpy"

    # ── Stage 1: Embedded scoring ───────────────────────────────────────
    from mlfs import CFIFSParams, fit_cfifs

    t0 = time.time()
    emb_params = CFIFSParams(
        alpha=alpha,
        use_instance_weights=use_weights,
        backend=backend,
        device=args.device,
    )
    emb_ranking, emb_info = fit_cfifs(X, Y, emb_params)
    emb_scores = np.asarray(emb_info["scores"], dtype=np.float64)
    t_emb = time.time() - t0
    print(f"[Embedded]  {t_emb:.2f}s  (α={alpha}, weights={'ON' if use_weights else 'OFF'})")
    print(f"  top features: {emb_ranking[:args.top_k].tolist()}")

    if not run_full:
        print(f"\nDone (embedded only).  Ranking length = {len(emb_ranking)}")
        return

    # ── Stage 2: Spectral scoring ───────────────────────────────────────
    from mlfs.spectral_mlfs import SLAGDParams, _fit_numpy as _slagd_np

    t1 = time.time()
    slagd_params = SLAGDParams(alpha=0.70, label_sim="jaccard",
                               backend=backend, device=args.device)
    if args.device == "cuda":
        try:
            from mlfs.spectral_mlfs import _fit_torch as _slagd_gpu
            import torch
            dev = torch.device("cuda")
            spec_ranking, spec_scores, _ = _slagd_gpu(X, Y, slagd_params, dev)
        except ImportError:
            spec_ranking, spec_scores, _ = _slagd_np(X, Y, slagd_params)
    else:
        spec_ranking, spec_scores, _ = _slagd_np(X, Y, slagd_params)
    spec_scores = np.asarray(spec_scores, dtype=np.float64)
    t_spec = time.time() - t1
    print(f"[Spectral]  {t_spec:.2f}s  —  top features: "
          f"{spec_ranking[:args.top_k].tolist()}")

    # ── Stage 3: Rank normalisation ─────────────────────────────────────
    emb_norm = _rank_uniform(emb_scores)
    spec_norm = _rank_uniform(spec_scores)

    # ── Stage 4: Choquet fusion with inner-CV capacity selection ────────
    mu_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    t2 = time.time()
    mu_e, mu_s, icv_score = _icv_select(
        emb_norm, spec_norm, X, Y,
        p_frac=0.20, cv=3, grid=mu_grid, mlknn_k=10,
        device=args.device, seed=42)
    t_icv = time.time() - t2

    fused = _choquet_2src(emb_norm, spec_norm, mu_e=mu_e, mu_s=mu_s)
    ranking = np.argsort(-fused, kind="mergesort") + 1  # 1-based

    interaction = 1.0 - mu_e - mu_s
    print(f"[Choquet]   {t_icv:.2f}s  —  μₑ={mu_e:.2f}  μₛ={mu_s:.2f}  "
          f"I={interaction:+.2f}  ICV-GM={icv_score:.4f}")
    print(f"\nFinal ranking (top {args.top_k}): {ranking[:args.top_k].tolist()}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

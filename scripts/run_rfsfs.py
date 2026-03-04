#!/usr/bin/env python3
"""
RFSFS Baseline Runner
=====================

Reimplementation of the RFSFS algorithm (Li et al., 2023) as a ranking-only
wrapper compatible with the benchmark pipeline.

The core update rule is:
  W ← W * (X'Y + α(WS) + γW) / (X'XW + α(WD) + β(Q⊙W) + γ(𝟏W))
where S,D are the cosine-similarity and degree matrices of the label space,
Q = 1/(2|W|+ε), and (α,β,γ) are regularisation weights.

We grid-search (α,β,γ) ∈ {0.01, 0.1}³ (8 combos) as in the original code,
averaging over 3 random initialisations (t=50 iterations each), and pick
the combo with the best training-set Micro-F1 at p=20%.

Output: 1-based full feature ranking (descending importance).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import sklearn.metrics.pairwise


def _label_correlation(Y):
    """Cosine similarity matrix and its degree matrix for labels."""
    C = sklearn.metrics.pairwise.cosine_similarity(Y.T)
    Dc = np.diag(np.sum(C, axis=1))
    return C, Dc


def _rfsfs_solve(X, Y, alpha, beta, gamma, t=50, seed=0):
    """Run RFSFS multiplicative updates and return per-feature importance."""
    n, d = X.shape
    _n, l = Y.shape
    eps = np.finfo(np.float32).eps

    S, D = _label_correlation(Y)
    dd = np.ones((d, d))

    XTX = X.T @ X
    XTY = X.T @ Y

    rng = np.random.RandomState(seed)
    W = rng.rand(d, l)

    for _i in range(t):
        Q = 1.0 / np.maximum(2.0 * np.abs(W), eps)
        Wu = XTY + alpha * (W @ S) + gamma * W
        Wd = XTX @ W + alpha * (W @ D) + beta * (Q * W) + gamma * (dd @ W)
        W = W * (Wu / np.maximum(Wd, eps))

    importance = np.linalg.norm(W, axis=1, ord=2)
    return importance


def rfsfs_ranking(X, Y, t=50, n_avg=3):
    """Run RFSFS with grid search over (α,β,γ) and return 1-based ranking.

    Grid search: (α,β,γ) ∈ {0.01, 0.1}³  (8 combos).
    Each combo is averaged over n_avg random initialisations.
    We pick the combo whose average importance vector yields the best
    training-set reconstruction ‖XW - Y‖_F  (lower is better).
    """
    best_importance = None
    best_resid = np.inf

    alphas = [0.01, 0.1]
    betas = [0.01, 0.1]
    gammas = [0.01, 0.1]

    n, d = X.shape

    for a in alphas:
        for b in betas:
            for g in gammas:
                imp_sum = np.zeros(d)
                resid_sum = 0.0
                for avg in range(n_avg):
                    imp = _rfsfs_solve(X, Y, a, b, g, t=t, seed=avg * 1000 + 7)
                    imp_sum += imp
                imp_avg = imp_sum / n_avg

                # Pick top-20% features and measure training reconstruction
                k = max(1, int(0.20 * d))
                top_k = np.argsort(-imp_avg)[:k]
                Xk = X[:, top_k]
                # Simple LS reconstruction residual
                W_ls, _, _, _ = np.linalg.lstsq(Xk, Y, rcond=None)
                resid = float(np.sum((Y - Xk @ W_ls) ** 2))

                if resid < best_resid:
                    best_resid = resid
                    best_importance = imp_avg.copy()

    # Sort by descending importance → 1-based ranking
    order = np.argsort(-best_importance)
    ranking = order + 1
    return ranking


def load_fold(data_dir: Path, dataset: str, fold: int):
    fp = data_dir / dataset / f"fold{fold}.mat"
    if not fp.exists():
        return None, None
    mat = sio.loadmat(str(fp))
    Xtr = np.asarray(mat["X_train"], dtype=np.float64)
    Ytr = np.asarray(mat["Y_train"], dtype=np.float64)
    return Xtr, Ytr


def main():
    ap = argparse.ArgumentParser(description="RFSFS baseline runner")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--method", type=str, default="RFSFS")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--t-iters", type=int, default=50, help="Number of MU iterations")
    ap.add_argument("--n-avg", type=int, default=3, help="Number of random inits to average")
    args = ap.parse_args()

    out_dir = args.results_dir / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets) * args.folds
    done = 0

    for ds in args.datasets:
        for fold in range(args.folds):
            done += 1
            out_rank = out_dir / f"{ds}_fold{fold}_ranking.csv"
            out_time = out_dir / f"{ds}_fold{fold}_time.txt"

            if out_rank.exists() and out_time.exists() and not args.overwrite:
                print(f"[{done}/{total}] SKIP {ds} fold{fold}")
                continue

            Xtr, Ytr = load_fold(args.data_dir, ds, fold)
            if Xtr is None:
                print(f"[{done}/{total}] SKIP {ds} fold{fold} (no data)")
                continue

            n, d = Xtr.shape
            t0 = time.time()
            ranking = rfsfs_ranking(Xtr, Ytr, t=args.t_iters, n_avg=args.n_avg)
            wall = time.time() - t0

            np.savetxt(out_rank, ranking, delimiter=",", fmt="%d")
            out_time.write_text(f"{wall:.6f}\n")
            print(f"[{done}/{total}] ✓ {ds} fold{fold}  d={d}  t={wall:.1f}s")

    print(f"\n✓ Done. Rankings saved to: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create 10-fold stratified MAT splits from dense_benchmark_v3 datasets.
Directly reads dataset.npz (X, Y dense arrays), creates iterative-stratified
10-fold splits with MinMax scaling, and saves fold{i}.mat + meta.json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skmultilearn.model_selection import IterativeStratification
from scipy.io import savemat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=Path("data/dense_benchmark_v3"))
    ap.add_argument("--out-root", type=Path, default=Path("data/panorama30_matlab_minmax_cv10"))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for ds in args.datasets:
        src = args.src_root / ds / "dataset.npz"
        if not src.exists():
            print(f"[SKIP] {ds}: no dataset.npz")
            continue

        data = np.load(str(src), allow_pickle=True)
        X = data["X"].astype(np.float64)
        Y = data["Y"].astype(np.int8)
        n, d = X.shape
        L = Y.shape[1]
        print(f"[{ds}] n={n}, d={d}, L={L}")

        out_dir = args.out_root / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        # Iterative stratified K-fold
        rng = np.random.RandomState(args.seed)
        order = rng.permutation(n)
        X_shuf = X[order]
        Y_shuf = Y[order]

        stratifier = IterativeStratification(
            n_splits=args.folds, order=2
        )
        fold_indices = []
        for train_idx, test_idx in stratifier.split(X_shuf, Y_shuf):
            fold_indices.append((train_idx, test_idx))

        for i, (train_idx, test_idx) in enumerate(fold_indices):
            X_train = X_shuf[train_idx].copy()
            X_test  = X_shuf[test_idx].copy()
            Y_train = Y_shuf[train_idx].copy()
            Y_test  = Y_shuf[test_idx].copy()

            # MinMax scaling (fit on train, transform both)
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)
            # Clip test to [0,1]
            np.clip(X_test, 0.0, 1.0, out=X_test)

            mat_path = out_dir / f"fold{i}.mat"
            savemat(str(mat_path), {
                "X_train": X_train,
                "Y_train": Y_train.astype(np.float64),
                "X_test":  X_test,
                "Y_test":  Y_test.astype(np.float64),
            }, do_compression=True)
            print(f"  fold{i}: train={len(train_idx)}, test={len(test_idx)} -> {mat_path}")

        meta = {
            "dataset": ds,
            "n_total": int(n),
            "n_features": int(d),
            "n_labels": int(L),
            "reference_split": {"train": int(len(fold_indices[0][0])), "test": int(len(fold_indices[0][1]))},
            "n_folds": int(args.folds),
            "seed": int(args.seed),
            "reference_test_size": round(len(fold_indices[0][1]) / n, 4),
            "effective_test_size": round(1.0 / args.folds, 4),
            "scaler": "minmax",
            "split_mode": "kfold",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  -> {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create a "frozen" CV10 fold directory for a larger dataset panorama without
regenerating folds for datasets that already have canonical folds saved.

Workflow:
  1) Choose a canonical folds root (e.g., data/paper_matlab_minmax_cv10) that
     you promise never to rewrite.
  2) Build a panorama dataset root (NPZ) containing all datasets you want.
  3) Run this script to:
     - symlink canonical fold folders for overlapping datasets
     - generate CV10 folds only for the missing datasets into the frozen folder
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _symlink_tree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)


def _has_fold_dir(p: Path) -> bool:
    return p.is_dir() and (p / "fold0.mat").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-folds", type=Path, default=Path("data/paper_matlab_minmax_cv10"))
    ap.add_argument("--panorama-npz-root", type=Path, default=Path("data/panorama30_protocol"))
    ap.add_argument("--out-folds", type=Path, default=Path("data/panorama30_matlab_minmax_cv10_frozen"))
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scaler", choices=["none", "zscore", "minmax"], default="minmax")
    ap.add_argument("--datasets", nargs="*", default=None, help="Datasets to include (default: infer from panorama-npz-root).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    canonical = (repo_root / args.canonical_folds).resolve() if not args.canonical_folds.is_absolute() else args.canonical_folds
    npz_root = (repo_root / args.panorama_npz_root).resolve() if not args.panorama_npz_root.is_absolute() else args.panorama_npz_root
    out_root = (repo_root / args.out_folds).resolve() if not args.out_folds.is_absolute() else args.out_folds
    out_root.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = sorted([p.name for p in npz_root.iterdir() if p.is_dir()])

    # 1) Link canonical fold dirs when present.
    linked = 0
    missing = []
    for ds in datasets:
        src = canonical / ds
        dst = out_root / ds
        if _has_fold_dir(src):
            _symlink_tree(src, dst)
            linked += 1
        else:
            missing.append(ds)

    # 2) Generate folds for missing datasets.
    if missing:
        cmd = [
            "python3",
            "scripts/export_cv_splits_to_mat.py",
            "--data-root",
            str(npz_root),
            "--output-dir",
            str(out_root),
            "--folds",
            str(int(args.folds)),
            "--seed",
            str(int(args.seed)),
            "--scaler",
            str(args.scaler),
            "--split-mode",
            "kfold",
            "--datasets",
            *missing,
        ]
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(repo_root), check=True)

    print(f"✓ Frozen folds ready: {out_root}")
    print(f"  linked canonical: {linked}")
    print(f"  generated new:     {len(missing)}")


if __name__ == "__main__":
    main()


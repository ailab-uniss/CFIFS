#!/usr/bin/env python3
"""
CFIFS Solver Ablation Runner
============================

Runs ablation variants of the CFIFS embedded solver (NO spectral fusion, β=0):

  CFIFS_EMB       : α=0.35, instance weights ON  → full solver, no fusion
  ACSF_REG       : α=0.0,  instance weights ON  → regression-only, with weights
  ACSF_LOG       : α=1.0,  instance weights ON  → logistic-only, with weights
  ACSF_NOWT      : α=0.35, instance weights OFF → full solver, no weights
  ACSF_SIMPLE    : α=0.0,  instance weights OFF → simplest ℓ₂₁ regression (PLST)

All variants:
  1. Run the embedded solver with the specified configuration.
  2. Rank features by descending ||W_{j:}||_2.
  3. Save ranking.csv + info.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlfs.cfifs_embedded import CFIFSParams, fit_cfifs

# ---- variant configs --------------------------------------------------------

VARIANTS = {
    "CFIFS_EMB":    dict(alpha=0.35, use_instance_weights=True),
    "ACSF_REG":    dict(alpha=0.0,  use_instance_weights=True),
    "ACSF_LOG":    dict(alpha=1.0,  use_instance_weights=True),
    "ACSF_NOWT":   dict(alpha=0.35, use_instance_weights=False),
    "ACSF_SIMPLE": dict(alpha=0.0,  use_instance_weights=False),
}


# ---- helpers ----------------------------------------------------------------

def load_fold(data_dir: Path, dataset: str, fold: int):
    fp = data_dir / dataset / f"fold{fold}.mat"
    if not fp.exists():
        return None, None
    mat = sio.loadmat(str(fp))
    Xtr = np.asarray(mat["X_train"], dtype=np.float64)
    Ytr = np.asarray(mat["Y_train"], dtype=np.float64)
    return Xtr, Ytr


def find_datasets(data_dir: Path, explicit: list | None) -> list[str]:
    if explicit:
        return list(explicit)
    out = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "fold0.mat").exists():
            out.append(p.name)
    return out


# ---- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="CFIFS solver ablation (EMB-only, no fusion)")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--method", type=str, required=True,
                    choices=list(VARIANTS.keys()),
                    help="Ablation variant to run")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    vcfg = VARIANTS[args.method]
    datasets = find_datasets(args.data_dir, args.datasets)
    out_dir = args.results_dir / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    cfifs_params = CFIFSParams(
        alpha=vcfg["alpha"],
        use_instance_weights=vcfg["use_instance_weights"],
        backend="torch",
        device=args.device,
    )

    total = len(datasets) * args.folds
    done = 0

    for ds in datasets:
        for fold in range(args.folds):
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
            L = Ytr.shape[1]
            t0 = time.time()

            # Run embedded solver only (no spectral, no fusion)
            ranking, info = fit_cfifs(Xtr, Ytr, cfifs_params)
            wall = time.time() - t0

            # Save
            np.savetxt(out_rank, ranking, delimiter=",", fmt="%d")
            meta = {
                "dataset": ds,
                "fold": fold,
                "n": int(n),
                "d": int(d),
                "L": int(L),
                "method": args.method,
                "alpha": float(vcfg["alpha"]),
                "use_instance_weights": bool(vcfg["use_instance_weights"]),
                "beta_icv": 0.0,  # no fusion
                "wall_time_s": float(wall),
                "iterations": int(info.get("iterations", 0)),
            }
            out_info.write_text(json.dumps(meta, indent=2))
            print(f"[{done}/{total}] ✓ {ds} fold{fold}  d={d}  "
                  f"α={vcfg['alpha']}  wt={vcfg['use_instance_weights']}  "
                  f"t={wall:.1f}s")

    print(f"\n✓ Done. Rankings saved to: {out_dir}")


if __name__ == "__main__":
    main()

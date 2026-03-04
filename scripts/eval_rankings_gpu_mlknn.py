#!/usr/bin/env python3
"""
GPU-accelerated p-grid evaluation of feature rankings with ML-kNN.

Uses src/mlfs/ml_knn_gpu.py (torch backend, cosine metric) for fast
GPU-based ML-kNN evaluation.  Falls back to the same metric functions as
eval_rankings_py_pgrid_fast.py so the JSON outputs are fully compatible
with aggregate_kgrid_and_make_tables.py.

Inputs:
  results_dir/<METHOD>/<DATASET>_fold<fold>_ranking.csv
  data_dir/<DATASET>/fold<fold>.mat    (or --data-dirs for multiple roots)

Outputs:
  results_dir/<METHOD>/<DATASET>_fold<fold><out_suffix>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "mlfs"))

# Re-use the metric functions from the existing evaluator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_rankings_py_pgrid_fast import (
    avg_precision_example_based,
    hamming_loss,
    macro_f1,
    micro_f1,
    one_error,
    pr_auc_micro_macro,
)

# Import GPU ML-kNN
import importlib.util

_gpu_path = str(REPO_ROOT / "src" / "mlfs" / "ml_knn_gpu.py")
_spec = importlib.util.spec_from_file_location("ml_knn_gpu", _gpu_path)
_mod = importlib.util.module_from_spec(_spec)
# Fix: register the module in sys.modules so that dataclass can resolve __module__
sys.modules["ml_knn_gpu"] = _mod
_spec.loader.exec_module(_mod)
MLkNNConfig = _mod.MLkNNConfig
MLkNNModel = _mod.MLkNNModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fold(data_dirs: List[Path], dataset: str, fold: int):
    for dd in data_dirs:
        p = dd / dataset / f"fold{fold}.mat"
        if p.exists():
            mat = sio.loadmat(str(p))
            Xtr = np.asarray(mat["X_train"], dtype=np.float64)
            Ytr = np.asarray(mat["Y_train"], dtype=np.float64)
            Xte = np.asarray(mat["X_test"], dtype=np.float64)
            Yte = np.asarray(mat["Y_test"], dtype=np.float64)
            return Xtr, Ytr, Xte, Yte
    raise FileNotFoundError(f"fold{fold}.mat not found for {dataset} in any data dir")


def load_ranking(results_dir: Path, method: str, dataset: str, fold: int):
    path = results_dir / method / f"{dataset}_fold{fold}_ranking.csv"
    r = np.loadtxt(str(path), delimiter=",").astype(np.int64).reshape(-1)
    return r - 1  # 1-based → 0-based


def eval_mlknn_gpu_pgrid(
    Xtr: np.ndarray,  # (n_train, d)
    Ytr_pm: np.ndarray,  # (L, n_train) in {-1,+1}
    Xte: np.ndarray,  # (n_test, d)
    Yte_pm: np.ndarray,  # (L, n_test)  in {-1,+1}
    ranking: np.ndarray,  # 0-based feature indices
    k_values: List[int],  # sorted unique feature counts (p-grid)
    *,
    mlknn_k: int = 10,
    mlknn_s: float = 1.0,
    backend: str = "torch",
    device: str = "cuda",
) -> Dict[str, list]:
    """Evaluate ML-kNN at each feature budget in *k_values*."""
    L, n_train = Ytr_pm.shape
    n_test = Yte_pm.shape[1]

    # Convert Y from {-1,+1} (L, n) → {0,1} (n, L) sparse for the MLkNN API
    Ytr_01 = ((Ytr_pm.T + 1) // 2).astype(np.float32)  # (n_train, L)
    Ytr_sp = sparse.csr_matrix(Ytr_01)

    micro_list, macro_list, hl_list = [], [], []
    mi_pr_list, ma_pr_list = [], []
    ap_list, oe_list = [], []
    labels_used_last, labels_total_last = 0, int(L)

    for k_feat in k_values:
        sel = ranking[:k_feat]
        Xtr_sub = Xtr[:, sel]
        Xte_sub = Xte[:, sel]

        Xtr_sp = sparse.csr_matrix(Xtr_sub.astype(np.float32))
        Xte_sp = sparse.csr_matrix(Xte_sub.astype(np.float32))

        cfg = MLkNNConfig(
            k=min(mlknn_k, n_train - 1),
            s=mlknn_s,
            metric="cosine",
            backend=backend,
            device=device,
        )
        model = MLkNNModel(cfg)
        model.fit(Xtr_sp, Ytr_sp)
        probs = model.predict_proba(Xte_sp)  # (n_test, L)

        # probs → (L, n_test) for metric functions
        Outputs = probs.T  # (L, n_test)
        Pre = np.where(Outputs < 0.5, -1, 1).astype(np.int8)

        micro_list.append(micro_f1(Pre, Yte_pm))
        macro_list.append(macro_f1(Pre, Yte_pm))
        hl_list.append(hamming_loss(Pre, Yte_pm))
        mi_ap, ma_ap, lu, lt = pr_auc_micro_macro(Outputs, Yte_pm)
        mi_pr_list.append(mi_ap)
        ma_pr_list.append(ma_ap)
        labels_used_last = int(lu)
        labels_total_last = int(lt)
        ap_list.append(avg_precision_example_based(Outputs, Yte_pm))
        oe_list.append(one_error(Outputs, Yte_pm))

    return {
        "k_values": k_values,
        "micro_f1": micro_list,
        "macro_f1": macro_list,
        "hamming_loss": hl_list,
        "micro_pr_auc": mi_pr_list,
        "macro_pr_auc": ma_pr_list,
        "avg_precision": ap_list,
        "one_error": oe_list,
        "macro_pr_auc_labels_used": labels_used_last,
        "macro_pr_auc_labels_total": labels_total_last,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--data-dirs", nargs="+", type=Path, required=True,
                    help="One or more data root directories to search for fold .mat files.")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mlknn-k", type=int, default=10)
    ap.add_argument("--mlknn-smooth", type=float, default=1.0)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--p-max", type=float, default=0.50)
    ap.add_argument("--p-step", type=float, default=0.05)
    ap.add_argument("--p-target", type=float, default=0.20)
    ap.add_argument("--out-suffix", type=str, default="_pgrid_metrics.json")
    ap.add_argument("--skip-existing", action="store_true", default=False)
    ap.add_argument("--backend", type=str, default="torch", choices=["torch", "sklearn"])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    data_dirs: List[Path] = list(args.data_dirs)
    folds = int(args.folds)

    if args.datasets is None:
        # Auto-discover from first data dir
        datasets = sorted([p.name for p in data_dirs[0].iterdir()
                           if p.is_dir() and (p / "fold0.mat").exists()])
    else:
        datasets = list(args.datasets)

    p_values = np.arange(float(args.p_min), float(args.p_max) + 1e-12, float(args.p_step))
    p_values = p_values[(p_values > 0) & (p_values <= 1)]
    p_target = float(args.p_target)

    total = len(args.methods) * len(datasets) * folds
    done = 0

    for method in args.methods:
        method_dir = results_dir / method
        if not method_dir.exists():
            print(f"WARNING: method dir missing: {method_dir}")
            continue
        for ds in datasets:
            for fold in range(folds):
                done += 1
                base = f"{ds}_fold{fold}"
                ranking_path = method_dir / f"{base}_ranking.csv"
                out_path = method_dir / f"{base}{args.out_suffix}"

                if not ranking_path.exists():
                    continue
                if args.skip_existing and out_path.exists():
                    try:
                        obj = json.loads(out_path.read_text(encoding="utf-8"))
                        if "micro_f1" in obj and "p_values" in obj:
                            continue
                    except Exception:
                        pass

                Xtr, Ytr, Xte, Yte = load_fold(data_dirs, ds, fold)
                n_features = int(Xtr.shape[1])
                n_labels = int(Ytr.shape[1])

                ranking = load_ranking(results_dir, method, ds, fold)

                k_for_p = np.maximum(1, np.round(p_values * n_features).astype(int))
                k_for_p = np.minimum(k_for_p, n_features)
                k_unique = list(dict.fromkeys(k_for_p.tolist()))
                kmax = int(max(k_unique))

                # Y → {-1,+1} (L, n) layout (same convention as original evaluator)
                Ytr_pm = (Ytr > 0).astype(np.int8).T
                Ytr_pm[Ytr_pm == 0] = -1
                Yte_pm = (Yte > 0).astype(np.int8).T
                Yte_pm[Yte_pm == 0] = -1

                t0 = time.time()
                curve = eval_mlknn_gpu_pgrid(
                    Xtr, Ytr_pm, Xte, Yte_pm, ranking,
                    k_unique,
                    mlknn_k=int(args.mlknn_k),
                    mlknn_s=float(args.mlknn_smooth),
                    backend=args.backend,
                    device=args.device,
                )
                eval_time = time.time() - t0

                # Map back to p grid
                k_arr = np.asarray(curve["k_values"], dtype=int)
                metrics_keys = ["micro_f1", "macro_f1", "hamming_loss",
                                "micro_pr_auc", "macro_pr_auc", "avg_precision", "one_error"]
                grid = {}
                for mk in metrics_keys:
                    grid[mk] = np.zeros(p_values.size, dtype=np.float64)
                for i, kk in enumerate(k_for_p):
                    idx = int(np.where(k_arr == int(kk))[0][0])
                    for mk in metrics_keys:
                        grid[mk][i] = float(curve[mk][idx])

                idx_t = int(np.argmin(np.abs(p_values - p_target)))
                p_used = float(p_values[idx_t])

                obj = {
                    "grid_mode": "pgrid",
                    "p_values": [float(x) for x in p_values],
                    "k_values": [int(x) for x in k_for_p],
                }
                for mk in metrics_keys:
                    obj[mk] = [float(x) for x in grid[mk]]
                    obj[f"{mk}_at_p_target"] = float(grid[mk][idx_t])
                    obj[f"{mk}_mean"] = float(np.mean(grid[mk]))

                obj["p_target"] = float(p_target)
                obj["p_target_used"] = p_used
                obj["macro_pr_auc_labels_used"] = int(curve.get("macro_pr_auc_labels_used", 0))
                obj["macro_pr_auc_labels_total"] = int(curve.get("macro_pr_auc_labels_total", n_labels))
                obj["n_features"] = int(n_features)
                obj["n_labels"] = int(n_labels)
                obj["time_eval_seconds"] = float(eval_time)
                obj["evaluator"] = "gpu_mlknn"
                obj["mlknn_metric"] = "cosine"

                out_path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
                tag = f"MiF1={obj['micro_f1_at_p_target']:.4f}"
                print(f"[{done}/{total}] {method} | {ds} | fold{fold} ✓  {tag}  ({eval_time:.1f}s)")

    print("✓ Done.")


if __name__ == "__main__":
    main()

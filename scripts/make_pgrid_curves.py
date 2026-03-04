#!/usr/bin/env python3
"""
Aggregate p-grid ML-kNN curves and (optionally) plot them.

Inputs:
  results_dir/<METHOD>/<DATASET>_fold<FOLD>_pgrid_metrics.json

Each JSON contains:
  p_values: list[float]    (e.g., 0.05..0.50)
  micro_f1/macro_f1/hamming_loss: list[float] aligned to p_values
Optionally (when present in the JSONs):
  micro_pr_auc/macro_pr_auc: list[float]  (higher is better)
  avg_precision: list[float]             (example-based AP / LRAP; higher is better)
  one_error: list[float]                 (lower is better)

Outputs:
  - CSVs with mean-over-datasets curves
  - A single PDF with 3 subplots (Micro-F1, Macro-F1, Hamming Loss)
  - Optionally, a 2x3 PDF that also includes Micro/Macro PR-AUC and AvgPrec
  - A per-dataset grid PDF (16 rows × 3 cols by default)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


METRICS_MAIN = ["micro_f1", "macro_f1", "hamming_loss"]
METRICS_EXTRA = ["micro_pr_auc", "macro_pr_auc", "avg_precision", "one_error"]
METRICS_EXTRA_FIG = ["micro_pr_auc", "macro_pr_auc", "avg_precision"]


@dataclass(frozen=True)
class Curve:
    p: np.ndarray  # shape [m]
    y: np.ndarray  # shape [m]


def load_curve(path: Path, metric: str) -> Curve:
    obj = json.loads(path.read_text(encoding="utf-8"))
    p = np.asarray(obj["p_values"], dtype=float)
    y = np.asarray(obj[metric], dtype=float)
    if p.ndim != 1 or y.ndim != 1 or p.shape != y.shape:
        raise ValueError(f"Invalid curve shapes in {path.name}")
    return Curve(p=p, y=y)


def mean_curve(curves: List[Curve]) -> Curve:
    if not curves:
        raise ValueError("No curves to average.")
    p0 = curves[0].p
    for c in curves[1:]:
        if c.p.shape != p0.shape or np.max(np.abs(c.p - p0)) > 1e-9:
            raise ValueError("Inconsistent p grids across folds.")
    y = np.mean(np.stack([c.y for c in curves], axis=0), axis=0)
    return Curve(p=p0, y=y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=10, help="Number of folds (default: 10).")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--xscale",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="X-axis scale for % selected features (default: linear).",
    )
    ap.add_argument(
        "--per-dataset-grid",
        action="store_true",
        help="Also write a per-dataset plot grid (rows=datasets plus an optional mean row; cols=metrics).",
    )
    ap.add_argument(
        "--per-dataset-grid-pages",
        type=int,
        default=2,
        help="Number of pages to split the per-dataset grid into (default: 2).",
    )
    ap.add_argument(
        "--include-mean-row",
        action="store_true",
        default=False,
        help="When --per-dataset-grid is set, append a final row with the mean-over-datasets curve.",
    )
    ap.add_argument("--display-map", type=Path, default=None, help="JSON map method_id -> display name.")
    ap.add_argument(
        "--include-extra-metrics",
        action="store_true",
        default=False,
        help="Also aggregate and plot AvgPrec (example-based) and One-error if present in JSONs.",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    display_map: Dict[str, str] = {}
    if args.display_map is not None:
        display_map = json.loads(args.display_map.read_text(encoding="utf-8"))

    # Infer datasets if not provided.
    datasets: List[str]
    if args.datasets is None:
        ds_set = set()
        for m in args.methods:
            for f in (args.results_dir / m).glob("*_fold*_pgrid_metrics.json"):
                name = f.name
                # dataset may contain underscores; split from _fold
                ds = name.rsplit("_fold", 1)[0]
                ds_set.add(ds)
        datasets = sorted(ds for ds in ds_set if ds.lower() != "reference")
    else:
        datasets = list(args.datasets)

    # curves_by_metric[metric][method] = Curve(mean over datasets of fold means)
    metrics = list(METRICS_MAIN)
    if args.include_extra_metrics:
        metrics.extend(METRICS_EXTRA)
    curves_by_metric: Dict[str, Dict[str, Curve]] = {metric: {} for metric in metrics}
    # curves_by_metric_ds[metric][dataset][method] = Curve(mean over folds)
    curves_by_metric_ds: Dict[str, Dict[str, Dict[str, Curve]]] = {metric: {} for metric in metrics}

    for metric in metrics:
        for method in args.methods:
            per_dataset: List[Curve] = []
            for ds in datasets:
                per_fold: List[Curve] = []
                for fold in range(args.folds):
                    path = args.results_dir / method / f"{ds}_fold{fold}_pgrid_metrics.json"
                    if not path.exists():
                        raise SystemExit(f"Missing: {path}")
                    per_fold.append(load_curve(path, metric))
                ds_curve = mean_curve(per_fold)
                per_dataset.append(ds_curve)
                curves_by_metric_ds[metric].setdefault(ds, {})[method] = ds_curve
            curves_by_metric[metric][method] = mean_curve(per_dataset)

    # Write CSVs
    for metric in metrics:
        p = next(iter(curves_by_metric[metric].values())).p
        csv_path = out_dir / f"pgrid_curve_{metric}.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("p," + ",".join(args.methods) + "\n")
            for i, pv in enumerate(p):
                row = [f"{pv:.4f}"]
                for m in args.methods:
                    try:
                        row.append(f"{curves_by_metric[metric][m].y[i]:.6f}")
                    except IndexError:
                        print(f"Error on {m}, len y: {len(curves_by_metric[metric][m].y)}, list p: {len(p)}, metric: {metric}")
                        raise
                f.write(",".join(row) + "\n")

    if args.no_plot:
        print(f"✓ Wrote p-grid curve CSVs to: {out_dir}")
        return

    import matplotlib.pyplot as plt

    # Use both colors and markers to distinguish methods robustly.
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
    linestyles = ["-", "--", "-.", ":"]

    titles = {
        "micro_f1": "Micro-F1",
        "macro_f1": "Macro-F1",
        "hamming_loss": "Hamming Loss",
        "micro_pr_auc": "Micro PR-AUC",
        "macro_pr_auc": "Macro PR-AUC",
        "avg_precision": "AvgPrec (example-based)",
        "one_error": "One-error",
    }

    if args.include_extra_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2), constrained_layout=False)
        plot_metrics = list(METRICS_MAIN) + list(METRICS_EXTRA_FIG)
        for ax, metric in zip(axes.reshape(-1), plot_metrics):
            for method in args.methods:
                disp = display_map.get(method, method)
                c = curves_by_metric[metric][method]
                mi = args.methods.index(method)
                ax.plot(
                    100 * c.p,
                    c.y,
                    linewidth=1.6,
                    marker=markers[mi % len(markers)],
                    linestyle=linestyles[(mi // len(markers)) % len(linestyles)],
                    markersize=4.5,
                    label=disp,
                )
            ax.set_title(titles[metric], fontsize=11)
            ax.set_xlabel("% selected features", fontsize=10)
            if str(args.xscale).lower().strip() == "log":
                ax.set_xscale("log")
            ax.grid(True, alpha=0.25)
            if metric in ("hamming_loss", "one_error"):
                ax.set_ylabel("lower is better", fontsize=10)
            else:
                ax.set_ylabel("higher is better", fontsize=10)
            ax.tick_params(axis="both", labelsize=9)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=10,
        )
        fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.16, hspace=0.35, wspace=0.22)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=False)
        for ax, metric in zip(axes, METRICS_MAIN):
            for method in args.methods:
                disp = display_map.get(method, method)
                c = curves_by_metric[metric][method]
                mi = args.methods.index(method)
                ax.plot(
                    100 * c.p,
                    c.y,
                    linewidth=1.6,
                    marker=markers[mi % len(markers)],
                    linestyle=linestyles[(mi // len(markers)) % len(linestyles)],
                    markersize=4.5,
                    label=disp,
                )
            ax.set_title(titles[metric])
            ax.set_xlabel("% selected features")
            if str(args.xscale).lower().strip() == "log":
                ax.set_xscale("log")
            ax.grid(True, alpha=0.25)
            if metric.endswith("loss"):
                ax.set_ylabel("lower is better")
            else:
                ax.set_ylabel("higher is better")
            ax.tick_params(axis="both", labelsize=8)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.14),
            fontsize=9,
        )
        fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.32, wspace=0.20)

    pdf_path = out_dir / "pgrid_curves.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    if args.per_dataset_grid:
        grid_datasets = list(datasets)
        if args.include_mean_row:
            grid_datasets.append("MEAN")

        pages = max(1, int(args.per_dataset_grid_pages))
        # Split datasets across pages as evenly as possible.
        chunks: list[list[str]] = []
        for pi in range(pages):
            start = (len(grid_datasets) * pi) // pages
            end = (len(grid_datasets) * (pi + 1)) // pages
            chunk = grid_datasets[start:end]
            if chunk:
                chunks.append(chunk)

        for pi, chunk in enumerate(chunks, start=1):
            nrows = len(chunk)
            ncols = len(METRICS_MAIN)

            fig2, axes2 = plt.subplots(
                nrows,
                ncols,
                figsize=(10.0, max(10.0, 1.55 * nrows)),
                sharex="col",
                constrained_layout=False,
            )

            if nrows == 1:
                axes2 = np.asarray([axes2])
            if ncols == 1:
                axes2 = np.asarray([[ax] for ax in axes2])

            for ri, ds in enumerate(chunk):
                for ci, metric in enumerate(METRICS_MAIN):
                    ax = axes2[ri][ci]
                    if ds == "MEAN":
                        src = {m: curves_by_metric[metric][m] for m in args.methods}
                        title_ds = "Mean"
                    else:
                        src = curves_by_metric_ds[metric][ds]
                        s = str(ds)
                        title_ds = s.capitalize() if s else ""

                    for mi, method in enumerate(args.methods):
                        disp = display_map.get(method, method)
                        c = src[method]
                        ax.plot(
                            100 * c.p,
                            c.y,
                            linewidth=1.0,
                            marker=markers[mi % len(markers)],
                            linestyle=linestyles[(mi // len(markers)) % len(linestyles)],
                            markersize=3.2,
                            label=disp,
                        )

                    if ri == 0:
                        ax.set_title(titles[metric], fontsize=9)
                    if ci == 0:
                        ax.set_ylabel(title_ds, fontsize=8)
                    if str(args.xscale).lower().strip() == "log":
                        ax.set_xscale("log")
                    ax.grid(True, alpha=0.18)
                    ax.tick_params(axis="both", labelsize=7, pad=1.5)

            for ci in range(ncols):
                axes2[-1][ci].set_xlabel("% selected features", fontsize=9)

            handles2, labels2 = axes2[0][0].get_legend_handles_labels()
            fig2.legend(
                handles2,
                labels2,
                loc="lower center",
                ncol=min(4, len(labels2)),
                frameon=False,
                bbox_to_anchor=(0.5, 0.004),
                fontsize=8,
            )

            fig2.subplots_adjust(left=0.16, right=0.99, top=0.97, bottom=0.07, hspace=0.30, wspace=0.26)

            pdf_grid = out_dir / f"pgrid_curves_by_dataset_grid_part{pi}.pdf"
            fig2.savefig(pdf_grid, bbox_inches="tight")
            plt.close(fig2)

    print(f"✓ Wrote p-grid curve CSVs and plots to: {out_dir}")


if __name__ == "__main__":
    main()

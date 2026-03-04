#!/usr/bin/env python3
"""
Build compact ORBIT ablation LaTeX tables from an aggregated p=20 CSV.

Inputs:
  - benchmark_means_p20.csv produced by scripts/aggregate_kgrid_and_make_tables.py
    (dataset-level means, already averaged over folds).

Outputs (written to --out-dir):
  - table_ablation_summary.tex : mean ± std across datasets
  - table_ablation_steps.tex   : marginal step effects (Δ%, W/T/L, Wilcoxon p)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import wilcoxon


def _read_benchmark_means(csv_path: Path) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    """
    Returns:
      datasets: list of dataset names (as found in CSV)
      values[metric][dataset][method] = float
    """
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header.")
        for r in reader:
            rows.append(r)

    if not rows:
        raise SystemExit(f"Empty CSV: {csv_path}")

    datasets: list[str] = []
    values: dict[str, dict[str, dict[str, float]]] = {}
    for r in rows:
        ds = str(r["dataset"])
        metric = str(r["metric"])
        if ds not in datasets:
            datasets.append(ds)
        values.setdefault(metric, {}).setdefault(ds, {})
        for k, v in r.items():
            if k in ("dataset", "metric"):
                continue
            if v is None or v == "":
                continue
            values[metric][ds][k] = float(v)
    return datasets, values


def _mean_std_across_datasets(
    datasets: Iterable[str],
    values: dict[str, dict[str, float]],
    methods: list[str],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for m in methods:
        arr = np.array([float(values[ds][m]) for ds in datasets], dtype=np.float64)
        out[m] = (float(np.mean(arr)), float(np.std(arr, ddof=0)))
    return out


def _wtl_and_wilcoxon(
    datasets: Iterable[str],
    old_vals: dict[str, float],
    new_vals: dict[str, float],
    *,
    higher_is_better: bool,
    tie_tol: float,
) -> tuple[int, int, int, float]:
    diffs: list[float] = []
    wins = ties = losses = 0
    for ds in datasets:
        a = float(old_vals[ds])
        b = float(new_vals[ds])
        if abs(b - a) <= float(tie_tol):
            ties += 1
            diffs.append(0.0)
            continue
        better = (b > a) if bool(higher_is_better) else (b < a)
        if better:
            wins += 1
        else:
            losses += 1
        diffs.append(b - a)

    x = np.asarray(diffs, dtype=np.float64)
    if np.all(np.abs(x) <= float(tie_tol)):
        return wins, ties, losses, 1.0

    alt = "greater" if bool(higher_is_better) else "less"
    try:
        res = wilcoxon(x, alternative=alt, zero_method="wilcox")
        p = float(res.pvalue)
    except Exception:
        p = 1.0
    return wins, ties, losses, p


def _mean_relative_delta_pct(
    datasets: Iterable[str],
    old_vals: dict[str, float],
    new_vals: dict[str, float],
    *,
    higher_is_better: bool,
    eps: float,
) -> float:
    deltas: list[float] = []
    for ds in datasets:
        a = float(old_vals[ds])
        b = float(new_vals[ds])
        denom = max(abs(a), float(eps))
        if bool(higher_is_better):
            deltas.append((b - a) / denom)
        else:
            deltas.append((a - b) / denom)
    return float(100.0 * float(np.mean(np.asarray(deltas, dtype=np.float64))))


def _fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def _fmt_pct(x: float) -> str:
    s = f"{x:+.2f}\\%"
    # "+0.00%" looks odd in tables; keep a plain 0.00% instead.
    return "0.00\\%" if s in ("+0.00\\%", "-0.00\\%") else s


def _fmt_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="benchmark_means_p20.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--p", type=float, default=0.20)
    ap.add_argument("--tie-tol", type=float, default=1e-12)
    args = ap.parse_args()

    datasets, values = _read_benchmark_means(args.csv)
    p_pct = int(round(100 * float(args.p)))

    # Methods (in order) and friendly variant names used in the paper.
    methods = [
        "ORBIT_ABL_A0_base",
        "OrbitSpectralFS_v2",
    ]
    names = {
        "ORBIT_ABL_A0_base": r"A0 Embedded only",
        "OrbitSpectralFS_v2": r"A1 CFIFS (no fusion)",
    }

    # Metrics included in the compact ablation tables.
    metrics = [
        ("micro_f1", True, "Micro-F1"),
        ("macro_f1", True, "Macro-F1"),
        ("hamming_loss", False, "Hamming Loss"),
    ]

    # --- Summary table (mean ± std across datasets) ---
    summary_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for metric, _hib, _title in metrics:
        if metric not in values:
            raise SystemExit(f"Missing metric in CSV: {metric}")
        summary_stats[metric] = _mean_std_across_datasets(datasets, values[metric], methods)

    # Per-metric best on the mean.
    best_by_metric: dict[str, set[str]] = {}
    for metric, higher_is_better, _title in metrics:
        means = {m: summary_stats[metric][m][0] for m in methods}
        best_val = max(means.values()) if higher_is_better else min(means.values())
        tol = 1e-12
        best_by_metric[metric] = {m for m, v in means.items() if abs(v - best_val) <= tol}

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Micro-F1 & Macro-F1 & Hamming Loss \\")
    lines.append(r"\midrule")
    for m in methods:
        row = [names.get(m, m)]
        for metric, _hib, _title in metrics:
            mean, std = summary_stats[metric][m]
            cell = _fmt_pm(mean, std)
            if m in best_by_metric[metric]:
                cell = r"\textbf{" + cell + "}"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Component ablation at $p={p_pct}\%$ selected features (mean $\pm$ std across datasets; each dataset is first averaged over folds). "
        r"Micro/Macro-F1: higher is better; Hamming Loss: lower is better. Best per metric in bold.}"
    )
    lines.append(r"\label{tab:ablation_summary}")
    lines.append(r"\end{table}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "table_ablation_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- Steps table (marginal + cumulative effects) ---
    steps: list[tuple[str, str]] = [
        ("ORBIT_ABL_A0_base", "OrbitSpectralFS_v2"),
    ]
    # No separate cumulative row needed (only one step)
    cumulative: tuple[str, str] | None = None

    def _format_step_row(a: str, b: str, label: str | None = None) -> str:
        if label is None:
            label = f"{names.get(a,a)} $\\to$ {names.get(b,b)}"
        row = [label]
        for metric, higher_is_better, _title in metrics:
            old_vals = {ds: float(values[metric][ds][a]) for ds in datasets}
            new_vals = {ds: float(values[metric][ds][b]) for ds in datasets}
            delta_pct = _mean_relative_delta_pct(datasets, old_vals, new_vals, higher_is_better=higher_is_better, eps=1e-12)
            w, t, l, p = _wtl_and_wilcoxon(datasets, old_vals, new_vals, higher_is_better=higher_is_better, tie_tol=float(args.tie_tol))
            p_str = _fmt_p(p)
            if p < 0.05:
                p_str = r"\mathbf{" + p_str + "}"
            cell = (
                r"\begin{tabular}[c]{@{}c@{}}"
                + _fmt_pct(delta_pct)
                + r"\\"
                + f"{w}/{t}/{l}"
                + r"\\"
                + rf"$p={p_str}$"
                + r"\end{tabular}"
            )
            row.append(cell)
        return " & ".join(row) + r" \\"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Step & Micro ($\uparrow$) & Macro ($\uparrow$) & HL ($\downarrow$) \\")
    lines.append(r"\midrule")

    for si, (a, b) in enumerate(steps):
        lines.append(_format_step_row(a, b))
        if si != (len(steps) - 1):
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Component ablation at $p={p_pct}\%$ selected features. "
        r"$\Delta\%$ is the mean relative improvement across datasets (Micro/Macro-F1: $(new-old)/old$; HL: $(old-new)/old$). "
        r"Within each metric cell we report $(\Delta,\,\mathrm{W/T/L},\,p)$, where W/T/L counts datasets where the new variant is better/tied/worse and "
        r"$p$ is a one-sided paired Wilcoxon test ($p<0.05$ in bold).}"
    )
    lines.append(r"\label{tab:ablation_steps}")
    lines.append(r"\end{table}")

    (args.out_dir / "table_ablation_steps.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

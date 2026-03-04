#!/usr/bin/env python3
"""
Aggregate benchmark results and generate LaTeX tables + statistical tests.

Supported grid modes:
- kgrid: results_dir/<METHOD>/<DATASET>_fold<k>_kgrid_metrics.json
         (we summarize each fold by averaging metrics across k)
- pgrid: results_dir/<METHOD>/<DATASET>_fold<k>_pgrid_metrics.json
         (we summarize each fold at a target feature ratio p, e.g., p=0.20)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


METRICS = [
    "micro_f1",
    "macro_f1",
    "hamming_loss",
    "micro_pr_auc",
    "macro_pr_auc",
    "avg_precision",
    "one_error",
]

_FNAME_RE_KGRID = re.compile(r"^(?P<dataset>.+)_fold(?P<fold>\d+)_kgrid_metrics\.json$")
_FNAME_RE_PGRID = re.compile(r"^(?P<dataset>.+)_fold(?P<fold>\d+)_pgrid_metrics\.json$")


@dataclass(frozen=True)
class Cell:
    mean: float
    std: float


def holm_correction(pvals: List[Tuple[str, float]]) -> List[Tuple[str, float, float]]:
    m = len(pvals)
    if m == 0:
        return []

    indexed = list(enumerate(pvals))
    indexed_sorted = sorted(indexed, key=lambda t: t[1][1])

    adjusted: list[tuple[str, float, float] | None] = [None] * m
    max_adj = 0.0
    for rank, (orig_idx, (name, p)) in enumerate(indexed_sorted, start=1):
        adj = (m - rank + 1) * p
        adj = min(1.0, adj)
        max_adj = max(max_adj, adj)
        adjusted[orig_idx] = (name, p, max_adj)

    return adjusted  # type: ignore[return-value]


def load_results(
    results_dir: Path,
    methods: List[str] | None,
    *,
    grid_mode: str,
) -> Dict[str, Dict[str, Dict[int, dict]]]:
    """
    Returns:
      data[method][dataset][fold] = curve_metrics_dict
    """
    if methods is None:
        methods = sorted([p.name for p in results_dir.iterdir() if p.is_dir()])

    grid_mode = grid_mode.lower().strip()
    if grid_mode not in ("kgrid", "pgrid"):
        raise SystemExit("--grid-mode must be one of: kgrid, pgrid")

    data: Dict[str, Dict[str, Dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for method in methods:
        mdir = results_dir / method
        if not mdir.exists():
            raise SystemExit(f"Missing method dir: {mdir}")
        suffix = "*_kgrid_metrics.json" if grid_mode == "kgrid" else "*_pgrid_metrics.json"
        fname_re = _FNAME_RE_KGRID if grid_mode == "kgrid" else _FNAME_RE_PGRID
        for fpath in sorted(mdir.glob(suffix)):
            m = fname_re.match(fpath.name)
            if not m:
                continue
            dataset = m.group("dataset")
            fold = int(m.group("fold"))
            metrics = json.loads(fpath.read_text())
            data[method][dataset][fold] = metrics
    return data


def fold_value(metrics: dict, metric: str, *, grid_mode: str, p_target: float) -> float:
    grid_mode = grid_mode.lower().strip()
    if grid_mode == "kgrid":
        vals = metrics.get(metric)
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"Invalid {metric} list in kgrid_metrics.json")
        return float(np.mean([float(x) for x in vals]))

    if grid_mode == "pgrid":
        key = f"{metric}_at_p_target"
        if key in metrics:
            return float(metrics[key])
        # Fallback: derive from arrays.
        p_vals = metrics.get("p_values")
        v_vals = metrics.get(metric)
        if not isinstance(p_vals, list) or not isinstance(v_vals, list):
            raise ValueError("Invalid pgrid_metrics.json: missing p_values and metric arrays.")
        if len(p_vals) == 0 or len(v_vals) == 0 or len(p_vals) != len(v_vals):
            raise ValueError("Invalid pgrid_metrics.json: p_values and metric length mismatch.")
        p_arr = np.asarray(p_vals, dtype=float)
        v_arr = np.asarray(v_vals, dtype=float)
        idx = int(np.argmin(np.abs(p_arr - float(p_target))))
        return float(v_arr[idx])

    raise ValueError(f"Unknown grid_mode: {grid_mode}")


def aggregate(
    data: Dict[str, Dict[str, Dict[int, dict]]],
    methods_order: List[str] | None = None,
    *,
    grid_mode: str,
    p_target: float,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, Dict[str, Cell]]]]:
    methods = list(methods_order) if methods_order is not None else sorted(data.keys())
    datasets = sorted({ds for m in methods for ds in data[m].keys() if ds.lower() != "reference"})

    agg: Dict[str, Dict[str, Dict[str, Cell]]] = defaultdict(lambda: defaultdict(dict))

    for method in methods:
        for dataset in datasets:
            folds = sorted(data[method].get(dataset, {}).keys())
            if not folds:
                continue
            for metric in METRICS:
                vals = [fold_value(data[method][dataset][k], metric, grid_mode=grid_mode, p_target=p_target) for k in folds]
                agg[method][dataset][metric] = Cell(
                    mean=float(np.mean(vals)),
                    std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                )
    return methods, datasets, agg


def best_mask(values: Dict[str, float], higher_is_better: bool) -> Dict[str, bool]:
    if not values:
        return {}
    best_val = max(values.values()) if higher_is_better else min(values.values())
    tol = 1e-12
    return {k: (abs(v - best_val) <= tol) for k, v in values.items()}


def format_cell(c: Cell) -> str:
    return f"{c.mean:.4f} $\\pm$ {c.std:.4f}"


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain-text cells."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def load_display_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Missing display-map file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("display-map must be a JSON object: {\"method_id\": \"Display Name\", ...}")
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise SystemExit("display-map must map strings to strings")
        out[k] = v
    return out


def make_metric_table_tex(
    methods: List[str],
    datasets: List[str],
    agg: Dict[str, Dict[str, Dict[str, Cell]]],
    metric: str,
    caption: str,
    label: str,
    *,
    display_map: Dict[str, str] | None = None,
    split_cols: int = 0,
) -> str:
    higher_is_better = metric in ("micro_f1", "macro_f1", "micro_pr_auc", "macro_pr_auc", "avg_precision")

    display_map = display_map or {}
    display_methods = [display_map.get(m, m) for m in methods]
    methods_tex = [latex_escape(m) for m in display_methods]

    # Pre-compute best mask across ALL methods for each dataset.
    best_by_ds: Dict[str, Dict[str, bool]] = {}
    for ds in datasets:
        vals = {}
        for method in methods:
            if ds in agg[method]:
                vals[method] = agg[method][ds][metric].mean
        best_by_ds[ds] = best_mask(vals, higher_is_better)

    def format_ds(ds: str) -> str:
        # Paper convention: dataset names with only the initial capitalized.
        s = str(ds)
        if not s:
            return ""
        return latex_escape(s).capitalize()

    def build_rows(block_methods: List[str], metric: str) -> List[str]:
        lines = []
        for ds in datasets:
            row_cells = []
            for method in block_methods:
                c = agg[method][ds][metric]
                s = format_cell(c)
                if best_by_ds[ds].get(method, False):
                    s = f"\\textbf{{{s}}}"
                row_cells.append(s)
            lines.append(format_ds(ds) + " & " + " & ".join(row_cells) + " \\\\")
        return lines

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")

    if split_cols and split_cols > 0 and split_cols < len(methods):
        top_methods = methods[:split_cols]
        bot_methods = methods[split_cols:]
        top_methods_tex = methods_tex[:split_cols]
        bot_methods_tex = methods_tex[split_cols:]

        # Single tabular where the dataset list is repeated twice, once per method block.
        lines.append("\\begin{tabular}{l" + "c" * split_cols + "}")
        lines.append("\\toprule")

        header1 = "Dataset & " + " & ".join(top_methods_tex) + " \\\\"
        header2 = "Dataset & " + " & ".join(bot_methods_tex) + " \\\\"

        lines.append(header1)
        lines.append("\\midrule")
        lines.extend(build_rows(top_methods, metric))
        lines.append("\\midrule")
        lines.append(header2)
        lines.append("\\midrule")
        lines.extend(build_rows(bot_methods, metric))

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
    else:
        lines.append("\\begin{tabular}{l" + "c" * len(methods) + "}")
        lines.append("\\toprule")
        lines.append("Dataset & " + " & ".join(methods_tex) + " \\\\")
        lines.append("\\midrule")
        lines.extend(build_rows(methods, metric))
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


METRIC_DISPLAY_NAMES = {
    "micro_f1": "Micro-F1",
    "macro_f1": "Macro-F1",
    "hamming_loss": "Hamming Loss",
    "one_error": "One-error",
    "micro_pr_auc": "Micro PR-AUC",
    "macro_pr_auc": "Macro PR-AUC",
    "avg_precision": "AvgPrec",
}


def make_stacked_metric_table_tex(
    methods: List[str],
    datasets: List[str],
    agg: Dict[str, Dict[str, Dict[str, Cell]]],
    metrics: List[str],
    caption: str,
    label: str,
    *,
    display_map: Dict[str, str] | None = None,
) -> str:
    """Generate a landscape table with multiple metric blocks stacked vertically.

    Each block shows one metric with all datasets as rows and methods as columns.
    Format: mean ± std, best per dataset in bold.  Wrapped in landscape environment.
    """
    display_map = display_map or {}
    display_methods = [display_map.get(m, m) for m in methods]
    methods_tex = [latex_escape(m) for m in display_methods]
    n_methods = len(methods)

    def format_ds(ds: str) -> str:
        s = str(ds)
        if not s:
            return ""
        return latex_escape(s).capitalize()

    lines: List[str] = []
    lines.append("\\begin{landscape}")
    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l" + "c" * n_methods + "}")
    lines.append("\\toprule")

    # Header row: Dataset & Method1 & Method2 & ...
    lines.append("Dataset & " + " & ".join(methods_tex) + " \\\\")

    for mi, metric in enumerate(metrics):
        higher_is_better = metric in (
            "micro_f1", "macro_f1", "micro_pr_auc", "macro_pr_auc", "avg_precision",
        )
        # Best mask for this metric
        best_by_ds: Dict[str, Dict[str, bool]] = {}
        for ds in datasets:
            vals = {}
            for method in methods:
                if ds in agg[method]:
                    vals[method] = agg[method][ds][metric].mean
            best_by_ds[ds] = best_mask(vals, higher_is_better)

        # Metric label row
        metric_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{n_methods + 1}}}{{l}}{{\\textbf{{{metric_name}}}}} \\\\"
        )
        lines.append("\\midrule")

        # Data rows
        for ds in datasets:
            row_cells = []
            for method in methods:
                c = agg[method][ds][metric]
                s = format_cell(c)
                if best_by_ds[ds].get(method, False):
                    s = f"\\textbf{{{s}}}"
                row_cells.append(s)
            lines.append(format_ds(ds) + " & " + " & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("\\end{landscape}")
    return "\n".join(lines) + "\n"


def make_stats_tex(
    methods: List[str],
    datasets: List[str],
    data: Dict[str, Dict[str, Dict[int, dict]]],
    ref_method: str,
    *,
    grid_mode: str,
    p_target: float,
    label: str,
    caption_note: str = "",
    display_map: Dict[str, str] | None = None,
) -> str:
    def metric_display(metric: str) -> str:
        return {
            "micro_f1": "Micro-F1",
            "macro_f1": "Macro-F1",
            "hamming_loss": "Hamming Loss",
            "micro_pr_auc": "Micro PR-AUC",
            "macro_pr_auc": "Macro PR-AUC",
            "avg_precision": "AvgPrec",
            "one_error": "One-error",
        }[metric]

    def higher_is_better(metric: str) -> bool:
        return metric in ("micro_f1", "macro_f1", "micro_pr_auc", "macro_pr_auc", "avg_precision")

    display_map = display_map or {}
    lines: list[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{lclccccc}")
    lines.append("\\toprule")
    ref_disp = display_map.get(ref_method, ref_method)
    lines.append("Metric & Friedman $p$ & Compared & Gain (ref. adv.) & W/T/L & Wilcoxon $p$ & Holm $p$ & Sig. \\\\")
    lines.append("\\midrule")

    for metric in METRICS:
        # Build per-dataset mean scores (mean over folds), keeping only datasets present for all methods.
        ds_rows: list[tuple[str, np.ndarray]] = []
        for ds in datasets:
            means: list[float] = []
            ok = True
            for m in methods:
                folds = sorted(data[m].get(ds, {}).keys())
                if not folds:
                    ok = False
                    break
                vals = [fold_value(data[m][ds][k], metric, grid_mode=grid_mode, p_target=p_target) for k in folds]
                means.append(float(np.mean(vals)))
            if ok:
                ds_rows.append((ds, np.asarray(means, dtype=float)))

        if not ds_rows:
            continue

        mat = np.stack([row for _ds, row in ds_rows], axis=0)  # [n_datasets, n_methods]
        per_method = {m: mat[:, i] for i, m in enumerate(methods)}

        if len(methods) < 3:
            fried_p = float("nan")
            fried_stat = float("nan")
        else:
            arrays = [per_method[m].tolist() for m in methods]
            fried = friedmanchisquare(*arrays)
            fried_p = float(fried.pvalue)
            fried_stat = float(fried.statistic)

        if ref_method not in methods:
            continue
        else:
            ref_vals = np.asarray(per_method[ref_method], dtype=float)

            # Build raw Wilcoxon p-values for Holm correction.
            pvals: list[tuple[str, float]] = []
            p_raw: Dict[str, float] = {}
            for m in methods:
                if m == ref_method:
                    continue
                other = np.asarray(per_method[m], dtype=float)
                try:
                    # One-sided in the direction of the reference advantage:
                    # - higher-is-better: H1 median(ref-other) > 0
                    # - lower-is-better : H1 median(other-ref) > 0  (i.e., ref < other)
                    if higher_is_better(metric):
                        p = float(wilcoxon(ref_vals, other, zero_method="wilcox", alternative="greater").pvalue)
                    else:
                        p = float(wilcoxon(other, ref_vals, zero_method="wilcox", alternative="greater").pvalue)
                except ValueError:
                    p = 1.0
                p_raw[m] = p
                pvals.append((m, p))
            adj = holm_correction(pvals)

        # Add one block of rows per metric: (baseline, gain, wins/ties/losses, p, p_holm, significance).
        fried_txt = "--" if not np.isfinite(fried_p) else f"{fried_p:.3g}"
        metric_txt = metric_display(metric)

        # For "gain", make it positive when the reference is better.
        higher = higher_is_better(metric)
        tol = 1e-12

        def gain(ref: np.ndarray, other: np.ndarray) -> float:
            return float(np.mean(ref - other)) if higher else float(np.mean(other - ref))

        def wtl(ref: np.ndarray, other: np.ndarray) -> str:
            if higher:
                w = int(np.sum(ref > other + tol))
                l = int(np.sum(ref < other - tol))
            else:
                w = int(np.sum(ref < other - tol))
                l = int(np.sum(ref > other + tol))
            t = int(len(ref) - w - l)
            return f"{w}/{t}/{l}"

        # Emit a block header row for the metric (also carries Friedman p).
        lines.append(f"\\textbf{{{metric_txt}}} & {fried_txt} &  &  &  &  &  &  \\\\")

        for i, (name, p, p_adj) in enumerate(adj):
            disp = latex_escape(display_map.get(name, name))
            other_vals = np.asarray(per_method[name], dtype=float)
            g = gain(ref_vals, other_vals)
            g_txt = f"{g:.4f}"
            wtl_txt = wtl(ref_vals, other_vals)
            p_txt = f"{p_raw[name]:.2e}"
            p_adj_txt = f"{p_adj:.2e}"
            sig = "\\ding{51}" if p_adj <= 0.05 else ""

            # Keep the first two columns empty; the metric block line above already labels the metric.
            lines.append(f" &  & {disp} & {g_txt} & {wtl_txt} & {p_txt} & {p_adj_txt} & {sig} \\\\")

        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if grid_mode == "pgrid":
        pct = int(round(100 * float(p_target)))
        lines.append(
            f"\\caption{{Statistical tests across datasets at {pct}\\% selected features. "
            f"We report the Friedman test (over methods), followed by paired Wilcoxon post-hoc comparisons against the reference method ({latex_escape(ref_disp)}), "
            "with Holm correction for multiple comparisons. "
            "Each row compares the reference to one method. "
            "Gain is an \\emph{advantage} score defined as $(\\text{ref}-\\text{other})$ for higher-is-better metrics (Micro/Macro-F1, PR-AUC, AvgPrec) "
            "and as $(\\text{other}-\\text{ref})$ for lower-is-better metrics (Hamming Loss, One-error), so Gain $>0$ means the reference is better. "
            "W/T/L counts the number of datasets where the reference wins/ties/loses. "
            "Wilcoxon $p$ is one-sided in the direction of Gain; Holm $p$ is the corrected value; Sig. marks Holm $p\\le 0.05$.}"
        )
        if caption_note:
            lines[-1] = lines[-1][:-1] + " " + latex_escape(caption_note) + "}"
        lines.append(f"\\label{{{label}}}")
    else:
        lines.append(
            "\\caption{Statistical tests across datasets on k-grid-averaged scores (Friedman test over methods; "
            "Holm-corrected paired Wilcoxon comparing the reference method to each baseline).}"
        )
        if caption_note:
            lines[-1] = lines[-1][:-1] + " " + latex_escape(caption_note) + "}"
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def make_ranks_tex(
    methods: List[str],
    datasets: List[str],
    data: Dict[str, Dict[str, Dict[int, dict]]],
    *,
    grid_mode: str,
    p_target: float,
    label: str,
    caption_note: str = "",
    display_map: Dict[str, str] | None = None,
) -> str:
    def metric_display(metric: str) -> str:
        return {
            "micro_f1": "Micro-F1",
            "macro_f1": "Macro-F1",
            "hamming_loss": "Hamming Loss",
            "micro_pr_auc": "Micro PR-AUC",
            "macro_pr_auc": "Macro PR-AUC",
            "avg_precision": "AvgPrec",
            "one_error": "One-error",
        }[metric]

    def higher_is_better(metric: str) -> bool:
        return metric in ("micro_f1", "macro_f1", "micro_pr_auc", "macro_pr_auc", "avg_precision")

    display_map = display_map or {}

    avg_ranks_by_metric: Dict[str, Dict[str, float]] = {}
    fried_by_metric: Dict[str, Tuple[float, float]] = {}  # (stat, p)

    for metric in METRICS:
        ds_rows: list[np.ndarray] = []
        for ds in datasets:
            means: list[float] = []
            ok = True
            for m in methods:
                folds = sorted(data[m].get(ds, {}).keys())
                if not folds:
                    ok = False
                    break
                vals = [fold_value(data[m][ds][k], metric, grid_mode=grid_mode, p_target=p_target) for k in folds]
                means.append(float(np.mean(vals)))
            if ok:
                ds_rows.append(np.asarray(means, dtype=float))

        if not ds_rows:
            continue
        mat = np.stack(ds_rows, axis=0)  # [n_datasets, n_methods]

        # Average ranks (lower is better rank).
        sums = np.zeros((len(methods),), dtype=float)
        hib = higher_is_better(metric)
        for row in mat:
            ranks = rankdata(-row, method="average") if hib else rankdata(row, method="average")
            sums += ranks
        avg = sums / float(mat.shape[0])
        avg_ranks_by_metric[metric] = {m: float(avg[i]) for i, m in enumerate(methods)}

        if len(methods) < 3:
            fried_by_metric[metric] = (float("nan"), float("nan"))
        else:
            arrays = [mat[:, i].tolist() for i in range(mat.shape[1])]
            fried = friedmanchisquare(*arrays)
            fried_by_metric[metric] = (float(fried.statistic), float(fried.pvalue))

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l" + "c" * len(METRICS) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(metric_display(m) for m in METRICS) + " \\\\")
    lines.append("\\midrule")

    for m in methods:
        disp = latex_escape(display_map.get(m, m))
        ranks = []
        for metric in METRICS:
            r = avg_ranks_by_metric.get(metric, {}).get(m, float("nan"))
            ranks.append("--" if not np.isfinite(r) else f"{r:.2f}")
        lines.append(disp + " & " + " & ".join(ranks) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # Friedman stats in caption for quick lookup.
    parts = []
    for metric in METRICS:
        stat, p = fried_by_metric.get(metric, (float("nan"), float("nan")))
        if np.isfinite(p) and np.isfinite(stat):
            parts.append(f"{metric_display(metric)} ($\\chi^2$={stat:.2f}, $p$={p:.3g})")
    fried_txt = "; ".join(parts) if parts else "--"
    if grid_mode == "pgrid":
        pct = int(round(100 * float(p_target)))
        lines.append(
            f"\\caption{{Average ranks across datasets at {pct}\\% selected features (lower is better). "
            f"Friedman tests: {fried_txt}.}}"
        )
        if caption_note:
            lines[-1] = lines[-1][:-2] + " " + latex_escape(caption_note) + "}}"
        lines.append(f"\\label{{{label}}}")
    else:
        lines.append(f"\\caption{{Average ranks across datasets (lower is better). Friedman tests: {fried_txt}.}}")
        if caption_note:
            lines[-1] = lines[-1][:-2] + " " + latex_escape(caption_note) + "}}"
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--grid-mode", type=str, default="kgrid", choices=["kgrid", "pgrid"])
    ap.add_argument("--p-target", type=float, default=0.20, help="Only used when --grid-mode=pgrid.")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filter (names must match results).")
    ap.add_argument("--ref-method", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/tables"))
    ap.add_argument("--label-suffix", type=str, default="",
                    help="Optional suffix appended to all LaTeX labels, e.g. 'brlr' -> tab:microf1_pgrid_brlr.")
    ap.add_argument("--caption-note", type=str, default="",
                    help="Optional short note appended to all captions, e.g. '(BR-LR downstream)'.")
    ap.add_argument("--display-map", type=Path, default=None,
                    help="JSON file mapping method ids to display names.")
    ap.add_argument("--split-cols", type=int, default=0,
                    help="If >0, split method columns into two stacked tabulars of split-cols and remaining.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_mode = str(args.grid_mode)
    p_target = float(args.p_target)

    data = load_results(args.results_dir, args.methods, grid_mode=grid_mode)

    # Optional dataset filter must be applied before aggregation, otherwise missing
    # metrics in out-of-scope datasets can break table generation.
    if args.datasets:
        keep = set(args.datasets)
        for m in list(data.keys()):
            for ds in list(data[m].keys()):
                if ds not in keep:
                    del data[m][ds]

    methods, datasets, agg = aggregate(data, methods_order=args.methods, grid_mode=grid_mode, p_target=p_target)
    display_map = load_display_map(args.display_map)

    # Detect actual fold count from data for captions
    _n_folds = max(
        (max(data[m][ds].keys()) + 1 for m in data for ds in data[m] if data[m][ds]),
        default=5,
    )

    # CSV (mean only) for convenience
    csv_name = "benchmark_means_kgrid.csv" if grid_mode == "kgrid" else f"benchmark_means_p{int(round(100*p_target))}.csv"
    csv_path = out_dir / csv_name
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("dataset,metric," + ",".join(methods) + "\n")
        for metric in METRICS:
            for ds in datasets:
                row = [ds, metric]
                for m in methods:
                    row.append(f"{agg[m][ds][metric].mean:.6f}")
                f.write(",".join(row) + "\n")

    if grid_mode == "pgrid":
        pct = int(round(100 * float(p_target)))
        micro_caption = f"Micro-F1 at {pct}\\% selected features (mean $\\\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        macro_caption = f"Macro-F1 at {pct}\\% selected features (mean $\\\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        hl_caption = f"Hamming Loss at {pct}\\% selected features (mean $\\\\pm$ std over {_n_folds} folds). Best (lowest) per dataset in bold."
        micro_pr_caption = f"Micro PR-AUC at {pct}\\% selected features (mean $\\\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        macro_pr_caption = f"Macro PR-AUC at {pct}\\% selected features (mean $\\\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        avgprec_caption = f"AvgPrec at {pct}\\% selected features (mean $\\\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        oneerr_caption = f"One-error at {pct}\\% selected features (mean $\\\\pm$ std over {_n_folds} folds). Best (lowest) per dataset in bold."
        micro_label = "tab:microf1_pgrid"
        macro_label = "tab:macrof1_pgrid"
        hl_label = "tab:hamming_pgrid"
        micro_pr_label = "tab:micro_pr_auc_pgrid"
        macro_pr_label = "tab:macro_pr_auc_pgrid"
        avgprec_label = "tab:avgprec_pgrid"
        oneerr_label = "tab:oneerr_pgrid"
        stats_label = "tab:stats_pgrid"
        ranks_label = "tab:ranks_pgrid"
    else:
        micro_caption = f"Micro-F1 averaged over the feature-count grid (mean $\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        macro_caption = f"Macro-F1 averaged over the feature-count grid (mean $\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        hl_caption = f"Hamming Loss averaged over the feature-count grid (mean $\\pm$ std over {_n_folds} folds). Best (lowest) per dataset in bold."
        micro_pr_caption = f"Micro PR-AUC averaged over the feature-count grid (mean $\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        macro_pr_caption = f"Macro PR-AUC averaged over the feature-count grid (mean $\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        avgprec_caption = f"AvgPrec averaged over the feature-count grid (mean $\\pm$ std over {_n_folds} folds). Best per dataset in bold."
        oneerr_caption = f"One-error averaged over the feature-count grid (mean $\\pm$ std over {_n_folds} folds). Best (lowest) per dataset in bold."
        micro_label = "tab:microf1_kgrid"
        macro_label = "tab:macrof1_kgrid"
        hl_label = "tab:hamming_kgrid"
        micro_pr_label = "tab:micro_pr_auc_kgrid"
        macro_pr_label = "tab:macro_pr_auc_kgrid"
        avgprec_label = "tab:avgprec_kgrid"
        oneerr_label = "tab:oneerr_kgrid"
        stats_label = "tab:stats_kgrid"
        ranks_label = "tab:ranks_kgrid"

    label_suffix = str(args.label_suffix or "").strip()
    if label_suffix:
        micro_label = micro_label + "_" + label_suffix
        macro_label = macro_label + "_" + label_suffix
        hl_label = hl_label + "_" + label_suffix
        micro_pr_label = micro_pr_label + "_" + label_suffix
        macro_pr_label = macro_pr_label + "_" + label_suffix
        avgprec_label = avgprec_label + "_" + label_suffix
        oneerr_label = oneerr_label + "_" + label_suffix
        stats_label = stats_label + "_" + label_suffix
        ranks_label = ranks_label + "_" + label_suffix

    caption_note = str(args.caption_note or "").strip()
    if caption_note:
        micro_caption = micro_caption + " " + latex_escape(caption_note)
        macro_caption = macro_caption + " " + latex_escape(caption_note)
        hl_caption = hl_caption + " " + latex_escape(caption_note)
        micro_pr_caption = micro_pr_caption + " " + latex_escape(caption_note)
        macro_pr_caption = macro_pr_caption + " " + latex_escape(caption_note)
        avgprec_caption = avgprec_caption + " " + latex_escape(caption_note)
        oneerr_caption = oneerr_caption + " " + latex_escape(caption_note)

    (out_dir / "table_microf1.tex").write_text(
        make_metric_table_tex(
            methods,
            datasets,
            agg,
            metric="micro_f1",
            caption=micro_caption,
            label=micro_label,
            display_map=display_map,
            split_cols=int(args.split_cols or 0),
        ),
        encoding="utf-8",
    )
    (out_dir / "table_macrof1.tex").write_text(
        make_metric_table_tex(
            methods,
            datasets,
            agg,
            metric="macro_f1",
            caption=macro_caption,
            label=macro_label,
            display_map=display_map,
            split_cols=int(args.split_cols or 0),
        ),
        encoding="utf-8",
    )
    (out_dir / "table_hamming.tex").write_text(
        make_metric_table_tex(
            methods,
            datasets,
            agg,
            metric="hamming_loss",
            caption=hl_caption,
            label=hl_label,
            display_map=display_map,
            split_cols=int(args.split_cols or 0),
        ),
        encoding="utf-8",
    )
    (out_dir / "table_micro_pr_auc.tex").write_text(
        make_metric_table_tex(
            methods,
            datasets,
            agg,
            metric="micro_pr_auc",
            caption=micro_pr_caption,
            label=micro_pr_label,
            display_map=display_map,
            split_cols=int(args.split_cols or 0),
        ),
        encoding="utf-8",
    )
    (out_dir / "table_macro_pr_auc.tex").write_text(
        make_metric_table_tex(
            methods,
            datasets,
            agg,
            metric="macro_pr_auc",
            caption=macro_pr_caption,
            label=macro_pr_label,
            display_map=display_map,
            split_cols=int(args.split_cols or 0),
        ),
        encoding="utf-8",
    )
    (out_dir / "table_avg_precision.tex").write_text(
        make_metric_table_tex(
            methods,
            datasets,
            agg,
            metric="avg_precision",
            caption=avgprec_caption,
            label=avgprec_label,
            display_map=display_map,
            split_cols=int(args.split_cols or 0),
        ),
        encoding="utf-8",
    )
    (out_dir / "table_one_error.tex").write_text(
        make_metric_table_tex(
            methods,
            datasets,
            agg,
            metric="one_error",
            caption=oneerr_caption,
            label=oneerr_label,
            display_map=display_map,
            split_cols=int(args.split_cols or 0),
        ),
        encoding="utf-8",
    )
    (out_dir / "table_stats.tex").write_text(
        make_stats_tex(
            methods,
            datasets,
            data,
            ref_method=args.ref_method,
            grid_mode=grid_mode,
            p_target=p_target,
            label=stats_label,
            caption_note=caption_note,
            display_map=display_map,
        ),
        encoding="utf-8",
    )
    (out_dir / "table_ranks.tex").write_text(
        make_ranks_tex(
            methods,
            datasets,
            data,
            grid_mode=grid_mode,
            p_target=p_target,
            label=ranks_label,
            caption_note=caption_note,
            display_map=display_map,
        ),
        encoding="utf-8",
    )

    print(f"✓ Wrote tables to: {out_dir} (grid={grid_mode})")

    # ---- Stacked landscape tables (multiple metrics per table) ----
    if grid_mode == "pgrid":
        pct = int(round(100 * float(p_target)))
        stacked_tables = [
            {
                "metrics": ["micro_f1", "macro_f1"],
                "filename": "table_stacked_f1.tex",
                "caption": (
                    f"Main comparison at $p = {pct}\\%$ selected features "
                    f"(mean $\\pm$ std over {_n_folds} folds). "
                    f"Blocks report different metrics (higher is better). "
                    f"Best per dataset in bold."
                ),
                "label": f"tab:stacked_f1_{grid_mode}",
            },
            {
                "metrics": ["hamming_loss", "one_error"],
                "filename": "table_stacked_hloe.tex",
                "caption": (
                    f"Main comparison at $p = {pct}\\%$ selected features "
                    f"(mean $\\pm$ std over {_n_folds} folds). "
                    f"Blocks report different metrics (lower is better). "
                    f"Best per dataset in bold."
                ),
                "label": f"tab:stacked_hloe_{grid_mode}",
            },
            {
                "metrics": ["micro_pr_auc", "macro_pr_auc", "avg_precision"],
                "filename": "table_stacked_prauc.tex",
                "caption": (
                    f"Additional ranking-quality metrics at $p = {pct}\\%$ selected features "
                    f"(mean $\\pm$ std over {_n_folds} folds). "
                    f"All metrics in this table are higher-is-better. "
                    f"Best per dataset in bold."
                ),
                "label": f"tab:stacked_prauc_{grid_mode}",
            },
        ]
        for tbl in stacked_tables:
            lbl = tbl["label"]
            if label_suffix:
                lbl = lbl + "_" + label_suffix
            cap = tbl["caption"]
            if caption_note:
                cap = cap + " " + latex_escape(caption_note)
            (out_dir / tbl["filename"]).write_text(
                make_stacked_metric_table_tex(
                    methods,
                    datasets,
                    agg,
                    metrics=tbl["metrics"],
                    caption=cap,
                    label=lbl,
                    display_map=display_map,
                ),
                encoding="utf-8",
            )
        print(f"✓ Wrote stacked landscape tables to: {out_dir}")


if __name__ == "__main__":
    main()

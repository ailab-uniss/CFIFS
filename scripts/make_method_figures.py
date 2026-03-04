#!/usr/bin/env python3
"""
Generate illustrative figures for the CFIFS paper.

Figure 1 – Adaptive β(d) curve with real dataset markers.
Figure 2 – Embedded vs Spectral score complementarity (scatter, 3 datasets).
Figure 3 – Spectral mechanism: feature smoothness on the label graph (2 features).
Figure 4 – Top-k overlap between embedded and spectral rankings vs. k.
"""
from __future__ import annotations

import json
import glob
import math
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

OUT_DIR = Path("outputs/method_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- shared style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "sigmoid": "#2c3e50",
    "point": "#e74c3c",
    "emb": "#2980b9",
    "spec": "#27ae60",
    "fused": "#8e44ad",
    "top_only_emb": "#3498db",
    "top_only_spec": "#2ecc71",
    "top_both": "#e67e22",
    "top_neither": "#bdc3c7",
}


# ═══════════════════════════════════════════════
# FIGURE 1 – Adaptive β(d) sigmoid with dataset markers
# ═══════════════════════════════════════════════
def fig_adaptive_beta():
    print("Fig 1: Adaptive β(d) …")
    beta0, d0, tau = 0.35, 100, 15

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def beta_func(d):
        return beta0 * sigmoid((d - d0) / tau)

    # Smooth curve
    dd = np.linspace(0, 1900, 2000)
    bb = beta_func(dd)

    # Real dataset points
    infos = sorted(glob.glob("results/bench_orbit_final_lit16/OrbitSpectralFS_v2/*_fold0_info.json"))
    ds_data = []
    for fp in infos:
        with open(fp) as f:
            info = json.load(f)
        ds_data.append((info["dataset"].capitalize(), info["d"], info["beta_effective"]))

    # Filter to new subsets
    dagfs15 = {"Arts", "Business", "Education", "Entertain", "Health", "Recreation",
               "Computers", "Science", "Social", "Yelp", "Slashdot", "Human",
               "genbase", "medical", "yeast", "Birds", "Image", "Corel16k1", "Corel16k2", "Flags"}
    ds_data = [(n, d, b) for n, d, b in ds_data if n in dagfs15]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(dd, bb, color=COLORS["sigmoid"], linewidth=2, zorder=2)
    ax.axhline(beta0, color=COLORS["sigmoid"], linewidth=0.6, linestyle="--", alpha=0.4)
    ax.text(1850, beta0 + 0.008, rf"$\beta_0={beta0}$", ha="right", fontsize=7, color=COLORS["sigmoid"])

    # Shade low-d zone
    ax.axvspan(0, d0, alpha=0.06, color=COLORS["emb"], zorder=0)
    ax.text(d0 / 2, 0.33, "embedded\ndominates", ha="center", fontsize=7, color=COLORS["emb"], alpha=0.7)
    ax.axvspan(d0, 1900, alpha=0.04, color=COLORS["spec"], zorder=0)
    ax.text(900, 0.33, "spectral contributes", ha="center", fontsize=7, color=COLORS["spec"], alpha=0.7)

    # Vertical line at d0
    ax.axvline(d0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.text(d0 + 10, 0.01, rf"$d_0={d0}$", fontsize=7, color="gray")

    # Dataset markers
    for name, d, b in ds_data:
        ax.scatter(d, b, s=30, color=COLORS["point"], zorder=5, edgecolors="white", linewidths=0.4)
        # Label – offset to avoid overlap
        if d < 150:
            ax.annotate(name, (d, b), textcoords="offset points", xytext=(5, 5),
                       fontsize=6, color="#555")
        elif d > 1500:
            ax.annotate(name, (d, b), textcoords="offset points", xytext=(-5, -10),
                       fontsize=6, color="#555", ha="right")

    ax.set_xlabel("Number of features $d$")
    ax.set_ylabel(r"Blending weight $\beta(d)$")
    ax.set_xlim(-30, 1950)
    ax.set_ylim(-0.02, 0.42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT_DIR / "fig_adaptive_beta.pdf")
    plt.close(fig)
    print("  → fig_adaptive_beta.pdf")


# ═══════════════════════════════════════════════
# FIGURE 2 – Embedded vs Spectral score scatter (3 panels)
# ═══════════════════════════════════════════════
def fig_score_complementarity():
    print("Fig 2: Score complementarity …")
    datasets = ["Flags", "yeast", "Science"]
    titles_extra = {
        "Flags": r"$d=19$",
        "yeast": r"$d=103$",
        "Science": r"$d=743$",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=False)

    for ax, ds in zip(axes, datasets):
        data = np.load(f"outputs/fig_data_{ds}.npz", allow_pickle=True)
        emb = data["orbit_scores"]
        spec = data["slagd_scores"]
        d = int(data["d"])

        # Normalise to [0,1]
        def mm(v):
            lo, hi = v.min(), v.max()
            return (v - lo) / (hi - lo + 1e-12)

        emb_n = mm(emb)
        spec_n = mm(spec)

        # Color by quadrant: top-k = top 20%
        k = max(1, int(0.2 * d))
        top_emb = set(np.argsort(-emb_n)[:k])
        top_spec = set(np.argsort(-spec_n)[:k])

        colors = []
        for j in range(d):
            in_emb = j in top_emb
            in_spec = j in top_spec
            if in_emb and in_spec:
                colors.append(COLORS["top_both"])
            elif in_emb:
                colors.append(COLORS["top_only_emb"])
            elif in_spec:
                colors.append(COLORS["top_only_spec"])
            else:
                colors.append(COLORS["top_neither"])

        ax.scatter(emb_n, spec_n, c=colors, s=12, alpha=0.7, edgecolors="none", zorder=3)

        # Diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.3)

        ax.set_title(f"{ds.capitalize()}\n{titles_extra[ds]}", fontsize=8.5)
        ax.set_xlabel("Embedded score" if ds == "yeast" else "")
        if ds == datasets[0]:
            ax.set_ylabel("Spectral score")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Count unique features in top-k
        only_emb = len(top_emb - top_spec)
        only_spec = len(top_spec - top_emb)
        both = len(top_emb & top_spec)
        ax.text(0.97, 0.03, f"overlap {both}/{k}", transform=ax.transAxes,
                ha="right", fontsize=6.5, color="#555")

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["top_both"],
               markersize=5, label="Top-20% in both"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["top_only_emb"],
               markersize=5, label="Top-20% embedded only"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["top_only_spec"],
               markersize=5, label="Top-20% spectral only"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["top_neither"],
               markersize=5, label="Not in top-20%"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_score_complementarity.pdf")
    plt.close(fig)
    print("  → fig_score_complementarity.pdf")


# ═══════════════════════════════════════════════
# FIGURE 3 – Spectral mechanism: smooth vs noisy feature on label graph
# ═══════════════════════════════════════════════
def fig_spectral_smoothness():
    """
    For emotions dataset, sort instances by the Fiedler vector (spectral ordering
    of the label-affinity graph) and plot two features:
    - one with LOW Dirichlet energy → smooth curve
    - one with HIGH Dirichlet energy → noisy/random
    
    Top row: feature signal on spectral ordering.
    Bottom: bar chart of Dirichlet energy for all features, highlighting the two.
    """
    print("Fig 3: Spectral smoothness …")
    import sys
    sys.path.insert(0, "src")
    from mlfs.spectral_mlfs import _build_label_affinity

    ds = "Flags"
    fp = f"data/panorama30_matlab_minmax_cv10/{ds}/fold0.mat"
    mat = sio.loadmat(fp)
    X = np.asarray(mat["X_train"], dtype=np.float64)
    Y = np.asarray(mat["Y_train"], dtype=np.float64)
    if Y.min() < 0:
        Y = (Y > 0).astype(np.float64)
    n, d = X.shape

    # Label affinity → Laplacian → Fiedler vector
    A = _build_label_affinity(Y, "jaccard")
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 1e-10, 1.0 / np.sqrt(deg), 0.0)
    D_isq = np.diag(deg_inv_sqrt)
    L = np.eye(n) - D_isq @ A @ D_isq

    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    spectral_order = np.argsort(fiedler)

    # Dirichlet energy
    eps = 1e-10
    dirichlet = np.zeros(d)
    for f in range(d):
        xf = X[:, f]
        dirichlet[f] = (xf @ L @ xf) / (xf @ xf + eps)

    best_f = int(np.argmin(dirichlet))
    worst_f = int(np.argmax(dirichlet))
    print(f"  Best feature: {best_f} (E={dirichlet[best_f]:.4f})")
    print(f"  Worst feature: {worst_f} (E={dirichlet[worst_f]:.4f})")

    fig = plt.figure(figsize=(6, 4.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8], hspace=0.45, wspace=0.3)

    # --- Top row: feature signal on spectral order ---
    for col, fi, color, title in [
        (0, best_f, COLORS["spec"], f"Feature {best_f}: $E = {dirichlet[best_f]:.3f}$ (smooth)"),
        (1, worst_f, COLORS["point"], f"Feature {worst_f}: $E = {dirichlet[worst_f]:.3f}$ (noisy)"),
    ]:
        ax = fig.add_subplot(gs[0, col])
        xf_sorted = X[spectral_order, fi]
        ax.plot(np.arange(n), xf_sorted, linewidth=0.5, color=color, alpha=0.8)
        # Running average
        win = max(5, n // 30)
        running_avg = np.convolve(xf_sorted, np.ones(win) / win, mode="same")
        ax.plot(np.arange(n), running_avg, linewidth=1.8, color=color, alpha=0.9)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Instances (spectral order)", fontsize=7)
        if col == 0:
            ax.set_ylabel("Feature value", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6)

    # --- Bottom: bar chart of Dirichlet energy for all features ---
    ax_bar = fig.add_subplot(gs[1, :])
    sort_idx = np.argsort(dirichlet)
    bar_colors = [COLORS["top_neither"]] * d
    for i, idx in enumerate(sort_idx):
        if idx == best_f:
            bar_colors[i] = COLORS["spec"]
        elif idx == worst_f:
            bar_colors[i] = COLORS["point"]
    ax_bar.bar(np.arange(d), dirichlet[sort_idx], color=bar_colors, width=1.0, edgecolor="none")
    ax_bar.set_xlabel("Features (sorted by Dirichlet energy)", fontsize=7)
    ax_bar.set_ylabel("Dirichlet energy $E(f)$", fontsize=7)
    ax_bar.set_title(f"Flags dataset: {d} features", fontsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(labelsize=6)

    # Arrows to highlighted features
    best_pos = int(np.where(sort_idx == best_f)[0][0])
    worst_pos = int(np.where(sort_idx == worst_f)[0][0])
    ax_bar.annotate("smoothest", xy=(best_pos, dirichlet[best_f]),
                    xytext=(best_pos + 8, dirichlet[best_f] + 0.12),
                    fontsize=6.5, color=COLORS["spec"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["spec"], lw=0.8))
    ax_bar.annotate("noisiest", xy=(worst_pos, dirichlet[worst_f]),
                    xytext=(worst_pos - 12, dirichlet[worst_f] + 0.08),
                    fontsize=6.5, color=COLORS["point"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["point"], lw=0.8))

    fig.savefig(OUT_DIR / "fig_spectral_smoothness.pdf")
    plt.close(fig)
    print("  → fig_spectral_smoothness.pdf")


# ═══════════════════════════════════════════════
# FIGURE 4 – Top-k overlap curve (embedded vs spectral) across datasets
# ═══════════════════════════════════════════════
def fig_topk_overlap():
    """
    For each dataset, compute overlap(k) = |top_k_emb ∩ top_k_spec| / k
    for k from 1 to d.  Plot mean ± shaded std across datasets.
    Also show individual curves for a few representative datasets.
    """
    print("Fig 4: Top-k overlap …")
    datasets = ["Flags", "yeast", "Science"]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    for ds in datasets:
        data = np.load(f"outputs/fig_data_{ds}.npz", allow_pickle=True)
        emb = data["orbit_scores"]
        spec = data["slagd_scores"]
        d = int(data["d"])

        rank_emb = np.argsort(-emb)
        rank_spec = np.argsort(-spec)

        # Overlap at each k (as fraction of k)
        p_vals = np.arange(5, 55, 5) / 100.0  # 5% to 50%
        overlaps = []
        for p in p_vals:
            k = max(1, int(p * d))
            top_emb = set(rank_emb[:k])
            top_spec = set(rank_spec[:k])
            overlaps.append(len(top_emb & top_spec) / k)

        ax.plot(p_vals * 100, overlaps, marker="o", markersize=3, linewidth=1.5,
                label=f"{ds.capitalize()} ($d$={d})")

    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xlabel("% selected features ($k/d$)")
    ax.set_ylabel("Overlap fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(3, 52)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Agreement between embedded and spectral rankings", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_topk_overlap.pdf")
    plt.close(fig)
    print("  → fig_topk_overlap.pdf")


if __name__ == "__main__":
    fig_adaptive_beta()
    fig_score_complementarity()
    fig_spectral_smoothness()
    fig_topk_overlap()
    print("\nAll figures written to:", OUT_DIR)

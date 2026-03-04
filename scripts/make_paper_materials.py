import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

def holm_correction(pvals):
    m = len(pvals)
    if m == 0: return []
    pvals_sorted = sorted(pvals, key=lambda x: x[1])
    adj = []
    for k, (name, p) in enumerate(pvals_sorted, 1):
        corr = min(1.0, p * (m - k + 1))
        if k > 1: corr = max(corr, adj[-1][1])
        adj.append((name, corr))
    out = []
    for name, p in pvals:
        for n, c in adj:
            if n == name:
                out.append((name, p, c))
                break
    return out

def get_data(methods, datasets, metrics, results_dirs, folds=10, p_target=0.20):
    if not isinstance(results_dirs, list):
        results_dirs = [results_dirs]
    data = {m: {ds: {method: [] for method in methods} for ds in datasets} for m in metrics}
    for ds in datasets:
        for method in methods:
            for fold in range(folds):
                fp = None
                for rdir in results_dirs:
                    cand = rdir / method / f"{ds}_fold{fold}_pgrid_metrics.json"
                    if cand.exists():
                        fp = cand
                        break
                if fp is None: continue
                try:
                    with open(fp, 'r') as f:
                        d = json.load(f)
                    for m in metrics:
                        val = d.get(f"{m}_at_p_target", None)
                        if val is None:
                            k = f"p={p_target:.2f}"
                            if k in d and m in d[k]: val = d[k][m]
                        if val is not None:
                            data[m][ds][method].append(float(val))
                except Exception:
                    pass
    
    means = {m: {ds: {} for ds in datasets} for m in metrics}
    for m in metrics:
        for ds in datasets:
            for method in methods:
                vs = data[m][ds][method]
                if len(vs) == folds:
                    means[m][ds][method] = np.mean(vs)
    return means, data

def build_latex_table(means, data, methods, reference, datasets, metric, metric_name, higher_is_better, label_suffix=""):
    valid_ds = [ds for ds in datasets if all(m in means[metric][ds] for m in methods)]
    if not valid_ds: return ""
    
    other_m = [m for m in methods if m != reference]
    ref_arr = np.array([means[metric][ds][reference] for ds in valid_ds])
    
    pvals_raw = []
    win_ties_losses = {}
    for om in other_m:
        o_arr = np.array([means[metric][ds][om] for ds in valid_ds])
        diff = ref_arr - o_arr if higher_is_better else o_arr - ref_arr
        w = np.sum(diff > 0); t = np.sum(abs(diff) < 1e-6); l = np.sum(diff < 0)
        pval = 1.0
        if np.any(diff != 0):
            pval = wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue
        pvals_raw.append((om, pval))
        win_ties_losses[om] = f"{w}/{t}/{l}"
        
    adj_pvals = holm_correction(pvals_raw)
    sigs = {om: ("*" if p <= 0.05 else "") for om, _, p in adj_pvals}
    holm_ps = {om: hp for om, _, hp in adj_pvals}
    
    # Calculate means and ranks across datasets
    all_means = {mm: np.mean([means[metric][ds][mm] for ds in valid_ds]) for mm in methods}
    # Ranking
    ranks = {ds: {} for ds in valid_ds}
    for ds in valid_ds:
        # sort methods based on metric
        m_vals = [(mm, means[metric][ds][mm]) for mm in methods]
        if higher_is_better:
            m_vals.sort(key=lambda x: -x[1])
        else:
            m_vals.sort(key=lambda x: x[1])
        for r, (mm, v) in enumerate(m_vals, 1):
            ranks[ds][mm] = r
    avg_ranks = {mm: np.mean([ranks[ds][mm] for ds in valid_ds]) for mm in methods}
    
    lines = []
    if label_suffix == "_main":
        lines.append(r"\begin{landscape}")
    lines.append(r"\begin{table}[p]" if label_suffix == "_main" else r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    col_def = "l" + "c" * len(methods)
    lines.append(r"\begin{tabular}{" + col_def + "}")
    lines.append(r"\hline")
    def map_name(m):
        if label_suffix == "_main":
            if m in ("CFIFS", "CFIFS_CFI"):
                return r"\textbf{\projectname{}}"
            return m.replace("_", "\\_")
        elif label_suffix == "_abl_a":
            if m == "CFIFS_EMB": return r"Embedded Only"
            if m == "CFIFS_SPEC": return r"Spectral Only"
            if m == "CFIFS": return r"\textbf{+ Choquet Fusion}"
            return m.replace("_", "\\_")
        elif label_suffix == "_abl_b":
            if m == "CFIFS_CFI_FIX0505": return r"Fixed Cap. (0.5)"
            if m == "CFIFS_CFI_ADD": return r"Additive Cap."
            if m == "CFIFS_CFI": return r"Choquet (Min/Max)"
            if m == "CFIFS": return r"\textbf{Choquet (Rank)}"
            return m.replace("_", "\\_")
        else:
            if m in ("CFIFS", "CFIFS_CFI"):
                return r"\textbf{\projectname{}}"
            return m.replace("_", "\\_")
    
    header = "Dataset & " + " & ".join([map_name(m) for m in methods]) + r" \\"
    lines.append(header)
    lines.append(r"\hline")
    
    DS_DISPLAY = {"genbase": "Genbase", "medical": "Medical", "yeast": "Yeast"}
    for ds in valid_ds:
        row = [DS_DISPLAY.get(ds, ds)]
        m_vals = [means[metric][ds][mm] for mm in methods]
        best_val = max(m_vals) if higher_is_better else min(m_vals)
        for mm in methods:
            v = means[metric][ds][mm]
            s = f"{v:.4f}"
            std = np.std(data[metric][ds][mm]) if len(data[metric][ds][mm]) > 0 else 0.0
            if abs(v - best_val) < 1e-6:
                s = f"\\textbf{{{s}}}"
            s = f"{s} $\\pm$ {std:.4f}"
            row.append(s)
        lines.append(" & ".join(row) + r" \\")
        
    lines.append(r"\hline")
    # Means
    row = ["Average"]
    for mm in methods:
        row.append(f"{all_means[mm]:.4f}")
    lines.append(" & ".join(row) + r" \\")
    # Ranks
    row = ["Avg Rank"]
    for mm in methods:
        row.append(f"{avg_ranks[mm]:.2f}")
    lines.append(" & ".join(row) + r" \\")
    
    # W/T/L
    mapped_ref = map_name(reference)
    if mapped_ref.startswith(r"\textbf{") and mapped_ref.endswith("}"):
        mapped_ref = mapped_ref[8:-1]
    row = ["W/T/L vs " + mapped_ref]
    for mm in methods:
        if mm == reference:
            row.append("-")
        else:
            row.append(win_ties_losses[mm])
    lines.append(" & ".join(row) + r" \\")
    
    # p-values
    row = ["Holm $p$-value"]
    for mm in methods:
        if mm == reference:
            row.append("-")
        else:
            pstr = f"{holm_ps[mm]:.2e}"
            if sigs[mm] == "*":
                pstr = "\\textbf{" + pstr + "}"
            row.append(pstr)
    lines.append(" & ".join(row) + r" \\")
        
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{Performance comparison evaluated using the {metric_name} metric (higher is better). Mean results $\\pm$ standard deviation are reported across 10 cross-validation folds. The best value for each dataset is highlighted in bold. * indicates statistical significance according to the Wilcoxon paired test with Holm correction ($p < 0.05$).}}")
    lines.append(f"\\label{{tab:{metric}{label_suffix}}}")
    lines.append(r"\end{table}")
    if label_suffix == "_main":
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


def run():
    datasets = ['Arts', 'Birds', 'Business', 'Computers', 'Corel16k1', 'Corel16k2', 'Education', 'Entertain', 'Flags', 'Health', 'Human', 'Image', 'Recreation', 'Science', 'Slashdot', 'Social', 'Yelp', 'genbase', 'medical', 'yeast']
    results_dir = [Path("results/bench_panorama30_cv10"), Path("results/bench_10fold"), Path("results/bench_orbit_final_lit16")]
    
    # 1. Main comparison
    main_methods = ["GRRO", "SRFS", "RFSFS", "LRMFS", "LSMFS", "LRDG", "SCNMF", "CFIFS"]
    main_metrics_info = [
        ("micro_f1", "Micro-F1", True),
        ("macro_f1", "Macro-F1", True),
        ("hamming_loss", "Hamming Loss", False)
    ]
    main_means, main_data = get_data(main_methods, datasets, [m[0] for m in main_metrics_info], results_dir)
    
    main_tex = ""
    for m, mname, hib in main_metrics_info:
        main_tex += build_latex_table(main_means, main_data, main_methods, "CFIFS", datasets, m, mname, hib, "_main") + "\n\n"
        
    with open("outputs/paper_materials/table_main_results.tex", "w") as f:
        f.write(main_tex)
        
    # 2. Ablation Comparison
    abl_methods = ["CFIFS_EMB", "CFIFS_CFI", "CFIFS_CFI_ADD", "CFIFS_CFI_FIX0505", "CFIFS"]
    # Check if we have CFIFS_ADD ? No. So we compare CFIFS against CFIFS_CFI and CFIFS_EMB.
    # For additive vs non-add we can use CFIFS_CFI vs CFIFS_CFI_ADD vs CFIFS_CFI_FIX0505
    # Let's do two ablation tables.
    abl_metrics = [("micro_f1", "Micro-F1", True), ("macro_f1", "Macro-F1", True)]
    
    # Ablation Set A: Component Ablation (Base Modalities vs Fusion)
    set_a = ["CFIFS_EMB", "CFIFS_SPEC", "CFIFS"]
    means_a, data_a = get_data(set_a, datasets, [m[0] for m in abl_metrics], results_dir)
    tex_a = ""
    for m, mname, hib in abl_metrics:
        tex_a += build_latex_table(means_a, data_a, set_a, "CFIFS", datasets, m, mname + " (Component Ablation)", hib, "_abl_a") + "\n\n"
        
    # Ablation Set B: Fusion Strategies Ablation
    set_b = ["CFIFS_CFI_FIX0505", "CFIFS_CFI_ADD", "CFIFS_CFI", "CFIFS"]
    means_b, data_b = get_data(set_b, datasets, [m[0] for m in abl_metrics], results_dir)
    tex_b = ""
    for m, mname, hib in abl_metrics:
        tex_b += build_latex_table(means_b, data_b, set_b, "CFIFS", datasets, m, mname + " (Capacity Ablation)", hib, "_abl_b") + "\n\n"
        
    with open("outputs/paper_materials/table_ablation.tex", "w") as f:
        f.write("% --- Structural Ablation ---\n")
        f.write(tex_a)
        f.write("% --- Capacity Ablation ---\n")
        f.write(tex_b)
        
    # 3. Parameter Analysis (Capacities & Interaction)
    out_cap = []
    for ds in datasets:
        mu_e_list, mu_s_list, int_list = [], [], []
        for fold in range(10):
            fp = None
            for rdir in results_dir:
                cand = rdir / "CFIFS" / f"{ds}_fold{fold}_info.json"
                if cand.exists():
                    fp = cand
                    break
            if fp is not None:
                try:
                    with open(fp, "r") as f:
                        d = json.load(f)
                    mu_e_list.append(d.get("mu_e", 0.0))
                    mu_s_list.append(d.get("mu_s", 0.0))
                    int_list.append(d.get("interaction_I", 0.0))
                except: pass
        if mu_e_list:
            out_cap.append({
                "dataset": ds,
                "mu_e_mean": np.mean(mu_e_list),
                "mu_s_mean": np.mean(mu_s_list),
                "interaction_mean": np.mean(int_list),
                "interaction_std": np.std(int_list)
            })
    
    with open("outputs/paper_materials/capacities.json", "w") as f:
        json.dump(out_cap, f, indent=2)
        
    print("Paper materials generated in outputs/paper_materials/")

if __name__ == '__main__':
    run()

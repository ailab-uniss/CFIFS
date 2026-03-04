# Scripts

This folder contains the full reproducible pipeline for the CFIFS paper:

1. prepare multi-label cross-validation folds,
2. run CFIFS on training folds to obtain feature rankings,
3. evaluate rankings over a feature-ratio grid with ML-kNN,
4. aggregate results, statistical tests, and generate LaTeX tables / figures.

## Main entrypoints

| Script | Purpose |
|--------|---------|
| `run_cfifs.py` | Compute CFIFS rankings for all datasets × folds |
| `run_cfifs_ablation.py` | Ablation study (single-source and fusion variants) |
| `eval_rankings_gpu_mlknn.py` | Evaluate rankings with GPU ML-kNN on a p-grid |
| `eval_rankings_py_pgrid_fast.py` | Evaluate rankings (CPU fallback) |
| `aggregate_kgrid_and_make_tables.py` | Aggregate metrics, compute ranks/tests, LaTeX tables |
| `make_pgrid_curves.py` | Plot mean p-grid curves |
| `make_paper_materials.py` | End-to-end paper materials (tables + figures) |
| `make_method_figures.py` | Generate method illustration figures |
| `make_cfifs_ablation_tables.py` | Ablation-specific LaTeX tables |

## Dataset preparation utilities

| Script | Purpose |
|--------|---------|
| `build_panorama30_root.py` | Build the Panorama-30 benchmark root |
| `freeze_panorama_cv10.py` | Freeze 10-fold CV splits |
| `export_cv_splits_to_mat.py` | Export folds to `.mat` for MATLAB baselines |
| `export_dense_benchmark_to_mat.py` | Export dense benchmark to `.mat` |

## Other

| Script | Purpose |
|--------|---------|
| `run_rfsfs.py` | Run the RFSFS baseline for comparison |

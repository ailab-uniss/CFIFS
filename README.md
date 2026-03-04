# CFIFS — Choquet Fuzzy-Integral Feature Selection for Multi-Label Data

Reference Python / PyTorch implementation of the method described in:

> **Choquet-Based Fusion of Embedded and Spectral Scores for Multi-Label Feature Selection**
> Submitted

CFIFS produces a **single feature ranking per training fold** by:

1. **Embedded scoring** — a convex group-sparse solver that combines squared-reconstruction and per-label logistic losses with rarity-aware instance weighting (§ 3.1–3.2 of the paper).
2. **Spectral scoring** — SLAGD: Dirichlet energy on a label-affinity graph combined with an HSIC term (§ 3.3).
3. **Rank normalisation** — empirical-CDF mapping to [0, 1] so that the two channels are commensurate (§ 3.4).
4. **Choquet integral fusion** — a 2-source non-additive fuzzy integral with free singleton capacities (μₑ, μₛ), selected by an inner K-fold CV loop that maximises the geometric mean of Micro-F1 and Macro-F1 evaluated with ML-kNN on the top-*p*% features (§ 3.4–3.5).

---

## What is included

| Item | Description |
|------|-------------|
| `src/mlfs/` | Core library — embedded solver, spectral scorer, GPU ML-kNN |
| `scripts/run_cfifs.py` | **Main runner** (embedding → spectral → Choquet fusion) |
| `scripts/run_cfifs_ablation.py` | Ablation variants (EMB-only, REG-only, no weights, …) |
| `scripts/eval_rankings_gpu_mlknn.py` | Evaluate a set of rankings on a *p*-grid with ML-kNN |
| `scripts/make_paper_materials.py` | Generate LaTeX tables and figures from result JSONs |
| `scripts/make_method_figures.py` | Produce the illustrative method figures |
| `scripts/make_pgrid_curves.py` | *p*-grid performance curves |
| `baselines/` | MATLAB wrappers for the 7 baseline methods (GRRO, SRFS, RFSFS, LRMFS, LSMFS, LRDG, SCNMF) |

## What is **not** included

**Datasets** are not redistributed because the original licences may not permit it.  All 20 benchmarks used in the paper are publicly available:

| Domain | Datasets |
|--------|----------|
| Text (Yahoo) | Arts, Business, Computers, Education, Entertain, Health, Recreation, Science, Social |
| Image | Image, Birds, Corel16k1, Corel16k2 |
| Other | Flags, Slashdot, Yelp |
| Biology | genbase, medical, yeast, Human |

Standard sources: [Mulan](http://mulan.sourceforge.net/datasets-mlc.html), [KDIS](https://www.uco.es/kdis/mllresources/), [Cometa](http://www.omsz.eu/en/cometa/).

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[experiments]"
```

**Core** (`src/mlfs/`) only requires **NumPy ≥ 1.23**.
The `[experiments]` extra pulls in SciPy, scikit-learn, scikit-multilearn, and Matplotlib.
**GPU acceleration** requires [PyTorch](https://pytorch.org/get-started/locally/) — install it separately matching your CUDA version.

---

## Quickstart (Python API)

```python
import numpy as np
from mlfs import CFIFSParams, fit_cfifs

# X: (n_samples, n_features) float array, preferably scaled to [0,1]
# Y: (n_samples, n_labels)   binary {0,1}

params = CFIFSParams(
    alpha=0.35,          # trade-off: regression vs. logistic loss
    beta=0.01,           # group-lasso strength (ℓ₂₁ penalty)
    rho=1e-4,            # ridge regulariser
    rank=50,             # label-embedding dimension
    backend="torch",     # "numpy" or "torch"
    device="cuda",       # "cpu" or "cuda"
)
ranking, info = fit_cfifs(X, Y, params)
# ranking: 1-based feature indices sorted best → worst
# info["scores"]: per-feature importance (row norms of W)
```

---

## Reproducing the paper results

### 1. Prepare the data

Arrange each dataset in the **folded layout** expected by the scripts:

```
data/<DatasetName>/
  fold0.mat        # contains X_train, Y_train, X_test, Y_test
  fold1.mat
  ...
  fold9.mat
```

Each `.mat` file must contain four variables:
- `X_train` — `(n_train, d)` float, min-max normalised to [0, 1] on the training set
- `Y_train` — `(n_train, L)` binary `{0, 1}`
- `X_test`  — `(n_test, d)` float, normalised using the training statistics
- `Y_test`  — `(n_test, L)` binary `{0, 1}`

Use `scripts/freeze_panorama_cv10.py` to create 10-fold stratified splits from raw ARFF/CSV sources.

### 2. Run CFIFS (all datasets, 10 folds)

```bash
python scripts/run_cfifs.py \
    --data-dir data/panorama30_matlab_minmax_cv10 \
    --results-dir results/bench_panorama30_cv10 \
    --method CFIFS \
    --score-norm rank \
    --capacity-mode free \
    --mu-grid "0.0,0.2,0.4,0.6,0.8,1.0" \
    --icv-criterion gm \
    --icv-aggregate hard \
    --folds 10 \
    --device cuda
```

This writes, for each dataset × fold:
- `<results>/CFIFS/<dataset>_fold<k>_ranking.csv` — 1-based feature ranking
- `<results>/CFIFS/<dataset>_fold<k>_info.json` — metadata (capacity values, timing, …)

### 3. Run ablation variants

```bash
for variant in CFIFS_EMB ACSF_REG ACSF_LOG ACSF_NOWT ACSF_SIMPLE; do
    python scripts/run_cfifs_ablation.py \
        --data-dir data/panorama30_matlab_minmax_cv10 \
        --results-dir results/bench_panorama30_cv10 \
        --method $variant \
        --folds 10 \
        --device cuda
done
```

### 4. Run baselines (MATLAB)

```matlab
cd baselines
run_suite('../results/bench_panorama30_cv10', ...
          '../data/panorama30_matlab_minmax_cv10', ...
          {'Arts','Birds','Business',...}, ...   % dataset list
          {'GRRO','SRFS','RFSFS','LRMFS','LSMFS','LRDG','SCNMF'}, ...
          'folds', 10, 'rank_only', true);
```

### 5. Evaluate rankings on a *p*-grid

```bash
python scripts/eval_rankings_gpu_mlknn.py \
    --results-dir results/bench_panorama30_cv10 \
    --data-dirs data/panorama30_matlab_minmax_cv10 \
    --methods CFIFS CFIFS_EMB GRRO SRFS RFSFS LRMFS LSMFS LRDG SCNMF \
    --folds 10 \
    --p-min 0.05 --p-max 0.50 --p-step 0.05 --p-target 0.20 \
    --device cuda
```

### 6. Generate tables and figures

```bash
python scripts/make_paper_materials.py \
    --results-dir results/bench_panorama30_cv10 \
    --out-dir outputs/paper_materials

python scripts/make_pgrid_curves.py \
    --results-dir results/bench_panorama30_cv10 \
    --out-dir outputs/paper_materials

python scripts/make_method_figures.py \
    --out-dir outputs/paper_materials
```

---

## Repository structure

```
src/mlfs/
  __init__.py                   # public API: CFIFSParams, fit_cfifs
  cfifs_embedded.py             # embedded scoring (NumPy + PyTorch solvers)
  spectral_mlfs.py              # spectral scoring (SLAGD)
  ml_knn_gpu.py                 # GPU-accelerated ML-kNN classifier
  _components/
    instance_weights.py         # rarity-aware instance weighting

scripts/
  run_cfifs.py                  # main CFIFS runner (embedding + spectral + Choquet)
  run_cfifs_ablation.py         # embedded-only ablation variants
  run_rfsfs.py                  # Python re-implementation of RFSFS baseline
  eval_rankings_gpu_mlknn.py    # evaluate rankings on a p-grid (GPU ML-kNN)
  eval_rankings_py_pgrid_fast.py# helper: p-grid metric computation
  make_paper_materials.py       # generate LaTeX tables & figures
  make_method_figures.py        # illustrative method diagrams
  make_pgrid_curves.py          # p-grid performance curves
  make_cfifs_ablation_tables.py # ablation result tables
  aggregate_kgrid_and_make_tables.py  # aggregate grid results
  export_dense_benchmark_to_mat.py    # convert datasets to .mat
  export_cv_splits_to_mat.py    # export stratified CV splits
  build_panorama30_root.py      # assemble the multi-source benchmark
  freeze_panorama_cv10.py       # freeze 10-fold stratified splits

baselines/                      # MATLAB wrappers for 7 baseline methods
paper_latex/                    # LaTeX sources of the paper
```

---

## Reproducibility notes

- **Training-only preprocessing:** every statistic (label embedding, rarity weights, affinity graph) is computed on the training fold only — no test information leaks.
- **Scaling:** data is min-max normalised to [0, 1] per fold (fit on train, applied to test).
- **Stratification:** folds are created with iterative stratification to preserve label proportions.
- **Determinism:** all random seeds are fixed (default `--seed 42`).  NumPy RNG is used even in the PyTorch solver to guarantee cross-backend reproducibility.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{casu2026cfifs,
  title   = {Choquet-Based Fusion of Embedded and Spectral Scores for Multi-Label Feature Selection},
  author  = {Casu, Filippo and Lagorio, Andrea and Trunfio, Giuseppe A.},
  note    = {Submitted},
  year    = {2026},
}
```

## License

[MIT](LICENSE)

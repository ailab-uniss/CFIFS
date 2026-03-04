# CFIFS — Choquet Fuzzy-Integral Feature Selection for Multi-Label Data

Reference Python / PyTorch implementation of the method described in:

> **Choquet-Based Fusion of Embedded and Spectral Scores for Multi-Label Feature Selection**
> Submitted

CFIFS produces a **single feature ranking** for a multi-label dataset by:

1. **Embedded scoring** — a convex group-sparse solver that combines squared-reconstruction and per-label logistic losses with rarity-aware instance weighting (§ 3.1–3.2).
2. **Spectral scoring** — SLAGD: Dirichlet energy on a label-affinity graph combined with an HSIC term (§ 3.3).
3. **Rank normalisation** — empirical-CDF mapping to [0, 1] so that the two channels are commensurate (§ 3.4).
4. **Choquet integral fusion** — a 2-source non-additive fuzzy integral with free singleton capacities (μₑ, μₛ), selected by an inner K-fold CV loop that maximises the geometric mean of Micro-F1 and Macro-F1 evaluated with ML-kNN on the top-*p*% features (§ 3.4–3.5).

---

## Installation

```bash
git clone https://github.com/ailab-uniss/CFIFS.git
cd CFIFS
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[experiments]"
```

**Core** (`src/mlfs/`) only requires **NumPy ≥ 1.23**.
The `[experiments]` extra pulls in SciPy, scikit-learn, scikit-multilearn, and Matplotlib.
**GPU acceleration** requires [PyTorch](https://pytorch.org/get-started/locally/) — install it separately matching your CUDA version.

---

## Quick demo

A small example dataset (*emotions* — 538 samples, 72 features, 6 labels) is
bundled under `example/`.  Run CFIFS end-to-end in **one command**:

```bash
# Embedded scoring only (fast, ~0.1 s on CPU)
python cfifs_demo.py

# Full pipeline: embedded + spectral + Choquet fusion (~3 s on CPU)
python cfifs_demo.py --full

# GPU acceleration
python cfifs_demo.py --full --device cuda
```

Expected output (`--full`):

```
Dataset:  emotions_fold0
  samples = 538,  features = 72,  labels = 6

[Embedded]  0.13s  —  top features: [5, 36, 4, 20, 3, 35, ...]
[Spectral]  0.44s  —  top features: [48, 2, 47, 46, 18, ...]
[Choquet]   1.87s  —  μₑ=0.80  μₛ=0.80  I=-0.60  ICV-GM=0.6112

Final ranking (top 15): [5, 48, 36, 18, 4, 3, 65, 17, 47, ...]
Total time: 2.67s
```

### Using your own data

Prepare a `.mat` file with at least `X_train` *(n × d, float)* and `Y_train` *(n × L, binary {0,1})*:

```bash
python cfifs_demo.py --mat path/to/my_data.mat --full --device cuda
```

---

## Python API

```python
import numpy as np
from mlfs import CFIFSParams, fit_cfifs

# X: (n_samples, n_features) float, preferably scaled to [0,1]
# Y: (n_samples, n_labels)   binary {0,1}

params = CFIFSParams(
    alpha=0.35,          # trade-off: reconstruction vs. logistic loss
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

The spectral channel can also be used standalone:

```python
from mlfs.spectral_mlfs import SLAGDParams, fit_slagd

params = SLAGDParams(alpha=0.70, label_sim="jaccard",
                     backend="torch", device="cuda")
ranking, info = fit_slagd(X, Y, params)
```

---

## Repository structure

```
cfifs_demo.py                   # CLI demo — run CFIFS on bundled or custom data
example/
  emotions_fold0.mat            # bundled example (emotions, fold 0)
src/mlfs/
  __init__.py                   # public API: CFIFSParams, fit_cfifs
  cfifs_embedded.py             # embedded scoring (NumPy + PyTorch solvers)
  spectral_mlfs.py              # spectral scoring (SLAGD)
  ml_knn_gpu.py                 # GPU-accelerated ML-kNN classifier
  _components/
    instance_weights.py         # rarity-aware instance weighting
pyproject.toml                  # package metadata (pip install -e .)
```

---

## Reproducibility notes

- **Training-only preprocessing:** every statistic (label embedding, rarity weights, affinity graph) is computed on the training fold only — no test information leaks.
- **Scaling:** data is min-max normalised to [0, 1] per fold (fit on train, applied to test).
- **Determinism:** all random seeds are fixed (default seed 42).  NumPy RNG is used even in the PyTorch solver to guarantee cross-backend reproducibility.

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

"""GPU-accelerated Multi-Label k-Nearest Neighbours (ML-kNN).

This module provides a fast ML-kNN classifier with two backends:

* **sklearn** — cosine-metric brute-force neighbours via
  ``sklearn.neighbors.NearestNeighbors``; works on sparse CSR input.
* **torch** — dense GPU-accelerated neighbours via PyTorch
  matrix multiplication; recommended for n ≤ 10 000.

The model is used inside CFIFS’s inner cross-validation loop (ICV)
to evaluate candidate feature subsets.

Public API
----------
- ``MLkNNConfig``  — frozen dataclass with all configuration knobs.
- ``MLkNNModel``   — fit/predict model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class MLkNNConfig:
    """Configuration for the ML-kNN classifier.

    Attributes
    ----------
    k : int
        Number of nearest neighbours (default 10).
    s : float
        Laplace smoothing parameter (default 1.0).
    metric : str
        Distance metric for the sklearn backend (default ``'cosine'``).
    backend : str
        ``'auto'`` | ``'torch'`` | ``'sklearn'``.
    device : str
        ``'auto'`` | ``'cpu'`` | ``'cuda'``.
    label_adaptive_k : bool
        Use smaller *k* for rare labels (default ``False``).
    label_k_min : int
        Minimum *k* when ``label_adaptive_k`` is ``True``.
    label_k_power : float
        Power exponent for label-adaptive *k* scaling.
    torch_max_train_samples : int
        Auto backend heuristic: skip torch if n > this (0 = disable).
    torch_min_density : float
        Auto backend heuristic: skip torch if density < this.
    torch_max_dense_mb : int
        Auto backend heuristic: skip torch if estimated dense
        memory exceeds this many MiB.
    """
    k: int = 10
    s: float = 1.0
    metric: str = "cosine"
    backend: str = "auto"  # auto | torch | sklearn
    device: str = "auto"  # auto | cpu | cuda
    # Label-adaptive k: use smaller k for rare labels in ML-kNN.
    label_adaptive_k: bool = False
    label_k_min: int = 3
    label_k_power: float = 0.5
    # Heuristics for backend=auto:
    # - Torch backend densifies X and computes full similarity matrices, so it is only
    #   beneficial for relatively small / dense problems.
    # Note: these are heuristics to avoid slowdowns/oom from densification.
    # If you want "auto ≈ torch unless too big", set max_train_samples=0 and min_density=0.
    torch_max_train_samples: int = 0  # 0 disables this check
    torch_min_density: float = 0.0  # 0 disables this check
    torch_max_dense_mb: int = 1024  # cap (rough) for X + similarity matrices on torch backend


class MLkNNModel:
    """Multi-Label k-Nearest Neighbours classifier.

    Scope
    -----
    Implements the full ML-kNN algorithm (Zhang & Zhou, 2007): fit
    computes prior and conditional probability tables from training
    neighbours; predict_proba returns posterior label probabilities.
    """

    def __init__(self, cfg: MLkNNConfig) -> None:
        """Initialise the ML-kNN model.

        Parameters
        ----------
        cfg : MLkNNConfig
            Configuration object.

        Preconditions
        -------------
        * ``cfg.k > 0`` and ``cfg.s > 0``.
        * ``cfg.backend`` ∈ {``'auto'``, ``'torch'``, ``'sklearn'``}.
        * ``cfg.device``  ∈ {``'auto'``, ``'cpu'``, ``'cuda'``}.

        Postconditions
        --------------
        * The model is initialised but **not** fitted.
        """
        if cfg.k <= 0:
            raise ValueError("k must be > 0")
        if cfg.s <= 0:
            raise ValueError("s must be > 0")
        self.cfg = cfg
        # Torch backend attributes
        self._use_torch = False
        self._device = None
        self._backend_selected: str | None = None
        
        backend = str(cfg.backend).strip().lower()
        if backend not in {"auto", "torch", "sklearn"}:
            raise ValueError(f"Unsupported MLkNNConfig.backend={cfg.backend!r} (auto|torch|sklearn)")
        self._backend_requested = backend

        device = str(cfg.device).strip().lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unsupported MLkNNConfig.device={cfg.device!r} (auto|cpu|cuda)")
        self._device_requested = device

        # Torch is optional: only attempt import if it might be used.
        if self._backend_requested in {"auto", "torch"}:
            try:
                import torch

                if self._device_requested == "cpu":
                    self._device = torch.device("cpu")
                elif self._device_requested == "cuda":
                    self._device = torch.device("cuda")
                else:
                    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._use_torch = True
            except ImportError:
                self._use_torch = False

        # Model state
        self._prior_true: np.ndarray | None = None
        self._prior_false: np.ndarray | None = None
        self._cond_true: np.ndarray | None = None
        self._cond_false: np.ndarray | None = None
        
        # Stored training data (kept on CPU sparse until needed, or dense GPU if small enough?)
        # Since we refit often (feature selection), we only store fit results here.
        # But wait, MLkNN logic requires finding neighbors *in the specific feature subspace*.
        # So 'fit' actually does the heavy lifting of neighbor search. 
        # But in a wrapper setting, we instantiate a new model for each mask.
        self._y_train_dense: np.ndarray | None = None
        
        # Legacy fallback
        self._nn: NearestNeighbors | None = None
        self._y_train: sparse.csr_matrix | None = None

    def _select_backend(self, x_train: sparse.csr_matrix) -> str:
        """Choose between 'torch' and 'sklearn' for the given training data.

        Scope
        -----
        Applies heuristics based on matrix size, density, and estimated
        GPU memory to decide which backend is most efficient.

        Parameters
        ----------
        x_train : csr_matrix of shape (n, d)
            Training features.

        Preconditions
        -------------
        * Called before ``_fit_torch`` / sklearn fit.

        Postconditions
        --------------
        * Returns ``'torch'`` or ``'sklearn'``.
        """
        # Explicit choice always wins.
        if self._backend_requested == "sklearn":
            return "sklearn"
        if self._backend_requested == "torch":
            if not self._use_torch:
                raise RuntimeError("MLkNNConfig.backend='torch' requested but torch is not available.")
            return "torch"

        # auto: prefer sklearn for large / sparse matrices (torch densifies).
        n, d = x_train.shape
        if d == 0 or n == 0:
            return "sklearn"
        if bool(getattr(self.cfg, "label_adaptive_k", False)):
            return "sklearn"
        nnz = int(getattr(x_train, "nnz", 0))
        density = float(nnz) / float(n * d)
        # Torch backend densifies X and computes similarity matrices:
        # - xt_t: (n, d) float32
        # - sim train: (n, n) float32
        # - sim val: (n_val, n) float32 (during predict)
        dense_mb = (n * d * 4 + n * n * 4) / (1024.0 * 1024.0)
        max_n = int(self.cfg.torch_max_train_samples)
        min_den = float(self.cfg.torch_min_density)
        if dense_mb > float(self.cfg.torch_max_dense_mb):
            return "sklearn"
        if max_n > 0 and n > max_n:
            return "sklearn"
        if min_den > 0.0 and density < min_den:
            return "sklearn"
        if not self._use_torch:
            return "sklearn"
        return "torch"

    def _fit_torch(self, x_train: sparse.csr_matrix, y_train: sparse.csr_matrix) -> None:
        """Fit ML-kNN using dense GPU operations.

        Scope
        -----
        Densifies *x_train* and *y_train*, computes full pairwise
        cosine similarity on the GPU, extracts *k* nearest
        neighbours, and builds the prior/conditional tables.

        Parameters
        ----------
        x_train : csr_matrix of shape (n, d)
            Training features.
        y_train : csr_matrix of shape (n, L)
            Training labels.

        Preconditions
        -------------
        * PyTorch is available and ``self._device`` is set.
        * Matrices fit in GPU memory.

        Postconditions
        --------------
        * ``self._prior_true_t``, ``self._prior_false_t``,
          ``self._cond_true_t``, ``self._cond_false_t`` are set as
          GPU tensors.
        * ``self._xt_train`` and ``self._yt_train`` are stored for
          prediction.
        """
        import torch
        
        k = int(self.cfg.k)
        s = float(self.cfg.s)
        n, _ = x_train.shape
        m = y_train.shape[1]
        
        # Move to GPU/Tensor
        # For feature selection, fit is called with sliced X.
        # X is usually (N, F_sub).
        xt_t = torch.from_numpy(x_train.toarray()).float().to(self._device)
        yt_t = torch.from_numpy(y_train.toarray()).float().to(self._device)
        
        self._xt_train = xt_t # Keep for predict
        self._yt_train = yt_t
        
        # Normalize for cosine distance: sim = (a . b) / (|a|*|b|)
        # If we normalize vectors first, it is just dot product.
        xt_norm = torch.nn.functional.normalize(xt_t, p=2, dim=1)
        
        # Compute pairwise similarity (N, N)
        # heavy op, but fast on GPU for N=2000
        sim_matrix = torch.mm(xt_norm, xt_norm.t())
        
        # We need top k+1 neighbors (including self usually at value 1.0)
        # Using topk is faster than sort
        # largest=True because cosine similarity 1.0 is best.
        target_k = min(k + 1, n)
        _, indices = torch.topk(sim_matrix, k=target_k, dim=1)
        
        # Drop self (first column)?
        # Verify if self is always first. Usually yes, but with identical points it might vary.
        # Strict ML-KNN excludes the instance itself from neighbors.
        # We simply exclude the first column, assuming it is self (or equivalent).
        # indices shape: (N, K+1)
        if indices.shape[1] > k:
             neighbor_indices = indices[:, 1:] 
        else:
             neighbor_indices = indices
             
        # neighbor_indices: (N, K)
        # Count labels of neighbors
        # yt_t: (N, M)
        # Gather neighbors' labels.
        # We want count[i, l] = sum(yt_t[neighbor_indices[i], l])
        
        # Expand yt_t for gathering? Or loop?
        # Vectorized gather:
        # neighbor_indices is (N, K). Flatten -> (N*K)
        flat_neighbors = neighbor_indices.reshape(-1)
        neighbor_labels = torch.index_select(yt_t, 0, flat_neighbors) # (N*K, M)
        neighbor_labels = neighbor_labels.view(n, k, m)
        label_counts = neighbor_labels.sum(dim=1) # (N, M)
        
        # Calculations for Prior/Cond
        # pos[l]: total positive examples for label l
        pos = yt_t.sum(dim=0) # (M)
        neg = n - pos
        
        prior_true = (s + pos) / (2.0 * s + n)
        prior_false = 1.0 - prior_true
        
        # Conditional probabilities
        # c_true[l, c]: P(C_l == c | H_1^l) -> fraction of true instances having exactly c neighbors with label l
        cond_true = torch.zeros((m, k + 1), device=self._device)
        cond_false = torch.zeros((m, k + 1), device=self._device)
        
        # This part is harder to vectorize fully without creating huge tensors (N, M, K+1).
        # But M is small (labels ~10-20), N ~2000.
        
        label_counts = label_counts.long() # (N, M) taking values 0..k
        yt_bool = yt_t.bool()
        
        # We can iterate over c in 0..k
        for c in range(k + 1):
            # Mask of instances having exactly c neighbors for each label
            mask_c = (label_counts == c) # (N, M)
            
            # Intersection with y=1
            ct = (mask_c & yt_bool).sum(dim=0).float() # (M,)
            # Intersection with y=0
            cf = (mask_c & (~yt_bool)).sum(dim=0).float() # (M,)
            
            cond_true[:, c] = (s + ct) / (s * (k + 1) + pos)
            cond_false[:, c] = (s + cf) / (s * (k + 1) + neg)
            
        self._prior_true_t = prior_true
        self._prior_false_t = prior_false
        self._cond_true_t = cond_true
        self._cond_false_t = cond_false

    def _predict_torch(self, x_val: sparse.csr_matrix) -> np.ndarray:
        """Predict label probabilities using the GPU backend.

        Scope
        -----
        Computes pairwise cosine similarity between validation
        samples and stored training data, counts neighbour labels,
        and applies the ML-kNN posterior formula.

        Parameters
        ----------
        x_val : csr_matrix of shape (n_val, d)
            Validation features.

        Preconditions
        -------------
        * The model has been fitted via ``_fit_torch``.

        Postconditions
        --------------
        * Returns a float NumPy array of shape (n_val, L) with
          values in (0, 1).
        """
        import torch
        
        xv_t = torch.from_numpy(x_val.toarray()).float().to(self._device)
        n_val = xv_t.shape[0]
        k = int(self.cfg.k)
        
        # Normalize query
        xv_norm = torch.nn.functional.normalize(xv_t, p=2, dim=1)
        xt_norm = torch.nn.functional.normalize(self._xt_train, p=2, dim=1)
        
        # Val x Train similarity
        sim_matrix = torch.mm(xv_norm, xt_norm.t()) # (N_val, N_train)
        
        target_k = min(k, self._xt_train.shape[0])
        _, indices = torch.topk(sim_matrix, k=target_k, dim=1)
        
        # Count neighbors
        flat_neighbors = indices.reshape(-1)
        neighbor_labels = torch.index_select(self._yt_train, 0, flat_neighbors)
        m = self._yt_train.shape[1]
        
        neighbor_labels = neighbor_labels.view(n_val, target_k, m)
        label_counts = neighbor_labels.sum(dim=1).long() # (N_val, M) values 0..k
        
        # Standard ML-KNN inference
        # If count is c, P(H1|E) ~ P(H1) * P(E|H1)
        
        # Gather conditional probs
        # cond_true_t is (M, K+1). We want to select column based on label_counts
        # label_counts is (N_val, M)
        
        # We need to broadcast index selection.
        # Let's do it per label or simpler iteration?
        # Vectorized gather:
        # We need to gather from cond_true_t which is (M, K+1) using indices from label_counts (N_val, M)
        # Result should be (N_val, M)
        
        # Transpose cond to (K+1, M) maybe easier?
        # cond_true_t.t() -> (K+1, M)
        # gather logic: out[i, j] = input[index[i, j], j]
        # Torch gather works on specified dim.
        
        ct_t = self._cond_true_t.t() # (K+1, M)
        cf_t = self._cond_false_t.t() # (K+1, M)
        
        # We extend ct_t to (N_val, K+1, M) -> impractical? NO.
        # We treat M as batch?
        
        # Easier: loop over M labels or reshape. M is small.
        # or use gather with expanded dims.
        
        # Try gather:
        # input: (K+1, M)
        # index: (N_val, M) -> values in 0..K
        # We want output (N_val, M).
        # Since 'gather' expects matching dimensions, we cannot directly gather (K+1, M) with (N_val, M).
        
        # But we can transpose input to (M, K+1).
        # index is (M, N_val) (transposed label_counts).
        # then gather along dim 1.
        
        counts_t = label_counts.t() # (M, N_val)
        
        pt_neigh = torch.gather(self._cond_true_t, 1, counts_t) # (M, N_val)
        pf_neigh = torch.gather(self._cond_false_t, 1, counts_t) # (M, N_val)
        
        prior_t = self._prior_true_t.unsqueeze(1) # (M, 1)
        prior_f = self._prior_false_t.unsqueeze(1) # (M, 1)
        
        prob_true = prior_t * pt_neigh
        prob_false = prior_f * pf_neigh
        
        denom = prob_true + prob_false
        # avoid div zero
        probs = prob_true / (denom + 1e-10)
        
        return probs.t().cpu().numpy() # (N_val, M)

    def fit(self, x_train: sparse.csr_matrix, y_train: sparse.csr_matrix) -> "MLkNNModel":
        """Fit the ML-kNN model on training data.

        Scope
        -----
        Selects the backend, finds *k* nearest neighbours, and
        computes the prior and conditional probability tables
        required for Bayesian label prediction.

        Parameters
        ----------
        x_train : csr_matrix of shape (n, d)
            Training features.
        y_train : csr_matrix of shape (n, L)
            Training labels (binary).

        Preconditions
        -------------
        * Both matrices have the same number of rows.
        * Labels are binary {0, 1}.

        Postconditions
        --------------
        * The model is ready for ``predict_proba``.
        * Returns ``self`` for method chaining.
        """
        self._backend_selected = self._select_backend(x_train)
        if self._backend_selected == "torch":
            self._fit_torch(x_train, y_train)
            return self
            
        # Legacy Scikit-learn
        x_train = x_train.tocsr()
        y_train = y_train.tocsr()

        n, _d = x_train.shape
        m = y_train.shape[1]
        k = int(self.cfg.k)
        s = float(self.cfg.s)

        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric=self.cfg.metric, algorithm="brute")
        nn.fit(x_train)
        neigh = nn.kneighbors(x_train, return_distance=False)

        # Drop self when present (first neighbor in cosine metric is usually self).
        if neigh.shape[1] > k:
            neigh = neigh[:, 1 : k + 1]

        y_dense = y_train.toarray().astype(np.int8, copy=False)
        label_adaptive = bool(getattr(self.cfg, "label_adaptive_k", False))
        if label_adaptive:
            label_freq = y_dense.mean(axis=0)
            nonzero = label_freq[label_freq > 0]
            base = float(np.median(nonzero)) if nonzero.size else 1e-3
            power = float(getattr(self.cfg, "label_k_power", 0.5))
            k_min = int(getattr(self.cfg, "label_k_min", 3))
            k_l = np.clip(np.round(k * (label_freq / max(base, 1e-6)) ** power), k_min, k).astype(int)
            self._label_k = k_l
            neighbor_labels = y_dense[neigh]  # (n, k, m)
            cum_counts = np.cumsum(neighbor_labels, axis=1)
            label_counts = np.zeros((n, m), dtype=np.int16)
            for l in range(m):
                kl = int(k_l[l])
                if kl <= 0:
                    continue
                label_counts[:, l] = cum_counts[:, kl - 1, l]
        else:
            label_counts = np.zeros((n, m), dtype=np.int16)
            for i in range(n):
                idxs = neigh[i]
                if idxs.size == 0:
                    continue
                label_counts[i] = y_dense[idxs].sum(axis=0)

        pos = y_dense.sum(axis=0).astype(np.int64)
        neg = (n - pos).astype(np.int64)
        prior_true = (s + pos) / (2.0 * s + n)
        prior_false = 1.0 - prior_true

        cond_true = np.zeros((m, k + 1), dtype=np.float64)
        cond_false = np.zeros((m, k + 1), dtype=np.float64)
        for l in range(m):
            lc = label_counts[:, l]
            yt = y_dense[:, l].astype(bool)
            kl = int(self._label_k[l]) if label_adaptive else k
            for c in range(kl + 1):
                ct = int(np.sum((lc == c) & yt))
                cf = int(np.sum((lc == c) & (~yt)))
                cond_true[l, c] = (s + ct) / (s * (kl + 1) + pos[l])
                cond_false[l, c] = (s + cf) / (s * (kl + 1) + neg[l])

        self._nn = nn
        self._y_train = y_train
        self._prior_true = prior_true
        self._prior_false = prior_false
        self._cond_true = cond_true
        self._cond_false = cond_false
        return self

    def predict_proba(self, x: sparse.csr_matrix) -> np.ndarray:
        """Return posterior label probabilities for new samples.

        Scope
        -----
        For each sample, finds *k* nearest training neighbours,
        counts label occurrences, and applies Bayes’ rule with
        the fitted prior/conditional tables.

        Parameters
        ----------
        x : csr_matrix of shape (n_val, d)
            Validation or test features.

        Preconditions
        -------------
        * The model has been fitted via ``fit()``.

        Postconditions
        --------------
        * Returns a float64 ndarray of shape (n_val, L) with
          values in (0, 1).
        """
        if self._backend_selected == "torch":
            return self._predict_torch(x)
            
        if self._nn is None or self._y_train is None:
            raise RuntimeError("Model not fitted.")
        prior_true = self._prior_true
        prior_false = self._prior_false
        cond_true = self._cond_true
        cond_false = self._cond_false
        assert prior_true is not None and prior_false is not None and cond_true is not None and cond_false is not None

        x = x.tocsr()
        n = x.shape[0]
        m = self._y_train.shape[1]
        k = int(self.cfg.k)

        neigh = self._nn.kneighbors(x, return_distance=False)
        # When predicting, there is no self to drop; ensure exactly k neighbors if possible.
        if neigh.shape[1] > k:
            neigh = neigh[:, :k]

        y_dense = self._y_train.toarray().astype(np.int8, copy=False)
        probs = np.zeros((n, m), dtype=np.float64)
        label_adaptive = bool(getattr(self.cfg, "label_adaptive_k", False)) and hasattr(self, "_label_k")
        if label_adaptive:
            neighbor_labels = y_dense[neigh]  # (n, k, m)
            cum_counts = np.cumsum(neighbor_labels, axis=1)
            for i in range(n):
                for l in range(m):
                    kl = int(self._label_k[l])
                    if kl <= 0:
                        probs[i, l] = prior_true[l]
                        continue
                    c = int(cum_counts[i, kl - 1, l])
                    pt = prior_true[l] * cond_true[l, c]
                    pf = prior_false[l] * cond_false[l, c]
                    denom = pt + pf
                    probs[i, l] = 0.5 if denom == 0 else (pt / denom)
        else:
            for i in range(n):
                idxs = neigh[i]
                if idxs.size == 0:
                    probs[i] = prior_true
                    continue
                counts = y_dense[idxs].sum(axis=0).clip(0, k)
                for l in range(m):
                    c = int(counts[l])
                    pt = prior_true[l] * cond_true[l, c]
                    pf = prior_false[l] * cond_false[l, c]
                    denom = pt + pf
                    probs[i, l] = 0.5 if denom == 0 else (pt / denom)
        return probs


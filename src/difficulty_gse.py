from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import NearestNeighbors



@dataclass
class SpatialDynamicsField:
   
    coords: np.ndarray          
    difficulty_score: np.ndarray  

    @property
    def N(self) -> int:
        return len(self.difficulty_score)


def build_spatial_adjacency(
    coords: np.ndarray,
    k: int = 6,
    weight: str = "binary",   # "binary" | "distance" | "gaussian"
    sigma: float = 1.0,
) -> csr_matrix:
    """
    Build a sparse k-NN adjacency matrix from 2D spot coordinates.

    Parameters
    ----------
    coords  : (N, 2) pixel coordinates
    k       : number of nearest neighbours
    weight  : edge weighting scheme
                "binary"   → 1 for all edges
                "distance" → 1 / (dist + 1e-8)
                "gaussian" → exp(-dist² / (2σ²))
    sigma   : bandwidth for gaussian weights (in same units as coords)

    Returns
    -------
    A : (N, N) symmetric sparse adjacency matrix
    """
    N = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    rows, cols, vals = [], [], []
    for i in range(N):
        for j_pos in range(1, k + 1):          # skip self (index 0)
            j   = indices[i, j_pos]
            d   = distances[i, j_pos]

            if weight == "binary":
                w = 1.0
            elif weight == "distance":
                w = 1.0 / (d + 1e-8)
            else:                               # gaussian
                w = float(np.exp(-(d ** 2) / (2 * sigma ** 2 + 1e-10)))

            rows.append(i); cols.append(j); vals.append(w)
            rows.append(j); cols.append(i); vals.append(w)   # symmetrise

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A


def graph_laplacian(A: csr_matrix) -> csr_matrix:
    """
    Unnormalized graph Laplacian  L = D - A
    where D is the diagonal degree matrix.
    """
    degree = np.asarray(A.sum(axis=1)).ravel()
    D = diags(degree)
    return D - A


# ---------------------------------------------------------------------------
# Graph Signal Energy difficulty
# ---------------------------------------------------------------------------

def graph_signal_energy_difficulty(
    base_score: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
    alpha: float = 0.5,
    weight: str = "binary",
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Topology-aware difficulty via Graph Signal Energy (GSE).

    Algorithm
    ---------
    1. Build k-NN spatial graph → adjacency A → Laplacian L = D - A
    2. Apply Laplacian to base difficulty:  s = L @ base_score
    3. Local energy:                        e_i = |s_i|
       High e_i ↔ spot sits at a sharp boundary in difficulty space.
    4. Combine:
       d_i = α · normalize(base_score_i) + (1-α) · normalize(e_i)

    Parameters
    ----------
    base_score : (N,) existing difficulty estimate (e.g. your D_bar-based score)
    coords     : (N, 2) pixel coordinates
    k          : spatial graph neighbours
    alpha      : weight for base score vs local energy
                 0 → pure topology (boundary-only)
                 1 → pure base score (ignores graph)
                 0.5 → balanced (recommended)
    weight     : adjacency weighting ("binary", "distance", "gaussian")
    sigma      : gaussian bandwidth (pixels)

    Returns
    -------
    gse_difficulty : (N,) array in [0, 1], higher = harder
    """
    base_score = np.asarray(base_score, dtype=np.float64)
    N = len(base_score)

    # 1. Graph + Laplacian
    A = build_spatial_adjacency(coords, k=k, weight=weight, sigma=sigma)
    L = graph_laplacian(A)

    # 2. Graph signal: apply Laplacian to base difficulty
    s = L @ base_score                    # (N,) — signed local variation

    # 3. Local energy: magnitude of Laplacian response
    e = np.abs(s)                         # (N,) — boundary sharpness

    # 4. Normalize both components to [0, 1]
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-10:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    base_n  = _norm(base_score)
    energy_n = _norm(e)

    # 5. Combine
    gse_difficulty = alpha * base_n + (1.0 - alpha) * energy_n

    return gse_difficulty.astype(np.float32)


# ---------------------------------------------------------------------------
# Drop-in replacement that takes a SpatialDynamicsField
# ---------------------------------------------------------------------------

def gse_difficulty_from_field(
    field,                    # SpatialDynamicsField
    k: int = 6,
    alpha: float = 0.5,
    weight: str = "binary",
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Convenience wrapper: compute GSE difficulty directly from a
    SpatialDynamicsField, using field.difficulty_score as the base.

    Returns
    -------
    gse_score : (N,) array, same shape as field.difficulty_score
    """
    return graph_signal_energy_difficulty(
        base_score=field.difficulty_score,
        coords=field.coords,
        k=k,
        alpha=alpha,
        weight=weight,
        sigma=sigma,
    )


# ---------------------------------------------------------------------------
# Utility: compare both scores side by side
# ---------------------------------------------------------------------------

def compare_difficulty_scores(
    field,
    k: int = 6,
    alpha: float = 0.5,
) -> dict:
    """
    Compute both original and GSE difficulty scores and return summary stats.
    Useful for diagnosing how much topology changes the ordering.

    Returns
    -------
    dict with keys:
        original  : (N,) original difficulty_score
        gse       : (N,) GSE difficulty score
        rank_corr : Spearman rank correlation between the two
        top25_overlap : fraction of top-25% hard spots shared by both
    """
    from scipy.stats import spearmanr

    orig = np.asarray(field.difficulty_score, dtype=np.float32)
    gse  = gse_difficulty_from_field(field, k=k, alpha=alpha)

    rho, _ = spearmanr(orig, gse)

    N = len(orig)
    n_top = max(1, N // 4)
    top_orig = set(np.argsort(orig)[-n_top:])
    top_gse  = set(np.argsort(gse)[-n_top:])
    overlap  = len(top_orig & top_gse) / n_top

    print(f"[GSE] N={N} | alpha={alpha} | k={k}")
    print(f"  Spearman rank corr (orig vs GSE): {rho:.3f}")
    print(f"  Top-25% hard spots overlap:       {overlap:.1%}")
    print(f"  GSE  mean={gse.mean():.3f}  std={gse.std():.3f}")
    print(f"  Orig mean={orig.mean():.3f}  std={orig.std():.3f}")

    return {
        "original":       orig,
        "gse":            gse,
        "rank_corr":      float(rho),
        "top25_overlap":  float(overlap),
    }


# ---------------------------------------------------------------------------
# Topological difficulty from raw data (no training needed)
# ---------------------------------------------------------------------------

def topological_difficulty_from_data(
    expression: np.ndarray,
    coords: np.ndarray,
    patches: Optional[np.ndarray] = None,
    k_neighbours: int = 6,
    alpha_expr: float = 0.4,
    alpha_hist: float = 0.3,
    alpha_spatial: float = 0.3,
) -> np.ndarray:
    """
    Compute difficulty score directly from raw data using graph signal energy.

    Difficulty captures three signals:
    1. Expression heterogeneity (entropy per spot)
    2. Histology heterogeneity (patch variance per spot)  [if patches provided]
    3. Spatial discontinuity (graph Laplacian on features)

    No training or warm-up needed.

    Parameters
    ----------
    expression  : (N, G) log-normalized gene expression
    coords      : (N, 2) spatial coordinates
    patches     : (N, C, H, W) or (N, D) image patches, optional
    k_neighbours : k for spatial graph
    alpha_expr  : weight for expression heterogeneity
    alpha_hist  : weight for histology heterogeneity (ignored if no patches)
    alpha_spatial : weight for spatial discontinuity

    Returns
    -------
    difficulty : (N,) array in [0, 1], higher = harder
    """
    N = coords.shape[0]
    exp = np.asarray(expression, dtype=np.float64)

    # 1. Expression entropy per spot
    exp_pos = exp + 1e-10
    exp_sum = exp_pos.sum(axis=1, keepdims=True)
    p = exp_pos / exp_sum
    entropy = (-p * np.log(p)).sum(axis=1).astype(np.float32)

    # 2. Patch variance (if patches available)
    if patches is not None:
        pf = np.asarray(patches, dtype=np.float32)
        if pf.ndim >= 3:
            pf = pf.reshape(N, -1)
        patch_var = pf.var(axis=1).astype(np.float32)
    else:
        patch_var = np.zeros(N, dtype=np.float32)

    # 3. Normalize helper
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-10:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    entropy_n = _norm(entropy)
    patch_n = _norm(patch_var) if patches is not None else np.zeros(N, dtype=np.float32)

    # 4. Build spatial graph + Laplacian
    A = build_spatial_adjacency(coords, k=k_neighbours, weight="binary")
    L = graph_laplacian(A)

    # 5. Boundary energy for each feature
    def _boundary_energy(feat: np.ndarray) -> np.ndarray:
        s = L @ feat
        return np.abs(s).astype(np.float32)

    e_entropy = _norm(_boundary_energy(entropy_n))
    e_patch = _norm(_boundary_energy(patch_n)) if patches is not None else np.zeros(N, dtype=np.float32)

    # 6. Combine signals
    if patches is not None:
        base = 0.5 * entropy_n + 0.5 * patch_n
        spatial_energy = 0.5 * e_entropy + 0.5 * e_patch
        difficulty = (alpha_expr * entropy_n + alpha_hist * patch_n + alpha_spatial * spatial_energy)
    else:
        base = entropy_n
        spatial_energy = e_entropy
        difficulty = (0.5 * entropy_n + 0.5 * e_entropy)

    return _norm(difficulty)


# ---------------------------------------------------------------------------
# Niche-level difficulty (delegates to src.niche)
# ---------------------------------------------------------------------------

def niche_difficulty_from_data(
    expression: np.ndarray,
    coords: np.ndarray,
    alpha: float = 0.40,
    beta: float = 0.30,
    gamma: float = 0.15,
    delta: float = 0.15,
    method: str = "spatial_leiden",
    resolution: float = 1.0,
    n_niches: Optional[int] = None,
    spatial_weight: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-call convenience: build niches and compute niche-level difficulty.

    Parameters
    ----------
    expression  : (N, G) log-normalised gene expression.
    coords      : (N, 2) spatial coordinates.
    alpha–delta : niche difficulty component weights (see
                  ``compute_niche_difficulty``).
    method      : niche construction method.
    resolution  : Leiden resolution.
    n_niches    : fixed niche count (only for k-means).
    spatial_weight : spatial vs expression weight in joint feature space.

    Returns
    -------
    niche_labels  : (N,) int — which niche each spot belongs to.
    niche_scores  : (K,) float — difficulty of each niche.
    spot_scores   : (N,) float — difficulty of each spot (mapped from niche).
    """
    from .niche import build_spatial_niches, compute_niche_difficulty

    niche_labels = build_spatial_niches(
        expression=expression,
        coords=coords,
        method=method,
        resolution=resolution,
        n_niches=n_niches,
        spatial_weight=spatial_weight,
    )

    niche_scores, spot_scores = compute_niche_difficulty(
        expression=expression,
        coords=coords,
        niche_labels=niche_labels,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )

    return niche_labels, niche_scores, spot_scores
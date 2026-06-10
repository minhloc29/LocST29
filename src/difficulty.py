"""Difficulty computation for spatial transcriptomics.

Provides both **spot-level** and **niche-level** (microenvironment)
difficulty scoring, along with the spatial graph utilities they share.

All difficulty scores are in [0, 1], where higher = harder to learn.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors


# ===================================================================
#  Spatial graph utilities
# ===================================================================

def build_spatial_adjacency(
    coords: np.ndarray,
    k: int = 6,
    weight: str = "binary",   # "binary" | "distance" | "gaussian"
    sigma: float = 1.0,
) -> csr_matrix:
    """Build a sparse k-NN adjacency matrix from 2D spot coordinates.

    Parameters
    ----------
    coords  : (N, 2) pixel coordinates.
    k       : number of nearest neighbours.
    weight  : edge weighting scheme: ``"binary"`` (1 for all edges),
              ``"distance"`` (1 / dist), or ``"gaussian"``.
    sigma   : bandwidth for gaussian weights (same units as coords).

    Returns
    -------
    A : (N, N) symmetric sparse adjacency matrix.
    """
    N = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    rows, cols, vals = [], [], []
    for i in range(N):
        for j_pos in range(1, k + 1):          # skip self (index 0)
            j = indices[i, j_pos]
            d = distances[i, j_pos]

            if weight == "binary":
                w = 1.0
            elif weight == "distance":
                w = 1.0 / (d + 1e-8)
            else:                               # gaussian
                w = float(np.exp(-(d ** 2) / (2 * sigma ** 2 + 1e-10)))

            rows.append(i); cols.append(j); vals.append(w)
            rows.append(j); cols.append(i); vals.append(w)

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A


def graph_laplacian(A: csr_matrix) -> csr_matrix:
    """Unnormalized graph Laplacian  L = D - A."""
    degree = np.asarray(A.sum(axis=1)).ravel()
    D = diags(degree)
    return D - A


def _norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-10:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


# ===================================================================
#  Spot-level difficulty
# ===================================================================

def graph_signal_energy_difficulty(
    base_score: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
    alpha: float = 0.5,
    weight: str = "binary",
    sigma: float = 1.0,
) -> np.ndarray:
    """Topology-aware difficulty via Graph Signal Energy (GSE).

    Algorithm
    ---------
    1. Build k-NN spatial graph → Laplacian L.
    2. Apply Laplacian to base difficulty:  s = L @ base_score.
    3. Local energy: e_i = |s_i|  (high ↔ sharp boundary).
    4. Combine: d_i = α · base_norm_i + (1-α) · energy_norm_i.

    Returns
    -------
    gse_difficulty : (N,) array in [0, 1], higher = harder.
    """
    base_score = np.asarray(base_score, dtype=np.float64)
    N = len(base_score)

    A = build_spatial_adjacency(coords, k=k, weight=weight, sigma=sigma)
    L = graph_laplacian(A)

    s = L @ base_score
    e = np.abs(s)

    base_n = _norm(base_score)
    energy_n = _norm(e)

    gse_difficulty = alpha * base_n + (1.0 - alpha) * energy_n
    return gse_difficulty.astype(np.float32)


def topological_difficulty_from_data(
    expression: np.ndarray,
    coords: np.ndarray,
    patches: Optional[np.ndarray] = None,
    k_neighbours: int = 6,
    alpha_expr: float = 0.4,
    alpha_hist: float = 0.3,
    alpha_spatial: float = 0.3,
) -> np.ndarray:
    """Spot-level difficulty computed directly from raw data (no training).

    Captures three signals:
    1. Expression heterogeneity (entropy per spot).
    2. Histology heterogeneity (patch variance per spot) — optional.
    3. Spatial discontinuity (graph Laplacian on features).

    Returns
    -------
    difficulty : (N,) array in [0, 1], higher = harder.
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

    entropy_n = _norm(entropy)
    patch_n = _norm(patch_var) if patches is not None else np.zeros(N, dtype=np.float32)

    # 3. Spatial graph + Laplacian
    A = build_spatial_adjacency(coords, k=k_neighbours, weight="binary")
    L = graph_laplacian(A)

    def _boundary_energy(feat: np.ndarray) -> np.ndarray:
        s = L @ feat
        return np.abs(s).astype(np.float32)

    e_entropy = _norm(_boundary_energy(entropy_n))
    e_patch = _norm(_boundary_energy(patch_n)) if patches is not None else np.zeros(N, dtype=np.float32)

    # 4. Combine
    if patches is not None:
        difficulty = (
            alpha_expr * entropy_n
            + alpha_hist * patch_n
            + alpha_spatial * (0.5 * e_entropy + 0.5 * e_patch)
        )
    else:
        difficulty = 0.5 * entropy_n + 0.5 * e_entropy

    return _norm(difficulty)



def build_spatial_niches(
    expression: np.ndarray,
    coords: np.ndarray,
    method: str = "spatial_leiden",
    resolution: float = 1.0,
    n_niches: Optional[int] = None,
    spatial_weight: float = 0.3,
    n_neighbors: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster spots into niches using joint [expression, space] features.

    Parameters
    ----------
    method : ``"spatial_leiden"``, ``"spatial_kmeans"``, or ``"louvain"``.

    Returns
    -------
    niche_labels : (N,) int array, values in [0, K-1].
    """
    N = expression.shape[0]

    exp_std = (expression - expression.mean(axis=0, keepdims=True)) / (
        expression.std(axis=0, keepdims=True) + 1e-10
    )
    coord_std = (coords - coords.mean(axis=0, keepdims=True)) / (
        coords.std(axis=0, keepdims=True) + 1e-10
    )

    joint = np.concatenate(
        [exp_std * (1.0 - spatial_weight), coord_std * spatial_weight], axis=1
    )

    if method == "spatial_leiden":
        return _niches_via_leiden(joint, resolution, n_neighbors, random_state)
    elif method == "spatial_kmeans":
        K = n_niches if n_niches is not None else min(20, N // 5)
        labels = KMeans(
            n_clusters=K, random_state=random_state, n_init=10
        ).fit_predict(joint)
        return labels.astype(np.int32)
    elif method == "louvain":
        return _niches_via_leiden(joint, resolution, n_neighbors, random_state,
                                  use_louvain=True)
    else:
        raise ValueError(f"Unknown niche method: {method}")


def _niches_via_leiden(
    joint: np.ndarray,
    resolution: float,
    n_neighbors: int,
    random_state: int,
    use_louvain: bool = False,
) -> np.ndarray:
    """Leiden / Louvain on a k-NN graph in joint space."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(joint)
    adj = nbrs.kneighbors_graph(joint, mode="connectivity")

    try:
        import scanpy as sc
        import anndata as ad
    except ImportError:
        raise ImportError(
            "scanpy is required for Leiden/Louvain niche construction. "
            "Install with: pip install scanpy"
        )

    adata = ad.AnnData(joint)
    adata.obsp["connectivities"] = adj

    if use_louvain:
        sc.tl.louvain(adata, resolution=resolution, random_state=random_state)
        labels = np.array(adata.obs["louvain"].astype(int).values)
    else:
        sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
        labels = np.array(adata.obs["leiden"].astype(int).values)

    return labels.astype(np.int32)


# ===================================================================
#  Niche-level difficulty metrics
# ===================================================================

def niche_heterogeneity(
    expression: np.ndarray,
    niche_labels: np.ndarray,
) -> np.ndarray:
    """Internal transcriptional diversity of each niche.

    High heterogeneity → niche contains dissimilar spots → harder.
    Returns (K,) array.
    """
    K = int(niche_labels.max()) + 1
    het = np.zeros(K, dtype=np.float32)

    for k in range(K):
        mask = niche_labels == k
        niche_expr = expression[mask]
        n = len(niche_expr)
        if n > 1:
            dists = cosine_distances(niche_expr)
            het[k] = float(dists[np.triu_indices_from(dists, k=1)].mean())
        else:
            het[k] = 0.0
    return het


def niche_topology_difficulty(
    niche_labels: np.ndarray,
    coords: np.ndarray,
    expression: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Topological difficulty on the niche-level graph.

    Builds a graph of niche centroids, then applies the Laplacian to
    niche-averaged expression.  High energy → niche sits on a boundary.

    Returns (K,) array.
    """
    K = int(niche_labels.max()) + 1
    centroids = np.zeros((K, 2), dtype=np.float64)
    niche_expr = np.zeros((K, expression.shape[1]), dtype=np.float64)

    for k in range(K):
        mask = niche_labels == k
        centroids[k] = coords[mask].mean(axis=0)
        niche_expr[k] = expression[mask].mean(axis=0)

    A = build_spatial_adjacency(centroids, k=min(k, K - 1), weight="binary")
    L = graph_laplacian(A)

    s = L @ niche_expr
    energy = np.abs(s).mean(axis=1)
    return energy.astype(np.float32)


def niche_ambiguity(
    expression: np.ndarray,
    niche_labels: np.ndarray,
) -> np.ndarray:
    """Per-spot ambiguity of niche membership.

    High when a spot is nearly as close to another niche's centroid
    as to its own — i.e. it lies near a niche boundary.

    Returns (N,) array.
    """
    K = int(niche_labels.max()) + 1
    centroids = np.zeros((K, expression.shape[1]), dtype=np.float64)

    for k in range(K):
        mask = niche_labels == k
        centroids[k] = expression[mask].mean(axis=0)

    dist_to_own = np.zeros(len(expression), dtype=np.float32)
    dist_to_nearest_other = np.zeros(len(expression), dtype=np.float32)

    for i in range(len(expression)):
        own = int(niche_labels[i])
        d_own = euclidean_distances(
            expression[i:i + 1], centroids[own:own + 1]
        )[0, 0]
        dist_to_own[i] = d_own

        other_mask = np.ones(K, dtype=bool)
        other_mask[own] = False
        d_other = euclidean_distances(
            expression[i:i + 1], centroids[other_mask]
        ).min()
        dist_to_nearest_other[i] = d_other

    return (dist_to_own / (dist_to_nearest_other + 1e-10)).astype(np.float32)


def compute_niche_difficulty(
    expression: np.ndarray,
    coords: np.ndarray,
    niche_labels: np.ndarray,
    alpha: float = 0.40,
    beta: float = 0.30,
    gamma: float = 0.15,
    delta: float = 0.15,
    niche_dynamics_uncertainty: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Composite niche-level difficulty score.

    D_niche = α·heterogeneity + β·topology + γ·ambiguity [+ δ·uncertainty]

    Parameters
    ----------
    expression  : (N, G) gene expression.
    coords      : (N, 2) spatial coordinates.
    niche_labels : (N,) niche assignment per spot.
    alpha–delta : component weights.
    niche_dynamics_uncertainty : (K,) optional training-dynamics uncertainty.

    Returns
    -------
    niche_scores : (K,) difficulty per niche.
    spot_scores  : (N,) difficulty propagated to each spot.
    """
    K = int(niche_labels.max()) + 1

    het = niche_heterogeneity(expression, niche_labels)
    topo = niche_topology_difficulty(niche_labels, coords, expression)

    amb_per_spot = niche_ambiguity(expression, niche_labels)
    amb = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask = niche_labels == k
        amb[k] = amb_per_spot[mask].mean()

    het_n = _norm(het)
    topo_n = _norm(topo)
    amb_n = _norm(amb)

    niche_scores = alpha * het_n + beta * topo_n + gamma * amb_n

    if niche_dynamics_uncertainty is not None:
        unc_n = _norm(niche_dynamics_uncertainty)
        total = alpha + beta + gamma + delta
        niche_scores = (
            (alpha / total) * het_n
            + (beta / total) * topo_n
            + (gamma / total) * amb_n
            + (delta / total) * unc_n
        )

    # Propagate to spots
    spot_scores = np.array(
        [niche_scores[int(lbl)] for lbl in niche_labels], dtype=np.float32
    )
    return niche_scores, spot_scores


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
    """One-call: build niches + compute niche-level difficulty.

    Returns
    -------
    niche_labels : (N,) int
    niche_scores : (K,) float
    spot_scores  : (N,) float
    """
    niche_labels = build_spatial_niches(
        expression=expression, coords=coords,
        method=method, resolution=resolution,
        n_niches=n_niches, spatial_weight=spatial_weight,
    )
    niche_scores, spot_scores = compute_niche_difficulty(
        expression=expression, coords=coords,
        niche_labels=niche_labels,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
    )
    return niche_labels, niche_scores, spot_scores


# ===================================================================
#  Niche summary
# ===================================================================

def summarise_niches(
    niche_labels: np.ndarray,
    niche_scores: np.ndarray,
) -> dict:
    """Print and return per-niche summary statistics."""
    K = len(niche_scores)
    counts = np.bincount(niche_labels, minlength=K)
    stats = {
        "n_niches": K,
        "mean_niche_size": float(counts.mean()),
        "std_niche_size": float(counts.std()),
        "mean_niche_difficulty": float(niche_scores.mean()),
        "std_niche_difficulty": float(niche_scores.std()),
        "frac_hard_niches": float((niche_scores > 0.75).mean()),
    }
    print(f"[Difficulty] Niches: K={K} | mean_size={stats['mean_niche_size']:.1f} | "
          f"mean_difficulty={stats['mean_niche_difficulty']:.3f} | "
          f"hard_frac={stats['frac_hard_niches']:.1%}")
    return stats


# ===================================================================
#  Convenience (takes a SpatialDynamicsField)
# ===================================================================

def gse_difficulty_from_field(field, k: int = 6, alpha: float = 0.5,
                              weight: str = "binary", sigma: float = 1.0
                              ) -> np.ndarray:
    """GSE difficulty directly from a ``SpatialDynamicsField``."""
    return graph_signal_energy_difficulty(
        base_score=field.difficulty_score, coords=field.coords,
        k=k, alpha=alpha, weight=weight, sigma=sigma,
    )



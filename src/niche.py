"""Niche / Microenvironment construction and analysis for spatial transcriptomics.

A *niche* is a spatially contiguous group of spots that share similar
gene expression profiles — a biological microenvironment.

Niche-level operations are the foundation of **biologically-aware
curriculum learning**, where difficulty is assigned to microenvironments
rather than individual spots.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors

from .difficulty_gse import build_spatial_adjacency, graph_laplacian


# ---------------------------------------------------------------------------
# Niche construction
# ---------------------------------------------------------------------------

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
    """Build niche labels by clustering on joint [expression, space] features.

    Parameters
    ----------
    expression  : (N, G) log-normalized gene expression.
    coords      : (N, 2) spatial coordinates.
    method      : one of ``"spatial_leiden"``, ``"spatial_kmeans"``,
                  ``"louvain"``.
    resolution  : Leiden resolution (larger → more niches).
    n_niches    : fixed number of niches (for k-means).  If None and method
                  is k-means, defaults to 20.
    spatial_weight : weight of spatial coordinates in the joint feature space.
                     0 → pure expression; 1 → pure spatial.
    n_neighbors : number of neighbours for graph construction.
    random_state : random seed.

    Returns
    -------
    niche_labels : (N,) int array, values in [0, K-1].
    """
    N = expression.shape[0]

    # --- Normalise expression & coordinates ---
    exp_std = (expression - expression.mean(axis=0, keepdims=True)) / (
        expression.std(axis=0, keepdims=True) + 1e-10
    )
    coord_std = (coords - coords.mean(axis=0, keepdims=True)) / (
        coords.std(axis=0, keepdims=True) + 1e-10
    )

    # --- Joint features ---
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
    """Run Leiden or Louvain on a k-NN graph built from *joint*."""
    # Build k-NN graph in joint space
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


def build_expression_niches(
    expression: np.ndarray,
    coords: np.ndarray,
    method: str = "spatial_leiden",
    resolution: float = 1.0,
    n_niches: Optional[int] = None,
    spatial_weight: float = 0.3,
    n_neighbors: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Alias for ``build_spatial_niches``."""
    return build_spatial_niches(
        expression=expression,
        coords=coords,
        method=method,
        resolution=resolution,
        n_niches=n_niches,
        spatial_weight=spatial_weight,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Niche-level difficulty metrics
# ---------------------------------------------------------------------------

def niche_heterogeneity(
    expression: np.ndarray,
    niche_labels: np.ndarray,
) -> np.ndarray:
    """Internal diversity of each niche.

    High heterogeneity → the niche contains transcriptionally dissimilar
    spots → likely harder for the model to learn as a unit.

    Returns
    -------
    het : (K,) array, higher = more heterogeneous.
    """
    K = int(niche_labels.max()) + 1
    het = np.zeros(K, dtype=np.float32)

    for k in range(K):
        mask = niche_labels == k
        niche_expr = expression[mask]
        n = len(niche_expr)
        if n > 1:
            # Mean pairwise cosine distance within the niche
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

    Builds a graph whose nodes are niche **centroids**, then applies the
    graph Laplacian to niche-averaged expression.  High energy → niche
    sits on a sharp expression boundary in niche-space.

    Returns
    -------
    energy : (K,) array, higher = more boundary-like.
    """
    K = int(niche_labels.max()) + 1
    centroids = np.zeros((K, 2), dtype=np.float64)
    niche_expr = np.zeros((K, expression.shape[1]), dtype=np.float64)

    for k in range(K):
        mask = niche_labels == k
        centroids[k] = coords[mask].mean(axis=0)
        niche_expr[k] = expression[mask].mean(axis=0)

    # Graph on niche centroids
    A = build_spatial_adjacency(centroids, k=min(k, K - 1), weight="binary")
    L = graph_laplacian(A)

    # Apply Laplacian to mean expression
    s = L @ niche_expr  # (K, G)
    energy = np.abs(s).mean(axis=1)  # (K,) — average across genes

    return energy.astype(np.float32)


def niche_ambiguity(
    expression: np.ndarray,
    niche_labels: np.ndarray,
) -> np.ndarray:
    """Per-spot ambiguity of niche membership.

    A spot is *ambiguous* when it is nearly as close to a different niche's
    centroid as it is to its own — i.e. it sits near a niche boundary in
    expression space.

    Returns
    -------
    ambiguity : (N,) array, higher = more ambiguous (harder).
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
        # Distance to own centroid
        d_own = euclidean_distances(
            expression[i:i + 1], centroids[own:own + 1]
        )[0, 0]
        dist_to_own[i] = d_own

        # Minimum distance to any *other* centroid
        other_mask = np.ones(K, dtype=bool)
        other_mask[own] = False
        d_other = euclidean_distances(
            expression[i:i + 1], centroids[other_mask]
        ).min()
        dist_to_nearest_other[i] = d_other

    # Ambiguity: own-dist / nearest-other-dist
    # High when the spot is as close to another niche as to its own
    ambiguity = dist_to_own / (dist_to_nearest_other + 1e-10)
    return ambiguity.astype(np.float32)


# ---------------------------------------------------------------------------
# Composite niche difficulty
# ---------------------------------------------------------------------------

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
    """Compute composite difficulty at the niche level.

    Parameters
    ----------
    expression  : (N, G) log-normalized gene expression.
    coords      : (N, 2) spatial coordinates.
    niche_labels : (N,) niche assignment per spot.
    alpha       : weight for niche *heterogeneity*.
    beta        : weight for niche *topology* (boundary energy).
    gamma       : weight for niche *ambiguity*.
    delta       : weight for *uncertainty* from training dynamics (optional).
    niche_dynamics_uncertainty : (K,) optional uncertainty from training.

    Returns
    -------
    niche_scores : (K,) difficulty per niche (0–1, higher = harder).
    spot_scores  : (N,) difficulty propagated to each spot.
    """
    K = int(niche_labels.max()) + 1

    # 1. Heterogeneity
    het = niche_heterogeneity(expression, niche_labels)

    # 2. Topology
    topo = niche_topology_difficulty(niche_labels, coords, expression)

    # 3. Ambiguity — aggregate per-niche mean
    amb_per_spot = niche_ambiguity(expression, niche_labels)
    amb = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask = niche_labels == k
        amb[k] = amb_per_spot[mask].mean()

    # 4. Normalise each component to [0, 1]
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-10:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    het_n = _norm(het)
    topo_n = _norm(topo)
    amb_n = _norm(amb)

    # 5. Weighted combination
    niche_scores = alpha * het_n + beta * topo_n + gamma * amb_n

    if niche_dynamics_uncertainty is not None:
        unc_n = _norm(niche_dynamics_uncertainty)
        # Renormalise weights
        total = alpha + beta + gamma + delta
        niche_scores = (
            (alpha / total) * het_n
            + (beta / total) * topo_n
            + (gamma / total) * amb_n
            + (delta / total) * unc_n
        )

    # 6. Propagate to spots
    spot_scores = np.array([niche_scores[int(lbl)] for lbl in niche_labels],
                           dtype=np.float32)

    return niche_scores, spot_scores


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

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
    print(f"[Niche] K={K} | mean_size={stats['mean_niche_size']:.1f} | "
          f"mean_difficulty={stats['mean_niche_difficulty']:.3f} | "
          f"hard_frac={stats['frac_hard_niches']:.1%}")
    return stats

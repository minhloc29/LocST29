from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, diags
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

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
    weight: str = "binary",   
    sigma: float = 1.0,
) -> csr_matrix:
   
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

            rows.append(i)
            cols.append(j)
            vals.append(w)
            rows.append(j)
            cols.append(i)
            vals.append(w)   

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A


def graph_laplacian(A: csr_matrix) -> csr_matrix:
   
    degree = np.asarray(A.sum(axis=1)).ravel()
    D = diags(degree)
    return D - A


def niche_difficulty_from_data(
    expression: np.ndarray,
    coords: np.ndarray,
    method: str = "spatial_leiden",
    resolution: float = 1.0,
    n_niches: Optional[int] = None,
    spatial_weight: float = 0.3,
    aggregation: str = "correlation_weighted",
    per_spot_error: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-call convenience: build niches and compute niche-level difficulty.

    Parameters
    ----------
    expression  : (N, G) log-normalised gene expression.
    coords      : (N, 2) spatial coordinates.
    alpha–delta : niche difficulty component weights (only used when
                  ``aggregation="linear"``).
    method      : niche construction method.
    resolution  : Leiden resolution.
    n_niches    : fixed niche count (only for k-means).
    spatial_weight : spatial vs expression weight in joint feature space.
    aggregation : aggregation strategy for combining difficulty components
                  (passed to ``compute_niche_difficulty``).
    per_spot_error : (N,) optional — per-spot prediction error for
                     correlation-weighted aggregation.

    Returns
    -------
    niche_labels  : (N,) int — which niche each spot belongs to.
    niche_scores  : (K,) float — difficulty of each niche.
    spot_scores   : (N,) float — difficulty of each spot (mapped from niche).
    """

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
        aggregation=aggregation,
        per_spot_error=per_spot_error,
    )

    return niche_labels, niche_scores, spot_scores


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
    n_pcs = min(30, exp_std.shape[1])

    exp_pca = PCA(n_components=n_pcs, random_state=random_state).fit_transform(exp_std)

    joint = np.concatenate([exp_pca * (1.0 - spatial_weight), coord_std * spatial_weight], axis=1)

    if method == "spatial_leiden":
        return _niches_via_leiden(joint, coord_std, resolution, n_neighbors, random_state)
    elif method == "spatial_kmeans":
        K = n_niches if n_niches is not None else min(20, N // 5)
        labels = KMeans(
            n_clusters=K, random_state=random_state, n_init=10
        ).fit_predict(joint)
        return labels.astype(np.int32)
    elif method == "louvain":
        return _niches_via_leiden(joint, coord_std, resolution, n_neighbors, random_state,
                                  use_louvain=True)
    else:
        raise ValueError(f"Unknown niche method: {method}")


def _niches_via_leiden(
    joint: np.ndarray,
    coord_std,
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
        sc.tl.louvain(adata, resolution=resolution, random_state=random_state, adjacency=adata.obsp["connectivities"])
        labels = np.array(adata.obs["louvain"].astype(int).values)
    else:
        sc.tl.leiden(adata, resolution=resolution, random_state=random_state, adjacency=adata.obsp["connectivities"])
        labels = np.array(adata.obs["leiden"].astype(int).values)

    return labels.astype(np.int32)


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

    for niche_idx in range(K):
        mask = niche_labels == niche_idx
        centroids[niche_idx] = coords[mask].mean(axis=0)
        niche_expr[niche_idx] = expression[mask].mean(axis=0)

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

    ambiguity = dist_to_own / (dist_to_nearest_other + 1e-10)
    return ambiguity.astype(np.float32)


def _rank(x: np.ndarray) -> np.ndarray:
    """Convert values to ranks in [0, 1].  Lower rank = lower original value."""
    from scipy.stats import rankdata
    ranks = rankdata(x, method="average")  # 1-based
    return (ranks - 1.0) / (len(ranks) - 1.0) if len(ranks) > 1 else np.zeros_like(ranks)


def _normalise(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = x.min(), x.max()
    return np.zeros_like(x, dtype=np.float32) if hi - lo < 1e-10 else ((x - lo) / (hi - lo)).astype(np.float32)


def rank_aggregate_correlation_weighted(
    het: np.ndarray,
    topo: np.ndarray,
    amb_per_spot: np.ndarray,
    niche_labels: np.ndarray,
    per_spot_error: Optional[np.ndarray] = None,
    *,
    flip_negative_topo: bool = True,
    epsilon: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    
    from scipy.stats import spearmanr

    K = int(niche_labels.max()) + 1
    N = len(niche_labels)

    amb_niche = np.array([amb_per_spot[niche_labels == k].mean() for k in range(K)])

    if per_spot_error is not None and per_spot_error.std() > 1e-10:
        # Compute correlations at the per-spot level
        spot_het = np.array([het[lbl] for lbl in niche_labels])
        spot_topo = np.array([topo[lbl] for lbl in niche_labels])

        r_het, _ = spearmanr(spot_het, per_spot_error)
        r_topo, _ = spearmanr(spot_topo, per_spot_error)
        r_amb, _ = spearmanr(amb_per_spot, per_spot_error)

        abs_sum = abs(r_het) + abs(r_topo) + abs(r_amb)
        if abs_sum < 1e-10:
            w_het = w_topo = w_amb = 1.0 / 3.0
        else:
            w_het = abs(r_het) / abs_sum
            w_topo = abs(r_topo) / abs_sum
            w_amb = abs(r_amb) / abs_sum

        for w in [w_het, w_topo, w_amb]:
            if w < epsilon:
                w = epsilon

        w_sum = w_het + w_topo + w_amb
        w_het /= w_sum
        w_topo /= w_sum
        w_amb /= w_sum

        flip_topo = r_topo < 0 and flip_negative_topo
    else:
        w_het = w_topo = w_amb = 1.0 / 3.0
        flip_topo = False

    het_rank = _rank(het)
    topo_rank = _rank(topo)
    amb_rank = _rank(amb_niche)

    if flip_topo:
        topo_rank = 1.0 - topo_rank

    niche_scores = (
        w_het * het_rank + w_topo * topo_rank + w_amb * amb_rank
    )
    niche_scores = _normalise(niche_scores)

    amb_n_spot = _normalise(amb_per_spot)
    spot_scores = np.array([niche_scores[int(lbl)] for lbl in niche_labels], dtype=np.float32)
    spot_scores = _normalise(spot_scores + 0.3 * amb_n_spot)

    return niche_scores.astype(np.float32), spot_scores.astype(np.float32)


def rank_aggregate_median(
    het: np.ndarray,
    topo: np.ndarray,
    amb_per_spot: np.ndarray,
    niche_labels: np.ndarray,
    *,
    flip_negative_topo: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    
    K = int(niche_labels.max()) + 1
    amb_niche = np.array([amb_per_spot[niche_labels == k].mean() for k in range(K)])

    het_n = _normalise(het)
    topo_n = _normalise(topo)
    amb_n = _normalise(amb_niche)

    if flip_negative_topo:
        topo_n = 1.0 - topo_n

    stacked = np.column_stack([het_n, topo_n, amb_n])  # (K, 3)
    niche_scores = np.median(stacked, axis=1)
    niche_scores = _normalise(niche_scores)

    spot_scores = np.array([niche_scores[int(lbl)] for lbl in niche_labels], dtype=np.float32)
    amb_n_spot = _normalise(amb_per_spot)
    spot_scores = _normalise(spot_scores + 0.3 * amb_n_spot)

    return niche_scores.astype(np.float32), spot_scores.astype(np.float32)


def rank_aggregate_borda(
    het: np.ndarray,
    topo: np.ndarray,
    amb_per_spot: np.ndarray,
    niche_labels: np.ndarray,
    *,
    flip_negative_topo: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Borda rank aggregation (Strategy 2).

    Simple average of the three component ranks — no weights, no training.
    Surprisingly robust in ensemble settings.

    Returns
    -------
    niche_scores : (K,)
    spot_scores  : (N,)
    """
    K = int(niche_labels.max()) + 1
    amb_niche = np.array([amb_per_spot[niche_labels == k].mean() for k in range(K)])

    het_rank = _rank(het)
    topo_rank = _rank(topo)
    amb_rank = _rank(amb_niche)

    if flip_negative_topo:
        topo_rank = 1.0 - topo_rank

    niche_scores = (het_rank + topo_rank + amb_rank) / 3.0
    niche_scores = _normalise(niche_scores)

    spot_scores = np.array([niche_scores[int(lbl)] for lbl in niche_labels], dtype=np.float32)
    amb_n_spot = _normalise(amb_per_spot)
    spot_scores = _normalise(spot_scores + 0.3 * amb_n_spot)

    return niche_scores.astype(np.float32), spot_scores.astype(np.float32)



def compute_niche_difficulty(
    expression, coords, niche_labels,
    aggregation: str = "correlation_weighted",
    per_spot_error: Optional[np.ndarray] = None,
):
   
    K = int(niche_labels.max()) + 1
    het = niche_heterogeneity(expression, niche_labels)
    topo = niche_topology_difficulty(niche_labels, coords, expression)
    amb_per_spot = niche_ambiguity(expression, niche_labels)

    if aggregation in ("correlation_weighted",):
        return rank_aggregate_correlation_weighted(
            het, topo, amb_per_spot, niche_labels,
            per_spot_error=per_spot_error,
        )
    elif aggregation == "borda":
        return rank_aggregate_borda(
            het, topo, amb_per_spot, niche_labels,
        )
    elif aggregation == "median":
        return rank_aggregate_median(
            het, topo, amb_per_spot, niche_labels,
        )
        

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


def plot_niche_quality(
    expression: np.ndarray,
    coords: np.ndarray,
    niche_labels: np.ndarray,
    niche_scores: np.ndarray,
    spot_scores: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (18, 12),
    cmap: str = "plasma",
    title: str = "Niche Quality Dashboard",
):
   

    K = int(niche_labels.max()) + 1
    counts = np.bincount(niche_labels, minlength=K)

    # --- Compute auxiliary metrics on the fly ---
    sil_values = _per_niche_silhouette(expression, coords, niche_labels)
    spat_coh = _spatial_coherence_per_spot(niche_labels, coords)
    het = niche_heterogeneity(expression, niche_labels)
    topo = niche_topology_difficulty(niche_labels, coords, expression)
    amb_per_spot = niche_ambiguity(expression, niche_labels)
    amb = np.array([amb_per_spot[niche_labels == k].mean() for k in range(K)])

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---- 1. Spatial map (discrete niches) ----
    ax = axes[0, 0]
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=niche_labels, cmap="tab20",
        s=8, alpha=0.7, edgecolors="none"
    )
    ax.set_title(f"Spatial Niche Map (K={K})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(K), shrink=0.7)
    cbar.set_label("Niche label")

    ax = axes[0, 1]
    norm = Normalize(vmin=spot_scores.min(), vmax=spot_scores.max())
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=spot_scores, cmap=cmap,
        s=8, alpha=0.7, edgecolors="none", norm=norm,
    )
    ax.set_title("Per-Spot Difficulty")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Difficulty")

    # ---- 3. Niche composition ----
    ax = axes[0, 2]
    colors = [plt.cm.tab20(i % 20) for i in range(K)]
    ax.bar(range(K), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Niche Size Distribution")
    ax.set_xlabel("Niche")
    ax.set_ylabel("Spot count")
    ax.set_xticks(range(K))
    mean_size = counts.mean()
    ax.axhline(mean_size, color="red", ls="--", lw=0.8, label=f"Mean = {mean_size:.0f}")
    ax.legend(fontsize=8)

    # ---- 4. Difficulty components ----
    ax = axes[1, 0]
    x = np.arange(K)
    width = 0.25
    ax.bar(x - width, het, width, label="Heterogeneity", alpha=0.8)
    ax.bar(x, topo, width, label="Topology", alpha=0.8)
    ax.bar(x + width, amb, width, label="Ambiguity", alpha=0.8)
    ax.set_title("Niche Difficulty Components")
    ax.set_xlabel("Niche")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.legend(fontsize=7)

    # Overlay composite score as line
    ax_twin = ax.twinx()
    ax_twin.plot(x, niche_scores, "ko-", markersize=3, linewidth=1.2,
                 label="Composite")
    ax_twin.set_ylabel("Composite difficulty", fontsize=9)
    ax_twin.legend(fontsize=7, loc="upper right")

    # ---- 5. Spatial coherence histogram ----
    ax = axes[1, 1]
    ax.hist(spat_coh, bins=30, color="steelblue", edgecolor="white",
            alpha=0.8, density=True)
    ax.axvline(spat_coh.mean(), color="red", ls="--", lw=1.2,
               label=f"Mean = {spat_coh.mean():.3f}")
    ax.set_title("Spatial Coherence per Spot")
    ax.set_xlabel("Fraction of neighbours sharing niche")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # ---- 6. Per-niche silhouette ----
    ax = axes[1, 2]
    valid = ~np.isnan(sil_values)
    if valid.sum() > 0:
        sil_plot = np.full(K, np.nan)
        sil_plot[valid] = sil_values[valid]
        ax.bar(range(K), sil_plot, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title("Per-Niche Silhouette")
        ax.set_xlabel("Niche")
        ax.set_ylabel("Silhouette score")
        ax.set_xticks(range(K))
        mean_sil = np.nanmean(sil_values)
        ax.axhline(mean_sil, color="red", ls="--", lw=0.8,
                   label=f"Mean = {mean_sil:.3f}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not enough niches\nfor silhouette (< 2)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Per-Niche Silhouette")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Niche] Dashboard saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_niche_spatial(
    coords: np.ndarray,
    niche_labels: np.ndarray,
    spot_scores: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Niche Spatial View",
):

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    K = int(niche_labels.max()) + 1

    # Left: discrete niches
    sc1 = ax1.scatter(
        coords[:, 0], coords[:, 1], c=niche_labels,
        cmap="tab20", s=10, alpha=0.7, edgecolors="none",
    )
    ax1.set_title(f"Niche Labels (K={K})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    cbar1 = fig.colorbar(sc1, ax=ax1, ticks=range(K), shrink=0.7)
    cbar1.set_label("Niche")

    # Right: difficulty overlay
    if spot_scores is not None:
        sc2 = ax2.scatter(
            coords[:, 0], coords[:, 1], c=spot_scores,
            cmap="plasma", s=10, alpha=0.7, edgecolors="none",
        )
        ax2.set_title("Spot Difficulty (composite)")
        fig.colorbar(sc2, ax=ax2, shrink=0.7, label="Difficulty")
    else:
        ax2.scatter(
            coords[:, 0], coords[:, 1], c="gray",
            s=10, alpha=0.5, edgecolors="none",
        )
        ax2.set_title("Spatial Layout")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_niche_difficulty_components(
    niche_scores: np.ndarray,
    heterogeneity: np.ndarray,
    topology: np.ndarray,
    ambiguity: np.ndarray,
    niche_labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Niche Difficulty Breakdown",
):

    K = len(niche_scores)
    x = np.arange(K)

    fig, ax = plt.subplots(figsize=(max(6, K * 0.4), 4.5))
    width = 0.25

    ax.bar(x - width, heterogeneity, width, label="Heterogeneity", alpha=0.85)
    ax.bar(x, topology, width, label="Topology (boundary)", alpha=0.85)
    ax.bar(x + width, ambiguity, width, label="Ambiguity", alpha=0.85)

    ax.plot(x, niche_scores, "ko-", markersize=4, linewidth=1.5,
            label="Composite", zorder=5)

    ax.set_title(title)
    ax.set_xlabel("Niche")
    ax.set_ylabel("Score (normalised 0–1)")
    ax.set_xticks(x)
    if niche_labels is not None:
        unique = sorted(set(niche_labels))
        ax.set_xticklabels([f"{lbl}" for lbl in unique])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _per_niche_silhouette(
    expression: np.ndarray,
    coords: np.ndarray,
    niche_labels: np.ndarray,
    spatial_weight: float = 0.3,
    random_state = 42
) -> np.ndarray:
    """Mean silhouette per niche on joint [expression, space] features."""

    K = int(niche_labels.max()) + 1
    if K < 2:
        return np.full(K, np.nan)

    exp_std = StandardScaler().fit_transform(expression)
    coord_std = StandardScaler().fit_transform(coords)
    n_pcs = min(30, exp_std.shape[1])

    exp_pca = PCA(n_components=n_pcs, random_state=random_state).fit_transform(exp_std)

    joint = np.concatenate([exp_pca * (1.0 - spatial_weight), coord_std * spatial_weight], axis=1)

    samples = silhouette_samples(joint, niche_labels)
    sil = np.array([samples[niche_labels == k].mean() for k in range(K)])
    return sil


def _spatial_coherence_per_spot(
    niche_labels: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> np.ndarray:

    N = len(niche_labels)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    agreements = np.zeros(N, dtype=np.float32)
    for i in range(N):
        neighbours = indices[i, 1:]
        agreements[i] = (niche_labels[neighbours] == niche_labels[i]).mean()
    return agreements

from __future__ import annotations

import argparse
import importlib.util as _util
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr



_niche_path = str(Path(__file__).resolve().parent.parent / "src" / "niche.py")
_niche_spec = _util.spec_from_file_location("niche_module", _niche_path)

_difficulty_gse_path = str(Path(__file__).resolve().parent.parent / "src" / "difficulty_gse.py")
_dgse_spec = _util.spec_from_file_location("difficulty_gse_module", _difficulty_gse_path)
_dgse_mod = _util.module_from_spec(_dgse_spec)
_dgse_spec.loader.exec_module(_dgse_mod)
sys.modules["src.difficulty_gse"] = _dgse_mod
sys.modules["difficulty_gse"] = _dgse_mod

_niche_mod = _util.module_from_spec(_niche_spec)
# Make '.' resolve to 'src'
_niche_mod.__package__ = "src"
# Pre-register so that 'from .difficulty_gse import ...' finds it
sys.modules["src.niche"] = _niche_mod
_niche_spec.loader.exec_module(_niche_mod)

compute_niche_difficulty = _niche_mod.compute_niche_difficulty
niche_topology_difficulty_func = _niche_mod.niche_topology_difficulty
niche_ambiguity_func = _niche_mod.niche_ambiguity
summarise_niches = _niche_mod.summarise_niches



def compute_silhouette_metrics(
    expression: np.ndarray,
    coords: np.ndarray,
    niche_labels: np.ndarray,
    spatial_weight: float = 0.3,
) -> Dict[str, float]:
    """Silhouette scores on the joint [expression, space] feature space.

    Returns global and per-niche silhouette stats.
    """
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import StandardScaler

    exp_std = StandardScaler().fit_transform(expression)
    coord_std = StandardScaler().fit_transform(coords)
    joint = np.concatenate(
        [exp_std * (1.0 - spatial_weight), coord_std * spatial_weight], axis=1
    )

    K = int(niche_labels.max()) + 1
    if K < 2:
        return {"silhouette_score": float("nan"), "n_niches": 1, "per_niche_silhouette": {}}

    global_sil = float(silhouette_score(joint, niche_labels))
    samples = silhouette_samples(joint, niche_labels)

    per_niche = {}
    for k in range(K):
        mask = niche_labels == k
        if mask.sum() < 2:
            per_niche[f"niche_{k}_sil"] = float("nan")
        else:
            per_niche[f"niche_{k}_sil"] = float(samples[mask].mean())
        per_niche[f"niche_{k}_n"] = int(mask.sum())

    # Complementary silhouette: expression-only and space-only
    sil_expr = float(silhouette_score(exp_std, niche_labels)) if K >= 2 else float("nan")
    sil_space = float(silhouette_score(coord_std, niche_labels)) if K >= 2 else float("nan")

    return {
        "silhouette_score": global_sil,
        "silhouette_expression": sil_expr,
        "silhouette_spatial": sil_space,
        "n_niches": K,
        "per_niche_silhouette": per_niche,
    }


def compute_spatial_coherence(
    niche_labels: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> Dict[str, float]:
    """Measure spatial coherence of niches using adjacency agreement.

    For each spot, what fraction of its k nearest neighbours share the same niche label?
    High values = spatially coherent niches.  This is essentially the 'purity' of
    the niche segmentation w.r.t. spatial proximity.
    """
    from sklearn.neighbors import NearestNeighbors

    N = len(niche_labels)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # Neighbour agreement: fraction of neighbours (excluding self) with same niche
    agreements = []
    for i in range(N):
        neighbours = indices[i, 1:]  # skip self
        same = (niche_labels[neighbours] == niche_labels[i]).mean()
        agreements.append(same)

    mean_agreement = float(np.mean(agreements))
    std_agreement = float(np.std(agreements))

    # Adjusted Rand Index between niche labels and spatial nearest-neighbour labels
    neighbour_labels = niche_labels[indices[:, 1]]  # label of nearest neighbour only
    from sklearn.metrics import adjusted_rand_score
    ari = float(adjusted_rand_score(niche_labels, neighbour_labels))

    return {
        "spatial_coherence_mean": mean_agreement,
        "spatial_coherence_std": std_agreement,
        "spatial_coherence_ari": ari,
    }


def compute_expression_consistency(
    expression: np.ndarray,
    niche_labels: np.ndarray,
) -> Dict[str, float]:
    """Within-niche expression consistency.

    For each niche, compute the mean pairwise correlation between its member spots.
    High correlation = transcriptionally coherent niche.

    Also computes the Calinski-Harabasz index (variance ratio criterion) — higher
    values mean niches are dense and well-separated.
    """
    from sklearn.metrics import calinski_harabasz_score

    K = int(niche_labels.max()) + 1

    # Per-niche mean pairwise Pearson correlation
    all_corrs = []
    for k in range(K):
        mask = niche_labels == k
        group = expression[mask]
        n = len(group)
        if n < 3:
            continue
        corr_matrix = np.corrcoef(group)
        # Upper triangle mean
        triu = np.triu_indices(n, k=1)
        all_corrs.append(float(corr_matrix[triu].mean()))

    mean_within_corr = float(np.mean(all_corrs)) if all_corrs else float("nan")

    # Calinski-Harabasz (higher = better separated, dense clusters)
    ch = float(calinski_harabasz_score(expression, niche_labels)) if K >= 2 else float("nan")

    return {
        "within_niche_corr": mean_within_corr,
        "calinski_harabasz": ch,
    }


def compute_niche_summary_stats(
    niche_labels: np.ndarray,
    niche_scores: np.ndarray,
) -> Dict[str, float]:
    """Basic summary statistics of the niche partition."""
    K = len(niche_scores)
    counts = np.bincount(niche_labels, minlength=K)

    return {
        "n_niches": K,
        "mean_niche_size": float(counts.mean()),
        "std_niche_size": float(counts.std()),
        "cv_niche_size": float(counts.std() / counts.mean()) if counts.mean() > 0 else float("nan"),
        "min_niche_size": float(counts.min()),
        "max_niche_size": float(counts.max()),
        "mean_difficulty": float(niche_scores.mean()),
        "std_difficulty": float(niche_scores.std()),
        "frac_hard_niches": float((niche_scores > 0.75).mean()),
    }


def compute_boundary_sharpness(
    expression: np.ndarray,
    niche_labels: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> Dict[str, float]:
    """Boundary sharpness via niche-level graph signal energy.

    Uses the same topology difficulty metric internally, but here we
    report how much of the total difficulty variance comes from boundaries.
    """
    topo = niche_topology_difficulty_func(niche_labels, coords, expression, k=k)

    return {
        "boundary_energy_mean": float(topo.mean()),
        "boundary_energy_std": float(topo.std()),
        "boundary_energy_max": float(topo.max()),
    }


def compute_ambiguity_profile(
    expression: np.ndarray,
    niche_labels: np.ndarray,
) -> Dict[str, float]:
    """Per-spot ambiguity metrics.

    Ambiguity is high near niche boundaries in expression space.
    We report what fraction of spots are 'ambiguous' (ambiguity > 0.8
    quantile threshold) as a proxy for niche boundary width.
    """
    amb = niche_ambiguity_func(expression, niche_labels)
    threshold = np.percentile(amb, 80)

    return {
        "ambiguity_mean": float(amb.mean()),
        "ambiguity_std": float(amb.std()),
        "ambiguity_max": float(amb.max()),
        "frac_high_ambiguity": float((amb > threshold).mean()),
    }


def compute_method_stability(
    expression: np.ndarray,
    coords: np.ndarray,
    methods: List[Tuple[str, dict]],
) -> Dict[str, Dict]:
    """Compare niche partitions across construction methods.

    Uses Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
    to measure pairwise agreement between methods.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    results = {}
    all_labels = {}

    for method_name, kwargs in methods:
        labels = build_niches(expression, coords, method=method_name, **kwargs)
        all_labels[method_name] = labels

        # Basic stats
        K = int(labels.max()) + 1
        counts = np.bincount(labels)
        results[method_name] = {
            "n_niches": K,
            "mean_size": float(counts.mean()),
        }

    # Pairwise ARI / NMI
    names = [m[0] for m in methods]
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            ari = adjusted_rand_score(all_labels[n1], all_labels[n2])
            nmi = normalized_mutual_info_score(all_labels[n1], all_labels[n2])
            results[f"ari_{n1}_vs_{n2}"] = float(ari)
            results[f"nmi_{n1}_vs_{n2}"] = float(nmi)

    return results


# =========================================================================
#  Core niche building wrapper (avoids scanpy dependency for synthetic)
# =========================================================================

def build_niches(
    expression: np.ndarray,
    coords: np.ndarray,
    method: str = "spatial_kmeans",
    n_niches: Optional[int] = None,
    spatial_weight: float = 0.3,
    resolution: float = 1.0,
) -> np.ndarray:
    """Build niche labels using sklearn-only methods.

    Uses KMeans directly; for Leiden/Louvain needs scanpy.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    N = expression.shape[0]
    exp_std = StandardScaler().fit_transform(expression)
    coord_std = StandardScaler().fit_transform(coords)
    joint = np.concatenate(
        [exp_std * (1.0 - spatial_weight), coord_std * spatial_weight], axis=1
    )

    if method == "spatial_kmeans":
        K = n_niches if n_niches is not None else min(20, N // 5)
        labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(joint)
        return labels.astype(np.int32)
    elif method in ("spatial_leiden", "louvain"):
        print("  [WARN] Using KMeans fallback (scanpy not available for Leiden)")
        K = n_niches if n_niches is not None else min(15, N // 5)
        labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(joint)
        return labels.astype(np.int32)
    else:
        raise ValueError(f"Unknown method: {method}")


# =========================================================================
#  Full Analysis
# =========================================================================

def run_full_analysis(
    expression: np.ndarray,
    coords: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    method: str = "spatial_kmeans",
    K_values: List[int] = [5, 8, 12, 16, 20, 30],
    name: str = "dataset",
) -> Dict:

    print(f"\n{'='*70}")
    print(f"  NICHE QUALITY ANALYSIS: {name}")
    print(f"  Spots: {expression.shape[0]},  Genes: {expression.shape[1]}")
    print(f"{'='*70}")

    all_results = {}

    for K in K_values:
        print(f"\n{'-'*60}")
        print(f"  K = {K} (spatial_kmeans, spatial_weight=0.3)")
        print(f"{'-'*60}")

        labels = build_niches(expression, coords, method="spatial_kmeans", n_niches=K)
        niche_scores, spot_scores = compute_niche_difficulty(expression, coords, labels)

        sil = compute_silhouette_metrics(expression, coords, labels)

        spatial = compute_spatial_coherence(labels, coords)

        consistency = compute_expression_consistency(expression, labels)

        stats = compute_niche_summary_stats(labels, niche_scores)

        boundary = compute_boundary_sharpness(expression, labels, coords)

        ambiguity = compute_ambiguity_profile(expression, labels)

        # Print summary
        print(f"    Silhouette (joint):     {sil['silhouette_score']:.4f}")
        print(f"    Silhouette (expr only): {sil['silhouette_expression']:.4f}")
        print(f"    Silhouette (space only):{sil['silhouette_spatial']:.4f}")
        print(f"    Spatial coherence:      {spatial['spatial_coherence_mean']:.4f}")
        print(f"    Spatial coherence ARI:  {spatial['spatial_coherence_ari']:.4f}")
        print(f"    Within-niche corr:      {consistency['within_niche_corr']:.4f}")
        print(f"    Calinski-Harabasz:      {consistency['calinski_harabasz']:.4f}")
        print(f"    Mean niche size:        {stats['mean_niche_size']:.1f} (CV={stats['cv_niche_size']:.2f})")
        print(f"    Mean difficulty:        {stats['mean_difficulty']:.4f}")
        print(f"    Boundary energy:        {boundary['boundary_energy_mean']:.4f}")
        print(f"    High ambiguity frac:    {ambiguity['frac_high_ambiguity']:.4f}")

        # If true_labels available, compute ARI and NMI against ground truth
        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari_gt = adjusted_rand_score(true_labels, labels)
            nmi_gt = normalized_mutual_info_score(true_labels, labels)
            print(f"    ARI vs ground truth:    {ari_gt:.4f}")
            print(f"    NMI vs ground truth:    {nmi_gt:.4f}")

        # Difficulty vs silhouette per-niche (are hard niches more diffuse?)
        per_niche_sil = sil["per_niche_silhouette"]
        sil_values = []
        for k in range(K):
            key = f"niche_{k}_sil"
            if key in per_niche_sil and not np.isnan(per_niche_sil[key]):
                sil_values.append(per_niche_sil[key])
            else:
                sil_values.append(float("nan"))

        sil_array = np.array(sil_values)
        valid = ~np.isnan(sil_array)
        if valid.sum() >= 3:
            rho, p = spearmanr(niche_scores[valid], sil_array[valid])
            print(f"    Difficulty vs Silhouette (Spearman ρ): {rho:.4f} (p={p:.4f})")

        out = {
            "method": method,
            "K": K,
            "niche_scores": niche_scores,
            "spot_scores": spot_scores,
            **sil,
            **spatial,
            **consistency,
            **stats,
            **boundary,
            **ambiguity,
        }

        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            out["ari_ground_truth"] = float(adjusted_rand_score(true_labels, labels))
            out["nmi_ground_truth"] = float(normalized_mutual_info_score(true_labels, labels))

        all_results[f"K={K}"] = out

    # --- Method stability (at K=12) ---
    print(f"\n{'='*60}")
    print("  METHOD COMPARISON (K=12)")
    print(f"{'='*60}")

    methods = [
        ("kmeans_default", {"method": "spatial_kmeans", "n_niches": 12, "spatial_weight": 0.3}),
        ("kmeans_expr90",  {"method": "spatial_kmeans", "n_niches": 12, "spatial_weight": 0.1}),
        ("kmeans_space50", {"method": "spatial_kmeans", "n_niches": 12, "spatial_weight": 0.5}),
    ]

    stability = compute_method_stability(expression, coords, methods)
    for key, val in stability.items():
        if isinstance(val, dict):
            print(f"  {key}: n_niches={val['n_niches']}, mean_size={val['mean_size']:.1f}")
        else:
            print(f"  {key}: {val:.4f}")

    all_results["method_stability"] = stability

    return all_results


def print_summary_table(all_results: Dict, true_labels_available: bool):
    """Print a compact table across K values."""
    print(f"\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"{'='*80}")

    header = f"  {'K':>4}  {'Sil(joint)':>10}  {'Sil(expr)':>9}  {'SpatCoh':>7}  "
    header += f"{'WithinC':>7}  {'CH':>9}  {'MeanDiff':>8}  {'BndryEn':>8}"
    if true_labels_available:
        header += f"  {'ARIgt':>6}"
    print(header)
    sep_len = 80
    print(f"  {'-' * (sep_len - 2)}")

    for k_key, v in sorted(all_results.items(), key=lambda x: int(x[0].split("=")[1]) if isinstance(x[0], str) and "=" in x[0] else 0):
        if "method_stability" in k_key:
            continue
        K = v.get("K", k_key)
        row = f"  {K:>4}  {v.get('silhouette_score', float('nan')):>10.4f}  "
        row += f"{v.get('silhouette_expression', float('nan')):>9.4f}  "
        row += f"{v.get('spatial_coherence_mean', float('nan')):>7.4f}  "
        row += f"{v.get('within_niche_corr', float('nan')):>7.4f}  "
        row += f"{v.get('calinski_harabasz', float('nan')):>9.1f}  "
        row += f"{v.get('mean_difficulty', float('nan')):>8.4f}  "
        row += f"{v.get('boundary_energy_mean', float('nan')):>8.4f}  "
        if true_labels_available:
            row += f"{v.get('ari_ground_truth', float('nan')):>6.3f}"
        print(row)

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Niche building quality analysis")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (uses synthetic data if omitted)")
    parser.add_argument("--K", type=int, nargs="+", default=[5, 8, 12, 16, 20, 30],
                        help="Niche counts to evaluate")
    parser.add_argument("--n_spots", type=int, default=1500,
                        help="Number of spots for synthetic data")
    parser.add_argument("--n_genes", type=int, default=100,
                        help="Number of genes for synthetic data")
    args = parser.parse_args()

    if args.config:
        print(f"Loading data from config: {args.config}")
        print("[WARN] Real data loading requires scanpy/anndata — falling back to synthetic")
        expression, coords, true_labels = generate_synthetic_st_data(
            n_spots=args.n_spots,
            n_genes=args.n_genes,
        )
        name = Path(args.config).stem
    else:
        expression, coords, true_labels = generate_synthetic_st_data(
            n_spots=args.n_spots,
            n_genes=args.n_genes,
        )
        name = "synthetic"

    results = run_full_analysis(
        expression, coords,
        true_labels=true_labels,
        K_values=args.K,
        name=name,
    )

    print_summary_table(results, true_labels is not None)

    print(f"\n  Niche quality analysis complete.\n")


if __name__ == "__main__":
    main()

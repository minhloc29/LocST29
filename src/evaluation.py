from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from scipy.stats import pearsonr, spearmanr

from .difficulty_gse import SpatialDynamicsField
from .curriculum import TrainingLog
from .utils import move_to_device, flatten_indices
from .utils import prepare_morans_adata, morans_i_scanpy_from_adata


def pearson_correlation_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    mode: str = "spot",
) -> float:
    """
    PCC averaged over spots (mode='spot', default) or genes (mode='gene').
 
    Spot-wise  — for each spot, correlate its predicted expression vector
                 with the true vector across genes, then average over spots.
                 This is the ST benchmarking standard (Hist2ST, iStar, etc.).
 
    Gene-wise  — for each gene, correlate predicted vs true values across
                 spots, then average over genes.  Kept for reference only.
 
    1-D inputs fall back to a single Pearson r.
    """
    if pred.ndim == 1:
        r, _ = pearsonr(pred, target)
        return float(r)
 
    if mode == "spot":
        # iterate over spots (axis 0), correlate across genes (axis 1)
        pccs = []
        for i in range(pred.shape[0]):
            p, t = pred[i], target[i]
            if t.std() < 1e-8 or p.std() < 1e-8:
                continue
            r, _ = pearsonr(p, t)
            pccs.append(r)
    else:
        # gene-wise: iterate over genes (axis 1), correlate across spots
        pccs = []
        for g in range(pred.shape[1]):
            p, t = pred[:, g], target[:, g]
            if t.std() < 1e-8 or p.std() < 1e-8:
                continue
            r, _ = pearsonr(p, t)
            pccs.append(r)
 
    return float(np.nanmean(pccs)) if pccs else float("nan")


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def calibration_error(
    pred: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
) -> float:
    
    flat_pred = pred.ravel()
    flat_target = target.ravel()
    bins = np.percentile(flat_pred, np.linspace(0, 100, n_bins + 1))
    errors = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (flat_pred >= lo) & (flat_pred < hi)
        if mask.sum() > 0:
            errors.append(np.abs(flat_pred[mask] - flat_target[mask]).mean())
    return float(np.mean(errors)) if errors else float("nan")


def compute_error_vector(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute per-spot MSE across genes.
    Accepts AnnData-like objects with .X or raw numpy arrays.
    """
    pred_X = pred.X if hasattr(pred, "X") else pred
    target_X = target.X if hasattr(target, "X") else target
    pred_X = np.asarray(pred_X)
    target_X = np.asarray(target_X)
    return np.mean((pred_X - target_X) ** 2, axis=1)

def build_difficulty_field(
    expression: np.ndarray,
    coords: np.ndarray,
    # --- Niche construction ---
    method: str = "spatial_leiden",
    resolution: float = 1.0,
    n_niches: Optional[int] = None,
    spatial_weight: float = 0.3,
    n_neighbors: int = 10,
    # --- Niche difficulty weights ---
    alpha: float = 0.40,   # heterogeneity
    beta: float = 0.30,    # topology / boundary energy
    gamma: float = 0.15,   # ambiguity
    delta: float = 0.15,   # uncertainty (ignored here — no training dynamics)
    # --- GSE blend ---
    use_gse: bool = True,
    gse_alpha: float = 0.5,
    k_neighbours: int = 6,
    random_state: int = 42,
) -> SpatialDynamicsField:
    """
    Build a DifficultyField from raw expression and coordinates.

    Pipeline
    --------
    1. Build spatial niches (Leiden / Louvain / K-means on joint features)
    2. Compute per-niche difficulty (heterogeneity + topology + ambiguity)
    3. Propagate niche scores to spots
    4. Optionally blend with Graph Signal Energy (GSE) for boundary sharpness
    5. Return DifficultyField with difficulty_score (N,) and coords (N, 2)

    Parameters
    ----------
    expression    : (N, G) log-normalized gene expression
    coords        : (N, 2) pixel / spatial coordinates
    method        : niche construction method
    resolution    : Leiden resolution (larger → more niches)
    n_niches      : fixed K for K-means; ignored for Leiden/Louvain
    spatial_weight: weight of coords vs expression in joint clustering space
    n_neighbors   : k-NN neighbors for graph construction in Leiden
    alpha         : weight for niche heterogeneity
    beta          : weight for niche topology (Laplacian boundary energy)
    gamma         : weight for niche ambiguity
    delta         : weight for training-dynamics uncertainty (set to 0 here)
    use_gse       : if True, blend spot_scores with graph signal energy
    gse_alpha     : blend weight (gse_alpha * niche_score + (1-gse_alpha) * boundary_energy)
    k_neighbours  : spatial graph k for GSE boundary computation
    random_state  : random seed for clustering

    Returns
    -------
    DifficultyField with .difficulty_score (N,), .coords (N, 2), .N
    """
    expression = np.nan_to_num(np.asarray(expression, dtype=np.float32), nan=0.0)
    coords = np.asarray(coords, dtype=np.float64)

    print(f"[DifficultyField] Building niches: N={len(expression)}, G={expression.shape[1]}, "
          f"method={method}, resolution={resolution}")

    # Step 1 + 2 + 3: niches → per-niche scores → spot scores
    niche_labels, niche_scores, spot_scores = niche_difficulty_from_data(
        expression=expression,
        coords=coords,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=0.0,          # no training dynamics available here
        method=method,
        resolution=resolution,
        n_niches=n_niches,
        spatial_weight=spatial_weight,
    )

    print(f"[DifficultyField] Niches built: K={len(niche_scores)} | "
          f"spot_scores: min={spot_scores.min():.4f} max={spot_scores.max():.4f} "
          f"mean={spot_scores.mean():.4f}")

    # Step 4: optionally blend with GSE for sharper boundary signal
    if use_gse:
        from .difficulty_gse import graph_signal_energy_difficulty
        gse_scores = graph_signal_energy_difficulty(
            base_score=spot_scores,
            coords=coords,
            k=k_neighbours,
            alpha=gse_alpha,
            weight="binary",
        )
        final_scores = gse_scores
        print(f"[DifficultyField] GSE blend applied: alpha={gse_alpha} | "
              f"final: min={final_scores.min():.4f} max={final_scores.max():.4f}")
    else:
        final_scores = spot_scores

    return SpatialDynamicsField(
        difficulty_score=final_scores.astype(np.float32),
        coords=coords.astype(np.float32),
    )

def convergence_speed(log: TrainingLog, threshold: float = 0.05) -> int:
    """
    Epoch at which validation loss first drops within threshold of its minimum.
    Returns total_epochs if never reached.
    """
    vals = np.array(log.val_loss)
    target = vals.min() + threshold * (vals.max() - vals.min())
    idx = np.where(vals <= target)[0]
    return int(idx[0]) if len(idx) > 0 else len(vals)


def per_region_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    region_name: str = "Hard",
) -> Dict[str, float]:
    """Compute MSE, MAE, PCC on a subset of spots defined by region_mask."""
    p = pred[region_mask]
    t = target[region_mask]
    if len(p) == 0:
        return {}
    return {
        f"{region_name}_MSE": mse(p, t),
        f"{region_name}_MAE": mae(p, t),
        f"{region_name}_PCC": pearson_correlation_coefficient(p, t),
        f"{region_name}_n_spots": int(region_mask.sum()),
    }


def boundary_hard_evaluation(
    pred_curriculum: np.ndarray,
    pred_baseline: np.ndarray,
    target: np.ndarray,
    field: SpatialDynamicsField,
    percentile: float = 75.0,
) -> Dict[str, float]:
    """Compare curriculum and baseline on the hardest *percentile* spots.

    Uses exact rank-based selection (takes top ``(100 - percentile)%`` of spots
    by difficulty_score) to avoid tie issues at the percentile boundary.
    """
    ds = field.difficulty_score
    N = len(ds)

    # Rank-based: select top (100 - percentile)% hardest spots
    n_hard = max(1, int(N * (100.0 - percentile) / 100.0))
    sorted_idx = np.argsort(ds)
    hard_local_idx = sorted_idx[-n_hard:]
    hard_mask = np.zeros(N, dtype=bool)
    hard_mask[hard_local_idx] = True

    print(f"  [DEBUG boundary_hard] difficulty_score: min={ds.min():.4f} max={ds.max():.4f} "
          f"mean={ds.mean():.4f} std={ds.std():.6f} unique={len(np.unique(ds))}")
    print(f"  [DEBUG boundary_hard] selecting top {n_hard}/{N} = {n_hard/N:.3f} hardest "
          f"(scores >= {ds[hard_local_idx[0]]:.4f})")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{q:3d}: {np.percentile(ds, q):.6f}")

    curr_metrics = per_region_metrics(pred_curriculum, target, hard_mask, "Curriculum_Hard")
    base_metrics = per_region_metrics(pred_baseline, target, hard_mask, "Baseline_Hard")

    gain = {}
    if curr_metrics and base_metrics:
        gain["MSE_gain"] = base_metrics["Baseline_Hard_MSE"] - curr_metrics["Curriculum_Hard_MSE"]
        gain["MAE_gain"] = base_metrics["Baseline_Hard_MAE"] - curr_metrics["Curriculum_Hard_MAE"]
        gain["PCC_gain"] = curr_metrics["Curriculum_Hard_PCC"] - base_metrics["Baseline_Hard_PCC"]

    return {**curr_metrics, **base_metrics, **gain, "hard_region_fraction": float(hard_mask.mean())}


@dataclass
class BiologicalAnnotations:
    """
    Optional per-spot biological masks (True = spot belongs to structure).
    """
    tumor_invasion: Optional[np.ndarray] = None
    immune_interface: Optional[np.ndarray] = None
    necrosis: Optional[np.ndarray] = None
    transition_state: Optional[np.ndarray] = None

    def available(self) -> Dict[str, np.ndarray]:
        out = {}
        for name in ("tumor_invasion", "immune_interface", "necrosis", "transition_state"):
            val = getattr(self, name)
            if val is not None:
                out[name] = val
        return out


def biological_overlap(
    field: SpatialDynamicsField,
    bio: BiologicalAnnotations,
    percentile: float = 75.0,
) -> Dict[str, float]:
    """
    For each biological structure, compute Jaccard overlap with the top
    percentile hardest spots.
    """
    hard_mask = field.difficulty_score > np.percentile(field.difficulty_score, percentile)
    results = {}
    for name, bio_mask in bio.available().items():
        intersection = (hard_mask & bio_mask).sum()
        union = (hard_mask | bio_mask).sum()
        jaccard = intersection / (union + 1e-10)
        results[f"jaccard_{name}"] = float(jaccard)
        results[f"overlap_frac_{name}"] = float(intersection / (bio_mask.sum() + 1e-10))
    return results


# ---------------------------------------------------------------------------
# Paper Metrics — dataclasses and functions for RQ1–RQ4
# ---------------------------------------------------------------------------

@dataclass
class DifficultyBinReport:
    """Per-bin metrics for both curriculum and baseline models."""
    bin_label: str
    n_spots: int
    curriculum_mse: float
    baseline_mse: float
    curriculum_pcc: float
    baseline_pcc: float
    mse_gain: float        # baseline - curriculum  (positive = curriculum better)
    pcc_gain: float        # curriculum - baseline


@dataclass
class PaperMetricsReport:
    """
    Paper-ready metrics organised by research question (RQ1–RQ4).
    All fields are plain Python types for easy JSON serialization.
    """
    # RQ1: Can training dynamics reveal spatial difficulty?
    difficulty_error_pearson: float = 0.0
    difficulty_error_spearman: float = 0.0
    difficulty_bins: List[DifficultyBinReport] = field(default_factory=list)

    # RQ2: What biological regions are difficult?
    interface_curriculum_mse: float = 0.0
    interface_baseline_mse: float = 0.0
    interface_curriculum_pcc: float = 0.0
    interface_baseline_pcc: float = 0.0
    n_interface_spots: int = 0
    cluster_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # RQ3: Does curriculum improve learning?
    aulc_curriculum: float = 0.0
    aulc_baseline: float = 0.0
    time_to_target_curriculum: int = -1
    time_to_target_baseline: int = -1

    # RQ4: Does curriculum help difficult regions?
    hard_bin_improvement_mse: float = 0.0
    hard_bin_improvement_pcc: float = 0.0
    overall_improvement_mse: float = 0.0
    overall_improvement_pcc: float = 0.0

    def print_rq1(self) -> None:
        print("\n" + "=" * 60)
        print("RQ1: Can Training Dynamics Reveal Spatial Difficulty?")
        print("=" * 60)
        print(f"  Difficulty-Error Correlation:")
        print(f"    Pearson r  = {self.difficulty_error_pearson:.4f}")
        print(f"    Spearman ρ = {self.difficulty_error_spearman:.4f}")
        print(f"\n  Difficulty Bin Performance (Curriculum):")
        print(f"  {'Bin':<20} {'N':>6} {'MSE':>10} {'PCC':>8}")
        print(f"  {'-' * 46}")
        for bin_rep in self.difficulty_bins:
            print(f"  {bin_rep.bin_label:<20} {bin_rep.n_spots:>6} "
                  f"{bin_rep.curriculum_mse:>10.4f} {bin_rep.curriculum_pcc:>8.4f}")

    def print_rq2(self) -> None:
        print("\n" + "=" * 60)
        print("RQ2: What Biological Regions Are Difficult?")
        print("=" * 60)
        if self.n_interface_spots > 0:
            print(f"  Interface Zone (n={self.n_interface_spots} spots):")
            print(f"    {'Metric':<20} {'Curriculum':>12} {'Baseline':>12} {'Delta':>10}")
            print(f"    {'-' * 56}")
            delta_mse = self.interface_baseline_mse - self.interface_curriculum_mse
            delta_pcc = self.interface_curriculum_pcc - self.interface_baseline_pcc
            print(f"    {'MSE':<20} {self.interface_curriculum_mse:>12.4f} "
                  f"{self.interface_baseline_mse:>12.4f} {delta_mse:>+10.4f}")
            print(f"    {'PCC':<20} {self.interface_curriculum_pcc:>12.4f} "
                  f"{self.interface_baseline_pcc:>12.4f} {delta_pcc:>+10.4f}")
        else:
            print("  No interface-zone spots identified.")
        if self.cluster_metrics:
            print(f"\n  Persistent Hard Clusters:")
            for cid, cm in sorted(self.cluster_metrics.items()):
                print(f"    Cluster {cid}: n={cm.get('n_spots', '?')} "
                      f"MSE(curr={cm.get('curriculum_mse', 0):.4f}, "
                      f"base={cm.get('baseline_mse', 0):.4f}) "
                      f"PCC(curr={cm.get('curriculum_pcc', 0):.4f}, "
                      f"base={cm.get('baseline_pcc', 0):.4f})")

    def print_rq3(self) -> None:
        print("\n" + "=" * 60)
        print("RQ3: Does Curriculum Improve Learning?")
        print("=" * 60)
        print(f"  {'Metric':<30} {'Curriculum':>12} {'Baseline':>12} {'Delta':>10}")
        print(f"  {'-' * 66}")
        delta_aulc = self.aulc_curriculum - self.aulc_baseline
        print(f"  {'AULC (val_loss)':<30} {self.aulc_curriculum:>12.4f} "
              f"{self.aulc_baseline:>12.4f} {delta_aulc:>+10.4f}")
        ttc = f"Epoch {self.time_to_target_curriculum}" if self.time_to_target_curriculum >= 0 else "N/A"
        ttb = f"Epoch {self.time_to_target_baseline}" if self.time_to_target_baseline >= 0 else "N/A"
        print(f"  {'Time-to-target (PCC>=thresh)':<30} {ttc:>12} {ttb:>12}")

    def print_rq4(self) -> None:
        print("\n" + "=" * 60)
        print("RQ4: Does Curriculum Help Difficult Regions?")
        print("=" * 60)
        print(f"  {'Bin':<20} {'N':>6} {'Curr MSE':>10} {'Base MSE':>10} "
              f"{'Δ MSE':>10} {'Δ PCC':>8}")
        print(f"  {'-' * 66}")
        for bin_rep in self.difficulty_bins:
            print(f"  {bin_rep.bin_label:<20} {bin_rep.n_spots:>6} "
                  f"{bin_rep.curriculum_mse:>10.4f} {bin_rep.baseline_mse:>10.4f} "
                  f"{bin_rep.mse_gain:>+10.4f} {bin_rep.pcc_gain:>+8.4f}")
        print(f"\n  Summary:")
        print(f"    Hardest bin MSE improvement: {self.hard_bin_improvement_mse:+.4f}")
        print(f"    Hardest bin PCC improvement: {self.hard_bin_improvement_pcc:+.4f}")
        print(f"    Overall MSE improvement:     {self.overall_improvement_mse:+.4f}")
        print(f"    Overall PCC improvement:     {self.overall_improvement_pcc:+.4f}")

    def print_all(self) -> None:
        self.print_rq1()
        self.print_rq2()
        self.print_rq3()
        self.print_rq4()


# ---------------------------------------------------------------------------
# Paper Metric Computation Functions
# ---------------------------------------------------------------------------

def difficulty_error_correlation(
    difficulty_score: np.ndarray,
    per_spot_error: np.ndarray,
) -> Tuple[float, float]:
    """
    Pearson and Spearman correlation between difficulty_score and per-spot error.

    Parameters
    ----------
    difficulty_score : (N,)  composite difficulty in [0, 1]
    per_spot_error   : (N,)  per-spot MSE across genes

    Returns
    -------
    pearson_r  : float
    spearman_rho : float
    """
    print(f"  [DEBUG corr] difficulty_score: min={difficulty_score.min():.4f} max={difficulty_score.max():.4f} mean={difficulty_score.mean():.4f} std={difficulty_score.std():.6f}")
    print(f"  [DEBUG corr] per_spot_error: min={per_spot_error.min():.4f} max={per_spot_error.max():.4f} mean={per_spot_error.mean():.4f} std={per_spot_error.std():.6f}")
    # Scatter: group by percentile
    for q_low, q_high, label in [(0, 25, "bottom25"), (25, 50, "mid_low"), (50, 75, "mid_high"), (75, 100, "top25")]:
        lo = np.percentile(difficulty_score, q_low)
        hi = np.percentile(difficulty_score, q_high)
        mask = (difficulty_score >= lo) & (difficulty_score < hi) if q_high < 100 else (difficulty_score >= lo) & (difficulty_score <= hi)
        if mask.sum() > 0:
            print(f"    {label}: n={mask.sum():>4}  mean_err={per_spot_error[mask].mean():.4f}  mean_ds={difficulty_score[mask].mean():.4f}")
    r_pearson, _ = pearsonr(difficulty_score, per_spot_error)
    r_spearman, _ = spearmanr(difficulty_score, per_spot_error)
    print(f"  [DEBUG corr] pearson={r_pearson:.4f}  spearman={r_spearman:.4f}")
    return float(r_pearson), float(r_spearman)


def compute_difficulty_bins(
    difficulty_score: np.ndarray,
    pred_curriculum: np.ndarray,
    pred_baseline: np.ndarray,
    target: np.ndarray,
    n_bins: int = 5,
) -> List[DifficultyBinReport]:
    """
    Partition spots into n_bins quantile groups by difficulty (ascending).

    Returns a DifficultyBinReport for each bin, where bin 0 = easiest, bin N = hardest.
    """
    N = len(difficulty_score)
    sorted_idx = np.argsort(difficulty_score)  # easiest first
    bin_size = N // n_bins
    default_labels = ["Easiest 20%", "20-40%", "40-60%", "60-80%", "Hardest 20%"]
    reports = []
    for i in range(n_bins):
        start = i * bin_size
        end = N if i == n_bins - 1 else start + bin_size
        bin_idx = sorted_idx[start:end]
        c_pred = pred_curriculum[bin_idx]
        b_pred = pred_baseline[bin_idx]
        tgt = target[bin_idx]
        cm = mse(c_pred, tgt)
        bm = mse(b_pred, tgt)
        cp = pearson_correlation_coefficient(c_pred, tgt)
        bp = pearson_correlation_coefficient(b_pred, tgt)
        reports.append(DifficultyBinReport(
            bin_label=default_labels[i] if i < len(default_labels) else f"Bin {i+1}",
            n_spots=len(bin_idx),
            curriculum_mse=cm,
            baseline_mse=bm,
            curriculum_pcc=cp,
            baseline_pcc=bp,
            mse_gain=bm - cm,
            pcc_gain=cp - bp,
        ))
    return reports


def compute_aulc(val_losses: List[float]) -> float:
    """Area Under the Loss Curve via trapezoidal integration."""
    losses = np.asarray(val_losses)
    if len(losses) < 2:
        return float("nan")
    return float(np.trapz(losses))


def time_to_target_pcc(
    epoch_pccs: List[float],
    threshold: float = 0.45,
) -> int:
    """First epoch index where PCC >= threshold. Returns -1 if never reached."""
    for i, pcc_val in enumerate(epoch_pccs):
        if pcc_val >= threshold:
            return i
    return -1


def evaluate_interface_zone(
    pred_curriculum: np.ndarray,
    pred_baseline: np.ndarray,
    target: np.ndarray,
    interface_mask: np.ndarray,
) -> Dict[str, float]:
    """Evaluate both models on interface-zone spots."""
    if interface_mask.sum() == 0:
        return {"interface_n_spots": 0}
    c_pred = pred_curriculum[interface_mask]
    b_pred = pred_baseline[interface_mask]
    tgt = target[interface_mask]
    return {
        "interface_n_spots": int(interface_mask.sum()),
        "interface_curriculum_mse": mse(c_pred, tgt),
        "interface_baseline_mse": mse(b_pred, tgt),
        "interface_curriculum_pcc": pearson_correlation_coefficient(c_pred, tgt),
        "interface_baseline_pcc": pearson_correlation_coefficient(b_pred, tgt),
    }


def evaluate_clusters(
    pred_curriculum: np.ndarray,
    pred_baseline: np.ndarray,
    target: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Per-DBSCAN-cluster MSE/PCC for both models."""
    unique = set(cluster_labels[cluster_labels >= 0])
    results = {}
    for cid in sorted(unique):
        mask = cluster_labels == cid
        c_pred = pred_curriculum[mask]
        b_pred = pred_baseline[mask]
        tgt = target[mask]
        results[str(cid)] = {
            "n_spots": int(mask.sum()),
            "curriculum_mse": mse(c_pred, tgt),
            "baseline_mse": mse(b_pred, tgt),
            "curriculum_pcc": pearson_correlation_coefficient(c_pred, tgt),
            "baseline_pcc": pearson_correlation_coefficient(b_pred, tgt),
        }
    return results


def compute_paper_metrics(
    pred_curriculum: np.ndarray,
    pred_baseline: np.ndarray,
    target: np.ndarray,
    field: SpatialDynamicsField,
    topology: Optional[object] = None,
    training_log_curriculum: Optional[TrainingLog] = None,
    epoch_pccs_curriculum: Optional[List[float]] = None,
    training_log_baseline: Optional[TrainingLog] = None,
    epoch_pccs_baseline: Optional[List[float]] = None,
    n_bins: int = 5,
    pcc_threshold: float = 0.45,
) -> PaperMetricsReport:
    """
    Compute all paper metrics (RQ1–RQ4) from predictions and dynamics data.
    """
    # --- RQ1 ---
    per_spot_error = compute_error_vector(pred_curriculum, target)
    r_pearson, r_spearman = difficulty_error_correlation(
        field.difficulty_score, per_spot_error
    )
    bins = compute_difficulty_bins(
        field.difficulty_score, pred_curriculum, pred_baseline, target, n_bins=n_bins
    )

    report = PaperMetricsReport(
        difficulty_error_pearson=r_pearson,
        difficulty_error_spearman=r_spearman,
        difficulty_bins=bins,
    )

    # --- RQ2 ---
    if topology is not None:
        im = evaluate_interface_zone(
            pred_curriculum, pred_baseline, target, topology.interface_mask
        )
        report.interface_curriculum_mse = im.get("interface_curriculum_mse", 0.0)
        report.interface_baseline_mse = im.get("interface_baseline_mse", 0.0)
        report.interface_curriculum_pcc = im.get("interface_curriculum_pcc", 0.0)
        report.interface_baseline_pcc = im.get("interface_baseline_pcc", 0.0)
        report.n_interface_spots = im.get("interface_n_spots", 0)
        report.cluster_metrics = evaluate_clusters(
            pred_curriculum, pred_baseline, target, topology.persistent_clusters
        )

    # --- RQ3 ---
    if training_log_curriculum is not None and len(training_log_curriculum.val_loss) > 0:
        report.aulc_curriculum = compute_aulc(training_log_curriculum.val_loss)
    if training_log_baseline is not None and len(training_log_baseline.val_loss) > 0:
        report.aulc_baseline = compute_aulc(training_log_baseline.val_loss)
    if epoch_pccs_curriculum is not None:
        report.time_to_target_curriculum = time_to_target_pcc(
            epoch_pccs_curriculum, threshold=pcc_threshold
        )
    if epoch_pccs_baseline is not None:
        report.time_to_target_baseline = time_to_target_pcc(
            epoch_pccs_baseline, threshold=pcc_threshold
        )

    # --- RQ4 ---
    if bins:
        hardest = bins[-1]
        report.hard_bin_improvement_mse = hardest.mse_gain
        report.hard_bin_improvement_pcc = hardest.pcc_gain
    report.overall_improvement_mse = mse(pred_baseline, target) - mse(pred_curriculum, target)
    report.overall_improvement_pcc = (
        pearson_correlation_coefficient(pred_curriculum, target)
        - pearson_correlation_coefficient(pred_baseline, target)
    )

    return report


@dataclass
class EvaluationReport:
    """Full Phase 5 evaluation report."""
    curriculum_PCC: float
    curriculum_MSE: float
    curriculum_MAE: float
    baseline_PCC: float
    baseline_MSE: float
    baseline_MAE: float
    calibration_error: float
    convergence_epoch: int
    hard_region_metrics: Dict[str, float] = field(default_factory=dict)
    biological_overlaps: Dict[str, float] = field(default_factory=dict)
    spatial_error_autocorr: Optional["SpatialAutocorrReport"] = None
    paper_metrics: Optional[PaperMetricsReport] = None
    # Niche-level evaluation (added in v2)
    niche_stratified: Dict[str, Dict[str, float]] = field(default_factory=dict)
    niche_boundary: Dict[str, float] = field(default_factory=dict)
    per_niche: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("PHASE 5 - EVALUATION REPORT")
        print("=" * 60)
        print("\n1. OVERALL PERFORMANCE")
        print(f"   {'Metric':<22} {'Curriculum':>12}  {'Baseline':>12}  {'Delta':>10}")
        print(f"   {'-' * 58}")
        for m, c_val, b_val in [
            ("PCC up", self.curriculum_PCC, self.baseline_PCC),
            ("MSE down", self.curriculum_MSE, self.baseline_MSE),
            ("MAE down", self.curriculum_MAE, self.baseline_MAE),
        ]:
            delta = c_val - b_val
            print(f"   {m:<22} {c_val:>12.4f}  {b_val:>12.4f}  {delta:>+10.4f}")
        print(f"   {'Calibration Err down':<22} {self.calibration_error:>12.4f}")
        print(f"   {'Convergence Epoch down':<22} {self.convergence_epoch:>12d}")

        if self.hard_region_metrics:
            print("\n2. BOUNDARY / HARD REGION PERFORMANCE")
            for k, v in self.hard_region_metrics.items():
                print(f"   {k:<35} {v:.4f}")

        if self.biological_overlaps:
            print("\n3. BIOLOGICAL MEANING (Jaccard / Overlap)")
            for k, v in self.biological_overlaps.items():
                print(f"   {k:<40} {v:.4f}")
        if self.spatial_error_autocorr is not None:
            r = self.spatial_error_autocorr
            print("\n4. SPATIAL AUTOCORRELATION (Moran's I)")
            print(f"   Moran's I: {r.morans_I:.4f} (p={r.morans_p:.4g})")
            print(
                f"   Random mean: {r.random_mean:.4f} ± {r.random_std:.4f} "
                f"[min {r.random_min:.4f}, max {r.random_max:.4f}]"
            )
            print(f"   Percentile vs random: {r.random_percentile:.1f}th")

        # --- Niche evaluation (v2) ---
        if self.niche_stratified:
            print("\n5. NICHE-STRATIFIED EVALUATION")
            print(f"   {'Stratum':<20} {'N':>6} {'Curr MSE':>10} {'Base MSE':>10} "
                  f"{'Δ MSE':>10} {'Curr PCC':>8} {'Base PCC':>8} {'Δ PCC':>8}")
            print(f"   {'-' * 82}")
            for label in ["easy_niches", "medium_niches", "hard_niches"]:
                if label in self.niche_stratified:
                    d = self.niche_stratified[label]
                    print(f"   {label:<20} {d['n_spots']:>6} "
                          f"{d['curriculum_mse']:>10.4f} {d['baseline_mse']:>10.4f} "
                          f"{d['mse_gain']:>+10.4f} "
                          f"{d['curriculum_pcc']:>8.4f} {d['baseline_pcc']:>8.4f} "
                          f"{d['pcc_gain']:>+8.4f}")

        if self.niche_boundary and self.niche_boundary.get("boundary_n_spots", 0) > 0:
            print(f"\n6. NICHE BOUNDARY EVALUATION")
            print(f"   Boundary spots: {self.niche_boundary['boundary_n_spots']} "
                  f"({self.niche_boundary['boundary_fraction']:.1%})")
            print(f"   Curriculum MSE: {self.niche_boundary['boundary_curriculum_mse']:.4f} "
                  f"| Baseline MSE: {self.niche_boundary['boundary_baseline_mse']:.4f} "
                  f"| Gain: {self.niche_boundary['boundary_mse_gain']:+.4f}")
            print(f"   Curriculum PCC: {self.niche_boundary['boundary_curriculum_pcc']:.4f} "
                  f"| Baseline PCC: {self.niche_boundary['boundary_baseline_pcc']:.4f} "
                  f"| Gain: {self.niche_boundary['boundary_pcc_gain']:+.4f}")

        if self.paper_metrics is not None:
            print("\n7. PAPER METRICS (RQ1–RQ4)")
            self.paper_metrics.print_all()
        print("=" * 60 + "\n")


@dataclass
class SpatialAutocorrReport:
    morans_I: float
    morans_p: float
    random_mean: float
    random_std: float
    random_min: float
    random_max: float
    random_percentile: float


def spatial_error_autocorrelation(
    pred: np.ndarray,
    target: np.ndarray,
    coords: np.ndarray,
    k_neighbours: int = 8,
    n_perm: int = 999,
    n_shuffles: int = 100,
) -> Tuple[SpatialAutocorrReport, np.ndarray]:
    """
    Compute Moran's I on per-spot errors and compare to shuffled errors.
    Returns a report and the shuffled Moran's I values.
    """
    errors = compute_error_vector(pred, target)
    
    adata = prepare_morans_adata(
        n_obs=len(errors),
        coords=coords,
        k_neighbours=k_neighbours,
    )
    
    

    morans_i_stat, pval = morans_i_scanpy_from_adata(adata, errors, n_perms=n_perm)

    random_morans = []
    for _ in range(n_shuffles):
        error_shuffled = np.random.permutation(errors)
        i_val, _ = morans_i_scanpy_from_adata(adata, error_shuffled, n_perms=None)
        random_morans.append(i_val)

    random_morans = np.asarray(random_morans, dtype=np.float32)
    random_mean = float(np.mean(random_morans))
    random_std = float(np.std(random_morans))
    random_min = float(np.min(random_morans))
    random_max = float(np.max(random_morans))
    random_percentile = float(
        np.sum(random_morans >= morans_i_stat) / len(random_morans) * 100
    )

    report = SpatialAutocorrReport(
        morans_I=float(morans_i_stat),
        morans_p=float(pval),
        random_mean=random_mean,
        random_std=random_std,
        random_min=random_min,
        random_max=random_max,
        random_percentile=random_percentile,
    )
    return report, random_morans


# ---------------------------------------------------------------------------
# Niche-level evaluation
# ---------------------------------------------------------------------------

def per_niche_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    niche_labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute MSE and PCC per niche.

    Returns
    -------
    results : {niche_label_str: {"n_spots": ..., "mse": ..., "pcc": ...}}
    """
    K = int(niche_labels.max()) + 1
    results: Dict[str, Dict[str, float]] = {}
    for k in range(K):
        mask = niche_labels == k
        if mask.sum() == 0:
            continue
        p, t = pred[mask], target[mask]
        results[f"niche_{k}"] = {
            "n_spots": int(mask.sum()),
            "mse": mse(p, t),
            "pcc": pearson_correlation_coefficient(p, t),
        }
    return results


def niche_stratified_evaluation(
    pred_curriculum: np.ndarray,
    pred_baseline: np.ndarray,
    target: np.ndarray,
    niche_labels: np.ndarray,
    niche_scores: np.ndarray,
    n_bins: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Group niches by difficulty (easy/medium/hard) and compare models.

    Key question: does curriculum learning improve *hard niches* more
    than baseline?
    """
    K = len(niche_scores)
    sorted_niches = np.argsort(niche_scores)
    bin_size = K // n_bins
    labels = ["easy_niches", "medium_niches", "hard_niches"]

    results: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(labels):
        start = i * bin_size
        end = K if i == n_bins - 1 else start + bin_size
        bin_niches = set(sorted_niches[start:end].tolist())
        mask = np.array([int(lbl) in bin_niches for lbl in niche_labels])

        if mask.sum() == 0:
            continue
        c_pred = pred_curriculum[mask]
        b_pred = pred_baseline[mask]
        tgt = target[mask]

        c_mse = mse(c_pred, tgt)
        b_mse = mse(b_pred, tgt)
        c_pcc = pearson_correlation_coefficient(c_pred, tgt)
        b_pcc = pearson_correlation_coefficient(b_pred, tgt)

        results[label] = {
            "n_spots": int(mask.sum()),
            "n_niches": int(mask.sum()),
            "curriculum_mse": c_mse,
            "baseline_mse": b_mse,
            "mse_gain": b_mse - c_mse,
            "curriculum_pcc": c_pcc,
            "baseline_pcc": b_pcc,
            "pcc_gain": c_pcc - b_pcc,
        }

    return results


def niche_boundary_evaluation(
    pred_curriculum: np.ndarray,
    pred_baseline: np.ndarray,
    target: np.ndarray,
    niche_labels: np.ndarray,
    coords: np.ndarray,
    spatial_k: int = 6,
) -> Dict[str, float]:
    """Evaluate spots that lie on *niche-niche boundaries*.

    A boundary spot is one whose neighbours belong to a different niche.
    These regions are biologically meaningful (tissue interfaces) and
    are expected to be the hardest.
    """
    from .utils import build_spatial_graph

    edge_index, _ = build_spatial_graph(coords, k=spatial_k)
    N = len(niche_labels)

    # Mark spots whose neighbourhood contains a different niche
    boundary = np.zeros(N, dtype=bool)
    for src, dst in zip(edge_index[0], edge_index[1]):
        if niche_labels[src] != niche_labels[dst]:
            boundary[src] = True
            boundary[dst] = True

    if boundary.sum() == 0:
        return {"boundary_n_spots": 0}

    c_pred = pred_curriculum[boundary]
    b_pred = pred_baseline[boundary]
    tgt = target[boundary]

    return {
        "boundary_n_spots": int(boundary.sum()),
        "boundary_fraction": float(boundary.mean()),
        "boundary_curriculum_mse": mse(c_pred, tgt),
        "boundary_baseline_mse": mse(b_pred, tgt),
        "boundary_mse_gain": mse(b_pred, tgt) - mse(c_pred, tgt),
        "boundary_curriculum_pcc": pearson_correlation_coefficient(c_pred, tgt),
        "boundary_baseline_pcc": pearson_correlation_coefficient(b_pred, tgt),
        "boundary_pcc_gain": pearson_correlation_coefficient(c_pred, tgt)
        - pearson_correlation_coefficient(b_pred, tgt),
    }


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  
    model.eval()
    all_pred, all_target, all_idx = [], [], []
    for x, y, idx in loader:
        x = move_to_device(x, device)
        out = model(x)

        pred = out

        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)
            
        pred = pred.detach().cpu().numpy()
        target = y.detach().cpu().numpy()
        idx_flat = flatten_indices(idx).cpu().numpy()

    
        if pred.ndim >= 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        assert pred.shape == y.shape, (
                    f"pred={pred.shape}, target={y.shape}"
                )
        
        all_pred.append(pred)
        all_target.append(target)
        all_idx.append(idx_flat)
    return (
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_target, axis=0),
        np.concatenate(all_idx, axis=0),
    )


def reorder_by_spot(
    values: np.ndarray,
    spot_indices: np.ndarray,
    n_spots: int,
) -> np.ndarray:
    """
    Re-order loader output (may be shuffled) back to canonical spot order.
    If a spot appears multiple times, the last value is kept.
    """
    shape = (n_spots,) + values.shape[1:]
    out = np.zeros(shape, dtype=values.dtype)
    out[spot_indices] = values
    return out


def run_evaluation(
    curriculum_model: nn.Module,
    baseline_model: nn.Module,
    test_loader,
    field: SpatialDynamicsField,
    training_log: TrainingLog,
    device: torch.device,
    bio_annotations: Optional[BiologicalAnnotations] = None,
    hard_percentile: float = 75.0,
    spatial_autocorr: bool = False,
    k_neighbours: int = 8,
    n_perm: int = 999,
    n_shuffles: int = 100,
    topology: Optional[TopologyResult] = None,
    training_log_baseline: Optional[TrainingLog] = None,
    epoch_pccs_curriculum: Optional[List[float]] = None,
    epoch_pccs_baseline: Optional[List[float]] = None,
    pcc_threshold: float = 0.45,
) -> EvaluationReport:

    N = field.N
    print("[Phase 5] Running inference on curriculum model...")
    cp, ct, ci = predict_all(curriculum_model, test_loader, device)
    cp = reorder_by_spot(cp, ci, N)
    ct = reorder_by_spot(ct, ci, N)

    print("[Phase 5] Running inference on baseline model...")
    bp, bt, bi = predict_all(baseline_model, test_loader, device)
    bp = reorder_by_spot(bp, bi, N)
    bt = reorder_by_spot(bt, bi, N)

    print("[Phase 5] Computing overall metrics...")
    report = EvaluationReport(
        curriculum_PCC=pearson_correlation_coefficient(cp, ct),
        curriculum_MSE=mse(cp, ct),
        curriculum_MAE=mae(cp, ct),
        baseline_PCC=pearson_correlation_coefficient(bp, bt),
        baseline_MSE=mse(bp, bt),
        baseline_MAE=mae(bp, bt),
        calibration_error=calibration_error(cp, ct),
        convergence_epoch=convergence_speed(training_log),
    )

    print("[Phase 5] Evaluating hard / boundary regions...")
    report.hard_region_metrics = boundary_hard_evaluation(
        cp, bp, ct, field, percentile=hard_percentile
    )

    if bio_annotations is not None:
        print("[Phase 5] Computing biological overlap...")
        report.biological_overlaps = biological_overlap(
            field, bio_annotations, percentile=hard_percentile
        )

    if spatial_autocorr:
        print("[Phase 5] Computing spatial autocorrelation of errors...")
        report.spatial_error_autocorr, _ = spatial_error_autocorrelation(
            cp,
            ct,
            field.coords,
            k_neighbours=k_neighbours,
            n_perm=n_perm,
            n_shuffles=n_shuffles,
        )

    print("[Phase 5] Computing paper metrics (RQ1–RQ4)...")
    report.paper_metrics = compute_paper_metrics(
        pred_curriculum=cp,
        pred_baseline=bp,
        target=ct,
        field=field,
        topology=topology,
        training_log_curriculum=training_log,
        epoch_pccs_curriculum=epoch_pccs_curriculum,
        training_log_baseline=training_log_baseline,
        epoch_pccs_baseline=epoch_pccs_baseline,
        pcc_threshold=pcc_threshold,
    )

    report.print_summary()
    return report


if __name__ == "__main__":
    """CLI: evaluate trained curriculum & baseline models on test data.

    Usage
    -----
    python -m src.evaluation \\
        --checkpoint_dir ./outputs/run1 \\
        --config ./configs/my_config.yaml

    This loads the curriculum model and baseline model from the checkpoint
    directory, runs inference on the test set, computes all evaluation
    metrics (MSE, PCC, MAE, difficulty bins, paper RQ1-RQ4 metrics), and
    prints a summary.
    """
    import argparse
    import json
    from pathlib import Path
    from importlib import import_module

    from torch.utils.data import DataLoader

    from src import (
        DataConfig,
        SpatialModelAdapter,
        MultiSlideAdapter,
        build_slide_loader,
        load_dataset,
        prepare_phase1,
    )
    from config.my_config import load_config

    parser = argparse.ArgumentParser(description="Evaluate trained curriculum models")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing curriculum_model.pt and baseline_model.pt")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.training.device)
    ckpt_dir = Path(args.checkpoint_dir)

    # ── Build model ──
    def _build_model(module_path, class_name, kwargs=None):
        if kwargs is None:
            kwargs = {}
        elif hasattr(kwargs, "__dict__"):
            kwargs = vars(kwargs)
        cls = getattr(import_module(module_path), class_name)
        return cls(**kwargs)

    model = _build_model(cfg.model.module, cfg.model.class_name, cfg.model.kwargs)
    baseline_model = _build_model(cfg.model.module, cfg.model.class_name, cfg.model.kwargs)
    if cfg.pipeline.wrap_model:
        model = SpatialModelAdapter(model)
        baseline_model = SpatialModelAdapter(baseline_model)

    # ── Load checkpoints ──
    ckpt_path = ckpt_dir / "curriculum_model.pt"
    base_ckpt_path = ckpt_dir / "baseline_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"[Eval] Loaded curriculum model from {ckpt_path}")
    else:
        print(f"[Eval] WARNING: {ckpt_path} not found — using random weights")
    if base_ckpt_path.exists():
        baseline_model.load_state_dict(torch.load(base_ckpt_path, map_location="cpu"))
        print(f"[Eval] Loaded baseline model from {base_ckpt_path}")
    else:
        print(f"[Eval] WARNING: {base_ckpt_path} not found — using random weights")

    model.to(device)
    baseline_model.to(device)

    # ── Build dataset & loader ──
    data_root = Path(cfg.dataset.data_root).resolve() if cfg.dataset.data_root else None
    data_cfg = DataConfig(
        dataset=cfg.dataset.name, fold=cfg.dataset.fold,
        adj=True, flatten=cfg.dataset.flatten, data_root=data_root,
    )
    test_base = load_dataset(data_cfg, train=False)
    if len(test_base.names) == 0:
        test_base = load_dataset(data_cfg, train=True)
        print("[Eval] No test set — using training set for evaluation")
    test_loader = build_slide_loader(
        test_base, slide_index=cfg.dataset.test_slide_index,
        batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
    )

    # ── Get data for field construction ──
    slide_name = test_base.names[cfg.dataset.test_slide_index]
    expression = test_base.exp_dict[slide_name]
    coords = test_base.center_dict[slide_name].astype(float)

    field = build_difficulty_field(
        expression=np.nan_to_num(expression, nan=0.0),
        coords=coords,
        k_neighbours=cfg.difficulty.k_neighbours,
    )

    # ── Run evaluation ──
    report = run_evaluation(
        curriculum_model=model,
        baseline_model=baseline_model,
        test_loader=test_loader,
        field=field,
        training_log=TrainingLog(),  # empty log — metrics that need it (AULC) will show nan
        device=device,
    )

    # ── Save results (optional) ──
    if args.output_json:
        from dataclasses import asdict
        output = asdict(report)
        # Convert numpy arrays / floats to plain Python types
        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        output = _convert(output)
        (ckpt_dir / args.output_json).write_text(
            json.dumps(output, indent=2, default=str)
        )
        print(f"[Eval] Results saved to {ckpt_dir / args.output_json}")

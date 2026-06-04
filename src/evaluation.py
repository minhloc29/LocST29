from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from scipy.stats import pearsonr

from .dynamics.dynamics import SpatialDynamicsField
from .curriculum.curriculum import TrainingLog
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
    
    hard_mask = field.D_bar > np.percentile(field.D_bar, percentile)

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
    hard_mask = field.D_bar > np.percentile(field.D_bar, percentile)
    results = {}
    for name, bio_mask in bio.available().items():
        intersection = (hard_mask & bio_mask).sum()
        union = (hard_mask | bio_mask).sum()
        jaccard = intersection / (union + 1e-10)
        results[f"jaccard_{name}"] = float(jaccard)
        results[f"overlap_frac_{name}"] = float(intersection / (bio_mask.sum() + 1e-10))
    return results


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


def run_phase5(
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

    report.print_summary()
    return report

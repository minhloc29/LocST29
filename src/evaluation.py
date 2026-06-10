"""Evaluation: overall metrics + niche-stratified + niche-boundary comparison."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from scipy.stats import pearsonr

from .curriculum import TrainingLog
from .utils import move_to_device, flatten_indices


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SpatialDynamicsField:
    """Holds spot difficulty scores and coordinates."""
    difficulty_score: np.ndarray  # (N,)  higher = harder
    coords: np.ndarray           # (N, 2)

    @property
    def N(self) -> int:
        return len(self.difficulty_score)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def pearson_correlation_coefficient(
    pred: np.ndarray, target: np.ndarray, mode: str = "spot",
) -> float:
    """PCC averaged over spots (mode='spot') or genes (mode='gene')."""
    if pred.ndim == 1:
        r, _ = pearsonr(pred, target)
        return float(r)

    pccs = []
    if mode == "spot":
        for i in range(pred.shape[0]):
            p, t = pred[i], target[i]
            if t.std() < 1e-8 or p.std() < 1e-8:
                continue
            r, _ = pearsonr(p, t)
            pccs.append(r)
    else:
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


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Curriculum vs baseline — overall + niche-stratified + boundary."""
    curriculum_PCC: float
    curriculum_MSE: float
    curriculum_MAE: float
    baseline_PCC: float
    baseline_MSE: float
    baseline_MAE: float
    niche_stratified: Dict[str, Dict[str, float]] = field(default_factory=dict)
    niche_boundary: Dict[str, float] = field(default_factory=dict)
    per_niche: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        # 1. Overall
        print("\n1. OVERALL PERFORMANCE")
        print(f"   {'Metric':<22} {'Curriculum':>12}  {'Baseline':>12}  {'Delta':>10}")
        print(f"   {'-' * 58}")
        for m, c, b in [
            ("PCC up", self.curriculum_PCC, self.baseline_PCC),
            ("MSE down", self.curriculum_MSE, self.baseline_MSE),
            ("MAE down", self.curriculum_MAE, self.baseline_MAE),
        ]:
            print(f"   {m:<22} {c:>12.4f}  {b:>12.4f}  {c - b:>+10.4f}")

        # 2. Niche-stratified
        if self.niche_stratified:
            print("\n2. NICHE-STRATIFIED EVALUATION")
            print(f"   {'Stratum':<20} {'N':>6} {'Curr MSE':>10} {'Base MSE':>10} "
                  f"{'Δ MSE':>10} {'Curr PCC':>8} {'Base PCC':>8} {'Δ PCC':>8}")
            print(f"   {'-' * 82}")
            for label in ["easy_niches", "medium_niches", "hard_niches"]:
                d = self.niche_stratified.get(label)
                if d is None:
                    continue
                print(f"   {label:<20} {d['n_spots']:>6} "
                      f"{d['curriculum_mse']:>10.4f} {d['baseline_mse']:>10.4f} "
                      f"{d['mse_gain']:>+10.4f} "
                      f"{d['curriculum_pcc']:>8.4f} {d['baseline_pcc']:>8.4f} "
                      f"{d['pcc_gain']:>+8.4f}")

        # 3. Niche boundary
        if self.niche_boundary and self.niche_boundary.get("boundary_n_spots", 0) > 0:
            b = self.niche_boundary
            print(f"\n3. NICHE BOUNDARY EVALUATION")
            print(f"   Boundary spots: {b['boundary_n_spots']} "
                  f"({b['boundary_fraction']:.1%} of total)")
            print(f"   {'Metric':<20} {'Curriculum':>12} {'Baseline':>12} {'Delta':>10}")
            print(f"   {'-' * 56}")
            print(f"   {'MSE':<20} {b['boundary_curriculum_mse']:>12.4f} "
                  f"{b['boundary_baseline_mse']:>12.4f} {b['boundary_mse_gain']:>+10.4f}")
            print(f"   {'PCC':<20} {b['boundary_curriculum_pcc']:>12.4f} "
                  f"{b['boundary_baseline_pcc']:>12.4f} {b['boundary_pcc_gain']:>+10.4f}")

        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Niche evaluation functions
# ---------------------------------------------------------------------------

def per_niche_metrics(
    pred: np.ndarray, target: np.ndarray, niche_labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
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
    """Group niches by difficulty and compare curriculum vs baseline."""
    K = len(niche_scores)
    sorted_niches = np.argsort(niche_scores)
    bin_size = max(1, K // n_bins)
    labels = ["easy_niches", "medium_niches", "hard_niches"]

    results: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(labels):
        start = i * bin_size
        end = K if i == n_bins - 1 else start + bin_size
        bin_niches = set(sorted_niches[start:end].tolist())
        mask = np.array([int(lbl) in bin_niches for lbl in niche_labels])
        if mask.sum() == 0:
            continue

        cm = mse(pred_curriculum[mask], target[mask])
        bm = mse(pred_baseline[mask], target[mask])
        cp = pearson_correlation_coefficient(pred_curriculum[mask], target[mask])
        bp = pearson_correlation_coefficient(pred_baseline[mask], target[mask])
        results[label] = {
            "n_spots": int(mask.sum()),
            "n_niches": bin_size,
            "curriculum_mse": cm, "baseline_mse": bm, "mse_gain": bm - cm,
            "curriculum_pcc": cp, "baseline_pcc": bp, "pcc_gain": cp - bp,
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
    """Evaluate spots on niche-niche boundaries (tissue interfaces)."""
    from .utils import build_spatial_graph

    edge_index, _ = build_spatial_graph(coords, k=spatial_k)
    boundary = np.zeros(len(niche_labels), dtype=bool)
    for src, dst in zip(edge_index[0], edge_index[1]):
        if niche_labels[src] != niche_labels[dst]:
            boundary[src] = boundary[dst] = True

    if boundary.sum() == 0:
        return {"boundary_n_spots": 0}

    cm = mse(pred_curriculum[boundary], target[boundary])
    bm = mse(pred_baseline[boundary], target[boundary])
    cp = pearson_correlation_coefficient(pred_curriculum[boundary], target[boundary])
    bp = pearson_correlation_coefficient(pred_baseline[boundary], target[boundary])
    return {
        "boundary_n_spots": int(boundary.sum()),
        "boundary_fraction": float(boundary.mean()),
        "boundary_curriculum_mse": cm, "boundary_baseline_mse": bm,
        "boundary_mse_gain": bm - cm,
        "boundary_curriculum_pcc": cp, "boundary_baseline_pcc": bp,
        "boundary_pcc_gain": cp - bp,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_all(
    model: nn.Module, loader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_pred, all_target, all_idx = [], [], []
    for x, y, idx in loader:
        x = move_to_device(x, device)
        out = model(x)
        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)
        pred = out.detach().cpu().numpy()
        target = y.detach().cpu().numpy()
        if pred.ndim >= 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        assert pred.shape == y.shape, f"pred={pred.shape}, target={y.shape}"
        all_pred.append(pred)
        all_target.append(target)
        all_idx.append(flatten_indices(idx).cpu().numpy())
    return (
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_target, axis=0),
        np.concatenate(all_idx, axis=0),
    )


def reorder_by_spot(values: np.ndarray, spot_indices: np.ndarray, n_spots: int) -> np.ndarray:
    shape = (n_spots,) + values.shape[1:]
    out = np.zeros(shape, dtype=values.dtype)
    out[spot_indices] = values
    return out


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    curriculum_model: nn.Module,
    baseline_model: nn.Module,
    test_loader,
    field: SpatialDynamicsField,
    training_log: TrainingLog,
    device: torch.device,
    niche_labels: Optional[np.ndarray] = None,
    niche_scores: Optional[np.ndarray] = None,
) -> EvaluationReport:

    N = field.N
    print("  Curriculum inference...")
    cp, ct, ci = predict_all(curriculum_model, test_loader, device)
    cp = reorder_by_spot(cp, ci, N)
    ct = reorder_by_spot(ct, ci, N)

    print("  Baseline inference...")
    bp, bt, bi = predict_all(baseline_model, test_loader, device)
    bp = reorder_by_spot(bp, bi, N)
    bt = reorder_by_spot(bt, bi, N)

    print("  Computing overall metrics...")
    report = EvaluationReport(
        curriculum_PCC=pearson_correlation_coefficient(cp, ct),
        curriculum_MSE=mse(cp, ct),
        curriculum_MAE=mae(cp, ct),
        baseline_PCC=pearson_correlation_coefficient(bp, bt),
        baseline_MSE=mse(bp, bt),
        baseline_MAE=mae(bp, bt),
    )

    if niche_labels is not None and niche_scores is not None:
        print("  Computing niche-stratified evaluation...")
        report.niche_stratified = niche_stratified_evaluation(
            cp, bp, ct, niche_labels, niche_scores, n_bins=3,
        )
        print("  Computing niche-boundary evaluation...")
        report.niche_boundary = niche_boundary_evaluation(
            cp, bp, ct, niche_labels, field.coords, spatial_k=6,
        )
        report.per_niche = per_niche_metrics(cp, ct, niche_labels)

    report.print_summary()
    return report

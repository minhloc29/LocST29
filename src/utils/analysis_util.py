from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import torch
import anndata as ad

from src.utils import Phase1Results, build_spatial_graph, normalise_difficulty, smooth_on_graph
from src.utils import move_to_device, flatten_indices


@dataclass
class DifficultyFactors:
    """Static difficulty factors derived from Phase 1 outputs and expression."""
    spot_mse: np.ndarray
    coords: np.ndarray
    entropy: np.ndarray
    sparsity: np.ndarray
    spatial_grad: np.ndarray
    morans_I: float
    morans_p: float


@dataclass
class DifficultyDynamics:
    """Epoch-wise difficulty (MSE) per spot."""
    epoch_mse: np.ndarray  # (T, N)
    coords: np.ndarray

    @property
    def T(self) -> int:
        return self.epoch_mse.shape[0]

    @property
    def N(self) -> int:
        return self.epoch_mse.shape[1]


class EpochMSECallback:
    """Collect per-epoch spot MSE from a model and validation loader."""

    def __init__(self, n_spots: int):
        self.n_spots = n_spots
        self._history: list[np.ndarray] = []

    @torch.no_grad()
    def record(self, model: torch.nn.Module, loader, device: torch.device) -> None:
        model.eval()
        sums = np.zeros(self.n_spots, dtype=np.float64)
        counts = np.zeros(self.n_spots, dtype=np.int64)

        for x, y, idx in loader:
            x = move_to_device(x, device)
            y = y.to(device)
            pred = model(x)
            diff = (pred - y) ** 2
            idx_flat = flatten_indices(idx)

            if pred.ndim >= 3 and pred.shape[0] == 1 and idx_flat.numel() == pred.shape[1]:
                per_spot = diff.mean(dim=-1).squeeze(0).detach().cpu().numpy()
                idx_np = idx_flat.cpu().numpy()
                for spot_id, mse_val in zip(idx_np, per_spot):
                    sums[int(spot_id)] += float(mse_val)
                    counts[int(spot_id)] += 1
            else:
                per_sample = diff.view(diff.shape[0], -1).mean(dim=1).detach().cpu().numpy()
                idx_np = idx_flat.cpu().numpy()
                for spot_id, mse_val in zip(idx_np, per_sample):
                    sums[int(spot_id)] += float(mse_val)
                    counts[int(spot_id)] += 1

        epoch = np.full(self.n_spots, np.nan, dtype=np.float32)
        mask = counts > 0
        epoch[mask] = (sums[mask] / counts[mask]).astype(np.float32)
        if np.isnan(epoch).any():
            fill = np.nanmean(epoch)
            epoch = np.where(np.isnan(epoch), fill, epoch)
        self._history.append(epoch)

    def to_dynamics(self, coords: np.ndarray) -> DifficultyDynamics:
        if not self._history:
            raise ValueError("No epochs recorded. Call record() first.")
        epoch_mse = np.stack(self._history, axis=0)
        return DifficultyDynamics(epoch_mse=epoch_mse, coords=coords)


def _entropy_factor(exp: np.ndarray) -> np.ndarray:
    exp = exp.astype(np.float64)
    exp_sum = exp.sum(axis=1, keepdims=True) + 1e-10
    p = exp / exp_sum
    return (-p * np.log(p + 1e-10)).sum(axis=1).astype(np.float32)


def _sparsity_factor(exp: np.ndarray, ratio_threshold: float = 0.3) -> np.ndarray:
    gene_mean = exp.mean(axis=0, keepdims=True) + 1e-8
    ratio = exp / gene_mean
    return (ratio < ratio_threshold).mean(axis=1).astype(np.float32)


def _spatial_gradient(values: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    edge_index, edge_weight = build_spatial_graph(coords, k=k)
    local_mean = smooth_on_graph(values, edge_index, edge_weight, n_iter=1)
    return (values - local_mean).astype(np.float32)


def run_difficulty_analysis(
    p1: Phase1Results,
    adata: ad.AnnData,
    dynamics: Optional[DifficultyDynamics] = None,
    k_neighbours: int = 6,
    n_clusters: int = 4,
) -> Dict[str, object]:
    
    _ = n_clusters  # reserved for future clustering extensions

    exp = adata.X
    if hasattr(exp, "toarray"):
        exp = exp.toarray()
    exp = np.asarray(exp)

    spot_mse = normalise_difficulty(p1.spot_mse)
    entropy = _entropy_factor(exp)
    sparsity = _sparsity_factor(exp)
    spatial_grad = _spatial_gradient(spot_mse, p1.coords, k=k_neighbours)

    factors = DifficultyFactors(
        spot_mse=spot_mse,
        coords=p1.coords,
        entropy=entropy,
        sparsity=sparsity,
        spatial_grad=spatial_grad,
        morans_I=p1.morans_I,
        morans_p=p1.morans_p,
    )

    out = {"factors": factors}
    if dynamics is not None:
        out["dynamics"] = dynamics

    print("[Phase 2] Factors computed:")
    print(f"  spots: {len(spot_mse)}")
    print(f"  Moran's I: {p1.morans_I:.4f} (p={p1.morans_p:.4g})")
    print(f"  entropy mean: {float(entropy.mean()):.4f}")
    print(f"  sparsity mean: {float(sparsity.mean()):.4f}")
    return out

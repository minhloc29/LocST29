from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple

from ..utils import build_spatial_graph, smooth_on_graph
from ..utils import (
    move_to_device,
    flatten_indices,
    select_active_inputs,
    select_active_targets,
)
from ..dynamics.dynamics import SpatialDynamicsField


@dataclass
class Config:
    """Hyper-parameters for the topology-aware curriculum."""
    tau_start: float = 0.3
    tau_end: float = 0.95
    tau_warmup_epochs: int = 5
    graph_diffusion_steps: int = 2
    k_neighbours: int = 6
    stability_weight: float = 0.5
    val_plateau_patience: int = 3
    min_mask_fraction: float = 0.10
    device: str = "cpu"


class AdaptiveThresholdScheduler:
    """
    Threshold tau(t) controlling spot inclusion.
    Starts at tau_start, rises to tau_end, accelerates on val plateaus.
    """

    def __init__(self, cfg: Config, total_epochs: int):
        self.cfg = cfg
        self.T = total_epochs
        self.tau = cfg.tau_start
        self._best_val = float("inf")
        self._plateau_ct = 0
        self._epoch = 0

    def step(self, val_loss: float) -> float:
        """Advance one epoch, return new threshold tau."""
        if self._epoch < self.cfg.tau_warmup_epochs:
            frac = self._epoch / max(self.cfg.tau_warmup_epochs, 1)
            self.tau = self.cfg.tau_start + frac * (self.cfg.tau_end - self.cfg.tau_start)
        else:
            self.tau = self.cfg.tau_end

        if val_loss < self._best_val:
            self._best_val = val_loss
            self._plateau_ct = 0
        else:
            self._plateau_ct += 1
            if self._plateau_ct >= self.cfg.val_plateau_patience:
                self.tau = min(self.tau + 0.05, self.cfg.tau_end)
                self._plateau_ct = 0

        self._epoch += 1
        return self.tau


class TopologyAwareCurriculumSampler:
    """
    Computes a binary mask over spots at each epoch:
    1. Select active spots: D_bar <= tau(t)
    2. Expand via graph diffusion to include neighbors
    3. Apply stability-exploration balance
    """

    def __init__(self, field: SpatialDynamicsField, cfg: Config):
        self.cfg = cfg
        self.D_bar = field.D_bar
        self.coords = field.coords
        self.N = field.N
        self.edge_index, self.edge_weight = build_spatial_graph(self.coords, k=cfg.k_neighbours)
        self._prev_mask: Optional[np.ndarray] = None

    def get_mask(self, tau: float) -> np.ndarray:
        """
        Compute spot inclusion mask for threshold tau.
        Returns bool array of shape (N,).
        """
        base_mask = (self.D_bar <= tau).astype(np.float32)

        if base_mask.mean() < self.cfg.min_mask_fraction:
            n_min = int(self.cfg.min_mask_fraction * self.N)
            top_easy = np.argsort(self.D_bar)[:n_min]
            base_mask[top_easy] = 1.0

        expanded = base_mask.copy()
        for _ in range(self.cfg.graph_diffusion_steps):
            expanded = smooth_on_graph(expanded, self.edge_index, self.edge_weight, n_iter=1)
            expanded = (expanded > 0.05).astype(np.float32)

        if self._prev_mask is not None:
            lam = self.cfg.stability_weight
            combined = lam * self._prev_mask + (1 - lam) * expanded
            final_mask = (combined > 0.5).astype(bool)
        else:
            final_mask = expanded.astype(bool)

        self._prev_mask = final_mask.astype(np.float32)
        return final_mask

    def mask_to_indices(self, mask: np.ndarray) -> np.ndarray:
        """Convert bool mask to integer spot indices."""
        return np.where(mask)[0]


@dataclass
class TrainingLog:
    """Records of training progress."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    tau_history: List[float] = field(default_factory=list)
    mask_sizes: List[int] = field(default_factory=list)
    epoch_masks: List[np.ndarray] = field(default_factory=list)

    def log(self, train: float, val: float, tau: float, mask: np.ndarray) -> None:
        self.train_loss.append(train)
        self.val_loss.append(val)
        self.tau_history.append(tau)
        self.mask_sizes.append(int(mask.sum()))
        self.epoch_masks.append(mask.copy())


def curriculum_train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    loader,
    active_indices: np.ndarray,
    device: torch.device,
) -> float:
    """
    One curriculum training epoch: only compute loss on active spots.

    Returns
    -------
    Mean training loss over active spots.
    """
    model.train()
    active_set = set(active_indices.tolist())
    total_loss, total_n = 0.0, 0

    for x, y, idx in loader:
        idx_flat = flatten_indices(idx)
        keep = torch.tensor(
            [i for i, sp in enumerate(idx_flat.tolist()) if sp in active_set],
            dtype=torch.long,
        )
        if len(keep) == 0:
            continue

        x_k = move_to_device(select_active_inputs(x, keep), device)
        y_k = select_active_targets(y, keep).to(device)

        optimizer.zero_grad()
        pred = model(x_k)
        loss = loss_fn(pred, y_k)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(keep)
        total_n += len(keep)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    """Standard (full) validation loop. Returns mean loss."""
    model.eval()
    total_loss, total_n = 0.0, 0
    for x, y, _ in loader:
        x = move_to_device(x, device)
        pred = model(x)
        loss = loss_fn(pred, y.to(device))
        if y.ndim >= 3 and y.shape[0] == 1:
            batch_n = y.shape[1]
        else:
            batch_n = y.shape[0]
        total_loss += loss.item() * batch_n
        total_n += batch_n
    return total_loss / max(total_n, 1)


def run_phase4(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    train_loader,
    val_loader,
    field: SpatialDynamicsField,
    cfg: Optional[Config] = None,
    total_epochs: int = 50,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, TrainingLog]:
    """
    Topology-aware curriculum training.

    The DataLoaders must yield (features, targets, spot_indices) tuples.
    """
    if cfg is None:
        cfg = Config()
    if device is None:
        device = torch.device(cfg.device)

    model.to(device)
    sampler = TopologyAwareCurriculumSampler(field, cfg)
    scheduler = AdaptiveThresholdScheduler(cfg, total_epochs)
    log = TrainingLog()

    print(f"Starting curriculum training for {total_epochs} epochs...")
    print(
        f"  tau: {cfg.tau_start:.2f} -> {cfg.tau_end:.2f} | "
        f"diffusion steps: {cfg.graph_diffusion_steps} | "
        f"stability lambda: {cfg.stability_weight:.2f}"
    )

    for epoch in range(total_epochs):
        val_loss = evaluate(model, val_loader, loss_fn, device)

        tau = scheduler.step(val_loss)
        mask = sampler.get_mask(tau)
        active_idx = sampler.mask_to_indices(mask)

        train_loss = curriculum_train_epoch(
            model, optimizer, loss_fn, train_loader, active_idx, device
        )

        log.log(train_loss, val_loss, tau, mask)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{total_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"tau={tau:.3f} | active spots={mask.sum()}/{field.N} "
                f"({100 * mask.mean():.1f}%)"
            )

    print("[Phase 4] Training complete.")
    return model, log

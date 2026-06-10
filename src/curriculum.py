from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Tuple

from .utils import build_spatial_graph, smooth_on_graph, expand_mask_k_hops
from .utils import (
    move_to_device,
    flatten_indices,
    select_active_inputs,
    select_active_targets,
)
from .dynamics import SpatialDynamicsField



class ThresholdScheduler:
    def __init__(self, cfg, total_epochs: int):
        self.cfg = cfg
        self.T = total_epochs
        self.tau = cfg.curriculum.tau_start          # was cfg.tau_start
        self._best_val = float("inf")
        self._plateau_ct = 0
        self._epoch = 0

    def step(self, val_loss: float) -> float:
        if self._epoch < self.cfg.curriculum.tau_warmup_epochs:   # was cfg.tau_warmup_epochs
            frac = self._epoch / max(self.cfg.curriculum.tau_warmup_epochs, 1)
            self.tau = self.cfg.curriculum.tau_start + frac * (
                self.cfg.curriculum.tau_end - self.cfg.curriculum.tau_start
            )
        else:
            self.tau = self.cfg.curriculum.tau_end

        if val_loss < self._best_val:
            self._best_val = val_loss
      
        self._epoch += 1
        return self.tau


class TopologyAwareCurriculumSampler:
    def __init__(self, field: SpatialDynamicsField, cfg):
        self.cfg = cfg
        self.difficulty_score = field.difficulty_score
        self.coords = field.coords
        self.N = field.N
        self.edge_index, self.edge_weight = build_spatial_graph(
            self.coords, k=cfg.difficulty.k_neighbours    # was cfg.difficulty.k_neighbours (now explicit)
        )
        self._prev_mask: Optional[np.ndarray] = None

    def get_mask(self, tau: float) -> np.ndarray:
        n_active = max(
            int(tau * self.N),
            int(self.cfg.curriculum.min_mask_fraction * self.N),   # was cfg.min_mask_fraction
        )
        idx = np.argsort(self.difficulty_score)
        base_mask = np.zeros(self.N, dtype=np.float32)
        base_mask[idx[:n_active]] = 1.0

        expanded = base_mask.astype(bool)
        for _ in range(self.cfg.graph.graph_diffusion_steps):      # was cfg.graph_diffusion_steps
            expanded = expand_mask_k_hops(
                expanded,
                self.edge_index,
                k=self.cfg.graph.graph_expansion_hops,             # was cfg.graph_expansion_hops
            )

        if self._prev_mask is not None:
            lam = self.cfg.curriculum.stability_weight             # was cfg.stability_weight
            combined = lam * self._prev_mask + (1 - lam) * expanded.astype(np.float32)
            final_mask = (combined >= 0.5).astype(bool)
        else:
            final_mask = expanded

        self._prev_mask = final_mask.astype(np.float32)
        return final_mask

    def mask_to_indices(self, mask: np.ndarray) -> np.ndarray:
        return np.where(mask)[0]


@dataclass
class TrainingLog:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_pcc: List[float] = field(default_factory=list)
    tau_history: List[float] = field(default_factory=list)
    mask_sizes: List[int] = field(default_factory=list)
    epoch_masks: List[np.ndarray] = field(default_factory=list)

    def log(self, train: float, val: float, tau: float, mask: np.ndarray,
            pcc: Optional[float] = None) -> None:
        self.train_loss.append(train)
        self.val_loss.append(val)
        self.val_pcc.append(pcc if pcc is not None else float("nan"))
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
    max_grad_norm: float = 1.0
) -> float:
   
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
        if isinstance(pred, tuple):
            pred = pred[0]
        
        if y_k.ndim == 3 and y_k.shape[0] == 1:
            y_k = y_k.squeeze(0)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        assert pred.shape == y_k.shape, (      # ← inside loop: correct
            f"pred={pred.shape}, target={y_k.shape}"
        )

        loss = loss_fn(pred, y_k)
        loss.backward()

        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
    model.eval()
    total_loss, total_n = 0.0, 0
    for x, y, _ in loader:
        x = move_to_device(x, device)
        pred = model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        y = y.to(device)
        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        assert pred.shape == y.shape, (        # ← inside loop: correct
            f"pred={pred.shape}, target={y.shape}"
        )

        loss = loss_fn(pred, y)
        total_loss += loss.item() * y.shape[0]
        total_n += y.shape[0]

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_with_pcc(
    model: nn.Module,
    loader,
    loss_fn: Callable,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on loader and return both validation loss and spot-wise PCC.
    PCC is averaged across spots (each spot is a gene expression vector).
    """
    model.eval()
    total_loss, total_n = 0.0, 0
    all_pred, all_target = [], []

    for x, y, _ in loader:
        x = move_to_device(x, device)
        pred = model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        y = y.to(device)
        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        assert pred.shape == y.shape, f"pred={pred.shape}, target={y.shape}"
        loss = loss_fn(pred, y)
        total_loss += loss.item() * y.shape[0]
        total_n += y.shape[0]
        all_pred.append(pred.cpu().numpy())
        all_target.append(y.cpu().numpy())

    val_loss = total_loss / max(total_n, 1)

    if all_pred:
        pred_np = np.concatenate(all_pred, axis=0)
        target_np = np.concatenate(all_target, axis=0)
        # spot-wise PCC
        from scipy.stats import pearsonr
        pccs = []
        for i in range(pred_np.shape[0]):
            p, t = pred_np[i], target_np[i]
            if t.std() < 1e-8 or p.std() < 1e-8:
                continue
            r, _ = pearsonr(p, t)
            pccs.append(r)
        val_pcc = float(np.nanmean(pccs)) if pccs else float("nan")
    else:
        val_pcc = float("nan")

    return val_loss, val_pcc



def save_init_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
) -> None:
   
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"[Checkpoint] Saved shared init to {path}")


def curriculum_train_epoch_v2(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    loader,
    difficulty_repo: dict,
    tau: float,
    device: torch.device,
    max_grad_norm: float = 1.0,
    min_mask_fraction: float = 0.10,
) -> float:
    """Train one epoch with curriculum-based spot selection.

    The *difficulty_repo* dict supports two modes:

    **Spot-level (original)** — ``{slide_id: np.ndarray(N_spots,)}``:
        Select the easiest *spots* up to fraction tau.

    **Niche-level (new)** — ``{slide_id: {"spot": ..., "niche": ...,
    "niche_labels": ...}}``:
        Select the easiest *niches* up to fraction tau; all spots in
        an active niche are trained together.
    """
    model.train()
    total_loss, total_n = 0.0, 0

    for x, y, idx, slide_id in loader:
        slide_id = int(slide_id)

        if slide_id in difficulty_repo:
            entry = difficulty_repo[slide_id]

            # Detect mode: dict with "niche" key → niche-level
            if isinstance(entry, dict) and "niche" in entry:
                # --- Niche-level selection ---
                niche_scores = entry["niche"]                # (K,)
                niche_labels = entry["niche_labels"]          # (N_spots,)
                n_niches = len(niche_scores)
                n_active = max(
                    int(tau * n_niches),
                    int(min_mask_fraction * n_niches),
                )
                # Easiest niches first
                active_niche_idx = np.argsort(niche_scores)[:n_active]
                active_set = set(active_niche_idx.tolist())
                keep = torch.tensor(
                    [i for i, lbl in enumerate(niche_labels) if int(lbl) in active_set],
                    dtype=torch.long,
                )
            else:
                # --- Spot-level selection (original) ---
                scores = entry if isinstance(entry, np.ndarray) else entry["spot"]
                n_spots = len(scores)
                n_active = max(
                    int(tau * n_spots),
                    int(min_mask_fraction * n_spots),
                )
                active_local = np.argsort(scores)[:n_active]
                keep = torch.tensor(active_local, dtype=torch.long)
        else:
            # Fallback: random tau-based sampling for slides not in repo
            n_spots = y.shape[1] if y.ndim == 3 else y.shape[0]
            n_active = max(int(tau * n_spots), 1)
            keep = torch.randperm(n_spots)[:n_active]

        if len(keep) == 0:
            continue

        x_k = move_to_device(select_active_inputs(x, keep), device)
        y_k = select_active_targets(y, keep).to(device)

        optimizer.zero_grad()
        pred = model(x_k)
        if isinstance(pred, tuple):
            pred = pred[0]
        if y_k.ndim == 3 and y_k.shape[0] == 1:
            y_k = y_k.squeeze(0)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        assert pred.shape == y_k.shape, f"pred={pred.shape}, y={y_k.shape}"
        loss = loss_fn(pred, y_k)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(keep)
        total_n += len(keep)

    return total_loss / max(total_n, 1)

def load_init_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> None:
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Init checkpoint not found: {path}\n"
            "Run save_init_checkpoint() after warm-up first."
        )
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Checkpoint] Loaded shared init from {path}")


def reset_optimizer(
    optimizer: torch.optim.Optimizer,
) -> None:
  
    optimizer.state.clear()
    print("[Checkpoint] Optimizer state reset (momentum/variance cleared).")



def train_curriculum(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    train_loader,
    val_loader,
    difficulty_repo,
    cfg,
    total_epochs: int = 50,
    device: Optional[torch.device] = None,
    init_checkpoint: Optional[str | Path] = None
) -> Tuple[nn.Module, TrainingLog]:

    if device is None:
        device = torch.device(cfg.training.device)

    print(init_checkpoint)
    init_checkpoint = None
    if init_checkpoint is not None:
        ckpt = torch.load(str(init_checkpoint), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Checkpoint] Loaded warm-up weights from {init_checkpoint}")

    model.to(device)
    scheduler = ThresholdScheduler(cfg, total_epochs)
    log = TrainingLog()

    # early stopping state
    best_val = float("inf")
    best_state: Optional[dict] = None

    print(f"Starting curriculum training for {total_epochs} epochs...")
    print(
        f"  tau: {cfg.curriculum.tau_start:.2f} -> {cfg.curriculum.tau_end:.2f} | "
        f"max_grad_norm: {cfg.training.max_grad_norm}"
    )

    for epoch in range(total_epochs):
        val_loss, val_pcc = evaluate_with_pcc(model, val_loader, loss_fn, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        tau = scheduler.step(val_loss)

        # Per-slide spot selection is handled inside curriculum_train_epoch_v2
        # using difficulty_repo scores directly. No global mask needed.
        train_loss = curriculum_train_epoch_v2(
            model, optimizer, loss_fn, train_loader, difficulty_repo, tau, device,
            max_grad_norm=cfg.training.max_grad_norm,
            min_mask_fraction=cfg.curriculum.min_mask_fraction,
        )

        log.log(train_loss, val_loss, tau, np.zeros(1), pcc=val_pcc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{total_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"PCC={val_pcc:.4f} | "
                f"tau={tau:.3f}"
            )

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Phase 4] Restored best val checkpoint (val={best_val:.4f}).")

    print("[Phase 4] Training complete.")
    return model, log
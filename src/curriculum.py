from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Tuple, Dict

from .utils import build_spatial_graph, smooth_on_graph, expand_mask_k_hops
from .utils import (
    move_to_device,
    flatten_indices,
    select_active_inputs,
    select_active_targets,
)
from .dynamics import SpatialDynamicsField



class LossTracker:
    """Tracks per-niche training loss with EMA smoothing and adaptive tau.

    This replaces the old ThresholdScheduler. Instead of a blind linear
    warmup of tau, we monitor the model's actual training loss on the
    active set and increase tau when progress plateaus — the model
    drives its own pace (self-paced learning).
    """

    def __init__(self, cfg, n_niches_per_slide: Dict[int, int]):
        self.cfg = cfg
        self.tau = cfg.curriculum.tau_start
        self.tau_end = cfg.curriculum.tau_end
        self.min_mask_fraction = cfg.curriculum.min_mask_fraction
        self.ema_alpha = getattr(cfg.curriculum, "spl_loss_ema_alpha", 0.9)
        self.plateau_threshold = getattr(cfg.curriculum, "spl_plateau_threshold", 0.005)
        self.tau_step = getattr(cfg.curriculum, "spl_tau_step", 0.02)

        # Per-slide: EMA of per-niche mean loss, shape (K,) for each slide
        # Initialized to None; first epoch fills them.
        self._ema: Dict[int, np.ndarray] = {}
        self._n_niches = n_niches_per_slide  # {slide_id: K}

        self._prev_active_loss: Optional[float] = None
        self._epoch = 0

    @property
    def active_fraction(self) -> float:
        """Current tau: fraction of easiest niches to train on."""
        active = max(int(self.tau * 100) / 100, self.min_mask_fraction)
        return min(active, 1.0)

    def update_ema(self, slide_id: int, per_niche_loss: np.ndarray) -> np.ndarray:
        """Update the EMA for a slide with the latest per-niche loss.

        Returns the smoothed per-niche loss array.
        """
        loss = np.asarray(per_niche_loss, dtype=np.float64)
        if slide_id not in self._ema:
            self._ema[slide_id] = loss.copy()
        else:
            self._ema[slide_id] = (
                self.ema_alpha * self._ema[slide_id]
                + (1.0 - self.ema_alpha) * loss
            )
        return self._ema[slide_id].copy()

    def get_smoothed_loss(self, slide_id: int) -> np.ndarray:
        """Return the current EMA loss for a slide, or zeros if not yet tracked."""
        if slide_id in self._ema:
            return self._ema[slide_id].copy()
        return np.zeros(self._n_niches.get(slide_id, 0), dtype=np.float64)

    def step_tau(self, active_loss: float) -> float:
        """Adaptively increase tau based on progress of active set.

        If the active-set loss hasn't dropped enough (plateau), we
        increase tau to let in more niches.  This is the 'self-paced'
        mechanism: the model drives the pace.
        """
        if self._prev_active_loss is not None:
            drop = self._prev_active_loss - active_loss
            if drop < self.plateau_threshold:
                # Plateau → let more niches in
                self.tau = min(self.tau_end, self.tau + self.tau_step)

        self._prev_active_loss = active_loss
        self._epoch += 1
        return self.tau


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


# ---------------------------------------------------------------------------
#  Helpers: compute per-niche loss from the model
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_niche_loss(
    model: nn.Module,
    loader,
    niche_labels_dict: Dict[int, np.ndarray],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """Run the model on *all* training spots and return per-niche mean loss.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode (caller is responsible for mode toggle).
    loader : DataLoader
        Training loader yielding (x, y, idx, slide_id).
    niche_labels_dict : dict
        {slide_id: (N_spots,) int array mapping each spot to its niche label}.
    device : torch.device

    Returns
    -------
    per_niche_loss : dict
        {slide_id: (K,) float64 array} — mean loss per niche.
    """
    model.eval()

    # Accumulators per slide: {slide_id: {niche_label: [sum_loss, count]}}
    accum: Dict[int, Dict[int, List[float]]] = {}

    for x, y, idx, slide_id in loader:
        sid = int(slide_id)
        if sid not in niche_labels_dict:
            continue

        niche_labels = niche_labels_dict[sid]          # (N_spots,)
        x = move_to_device(x, device)
        if isinstance(x, (list, tuple)):
            x = [move_to_device(xi, device) for xi in x]
        y = y.to(device)

        pred = model(x)
        if isinstance(pred, tuple):
            pred = pred[0]

        # Squeeze batch dim if needed
        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        # Per-spot MSE
        diff = (pred - y) ** 2                               # (N_spots, G)
        per_spot_loss = diff.mean(dim=1).detach().cpu().numpy()  # (N_spots,)

        # Map to niches
        if sid not in accum:
            accum[sid] = {}
        for spot_i, loss_val in enumerate(per_spot_loss):
            nlabel = int(niche_labels[spot_i])
            if nlabel not in accum[sid]:
                accum[sid][nlabel] = [0.0, 0.0]
            accum[sid][nlabel][0] += float(loss_val)
            accum[sid][nlabel][1] += 1.0

    # Aggregate to per-niche mean
    result: Dict[int, np.ndarray] = {}
    for sid, niches in accum.items():
        labels = sorted(niches.keys())
        n_niches = max(labels) + 1 if labels else 0
        per_niche = np.zeros(n_niches, dtype=np.float64)
        for lbl, (s, c) in niches.items():
            per_niche[lbl] = s / max(c, 1)
        result[sid] = per_niche

    return result


# ---------------------------------------------------------------------------
#  Training epoch functions
# ---------------------------------------------------------------------------

def curriculum_train_epoch_v2(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    loader,
    niche_labels_dict: Dict[int, np.ndarray],
    per_niche_loss: Dict[int, np.ndarray],
    tau: float,
    device: torch.device,
    max_grad_norm: float = 1.0,
    min_mask_fraction: float = 0.10,
) -> Tuple[float, float, int]:
    """Train one epoch with self-paced niche selection.

    Selection is based on the model's **own training loss** per niche
    (from the previous epoch), not a static difficulty score.

    Parameters
    ----------
    per_niche_loss : dict
        {slide_id: (K,) float64} — per-niche mean loss from previous epoch
        (or EMA-smoothed).  Niches with lower loss are selected first.

    Returns
    -------
    (avg_loss, active_loss, n_active_spots)
    """
    model.train()
    total_loss, total_n = 0.0, 0
    active_loss_sum, active_loss_count = 0.0, 0

    for x, y, idx, slide_id in loader:
        sid = int(slide_id)

        if sid in per_niche_loss:
            niche_scores = per_niche_loss[sid]                 # (K,) — current loss
            niche_labels = niche_labels_dict[sid]               # (N_spots,)
            n_niches = len(niche_scores)
            n_active = max(
                int(tau * n_niches),
                int(min_mask_fraction * n_niches),
            )
            # Lowest-loss niches first (easiest for the model right now)
            active_niche_idx = np.argsort(niche_scores)[:n_active]
            active_set = set(active_niche_idx.tolist())
            keep = torch.tensor(
                [i for i, lbl in enumerate(niche_labels) if int(lbl) in active_set],
                dtype=torch.long,
            )
        else:
            # Slide not in per_niche_loss — fallback: random tau-based selection
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
        active_loss_sum += loss.item() * len(keep)
        active_loss_count += len(keep)

    avg_loss = total_loss / max(total_n, 1)
    active_loss = active_loss_sum / max(active_loss_count, 1)
    return avg_loss, active_loss, total_n


def curriculum_train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    loader,
    active_indices: np.ndarray,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> float:
    """Legacy: train on a fixed set of active spots (used for warm-up)."""
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

        assert pred.shape == y_k.shape, (
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

        assert pred.shape == y.shape, (
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
    difficulty_repo,          # still needed for niche_labels_dict extraction
    cfg,
    total_epochs: int = 50,
    device: Optional[torch.device] = None,
    init_checkpoint: Optional[str | Path] = None,
) -> Tuple[nn.Module, TrainingLog]:

    if device is None:
        device = torch.device(cfg.training.device)

    if init_checkpoint is not None:
        ckpt = torch.load(str(init_checkpoint), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Checkpoint] Loaded warm-up weights from {init_checkpoint}")

    model.to(device)

    # Extract niche_labels_dict from difficulty_repo (needed for loss computation
    # and for the epoch training function).
    # difficulty_repo[slide_id] = {"spot": ..., "niche": ..., "niche_labels": ...}
    niche_labels_dict: Dict[int, np.ndarray] = {}
    n_niches_per_slide: Dict[int, int] = {}
    for sid, entry in difficulty_repo.items():
        if isinstance(entry, dict) and "niche" in entry:
            niche_labels_dict[sid] = entry["niche_labels"]
            n_niches_per_slide[sid] = len(entry["niche"])
        else:
            # Spot-level fallback: treat each spot as its own "niche"
            n = len(entry) if isinstance(entry, np.ndarray) else len(entry["spot"])
            niche_labels_dict[sid] = np.arange(n, dtype=np.int32)
            n_niches_per_slide[sid] = n

    # Initial ordering: use static difficulty as initial per_niche_loss if configured
    initial_ordering = getattr(cfg.curriculum, "spl_initial_ordering", "loss")
    if initial_ordering == "static":
        per_niche_loss: Dict[int, np.ndarray] = {}
        for sid, entry in difficulty_repo.items():
            if isinstance(entry, dict) and "niche" in entry:
                per_niche_loss[sid] = entry["niche"].astype(np.float64)
            elif isinstance(entry, np.ndarray):
                per_niche_loss[sid] = entry.astype(np.float64)
            else:
                per_niche_loss[sid] = entry["spot"].astype(np.float64)
    else:
        # Pure SPL: start with empty loss dict — first epoch will compute
        per_niche_loss = {}

    # Build loss tracker
    loss_tracker = LossTracker(cfg, n_niches_per_slide)
    log = TrainingLog()

    # early stopping state
    best_val = float("inf")
    best_state: Optional[dict] = None

    print(f"Starting SPL curriculum training for {total_epochs} epochs...")
    print(
        f"  tau: {cfg.curriculum.tau_start:.2f} -> {cfg.curriculum.tau_end:.2f} | "
        f"max_grad_norm: {cfg.training.max_grad_norm} | "
        f"initial_ordering: {initial_ordering}"
    )

    for epoch in range(total_epochs):
        val_loss, val_pcc = evaluate_with_pcc(model, val_loader, loss_fn, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # ── Step A: Compute per-niche loss on all training spots ──
        # This is the "self-paced" signal: the model evaluates itself on every niche.
        epoch_per_niche_loss = compute_per_niche_loss(
            model, train_loader, niche_labels_dict, device,
        )

        # ── Step B: Update EMA per slide ──
        for sid, loss_arr in epoch_per_niche_loss.items():
            loss_tracker.update_ema(sid, loss_arr)

        # ── Step C: Get smoothed loss for selection ──
        selection_loss: Dict[int, np.ndarray] = {}
        for sid in niche_labels_dict:
            selection_loss[sid] = loss_tracker.get_smoothed_loss(sid)

        tau = loss_tracker.active_fraction

        # ── Step D: Train on selected niches ──
        train_loss, active_loss, n_active = curriculum_train_epoch_v2(
            model, optimizer, loss_fn, train_loader,
            niche_labels_dict, selection_loss, tau, device,
            max_grad_norm=cfg.training.max_grad_norm,
            min_mask_fraction=cfg.curriculum.min_mask_fraction,
        )

        # ── Step E: Update tau adaptively ──
        loss_tracker.step_tau(active_loss)

        log.log(train_loss, val_loss, tau, np.zeros(1), pcc=val_pcc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{total_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"PCC={val_pcc:.4f} | "
                f"tau={tau:.3f} | "
                f"active_spots={n_active}"
            )

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Phase 4] Restored best val checkpoint (val={best_val:.4f}).")

    print("[Phase 4] Training complete.")
    return model, log

"""Trainer: build niches → assign difficulty → curriculum train → save model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Callable, Tuple

from .utils import Phase1Results, _seed_everything, move_to_device
from .curriculum import (
    train_curriculum, TrainingLog,
    evaluate as _evaluate,
)


def build_difficulty_repo(
    train_base,
    k_neighbours: int = 6,
    use_niches: bool = True,
    niche_cfg: Optional = None,
) -> dict:
    """
    Build per-slide difficulty scores from gene expression.

    If *use_niches* is True, difficulty is computed at the **niche**
    (microenvironment) level using ``niche_difficulty_from_data``.
    Otherwise the original spot-level ``topological_difficulty_from_data``
    is used (backward compatible).

    Returns
    -------
    difficulty_repo : dict
        If use_niches is False:
            {slide_idx (int): np.ndarray shape [N_spots]}
        If use_niches is True:
            {slide_idx (int): {
                "spot":         np.ndarray [N_spots],
                "niche":        np.ndarray [K],
                "niche_labels": np.ndarray [N_spots],
            }}
    """
    from .difficulty_gse import topological_difficulty_from_data, niche_difficulty_from_data

    repo = {}

    mode = "niche" if use_niches else "spot"
    print(f"[DifficultyRepo] Computing per-slide difficulty from expression "
          f"({len(train_base)} slides, mode={mode})...")

    for slide_idx in range(len(train_base)):
        slide_name = train_base.names[slide_idx]
        expression = train_base.exp_dict[slide_name]          # (N_spots, G)
        coords = train_base.center_dict[slide_name].astype(float)  # (N_spots, 2)

        if use_niches:
            # --- Niche-level difficulty ---
            kws = vars(niche_cfg) if niche_cfg is not None else {}
            niche_labels, niche_scores, spot_scores = niche_difficulty_from_data(
                expression=expression,
                coords=coords,
                alpha=kws.get("alpha", 0.40),
                beta=kws.get("beta", 0.30),
                gamma=kws.get("gamma", 0.15),
                delta=kws.get("delta", 0.15),
                method=kws.get("method", "spatial_leiden"),
                resolution=kws.get("resolution", 1.0),
                n_niches=kws.get("n_niches", None),
                spatial_weight=kws.get("spatial_weight", 0.3),
            )
            repo[slide_idx] = {
                "spot": spot_scores,
                "niche": niche_scores,
                "niche_labels": niche_labels,
            }
            print(f"  Slide {slide_name:>4} | N={len(spot_scores):>3} "
                  f"K={len(niche_scores):>2} | "
                  f"niche_mean={niche_scores.mean():.3f} | "
                  f"spot_mean={spot_scores.mean():.3f}")
        else:
            # --- Spot-level difficulty (original) ---
            scores = topological_difficulty_from_data(
                expression=expression,
                coords=coords,
                k_neighbours=k_neighbours,
            )
            repo[slide_idx] = scores
            print(f"  Slide {slide_name:>4} | N={len(scores):>3} | "
                  f"mean_score={scores.mean():.3f} | std={scores.std():.3f}")

    return repo


def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Shared uniform training loop used by warm-up and baseline."""
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        x, y = batch[0], batch[1]

        optimizer.zero_grad()
        x = move_to_device(x, device)
        pred = model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        y = y.to(device)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)
        assert pred.shape == y.shape, f"pred={pred.shape}, target={y.shape}"
        loss = loss_fn(pred, y)
        loss.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * y.shape[0]
        total_n += y.shape[0]
    return total_loss / max(total_n, 1)


class SpatialCurriculumTrainer:

    def __init__(self, p1: Phase1Results, cfg=None):
        self.p1 = p1
        self.cfg = cfg
        _seed_everything(self.cfg.training.seed)

        self.difficulty_repo: Optional[dict] = None
        self._training_log: Optional[TrainingLog] = None
        self._baseline_log: Optional[TrainingLog] = None

        # Set by train()
        self.model: Optional[nn.Module] = None
        self.baseline_model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.baseline_optimizer: Optional[torch.optim.Optimizer] = None

    def train_baseline(
        self,
        baseline_model: nn.Module,
        baseline_optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_loader,
        val_loader,
    ) -> Tuple[nn.Module, TrainingLog]:

        device = torch.device(self.cfg.training.device)

        if self.cfg.checkpoint.init_checkpoint is not None:
            ckpt = torch.load(self.cfg.checkpoint.init_checkpoint, map_location=device)
            baseline_model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Baseline] Loaded warm-up weights from {self.cfg.checkpoint.init_checkpoint}")
        else:
            print(
                "[Baseline] WARNING: no init_checkpoint set. "
                "Baseline starts from its current (possibly random) weights."
            )

        baseline_model.to(device)
        best_val = float("inf")
        best_state = None
        baseline_log = TrainingLog()

        print(f"[Baseline] Training for up to {self.cfg.training.total_epochs} epochs")

        for epoch in range(self.cfg.training.total_epochs):

            val_loss, val_pcc = _evaluate(baseline_model, val_loader, loss_fn, device)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in baseline_model.state_dict().items()
                }

            train_loss = _train_one_epoch(
                baseline_model, train_loader, baseline_optimizer, loss_fn,
                device, max_grad_norm=self.cfg.training.max_grad_norm,
            )

            baseline_log.log(train_loss, val_loss, tau=1.0, mask=np.ones(1))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1:3d}/{self.cfg.training.total_epochs} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f} | "
                    f"PCC={val_pcc:.4f} | "
                )

        if best_state is not None:
            baseline_model.load_state_dict(best_state)
            print(f"[Baseline] Best val checkpoint restored (val={best_val:.4f}).")

        return baseline_model, baseline_log

    def train(
        self,
        model: nn.Module,
        baseline_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        baseline_optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_base,
        train_loader,
        val_loader,
    ) -> dict:
        """
        Run full curriculum training.

        Returns a dict with trained models, training log, and difficulty repo.
        """
        self.model = model
        self.baseline_model = baseline_model
        self.optimizer = optimizer
        self.baseline_optimizer = baseline_optimizer

        device = torch.device(self.cfg.training.device)

        use_niches = getattr(self.cfg, "niche", None) is not None
        if use_niches:
            niche_cfg = self.cfg.niche
            enabled = getattr(niche_cfg, "enabled", True)
            use_niches = enabled

        print(f"\n[Trainer] Computing difficulty scores from expression data...")
        print(f"  slides: {len(train_base)} | spots: {self.p1.coords.shape[0]}"
              f" | mode={'niche' if use_niches else 'spot'}")

        # ── Step 1: build niche difficulty for all training slides ──
        self.difficulty_repo = build_difficulty_repo(
            train_base,
            k_neighbours=self.cfg.difficulty.k_neighbours,
            use_niches=use_niches,
            niche_cfg=self.cfg.niche if use_niches else None,
        )

        # ── Step 2: baseline (full-data, no curriculum) ──
        self.baseline_model, self._baseline_log = self.train_baseline(
            baseline_model, baseline_optimizer, loss_fn, train_loader, val_loader,
        )

        # ── Step 3: curriculum training ──
        self.model, self._training_log = train_curriculum(
            self.model, self.optimizer, loss_fn,
            train_loader, val_loader, self.difficulty_repo,
            cfg=self.cfg,
            total_epochs=self.cfg.training.total_epochs,
            device=device,
            init_checkpoint=self.cfg.checkpoint.init_checkpoint,
        )

        return {
            "model": self.model,
            "baseline_model": self.baseline_model,
            "training_log": self._training_log,
            "baseline_log": self._baseline_log,
            "difficulty_repo": self.difficulty_repo,
        }

    def save_checkpoint(self, output_dir: str | Path) -> None:
        """Save trained models to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(),          output_dir / "curriculum_model.pt")
        torch.save(self.baseline_model.state_dict(), output_dir / "baseline_model.pt")
        print(f"[Trainer] Models saved to {output_dir}")

"""Pipeline: build niches → assign difficulty → curriculum train → evaluate."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import anndata as ad
from pathlib import Path
from typing import Optional, Callable

from .utils import Phase1Results, _seed_everything, move_to_device
from .curriculum import (
    train_curriculum, TrainingLog,
    evaluate as _evaluate,
)
from .evaluation import (
    BiologicalAnnotations, EvaluationReport, run_evaluation
)


def build_difficulty_repo(
    train_base,
    niche_cfg,
) -> dict:
    """Build per-slide niche-level difficulty from gene expression.

    For each slide:
      1. Cluster spots into niches (spatial + expression).
      2. Compute niche-level difficulty (heterogeneity + topology + ambiguity).
      3. Map difficulty back to each spot.

    Returns
    -------
    difficulty_repo : {slide_idx (int): {
        "spot":         np.ndarray [N_spots],
        "niche":        np.ndarray [K],
        "niche_labels": np.ndarray [N_spots],
    }}
    """
    from .difficulty import niche_difficulty_from_data

    kws = vars(niche_cfg) if niche_cfg is not None else {}
    repo = {}

    print(f"[DifficultyRepo] Computing niche difficulty "
          f"({len(train_base)} slides)...")

    for slide_idx in range(len(train_base)):
        slide_name = train_base.names[slide_idx]
        expression = train_base.exp_dict[slide_name]                 # (N, G)
        coords = train_base.center_dict[slide_name].astype(float)    # (N, 2)

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
        print(f"  {slide_name:>4} | N={len(spot_scores):>3} "
              f"K={len(niche_scores):>2} | "
              f"niche_diff={niche_scores.mean():.3f} | "
              f"spot_diff={spot_scores.mean():.3f}")

    n_total = sum(v["niche"].shape[0] for v in repo.values())
    mean_k = np.mean([len(v["niche"]) for v in repo.values()])
    print(f"  Total niches across slides: ~{int(n_total)}  "
          f"mean K/slide: {mean_k:.1f}")
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


class SpatialCurriculumPipeline:

    def __init__(self, p1: Phase1Results, cfg=None):
        self.p1 = p1
        self.cfg = cfg
        _seed_everything(self.cfg.training.seed)

        self._training_log: Optional[TrainingLog] = None

    def train_baseline(
        self,
        baseline_model: nn.Module,
        baseline_optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_loader,
        val_loader,
    ) -> nn.Module:

        device = torch.device(self.cfg.training.device)

        if self.cfg.checkpoint.init_checkpoint is not None:
            ckpt = torch.load(self.cfg.checkpoint.init_checkpoint, map_location=device)
            baseline_model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Baseline] Loaded warm-up weights from {self.cfg.checkpoint.init_checkpoint}")
        else:
            print("[Baseline] Starting from current (possibly random) weights.")

        baseline_model.to(device)
        best_val = float("inf")
        best_state = None

        print(f"[Baseline] Training for {self.cfg.training.total_epochs} epochs "
              f"(full data, no curriculum)")

        for epoch in range(self.cfg.training.total_epochs):
            val_loss = _evaluate(baseline_model, val_loader, loss_fn, device)
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
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1:3d}/{self.cfg.training.total_epochs} | "
                      f"train={train_loss:.4f} | val={val_loss:.4f}")

        if best_state is not None:
            baseline_model.load_state_dict(best_state)
            print(f"[Baseline] Restored best val checkpoint (val={best_val:.4f}).")
        return baseline_model

    def run(
        self,
        adata: ad.AnnData,
        model: nn.Module,
        baseline_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        baseline_optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_base,
        train_loader,
        val_loader,
        test_loader,
        bio_annotations: Optional[BiologicalAnnotations] = None,
    ) -> EvaluationReport:

        device = torch.device(self.cfg.training.device)
        niche_cfg = getattr(self.cfg, "niche", None)
        if niche_cfg is None or not getattr(niche_cfg, "enabled", True):
            raise ValueError(
                "Pipeline requires niche configuration (cfg.niche.enabled=true). "
                "Add a [niche] section to your config."
            )

        print(f"\n[Pipeline] Computing niche difficulty from expression...")
        print(f"  training slides: {len(train_base)}  "
              f"test spots: {self.p1.coords.shape[0]}")

        # ── Step 1: build niche difficulty for all training slides ──
        difficulty_repo = build_difficulty_repo(train_base, niche_cfg)

        # ── Step 2: curriculum training ──
        model, log = train_curriculum(
            model, optimizer, loss_fn,
            train_loader, val_loader, difficulty_repo,
            cfg=self.cfg,
            total_epochs=self.cfg.training.total_epochs,
            device=device,
            init_checkpoint=self.cfg.checkpoint.init_checkpoint,
        )
        self._training_log = log

        # ── Step 3: baseline (full-data, no curriculum) ──
        baseline_model = self.train_baseline(
            baseline_model, baseline_optimizer, loss_fn, train_loader, val_loader,
        )

        # ── Step 4: build niche data for test slide evaluation ──
        from .difficulty import build_spatial_niches, compute_niche_difficulty

        test_expr = np.asarray(adata.X)
        test_coords = self.p1.coords

        niche_labels_eval = build_spatial_niches(
            expression=test_expr, coords=test_coords,
            method=getattr(niche_cfg, "method", "spatial_leiden"),
            resolution=getattr(niche_cfg, "resolution", 1.0),
            spatial_weight=getattr(niche_cfg, "spatial_weight", 0.3),
        )
        _, niche_scores_eval = compute_niche_difficulty(
            expression=test_expr, coords=test_coords,
            niche_labels=niche_labels_eval,
            alpha=getattr(niche_cfg, "alpha", 0.4),
            beta=getattr(niche_cfg, "beta", 0.3),
            gamma=getattr(niche_cfg, "gamma", 0.15),
            delta=getattr(niche_cfg, "delta", 0.15),
        )
        print(f"[Pipeline] Test slide: K={len(niche_scores_eval)} niches  "
              f"mean_diff={niche_scores_eval.mean():.3f}  "
              f"hard_frac={(niche_scores_eval > 0.75).mean():.1%}")

        # ── Step 5: evaluate ──
        evaluation_result = run_evaluation(
            model, baseline_model, test_loader,
            self._get_dummy_field(test_coords),
            self._training_log, device,
            bio_annotations=bio_annotations,
            hard_percentile=self.cfg.difficulty.hard_percentile,
            niche_labels=niche_labels_eval,
            niche_scores=niche_scores_eval,
        )
        return evaluation_result

    def _get_dummy_field(self, coords: np.ndarray):
        """Build a minimal SpatialDynamicsField for evaluation compatibility."""
        from .dynamics import SpatialDynamicsField
        N = coords.shape[0]
        return SpatialDynamicsField(
            D_field=np.empty((0, N), dtype=np.float32),
            D_bar=np.full(N, np.nan, dtype=np.float32),
            learning_speed=np.full(N, np.nan, dtype=np.float32),
            volatility=np.full(N, np.nan, dtype=np.float32),
            T_L=np.full(N, 0, dtype=np.int32),
            coords=coords,
            difficulty_score=np.zeros(N, dtype=np.float32),
        )

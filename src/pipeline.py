from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import anndata as ad
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from .utils import Phase1Results, _seed_everything, move_to_device
from .analysis import (
    run_difficulty_analysis
)
from .dynamics import build_difficulty_field, SpatialDynamicsField, TopologyResult
from .curriculum import (
    train_curriculum, TrainingLog,
    evaluate as _evaluate,
)
from .evaluation import (
    BiologicalAnnotations, EvaluationReport, run_evaluation
)


def build_difficulty_repo(
    train_base,
    k_neighbours: int = 6,
    use_niches: bool = False,
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


class SpatialCurriculumPipeline:
    
    def __init__(self, p1: Phase1Results, cfg: Optional = None):
        self.p1 = p1
        self.cfg = cfg
        _seed_everything(self.cfg.training.seed)

        self.difficulty_analysis_result = None
        self.difficulty_field = None
        self._training_log: Optional[TrainingLog] = None
        self._field: Optional[SpatialDynamicsField] = None


   
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
            # optimizer stays fresh (default Adam state) — no reset spike
            print(f"[Baseline] Loaded warm-up weights from {self.cfg.checkpoint.init_checkpoint}")
        else:
            print(
                "[Baseline] WARNING: no init_checkpoint set. "
                "Baseline starts from its current (possibly random) weights."
            )

        baseline_model.to(device)
        best_val = float("inf")
        best_state = None

        print(f"[Baseline] Training for up to {self.cfg.training.total_epochs} epochs")

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
                print(
                    f"  [Baseline] Epoch {epoch + 1:3d}/{self.cfg.training.total_epochs} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f}"
                )

        if best_state is not None:
            baseline_model.load_state_dict(best_state)
            print(f"[Baseline] Best val checkpoint restored (val={best_val:.4f}).")

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

        use_niches = getattr(self.cfg, "niche", None) is not None
        if use_niches:
            niche_cfg = self.cfg.niche
            enabled = getattr(niche_cfg, "enabled", True)
            use_niches = enabled

        print(f"\n[Pipeline] Computing difficulty scores from expression data...")
        print(f"  slides: {len(train_base)} | spots: {self.p1.coords.shape[0]}"
              f" | mode={'niche' if use_niches else 'spot'}")

        # Per-slide difficulty from expression (no training needed)
        difficulty_repo = build_difficulty_repo(
            train_base,
            k_neighbours=self.cfg.difficulty.k_neighbours,
            use_niches=use_niches,
            niche_cfg=self.cfg.niche if use_niches else None,
        )

        # Dummy dynamics — no warm-up training, so epoch_mse is empty.
        # The field below only uses coords; difficulty_score comes from expression.
        from .analysis import DifficultyDynamics
        dynamics = DifficultyDynamics(
            epoch_mse=np.empty((0, self.p1.coords.shape[0]), dtype=np.float32),
            coords=self.p1.coords,
        )

        self.difficulty_analysis_result = run_difficulty_analysis(self.p1, adata,
                                                       dynamics=dynamics,
                                                       k_neighbours=self.cfg.difficulty.k_neighbours,
                                                       n_clusters=self.cfg.difficulty.n_clusters)

        self.difficulty_field = build_difficulty_field(
            dynamics,
            learn_threshold=self.cfg.difficulty.learn_threshold,
            k_neighbours=self.cfg.difficulty.k_neighbours,
            dbscan_eps=self.cfg.difficulty.dbscan_eps,
            interface_percentile=self.cfg.difficulty.interface_pct,
            expression=np.asarray(adata.X),
            skip_topology=True,
        )
        self._field = self.difficulty_field["field"]

        mean_d = self.difficulty_field["stats"]["mean_difficulty"]
        std_d  = self.difficulty_field["stats"]["std_difficulty"]
        print(f"[Pipeline] Difficulty score: mean={mean_d:.3f}  std={std_d:.3f}")


        model, log = train_curriculum(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            difficulty_repo,
            cfg=self.cfg,
            total_epochs=self.cfg.training.total_epochs,
            device=device,
            init_checkpoint=self.cfg.checkpoint.init_checkpoint,
        )
        
        self._training_log = log
        
        baseline_model = self.train_baseline(
            baseline_model, baseline_optimizer, loss_fn, train_loader, val_loader,
        )

        evaluation_result = run_evaluation(
            model,
            baseline_model,
            test_loader,
            self._field,
            self._training_log,
            device,
            bio_annotations=bio_annotations,
            hard_percentile=self.cfg.difficulty.hard_percentile,
        )
        return evaluation_result
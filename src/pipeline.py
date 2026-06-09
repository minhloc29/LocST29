from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import anndata as ad
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from .utils import Phase1Results, _seed_everything, _make_seeded_generator, move_to_device
from .analysis import (
    DifficultyDynamics, EpochMSECallback, run_difficulty_analysis
)
from .dynamics import build_difficulty_field, SpatialDynamicsField, TopologyResult
from .curriculum import (
    train_curriculum, TrainingLog,
    save_init_checkpoint, load_init_checkpoint, reset_optimizer,
    evaluate as _evaluate,
)
from .evaluation import (
    BiologicalAnnotations, EvaluationReport, run_evaluation
)

def build_difficulty_repo(
    model: nn.Module,
    train_base,
    device: torch.device,
    warmup_passes: int,
    learn_threshold: float = 0.3,
) -> dict:
    """
    For each training slide, run warmup_passes forward passes and
    compute a difficulty_score vector of shape [N_spots].

    Returns
    -------
    difficulty_repo : {slide_idx (int): np.ndarray shape [N_spots]}
    """
    from .dataset import build_slide_loader
    from .dynamics import build_dynamics_field
    from .analysis import EpochMSECallback

    repo = {}
    model.eval()

    print(f"[DifficultyRepo] Building per-slide difficulty "
          f"({len(train_base)} slides × {warmup_passes} passes)...")

    for slide_idx in range(len(train_base)):
        slide_name = train_base.names[slide_idx]
        n_spots    = len(train_base.meta_dict[slide_name])
        coords     = train_base.center_dict[slide_name].astype(float)
        loader     = build_slide_loader(train_base, slide_index=slide_idx, batch_size=1)

        cb = EpochMSECallback(n_spots=n_spots)
        for _ in range(warmup_passes):
            cb.record(model, loader, device)

        dynamics = cb.to_dynamics(coords=coords)
        field    = build_dynamics_field(dynamics, learn_threshold=learn_threshold)
        repo[slide_idx] = field.difficulty_score  # [N_spots]

        fnl   = float((field.T_L == field.T).mean())
        d_bar = float(field.D_bar.mean())
        print(f"  Slide {slide_name:>4} | N={n_spots:>3} | "
              f"mean_score={field.difficulty_score.mean():.3f} | "
              f"mean_D_bar={d_bar:.3f} | frac_never_learned={fnl:.2f}")

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

        print("\n[Pipeline] Warm-up training to collect difficulty dynamics...")
        warmup_epochs = max(1, int(self.cfg.training.total_epochs * self.cfg.curriculum.warmup_ratio))
        print(f"  warm-up epochs: {warmup_epochs} "
              f"(ratio={self.cfg.curriculum.warmup_ratio}, total={self.cfg.training.total_epochs})")

        cb = EpochMSECallback(n_spots=self.p1.coords.shape[0])
        model.to(device)

        warmup_val_losses = []
        
        for ep in range(warmup_epochs):
            _train_one_epoch(
                model, train_loader, optimizer, loss_fn,
                device, max_grad_norm=self.cfg.training.max_grad_norm,
            )
            cb.record(model, val_loader, device)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  [Warm-up] Epoch {ep + 1}/{warmup_epochs}")

            val_loss = _evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            warmup_val_losses.append(float(val_loss))
        
        warmup_passes = max(20, warmup_epochs // max(len(train_base), 1))
        
        difficulty_repo = build_difficulty_repo(
            model, train_base, device,
            warmup_passes=warmup_passes,
            learn_threshold=self.cfg.difficulty.learn_threshold,
        ) 
        
        dynamics = cb.to_dynamics(self.p1.coords)
        self._warmup_val_losses = warmup_val_losses
        
        if len(warmup_val_losses) > 5:

            start_loss = warmup_val_losses[0]
            end_loss = warmup_val_losses[-1]

            improvement = (
                start_loss - end_loss
            ) / max(start_loss, 1e-8)

            print(
                f"[Warm-up] Validation improvement: "
                f"{100*improvement:.1f}%"
            )

            if improvement < 0.05:
                print(
                    "[Warm-up] WARNING: "
                    "Validation loss barely improved."
                )
                

        if self.cfg.checkpoint.init_checkpoint is not None:
            save_init_checkpoint(model, optimizer, self.cfg.checkpoint.init_checkpoint)
            print(f"[Pipeline] Shared init checkpoint saved → {self.cfg.checkpoint.init_checkpoint}")
        else:
            print(
                "[Pipeline] WARNING: init_checkpoint is None — curriculum and "
                "baseline will start from different weights."
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
        )
        self._field = self.difficulty_field["field"]
        

        fnl = self.difficulty_field["stats"]["frac_never_learned"]
        if fnl > self.cfg.difficulty.max_never_learned_frac:
            print(
                f"\n[Pipeline] WARNING: {fnl:.1%} of spots never learned during "
                f"warm-up (limit: {self.cfg.difficulty.max_never_learned_frac:.0%}).\n"
                f"  The difficulty map D(x,y,t) is likely unreliable — the\n"
                f"  curriculum will sort spots by noise, not biology.\n"
                f"  Fixes:\n"
                f"    • Increase warmup_ratio (current: {self.cfg.curriculum.warmup_ratio})\n"
                f"    • Lower learn_threshold (current: {self.cfg.difficulty.learn_threshold})\n"
                f"    • Check the model is actually converging during warm-up\n"
                f"  Continuing — inspect results carefully.\n"
            )


        model, log = train_curriculum(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            self._field,
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
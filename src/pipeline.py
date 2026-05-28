from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import anndata as ad
from dataclasses import dataclass, field
from typing import Optional, Callable

from .utils import Phase1Results
from .analysis.analysis import (
    DifficultyDynamics, EpochMSECallback, run_phase2
)
from .dynamics.dynamics import run_phase3, SpatialDynamicsField
from .curriculum.curriculum import Config, run_phase4, TrainingLog
from .evaluation import (
    BiologicalAnnotations, EvaluationReport, run_phase5
)
from .utils import move_to_device


@dataclass
class PipelineConfig:
    """Unified config for the full pipeline."""
    # Phase 2
    k_neighbours: int = 6
    n_clusters: int = 4

    # Phase 3
    learn_threshold: float = 0.3
    dbscan_eps: float = 50.0
    interface_pct: float = 80.0

    # Phase 4
    curriculum: Config = field(default_factory=Config)
    total_epochs: int = 50

    # Phase 5
    hard_percentile: float = 75.0

    # General
    device: str = "cpu"
    seed: int = 42


class SpatialCurriculumPipeline:
    """
    Orchestrates Phases 2-5 of the Spatially Adaptive Curriculum project.

    Parameters
    ----------
    p1  : Phase1Results - your existing Phase 1 outputs.
    cfg : PipelineConfig.
    """

    def __init__(self, p1: Phase1Results, cfg: Optional[PipelineConfig] = None):
        self.p1 = p1
        self.cfg = cfg or PipelineConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self._phase2_results = None
        self._phase3_results = None
        self._training_log: Optional[TrainingLog] = None
        self._field: Optional[SpatialDynamicsField] = None

    # ------------------------------------------------------------------
    # Phase 2 - only needs adata; no model training yet
    # ------------------------------------------------------------------

    def phase2(self, adata: ad.AnnData, dynamics: Optional[DifficultyDynamics] = None):
        """
        Run Phase 2: compute static factors and (optional) temporal dynamics.

        Pass dynamics if you have already collected epoch-wise MSE from a
        preliminary training run. Otherwise factors are computed and dynamics
        analysis is deferred until after Phase 4 training.
        """
        self._phase2_results = run_phase2(
            self.p1,
            adata,
            dynamics=dynamics,
            k_neighbours=self.cfg.k_neighbours,
            n_clusters=self.cfg.n_clusters,
        )
        return self._phase2_results

    # ------------------------------------------------------------------
    # Phase 3 - needs DifficultyDynamics
    # ------------------------------------------------------------------

    def phase3(self, dynamics: DifficultyDynamics):
        self._phase3_results = run_phase3(
            dynamics,
            learn_threshold=self.cfg.learn_threshold,
            k_neighbours=self.cfg.k_neighbours,
            dbscan_eps=self.cfg.dbscan_eps,
            interface_percentile=self.cfg.interface_pct,
        )
        self._field = self._phase3_results["field"]
        return self._phase3_results

    # ------------------------------------------------------------------
    # Phase 4 - train with curriculum
    # ------------------------------------------------------------------

    def phase4(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_loader,
        val_loader,
    ):
        assert self._field is not None, "Run phase3() first."
        device = torch.device(self.cfg.device)
        model, log = run_phase4(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            self._field,
            cfg=self.cfg.curriculum,
            total_epochs=self.cfg.total_epochs,
            device=device,
        )
        self._training_log = log
        return model, log

    # ------------------------------------------------------------------
    # Phase 5 - evaluate
    # ------------------------------------------------------------------

    def phase5(
        self,
        curriculum_model: nn.Module,
        baseline_model: nn.Module,
        test_loader,
        bio_annotations: Optional[BiologicalAnnotations] = None,
    ) -> EvaluationReport:
        assert self._field is not None, "Run phase3() first."
        assert self._training_log is not None, "Run phase4() first."
        device = torch.device(self.cfg.device)
        return run_phase5(
            curriculum_model,
            baseline_model,
            test_loader,
            self._field,
            self._training_log,
            device,
            bio_annotations=bio_annotations,
            hard_percentile=self.cfg.hard_percentile,
        )

    # ------------------------------------------------------------------
    # Convenience: run everything in one call
    # ------------------------------------------------------------------

    def run(
        self,
        adata: ad.AnnData,
        model: nn.Module,
        baseline_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_loader,
        val_loader,
        test_loader,
        bio_annotations: Optional[BiologicalAnnotations] = None,
    ) -> EvaluationReport:
        """
        Full pipeline: Phase 2 -> warm-up training to collect dynamics ->
        Phase 3 -> Phase 4 curriculum training -> Phase 5 evaluation.

        NOTE: This performs a short warm-up run with the provided model to
        collect DifficultyDynamics before switching to curriculum training.
        If you already have dynamics from Phase 1 / a separate run, call
        each phase method individually instead.
        """
        device = torch.device(self.cfg.device)

        # Warm-up: collect dynamics (short run, no curriculum)
        print("\n[Pipeline] Warm-up training to collect difficulty dynamics...")
        warmup_epochs = min(10, self.cfg.total_epochs // 5)
        cb = EpochMSECallback(n_spots=self.p1.coords.shape[0])
        model.to(device)
        model.train()
        for _ in range(warmup_epochs):
            for x, y, _ in train_loader:
                optimizer.zero_grad()
                x = move_to_device(x, device)
                loss = loss_fn(model(x), y.to(device))
                loss.backward()
                optimizer.step()
            cb.record(model, val_loader, device)

        dynamics = cb.to_dynamics(self.p1.coords)

        # Phase 2
        print("\n[Pipeline] -- Phase 2 --")
        self.phase2(adata, dynamics=dynamics)

        # Phase 3
        print("\n[Pipeline] -- Phase 3 --")
        self.phase3(dynamics)

        # Phase 4
        print("\n[Pipeline] -- Phase 4 --")
        model, _ = self.phase4(model, optimizer, loss_fn, train_loader, val_loader)

        # Phase 5
        print("\n[Pipeline] -- Phase 5 --")
        return self.phase5(model, baseline_model, test_loader, bio_annotations)

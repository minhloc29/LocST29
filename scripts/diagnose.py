"""
Diagnostic script: trace difficulty score computation and evaluation step by step.
Run this after training to see where the pipeline behaves unexpectedly.

Usage:
  python scripts/diagnose.py --checkpoint_dir /path/to/output --config /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config.my_config import load_config
from src import (
    DataConfig,
    SpatialCurriculumTrainer,
    SpatialModelAdapter,
    MultiSlideAdapter,
    build_slide_loader,
    load_dataset,
    prepare_phase1,
)
from src.dynamics import (
    SpatialDynamicsField,
    build_difficulty_field,
    build_dynamics_field,
    summarise_field,
)
from src.difficulty_gse import topological_difficulty_from_data, compare_difficulty_scores
from src.evaluation import (
    run_evaluation,
    compute_paper_metrics,
    boundary_hard_evaluation,
    difficulty_error_correlation,
    compute_error_vector,
    mse,
    pearson_correlation_coefficient,
    predict_all,
    reorder_by_spot,
)


def build_model(module_path, class_name, kwargs=None):
    from importlib import import_module
    if kwargs is None:
        kwargs = {}
    elif hasattr(kwargs, "__dict__"):
        kwargs = vars(kwargs)
    cls = getattr(import_module(module_path), class_name)
    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.training.device)
    out_dir = Path(args.checkpoint_dir)

    # ── 1. Load data ──
    print("=" * 70)
    print("STEP 1: Load data")
    print("=" * 70)
    data_root = Path(cfg.dataset.data_root).resolve() if cfg.dataset.data_root else None
    data_cfg = DataConfig(
        dataset=cfg.dataset.name, fold=cfg.dataset.fold,
        adj=True, flatten=cfg.dataset.flatten, data_root=data_root,
    )
    train_base = load_dataset(data_cfg, train=True)
    test_base = load_dataset(data_cfg, train=False)

    # Only one slide? Get the first slide's data
    slide_idx = 0
    slide_name = test_base.names[slide_idx]
    expression = test_base.exp_dict[slide_name]
    coords = test_base.center_dict[slide_idx].astype(float)
    N = coords.shape[0]
    print(f"  Slide: {slide_name} | N={N} spots | G={expression.shape[1]} genes")
    print(f"  Expression: min={expression.min():.2f} max={expression.max():.2f} "
          f"mean={expression.mean():.4f} std={expression.std():.4f}")

    # ── 2. Compute difficulty scores directly ──
    print("\n" + "=" * 70)
    print("STEP 2: topological_difficulty_from_data()")
    print("=" * 70)
    diff = topological_difficulty_from_data(expression, coords, k_neighbours=6)
    print(f"  difficulty_score: min={diff.min():.6f} max={diff.max():.6f} "
          f"mean={diff.mean():.6f} std={diff.std():.6f}")
    print(f"  unique values: {len(np.unique(diff))} / {N}")
    for q in [1, 5, 10, 25, 50, 75, 90, 99]:
        v = np.percentile(diff, q)
        print(f"  P{q:3d}: {v:.6f}")

    # Check the hard_region_fraction issue
    p75 = np.percentile(diff, 75, method='lower')
    hard_frac = (diff > p75).mean()
    print(f"\n  P75 (lower method)={p75:.6f}, hard_region_fraction={hard_frac:.3f}")
    print(f"  Count at P75 value: {(diff == p75).sum()} spots")

    # Histogram
    print(f"\n  Histogram (10 bins):")
    hist, edges = np.histogram(diff, bins=10)
    for i in range(len(hist)):
        bar = "#" * (hist[i] // 2)
        print(f"    [{edges[i]:.4f}, {edges[i+1]:.4f}): {hist[i]:>4}  {bar}")

    # ── 3. Compute per-spot error correlation ──
    print("\n" + "=" * 70)
    print("STEP 3: Predictions from saved checkpoints")
    print("=" * 70)

    # Build model
    model = build_model(cfg.model.module, cfg.model.class_name, cfg.model.kwargs)
    baseline_model = build_model(cfg.model.module, cfg.model.class_name, cfg.model.kwargs)
    if cfg.pipeline.wrap_model:
        model = SpatialModelAdapter(model)
        baseline_model = SpatialModelAdapter(baseline_model)

    # Load checkpoints
    ckpt_path = out_dir / "curriculum_model.pt"
    base_ckpt_path = out_dir / "baseline_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"  Loaded curriculum model from {ckpt_path}")
    else:
        print(f"  WARNING: {ckpt_path} not found — using random weights")

    if base_ckpt_path.exists():
        baseline_model.load_state_dict(torch.load(base_ckpt_path, map_location="cpu"))
        print(f"  Loaded baseline model from {base_ckpt_path}")
    else:
        print(f"  WARNING: {base_ckpt_path} not found — using random weights")

    model.to(device)
    baseline_model.to(device)

    # Build loaders
    val_loader = build_slide_loader(
        test_base, slide_index=cfg.dataset.test_slide_index,
        batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
    )

    # Run inference
    model.eval()
    baseline_model.eval()
    cp, ct, ci = predict_all(model, val_loader, device)
    bp, bt, bi = predict_all(baseline_model, val_loader, device)
    cp = reorder_by_spot(cp, ci, N)
    ct = reorder_by_spot(ct, ci, N)
    bp = reorder_by_spot(bp, bi, N)
    bt = reorder_by_spot(bt, bi, N)

    print(f"\n  Predictions shape: {cp.shape}")
    print(f"  Targets shape: {ct.shape}")
    print(f"  Model MSE: {mse(cp, ct):.6f}")

    # Per-spot error
    per_spot_err = compute_error_vector(cp, ct)
    print(f"  Per-spot error: min={per_spot_err.min():.4f} max={per_spot_err.max():.4f} "
          f"mean={per_spot_err.mean():.4f} std={per_spot_err.std():.4f}")

    # ── 4. Difficulty-Error Correlation ──
    print("\n" + "=" * 70)
    print("STEP 4: Difficulty-Error Correlation")
    print("=" * 70)
    from scipy.stats import pearsonr, spearmanr

    # Group by difficulty quartile and show mean error
    ds = diff
    for q_low, q_high, label in [(0, 25, "Easiest 0-25%"),
                                  (25, 50, "Easy 25-50%"),
                                  (50, 75, "Hard 50-75%"),
                                  (75, 100, "Hardest 75-100%")]:
        lo = np.percentile(ds, q_low)
        hi = np.percentile(ds, q_high)
        mask = (ds >= lo) & (ds < hi) if q_high < 100 else (ds >= lo) & (ds <= hi)
        if mask.sum() > 0:
            print(f"  {label:>20}: n={mask.sum():>4}  "
                  f"mean_error={per_spot_err[mask].mean():.4f}  "
                  f"mean_difficulty={ds[mask].mean():.4f}  "
                  f"model_PCC={pearson_correlation_coefficient(cp[mask], ct[mask]):.4f}")

    r_p, _ = pearsonr(ds, per_spot_err)
    r_s, _ = spearmanr(ds, per_spot_err)
    print(f"\n  Pearson:  {r_p:.4f}")
    print(f"  Spearman: {r_s:.4f}")

    # ── 5. Compare curriculum vs baseline per difficulty bin ──
    print("\n" + "=" * 70)
    print("STEP 5: Per-bin performance (curriculum vs baseline)")
    print("=" * 70)
    n_bins = 5
    sorted_idx = np.argsort(ds)
    bin_size = N // n_bins
    for i in range(n_bins):
        start = i * bin_size
        end = N if i == n_bins - 1 else start + bin_size
        bin_idx = sorted_idx[start:end]

        c_mse = mse(cp[bin_idx], ct[bin_idx])
        b_mse = mse(bp[bin_idx], bt[bin_idx])
        c_pcc = pearson_correlation_coefficient(cp[bin_idx], ct[bin_idx])
        b_pcc = pearson_correlation_coefficient(bp[bin_idx], bt[bin_idx])

        labels = ["Easiest", "20-40%", "40-60%", "60-80%", "Hardest"]
        print(f"  {labels[i]:>10} (n={len(bin_idx):>4}): "
              f"curr_MSE={c_mse:.4f} base_MSE={b_mse:.4f} "
              f"gain_MSE={b_mse-c_mse:+.4f} | "
              f"curr_PCC={c_pcc:.4f} base_PCC={b_pcc:.4f} "
              f"gain_PCC={c_pcc-b_pcc:+.4f}")

    # ── 6. Check the field used in evaluation ──
    print("\n" + "=" * 70)
    print("STEP 6: Build SpatialDynamicsField (expression path)")
    print("=" * 70)
    from src.analysis import DifficultyDynamics

    # Create a dummy dynamics to exercise the expression path
    dummy_dynamics = DifficultyDynamics(
        epoch_mse=np.empty((0, N), dtype=np.float32),
        coords=coords,
        N=N,
    )
    result = build_difficulty_field(
        dummy_dynamics,
        expression=expression,
        k_neighbours=6,
    )
    field = result["field"]
    print(f"  field.difficulty_score: min={field.difficulty_score.min():.4f} "
          f"max={field.difficulty_score.max():.4f} "
          f"mean={field.difficulty_score.mean():.4f} "
          f"std={field.difficulty_score.std():.6f}")
    print(f"  field.coords: {field.coords.shape}")
    print(f"  field.N: {field.N}")

    # ── 7. boundary_hard_evaluation ──
    print("\n" + "=" * 70)
    print("STEP 7: boundary_hard_evaluation()")
    print("=" * 70)
    ds_f = field.difficulty_score
    p75_f = np.percentile(ds_f, 75)
    hard_mask = ds_f > p75_f
    print(f"  difficulty_score: min={ds_f.min():.4f} max={ds_f.max():.4f} "
          f"mean={ds_f.mean():.4f} std={ds_f.std():.6f}")
    print(f"  P75={p75_f:.6f}, hard_mask sum={hard_mask.sum()}/{N} = {hard_mask.mean():.3f}")
    for q in [10, 25, 50, 75, 90, 99]:
        v = np.percentile(ds_f, q)
        print(f"  P{q:3d}: {v:.6f}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Check 1: difficulty score spread
    if diff.std() < 0.01:
        print("❌ Issue 1: Difficulty scores have very low variance — "
              "almost all spots look equally 'hard'.")
    else:
        print("✅ Diff scores spread is reasonable.")

    # Check 2: hard_region_fraction
    if hard_frac > 0.35:
        print(f"❌ Issue 2: hard_region_fraction={hard_frac:.2f} (should be ~0.25). "
              "Too many ties at the percentile boundary.")
    else:
        print("✅ hard_region_fraction looks normal.")

    # Check 3: correlation sign
    if r_p > 0:
        print(f"✅ Difficulty-Error correlation is positive ({r_p:.3f}) — as expected.")
    else:
        print(f"❌ Issue 3: Difficulty-Error correlation is NEGATIVE ({r_p:.3f}). "
              "Your difficulty metric inversely correlates with actual error.")

    # Check 4: curriculum vs baseline
    print(f"✅ Curriculum vs Baseline comparison available — check per-bin gains above.")

    print("\nRecommendations:\n"
          "1. If diff std < 0.01: add more signal to difficulty computation\n"
          "   (try graph_signal_energy_difficulty, or use more neighbours)\n"
          "2. If correlation is negative: the difficulty metric captures the wrong\n"
          "   thing — try GSE-based difficulty from build_difficulty_field()\n"
          "3. If hard_region_fraction > 0.4: fix percentile comparison to use\n"
          "   >= instead of >, or add jitter, or use deciles instead of percentile\n"
          "4. To get baseline AULC: save a TrainingLog from train_baseline()")


if __name__ == "__main__":
    main()

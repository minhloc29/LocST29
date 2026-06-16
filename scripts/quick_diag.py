"""
Quick diagnostic for difficulty score distribution.
Usage: python -m scripts.quick_diag --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from config.my_config import load_config
from src import DataConfig, load_dataset
from src.difficulty_gse import topological_difficulty_from_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_root = Path(cfg.dataset.data_root).resolve() if cfg.dataset.data_root else None
    data_cfg = DataConfig(
        dataset=cfg.dataset.name, fold=cfg.dataset.fold,
        adj=True, flatten=cfg.dataset.flatten, data_root=data_root,
    )

    # Load train data — test might be empty
    ds = load_dataset(data_cfg, train=True)

    if len(ds.names) == 0:
        print("No slides found!")
        return

    for slide_name in ds.names:
        print(f"\n{'='*60}")
        print(f"Slide: {slide_name}")
        print(f"{'='*60}")

        expr = ds.exp_dict[slide_name]
        coords = ds.center_dict[slide_name].astype(float)
        N = coords.shape[0]

        print(f"  Spots: {N},  Genes: {expr.shape[1]}")
        print(f"  NaNs in expression: {np.isnan(expr).sum()} / {expr.size}")
        print(f"  NaNs in coords:     {np.isnan(coords).sum()} / {coords.size}")

        # Clean expression
        expr_clean = np.nan_to_num(expr, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Difficulty score ---
        diff = topological_difficulty_from_data(expr_clean, coords, k_neighbours=6)
        print(f"\n  >>> topological_difficulty_from_data <<<")
        print(f"    min={diff.min():.6f}  max={diff.max():.6f}  "
              f"mean={diff.mean():.6f}  std={diff.std():.6f}")
        print(f"    unique values: {len(np.unique(diff))} / {N}")

        # Histogram
        hist, edges = np.histogram(diff, bins=10)
        print(f"    Histogram:")
        for i in range(len(hist)):
            bar = "█" * max(1, hist[i] // max(1, max(hist) // 30))
            print(f"      [{edges[i]:.4f}, {edges[i+1]:.4f}): {hist[i]:>4}  {bar}")

        # Percentiles
        for q in [1, 5, 10, 25, 50, 75, 90, 99]:
            v = np.percentile(diff, q)
            print(f"    P{q:3d}: {v:.6f}")

        # Hard region check
        n_hard = int(N * 0.25)
        sorted_idx = np.argsort(diff)
        hard_idx = sorted_idx[-n_hard:]
        hard_thresh = diff[hard_idx[0]] if n_hard > 0 else 0
        print(f"\n    Top-25% hardest threshold: {hard_thresh:.6f}")
        print(f"    Spots at or above threshold: {(diff >= hard_thresh).sum()} / {N}")

    print("\nDone.")


if __name__ == "__main__":
    main()

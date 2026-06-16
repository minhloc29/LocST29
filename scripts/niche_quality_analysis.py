"""
Analyse niche quality from a curriculum checkpoint.

Usage:
    python -m scripts.niche_quality_analysis --checkpoint <path> [--n_niches 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _load_niche_module():
    """Load src/niche.py bypassing src/__init__.py (which pulls in anndata)."""
    src = Path(__file__).resolve().parent.parent / "src"
    import importlib.util

    # Pre-load difficulty_gse so niche's relative import resolves
    dgse_spec = importlib.util.spec_from_file_location(
        "difficulty_gse", src / "difficulty_gse.py"
    )
    dgse_mod = importlib.util.module_from_spec(dgse_spec)
    sys.modules["src.difficulty_gse"] = dgse_mod
    dgse_spec.loader.exec_module(dgse_mod)

    niche_spec = importlib.util.spec_from_file_location(
        "src.niche", src / "niche.py"
    )
    niche_mod = importlib.util.module_from_spec(niche_spec)
    niche_mod.__package__ = "src"
    sys.modules["src.niche"] = niche_mod
    niche_spec.loader.exec_module(niche_mod)

    return niche_mod


def main():
    parser = argparse.ArgumentParser(description="Niche quality analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to curriculum checkpoint (.pt)")
    parser.add_argument("--n_niches", type=int, default=12,
                        help="Number of niches for k-means")
    parser.add_argument("--output", type=str, default="niche_quality.png",
                        help="Output path for the dashboard figure")
    parser.add_argument("--method", type=str, default="spatial_kmeans",
                        choices=["spatial_kmeans", "spatial_leiden"],
                        help="Niche construction method")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[Data] Loaded checkpoint: {args.checkpoint}")

    # Extract expression & coords — try common keys
    for key in ("expression", "expr", "X"):
        if key in ckpt:
            expr = ckpt[key]
            break
    else:
        expr = None

    for key in ("coords", "coordinates", "pos"):
        if key in ckpt:
            coords = ckpt[key]
            break
    else:
        coords = None

    if expr is None or coords is None:
        # Show available keys
        keys = [k for k in ckpt.keys() if isinstance(ckpt[k], np.ndarray)]
        print(f"Available array keys: {keys}")
        raise KeyError("Could not find expression/coords in checkpoint. "
                       "Pick the right keys and pass via --expr_key --coord_key")

    # Convert tensors to numpy
    if hasattr(expr, "numpy"):
        expr = expr.numpy()
    if hasattr(coords, "numpy"):
        coords = coords.numpy()

    expr = np.asarray(expr, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    print(f"[Data] Expression: {expr.shape},  Coords: {coords.shape}")

    # Load niche module and run analysis
    niche = _load_niche_module()
    labels = niche.build_spatial_niches(expr, coords, method=args.method, n_niches=args.n_niches)
    niche_scores, spot_scores = niche.compute_niche_difficulty(expr, coords, labels)
    niche.summarise_niches(labels, niche_scores)

    # Dashboard
    niche.plot_niche_quality(
        expr, coords, labels, niche_scores, spot_scores,
        save_path=args.output, show=False,
    )
    print(f"[Done] Dashboard saved to {args.output}")


if __name__ == "__main__":
    main()

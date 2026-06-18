
from __future__ import annotations
from src.niche import *
from src.dataset import *


import numpy as np
import torch

def _load_from_dataset(dataset_name: str, slide_index: int):

    cfg = DataConfig(
        dataset=dataset_name,
        fold=1, r=4, flatten=False, ori=False, adj=False
    )

    base = load_dataset(cfg, train=True)
    slide_name = base.names[slide_index]
    print(f"[Data] Slide: {slide_name}")

    expr = base.exp_dict[slide_name]  # (N, G) numpy
    coords = base.loc_dict[slide_name]  # (N, 2) numpy

    return np.asarray(expr, dtype=np.float32), np.asarray(coords, dtype=np.float32)






def main():

    checkpoint_path = r"runs\run1\curriculum_model.pt"
    n_niches = 12
    output = "niche_quality.png" 
    method = "spatial_leiden"
    dataset = "her2st"
    slide = 0
    
    expr, coords = _load_from_dataset(dataset, slide)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)


    if hasattr(expr, "numpy"):
        expr = expr.numpy()
    if hasattr(coords, "numpy"):
        coords = coords.numpy()

    expr = np.asarray(expr, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    print(f"[Data] Expression: {expr.shape},  Coords: {coords.shape}")

    labels = build_spatial_niches(expr, coords, method=method, n_niches=n_niches)
    niche_scores, spot_scores = compute_niche_difficulty(expr, coords, labels)
    summarise_niches(labels, niche_scores)

    plot_niche_quality(
        expr, coords, labels, niche_scores, spot_scores,
        save_path=output, show=False,
    )
    print(f"[Done] Dashboard saved to {output}")


if __name__ == "__main__":
    main()

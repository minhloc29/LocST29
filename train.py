from __future__ import annotations

import argparse
import shutil
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader


from src import (
    DataConfig,
    SpatialCurriculumTrainer,
    SpatialModelAdapter,
    MultiSlideAdapter,
    build_slide_loader,
    load_dataset,
    prepare_phase1,
)
from config.my_config import load_config



def build_model(
    module_path: str,
    class_name: str,
    kwargs=None
) -> nn.Module:

    if kwargs is None:
        kwargs = {}

    elif hasattr(kwargs, "__dict__"):
        kwargs = vars(kwargs)

    cls = getattr(import_module(module_path), class_name)

    return cls(**kwargs)


def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spatial curriculum training pipeline.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)


    data_root = Path(cfg.dataset.data_root).resolve() if cfg.dataset.data_root else None
    data_cfg = DataConfig(
        dataset=cfg.dataset.name,
        fold=cfg.dataset.fold,
        adj=True,
        flatten=cfg.dataset.flatten,
        data_root=data_root,
    )

    train_base = load_dataset(data_cfg, train=True)
    test_base  = load_dataset(data_cfg, train=False)

    # train_loader = build_slide_loader(
    #     train_base, slide_index=cfg.dataset.train_slide_index,
    #     batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
    # )
 
    train_loader = DataLoader(MultiSlideAdapter(train_base), batch_size=cfg.training.batch_size, shuffle=True)
    
    val_loader = build_slide_loader(
        test_base, slide_index=cfg.dataset.test_slide_index,
        batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
    )
    test_loader = build_slide_loader(
        test_base, slide_index=cfg.dataset.test_slide_index,
        batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
    )


    model_kwargs = vars(cfg.model.kwargs)
    model = build_model(cfg.model.module, cfg.model.class_name, model_kwargs)
    baseline_model = build_model(cfg.model.module, cfg.model.class_name, cfg.model.kwargs)

    if cfg.pipeline.wrap_model:
        model = SpatialModelAdapter(model)
    
    if cfg.pipeline.wrap_model:
        baseline_model = SpatialModelAdapter(baseline_model)
        
    if cfg.checkpoint.resume:
        model.load_state_dict(torch.load(cfg.checkpoint.resume, map_location="cpu"))


    loss_fn          = nn.MSELoss()
    optimizer        = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=cfg.training.lr)

    print("[Phase 1] Preparing Phase 1 outputs...")
    p1, adata = prepare_phase1(
        cfg=data_cfg,
        model=model,
        slide_index=cfg.dataset.test_slide_index,
        output_dir=cfg.pipeline.output_dir,
        device=cfg.training.device,
    )

    print("[Trainer] Running curriculum training...")
    trainer = SpatialCurriculumTrainer(p1, cfg)
    result = trainer.train(
        model=model,
        baseline_model=baseline_model,
        optimizer=optimizer,
        baseline_optimizer=baseline_optimizer,
        loss_fn=loss_fn,
        train_base=train_base,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    output_dir = Path(cfg.pipeline.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_checkpoint(output_dir)
    shutil.copy(args.config, output_dir / "config.yaml")

    print(f"\nSaved models → {output_dir}")
    print(f"Training log: {len(result['training_log'].train_loss)} epochs")
    print("[Trainer] Done.")


if __name__ == "__main__":
    main()
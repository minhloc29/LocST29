from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from src import (
    DataConfig,
    SpatialCurriculumPipeline,
    PipelineConfig,
    SpatialModelAdapter,
    build_slide_loader,
    load_dataset,
    prepare_phase1,
)

from src.utils import EarlyStopping

def parse_json_arg(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    return json.loads(value)


def build_model(module_path: str, class_name: str, kwargs: Dict[str, Any]) -> nn.Module:
    module = import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def train_baseline(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
) -> None:
    model.to(device)
    model.train()
    for _ in range(epochs):
        for x, y, _ in loader:
            optimizer.zero_grad()
            x = _move_to_device(x, device)
            y = y.to(device)
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]

            if y.ndim == 3 and y.shape[0] == 1:
                y = y.squeeze(0)
            
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            
            assert pred.shape == y.shape, (
                    f"pred={pred.shape}, target={y.shape}"
                )
            
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()


def _move_to_device(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        return tuple(_move_to_device(item, device) for item in batch)
    return batch.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spatial curriculum training pipeline.")
    parser.add_argument("--dataset", default="her2st", choices=["her2st", "cscc"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-slide-index", type=int, default=0)
    parser.add_argument("--test-slide-index", type=int, default=0)
    parser.add_argument("--total-epochs", type=int, default=200)
    parser.add_argument("--baseline-epochs", type=int, default=200)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flatten", action="store_true", default=False)
    parser.add_argument("--no-flatten", action="store_false", dest="flatten")

    parser.add_argument("--model-module", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--model-kwargs", default="")

    parser.add_argument("--baseline-module", default="")
    parser.add_argument("--baseline-class", default="")
    parser.add_argument("--baseline-kwargs", default="")

    parser.add_argument("--wrap-model", action="store_true", default=True)
    parser.add_argument("--no-wrap-model", action="store_false", dest="wrap_model")
    parser.add_argument("--output-dir", default="./runs/run1")
    parser.add_argument("--resume", default="")

    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root).resolve() if args.data_root else None
    data_cfg = DataConfig(
        dataset=args.dataset,
        fold=args.fold,
        adj=True,
        flatten=args.flatten,
        data_root=data_root,
    )

    train_base = load_dataset(data_cfg, train=True)
    test_base = load_dataset(data_cfg, train=False)

    train_loader = build_slide_loader(
        train_base,
        slide_index=args.train_slide_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = build_slide_loader(
        test_base,
        slide_index=args.test_slide_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = build_slide_loader(
        test_base,
        slide_index=args.test_slide_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_kwargs = parse_json_arg(args.model_kwargs)
    model = build_model(args.model_module, args.model_class, model_kwargs)
    if args.wrap_model:
        model = SpatialModelAdapter(model)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state)

    if args.baseline_module and args.baseline_class:
        baseline_kwargs = parse_json_arg(args.baseline_kwargs)
        baseline_model = build_model(args.baseline_module, args.baseline_class, baseline_kwargs)
        if args.wrap_model:
            baseline_model = SpatialModelAdapter(baseline_model)
    else:
        baseline_model = build_model(args.model_module, args.model_class, model_kwargs)
        if args.wrap_model:
            baseline_model = SpatialModelAdapter(baseline_model)

    init_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    
    model.load_state_dict(init_state)
    baseline_model.load_state_dict(init_state)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.baseline_epochs > 0:
        print("[Baseline] Training baseline model...")
        base_opt = torch.optim.Adam(baseline_model.parameters(), lr=args.lr)
        train_baseline(
            model=baseline_model,
            loader=train_loader,
            optimizer=base_opt,
            loss_fn=loss_fn,
            device=torch.device(args.device),
            epochs=args.baseline_epochs,
        )
    elif args.baseline_only:
        raise ValueError("baseline-only mode requires --baseline-epochs > 0")

    if args.baseline_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(baseline_model.state_dict(), output_dir / "baseline_model.pt")
        print("Baseline-only training complete.")
        return

    print("[Phase 1] Preparing Phase 1 outputs...")
    p1, adata = prepare_phase1(
        cfg=data_cfg,
        model=model,
        slide_index=args.test_slide_index,
        output_dir=args.output_dir,
        device=args.device,
    )

    cfg = PipelineConfig(total_epochs=args.total_epochs, device=args.device, seed=args.seed)
    pipe = SpatialCurriculumPipeline(p1, cfg)

    print("[Pipeline] Running curriculum training...")
    report = pipe.run(
        adata=adata,
        model=model,
        baseline_model=baseline_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    torch.save(model.state_dict(), output_dir / "curriculum_model.pt")
    torch.save(baseline_model.state_dict(), output_dir / "baseline_model.pt")

    report.print_summary()
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()







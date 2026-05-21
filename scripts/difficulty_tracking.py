import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
from torch.utils.data import DataLoader

from hist2st.config import TrainConfig
from hist2st.data import pk_load
from hist2st.model import build_model
from hist2st.utils.seeds import set_seed
from hist2st.difficulty.tracking import train_with_difficulty_tracking
from hist2st.difficulty.viz import plot_temporal_dynamics, plot_difficulty_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with temporal difficulty tracking")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="./difficulty_factor")
    parser.add_argument("--tag", type=str, default="5-7-2-8-4-16-32")
    parser.add_argument("--data", type=str, default="her2st")
    parser.add_argument("--neighbor", type=int, default=4)
    parser.add_argument("--prune", type=str, default="Grid")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--zinb", type=float, default=0.25)
    parser.add_argument("--bake", type=int, default=5)
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--policy", type=str, default="mean")
    parser.add_argument("--seed", type=int, default=12000)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    trainset = pk_load(args.fold, "train", False, args.data, neighs=args.neighbor, prune=args.prune)
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
    testset = pk_load(args.fold, "test", False, args.data, neighs=args.neighbor, prune=args.prune)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

    genes = 171 if args.data == "cscc" else 785
    model = build_model(
        tag=args.tag,
        genes=genes,
        lr=args.lr,
        label=None,
        dropout=args.dropout,
        zinb=args.zinb,
        nb=False,
        bake=args.bake,
        lamb=args.lamb,
        policy=args.policy,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    tracker = train_with_difficulty_tracking(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        args.epochs,
        device,
    )

    results = tracker.compute_metrics()
    plot_temporal_dynamics(results, testset, output_dir=args.output_dir)
    plot_difficulty_trajectories(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

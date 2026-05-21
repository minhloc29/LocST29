import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import ViT_HER2ST
from hist2st.model import build_model, load_checkpoint
from hist2st.predict import test_model
from hist2st.difficulty.factors import calculate_factors
from hist2st.difficulty.viz import plot_factor_grid
from hist2st.difficulty.tracking import DifficultyTracker
from hist2st.difficulty.viz import plot_temporal_dynamics, plot_difficulty_trajectories
from hist2st.difficulty import tracking as tracking_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute difficulty factors for Hist2ST")
    parser.add_argument("--folds", type=str, default="1-10", help="Fold range, e.g., 1-10 or 1,3,5")
    parser.add_argument("--tag", type=str, default="5-7-2-8-4-16-32")
    parser.add_argument("--checkpoint", type=str, default="./model/5-Hist2ST.ckpt")
    parser.add_argument("--output-dir", type=str, default="./difficulty_factor")
    parser.add_argument("--patch-radius", type=int, default=56)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--genes", type=int, default=785)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--zinb", type=float, default=0.25)
    parser.add_argument("--bake", type=int, default=5)
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--policy", type=str, default="mean")
    parser.add_argument("--prune", type=str, default="Grid")
    return parser.parse_args()


def parse_folds(folds_arg: str) -> list[int]:
    if "," in folds_arg:
        return [int(x) for x in folds_arg.split(",") if x.strip()]
    if "-" in folds_arg:
        start, end = folds_arg.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(folds_arg)]


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.std(x) < 1e-8:
        return np.nan, 1.0
    from scipy.stats import spearmanr
    return spearmanr(x, y)


def run_fold(args: argparse.Namespace, fold: int) -> dict[str, float]:
    print(f"\nProcessing fold {fold}...")
    testset = ViT_HER2ST(train=False, fold=fold, flatten=False, adj=True, ori=True, prune=args.prune)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

    model = build_model(
        tag=args.tag,
        genes=args.genes,
        lr=1e-5,
        label=None,
        dropout=args.dropout,
        zinb=args.zinb,
        nb=False,
        bake=args.bake,
        lamb=args.lamb,
        policy=args.policy,
    )
    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, resolve_device(args.device))
    else:
        print(f"Warning: checkpoint not found at {args.checkpoint}")
    model.to(resolve_device(args.device))
    model.eval()

    pred, gt = test_model(model, test_loader, resolve_device(args.device))
    H, S, M, B = calculate_factors(testset, args.patch_radius)
    error = np.mean((pred.X - gt.X) ** 2, axis=1)

    r_H, p_H = safe_corr(H, error)
    r_S, p_S = safe_corr(S, error)
    r_M, p_M = safe_corr(M, error)
    r_B, p_B = safe_corr(B, error)

    coords = testset.loc_dict[testset.names[0]]
    factors = {"H": H, "S": S, "M": M, "B": B}
    corrs = {"H": r_H, "S": r_S, "M": r_M, "B": r_B}

    output_file = os.path.join(args.output_dir, f"fold_{fold:02d}_difficulty_factors.png")
    plot_factor_grid(coords, factors, corrs, output_file)
    print(f"Saved: {output_file}")

    return {
        "fold": fold,
        "H_corr": r_H,
        "H_pval": p_H,
        "S_corr": r_S,
        "S_pval": p_S,
        "M_corr": r_M,
        "M_pval": p_M,
        "B_corr": r_B,
        "B_pval": p_B,
    }


def run_all(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    folds = parse_folds(args.folds)

    results = [run_fold(args, fold) for fold in folds]
    df_factors = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("DIFFICULTY FACTOR CORRELATIONS WITH PREDICTION ERROR")
    print("=" * 60)
    print(df_factors.to_string(index=False))

    csv_path = os.path.join(args.output_dir, "difficulty_factor_correlations.csv")
    df_factors.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print("\nMean Correlations:")
    for factor in ["H", "S", "M", "B"]:
        corr_col = f"{factor}_corr"
        pval_col = f"{factor}_pval"
        mean_corr = df_factors[corr_col].mean()
        sig_count = (df_factors[pval_col] < 0.05).sum()
        print(f"  {factor}: mean r={mean_corr:.4f}, significant folds={sig_count}/{len(folds)}")


def main() -> None:
    run_all(parse_args())

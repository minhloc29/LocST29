import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
from hist2st.model import build_model, load_checkpoint
from hist2st.predict import test_model
from dataset import ViT_HER2ST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch fold analysis with spatial statistics")
    parser.add_argument("--folds", type=str, default="1-10")
    parser.add_argument("--tag", type=str, default="5-7-2-8-4-16-32")
    parser.add_argument("--checkpoint", type=str, default="./model/5-Hist2ST.ckpt")
    parser.add_argument("--output-dir", type=str, default="./fold_analysis_results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prune", type=str, default="Grid")
    parser.add_argument("--genes", type=int, default=785)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--zinb", type=float, default=0.25)
    parser.add_argument("--bake", type=int, default=5)
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--policy", type=str, default="mean")
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


def compute_lisa(errors: np.ndarray, W_dense: np.ndarray) -> np.ndarray:
    n = len(errors)
    e = errors - errors.mean()
    variance = np.sum(e ** 2) / n
    lisa = np.zeros(n)
    for i in range(n):
        lisa[i] = (e[i] * np.sum(W_dense[i] * e)) / variance
    return lisa


def compute_morans_i(errors: np.ndarray, W_dense: np.ndarray, n_perm: int = 999) -> tuple[float, float]:
    n = len(errors)
    W_sum = W_dense.sum()
    e = errors - errors.mean()
    numerator = e @ W_dense @ e
    denominator = np.sum(e ** 2)
    morans_i = (n / W_sum) * (numerator / denominator)

    pvals = [
        (n / W_sum) * ((np.random.permutation(e) @ W_dense @ np.random.permutation(e)) / denominator)
        for _ in range(n_perm)
    ]
    pvals = np.array(pvals)
    p_value = (np.sum(pvals >= morans_i) + 1) / (len(pvals) + 1)
    return morans_i, p_value


def run_fold(args: argparse.Namespace, fold: int) -> dict[str, float]:
    device = resolve_device(args.device)
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
        load_checkpoint(model, args.checkpoint, device)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.to(device)
    model.eval()

    pred, gt = test_model(model, test_loader, device)
    sample = testset.names[0]
    coords = testset.loc_dict[sample]
    W_dense = np.asarray(testset.adj_dict[sample])
    error = np.mean((pred.X - gt.X) ** 2, axis=1)

    morans_i, p_val = compute_morans_i(error, W_dense)
    lisa = compute_lisa(error, W_dense)

    r_vals = [
        pearsonr(pred.X[:, g], gt.X[:, g])[0]
        for g in range(pred.shape[1])
        if not (np.isnan(pred.X[:, g]).any() or np.isnan(gt.X[:, g]).any())
    ]

    return {
        "fold": fold,
        "sample": sample,
        "n_spots": len(error),
        "mean_error": float(np.mean(error)),
        "std_error": float(np.std(error)),
        "median_error": float(np.median(error)),
        "morans_i": float(morans_i),
        "morans_p_value": float(p_val),
        "sig_clustered": "Yes" if p_val < 0.05 else "No",
        "lisa_mean": float(np.mean(lisa)),
        "lisa_high_spots": int(np.sum(lisa > np.percentile(lisa, 95))),
        "pearson_r": float(np.nanmean(r_vals)),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    folds = parse_folds(args.folds)

    results = []
    print("\n" + "=" * 80)
    print("BATCH FOLD ANALYSIS")
    print("=" * 80)

    for idx, fold in enumerate(folds):
        print(f"\n[{idx + 1}/{len(folds)}] Fold {fold}...", end=" ")
        row = run_fold(args, fold)
        results.append(row)
        print(f"Morans I={row['morans_i']:.4f}, p={row['morans_p_value']:.4f}")

    df = pd.DataFrame(results).sort_values("fold")
    csv_path = os.path.join(args.output_dir, "fold_statistics.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE:")
    print(df.to_string(index=False))
    print("\n" + "=" * 80)
    print(f"Mean Error:      {df['mean_error'].mean():.6f} ± {df['mean_error'].std():.6f}")
    print(f"Moran's I:       {df['morans_i'].mean():.4f} ± {df['morans_i'].std():.4f}")
    print(f"Sig. clustered:  {(df['morans_p_value'] < 0.05).sum()} / {len(df)} folds")
    print(f"Pearson R:       {df['pearson_r'].mean():.4f} ± {df['pearson_r'].std():.4f}")
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()

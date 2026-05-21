"""
Batch Analysis: Error heatmaps + Spatial Statistics (Moran's I, LISA, p-values) for all folds
Save as: batch_fold_analysis.py
Run: python batch_fold_analysis.py
"""

import os, sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from predict import test
from HIST2ST import Hist2ST
from dataset import ViT_HER2ST
from graph_construction import calcADJ

# ============== CONFIG ==============
folds = list(range(1, 20))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tag = '5-7-2-8-4-16-32'
k, p, d1, d2, d3, h, c = map(lambda x: int(x), tag.split('-'))
output_dir = './fold_analysis_results'
os.makedirs(output_dir, exist_ok=True)

def compute_lisa(errors, W_dense):

    n = len(errors)
    e = errors - errors.mean()
    variance = np.sum(e**2) / n
    lisa = np.zeros(n)
    for i in range(n):
        lisa[i] = (e[i] * np.sum(W_dense[i] * e)) / variance
    return lisa

def compute_morans_i(errors, W_dense, n_perm=999):

    n = len(errors)
    W_sum = W_dense.sum()
    e = errors - errors.mean()
    numerator = e @ W_dense @ e
    denominator = np.sum(e**2)
    morans_i = (n / W_sum) * (numerator / denominator)
    
    # Permutation test
    pvals = [(n / W_sum) * ((np.random.permutation(e) @ W_dense @ np.random.permutation(e)) / denominator) 
             for _ in range(n_perm)]
    pvals = np.array(pvals)
    p_value = (np.sum(pvals >= morans_i) + 1) / (len(pvals) + 1)
    return morans_i, p_value

# ============== MAIN LOOP ==============
results_list = []
print("\n" + "="*80)
print("BATCH FOLD ANALYSIS")
print("="*80)

for idx, fold in enumerate(folds):
    print(f"\n[{idx + 1}/{len(folds)}] Fold {fold}...", end=" ", flush=True)
    try:
        # Load & predict
        testset = ViT_HER2ST(train=False, fold=fold, flatten=False, adj=True, ori=True, prune='Grid')
        test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
        
        model = Hist2ST(depth1=d1, depth2=d2, depth3=d3, n_genes=785, kernel_size=k, 
                       patch_size=p, heads=h, channel=c, dropout=0.2, zinb=0.25, nb=False, bake=5, lamb=0.5)
        model.load_state_dict(torch.load(f'./model/5-Hist2ST.ckpt', map_location=device))
        pred, gt = test(model.to(device), test_loader, device=device)
        
        # Coords & error
        sample = testset.names[0]
        coords = testset.loc_dict[sample]
        W_dense = np.asarray(testset.adj_dict[sample])
        error = np.mean((pred.X - gt.X) ** 2, axis=1)
        
        # Spatial stats
        morans_i, p_val = compute_morans_i(error, W_dense)
        lisa = compute_lisa(error, W_dense)
        
        # Gene correlation
        r_vals = [pearsonr(pred.X[:, g], gt.X[:, g])[0] 
                  for g in range(pred.shape[1]) 
                  if not (np.isnan(pred.X[:, g]).any() or np.isnan(gt.X[:, g]).any())]
        
        results_list.append({
            'fold': fold, 'sample': sample, 'n_spots': len(error),
            'mean_error': np.mean(error), 'std_error': np.std(error), 'median_error': np.median(error),
            'morans_i': morans_i, 'morans_p_value': p_val, 'sig_clustered': 'Yes' if p_val < 0.05 else 'No',
            'lisa_mean': np.mean(lisa), 'lisa_high_spots': np.sum(lisa > np.percentile(lisa, 95)),
            'pearson_r': np.nanmean(r_vals)
        })
        
        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        scatter1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=error, cmap='magma_r', s=100)
        axes[0].set_title(f'Fold {fold}: Prediction Error', fontweight='bold')
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
        plt.colorbar(scatter1, ax=axes[0], label='MSE')
        
        scatter2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=lisa, cmap='RdBu_r', s=100)
        axes[1].set_title(f'Fold {fold}: LISA (Local Moran\'s I)', fontweight='bold')
        axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
        plt.colorbar(scatter2, ax=axes[1], label='LISA')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fold_{fold:02d}_error_lisa.png', dpi=150)
        plt.close()
        print(f"✓ Morans I={morans_i:.4f}, p={p_val:.4f}")
        
    except Exception as e:
        print(f"✗ {e}")

# ============== SUMMARY ==============
df = pd.DataFrame(results_list).sort_values('fold')
df.to_csv(f'{output_dir}/fold_statistics.csv', index=False)

print("\n" + "="*80)
print("SUMMARY TABLE:")
print(df.to_string(index=False))
print("\n" + "="*80)
print(f"Mean Error:      {df['mean_error'].mean():.6f} ± {df['mean_error'].std():.6f}")
print(f"Moran's I:       {df['morans_i'].mean():.4f} ± {df['morans_i'].std():.4f}")
print(f"Sig. clustered:  {(df['morans_p_value'] < 0.05).sum()} / {len(df)} folds")
print(f"Pearson R:       {df['pearson_r'].mean():.4f} ± {df['pearson_r'].std():.4f}")
print(f"\n✓ Results: {output_dir}/")
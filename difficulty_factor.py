import os, sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import entropy, spearmanr
import cv2
from torch.utils.data import DataLoader
from dataset import ViT_HER2ST
from HIST2ST import Hist2ST
from predict import test


folds = list(range(1, 11))  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tag = '5-7-2-8-4-16-32'
k, p, d1, d2, d3, h, c = map(lambda x: int(x), tag.split('-'))
checkpoint_path = './model/5-Hist2ST.ckpt'
patch_radius = 56  # patch size radius
output_dir = './difficulty_factor'

os.makedirs(output_dir, exist_ok=True)

def calculate_difficulty_factors(testset, fold, pred, gt):
    
    print(f"Check testset: {testset}")
    """
    Calculate spatial difficulty factors for a fold
    
    Returns:
        H (Cell-type Entropy): Diversity of cell types in neighborhood (N, )
        S (Transcript Sparsity): Gene expression sparsity per spot (N, )
        M (Morphology Ambiguity): Visual complexity of image patches (N, )
        B (Boundary Intensity): Distance to tissue boundary (N, )
    """
    sample = testset.names[0] # A1, A2, B1, B2
    coords = testset.loc_dict[sample]
    exp = testset.exp_dict[sample]  # normalized log expression
    img = testset.img_dict[sample]  # image tensor
    W = testset.adj_dict[sample]  # adjacency matrix
    W_dense = np.asarray(W)
    
    n_spots = len(coords)
    H = np.zeros(n_spots)  # Entropy
    S = np.zeros(n_spots)  # Sparsity
    M = np.zeros(n_spots)  # Morphology ambiguity
    B = np.zeros(n_spots)  # Boundary intensity
    
    # ============================================
    # 1. CELL-TYPE ENTROPY (H)
    # ============================================
    # Shannon entropy of gene expression in neighborhood
    for i in range(n_spots):
        neighbors = W_dense[i] > 0
        neighbor_exp = exp[neighbors, :]
        # Mean expression per gene in neighborhood
        mean_exp = np.mean(neighbor_exp, axis=0)
        # Normalize to probability
        mean_exp = mean_exp / (np.sum(mean_exp) + 1e-10)
        # Shannon entropy
        H[i] = -np.sum(mean_exp * np.log(mean_exp + 1e-10))
    
    # ============================================
    # 2. TRANSCRIPT SPARSITY (S)
    # ============================================
    # Fraction of zero or near-zero genes
    global_threshold = np.percentile(exp, 5)

    for i in range(n_spots):

        S[i] = (exp[i] < global_threshold).mean()
    
    # ============================================
    # 3. MORPHOLOGY AMBIGUITY (M)
    # ============================================
    # Visual entropy/complexity of image patches
    r = patch_radius
    img_np = img.numpy() if hasattr(img, 'numpy') else np.array(img)
    # Normalize image to [0, 255] if needed
    if img_np.max() <= 1.0:
        img_np = img_np * 255
    
    for i in range(n_spots):
        x, y = coords[i].astype(int)
        try:
            patch = img_np[max(0, x-r):min(img_np.shape[0], x+r), 
                          max(0, y-r):min(img_np.shape[1], y+r), :]
            
            # Convert to grayscale and compute entropy
            if len(patch.shape) == 3:
                patch_gray = np.mean(patch, axis=2)
            else:
                patch_gray = patch
            
            # Histogram-based entropy
            patch_gray_normalized = (patch_gray / patch_gray.max() * 255) if patch_gray.max() > 0 else patch_gray
            hist, _ = np.histogram(patch_gray_normalized.flatten(), bins=32, range=(0, 255))
            hist = hist / (hist.sum() + 1e-10)
            M[i] = -np.sum(hist[hist > 0] * np.log(hist[hist > 0] + 1e-10))
        except:
            M[i] = 0
    
    # ============================================
    # 4. BOUNDARY INTENSITY (B)
    # ============================================
    # Edge detection: gradient magnitude at spot location
    img_uint8 = np.uint8(np.clip(img_np, 0, 255))
    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
        img_cv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        img_cv = img_uint8 if len(img_uint8.shape) == 2 else img_uint8[:, :, 0]
    sobelx = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    for i in range(n_spots):
        x, y = coords[i].astype(int)
        x = np.clip(x, r, gradient.shape[0] - r)
        y = np.clip(y, r, gradient.shape[1] - r)
        window = gradient[x-10:x+10, y-10:y+10]

        B[i] = np.percentile(window, 95)
            
    return H, S, M, B

# ============================================
# Usage & Analysis
# ============================================
def safe_corr(x,y):

    if np.std(x)<1e-8:
        return np.nan,1

    return spearmanr(x,y)

results_list = []

for fold in folds:
    print(f'\nProcessing fold {fold}...')
    
    # Load testset
    testset = ViT_HER2ST(train=False, fold=fold, flatten=False, adj=True, ori=True, prune='Grid')
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    
    # Load model with proper checkpoint loading
    model = Hist2ST(depth1=d1, depth2=d2, depth3=d3, n_genes=785, kernel_size=k, 
                       patch_size=p, heads=h, channel=c, dropout=0.2, zinb=0.25, nb=False, bake=5, lamb=0.5)
        
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f'Warning: checkpoint not found at {checkpoint_path}')
    model.to(device)
    model.eval()
    
    # Get predictions
    pred, gt = test(model, test_loader, device)
    
    # Calculate difficulty factors
    H, S, M, B = calculate_difficulty_factors(testset, fold, pred, gt)
    error = np.mean((pred.X - gt.X) ** 2, axis=1)
    
    # Correlate with error
    r_H, p_H = safe_corr(H, error)
    r_S, p_S = safe_corr(S, error)
    r_M, p_M = safe_corr(M, error)
    r_B, p_B = safe_corr(B, error)
    
    results_list.append({
        'fold': fold,
        'H_corr': r_H, 'H_pval': p_H,
        'S_corr': r_S, 'S_pval': p_S,
        'M_corr': r_M, 'M_pval': p_M,
        'B_corr': r_B, 'B_pval': p_B,
    })
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    coords = testset.loc_dict[testset.names[0]]
    
    for ax, data, title, cmap in [
        (axes[0,0], H, f'Cell-type Entropy (H) - corr: {r_H:.3f}', 'viridis'),
        (axes[0,1], S, f'Transcript Sparsity (S) - corr: {r_S:.3f}', 'plasma'),
        (axes[1,0], M, f'Morphology Ambiguity (M) - corr: {r_M:.3f}', 'cool'),
        (axes[1,1], B, f'Boundary Intensity (B) - corr: {r_B:.3f}', 'hot'),
    ]:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=data, cmap=cmap, s=80)
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'fold_{fold:02d}_difficulty_factors.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_file}')
    plt.close()

# Summary statistics
df_factors = pd.DataFrame(results_list)
print('\n' + '='*60)
print('DIFFICULTY FACTOR CORRELATIONS WITH PREDICTION ERROR')
print('='*60)
print(df_factors.to_string(index=False))

# Save to CSV
csv_path = os.path.join(output_dir, 'difficulty_factor_correlations.csv')
df_factors.to_csv(csv_path, index=False)
print(f'\nSaved: {csv_path}')

# Print summary
print('\nMean Correlations:')
for factor in ['H', 'S', 'M', 'B']:
    corr_col = f'{factor}_corr'
    pval_col = f'{factor}_pval'
    mean_corr = df_factors[corr_col].mean()
    sig_count = (df_factors[pval_col] < 0.05).sum()
    print(f'  {factor}: mean r={mean_corr:.4f}, significant folds={sig_count}/10')
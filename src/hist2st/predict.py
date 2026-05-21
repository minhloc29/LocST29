import numpy as np
import scanpy as sc
import anndata as ad
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score


def test_model(model, loader, device="cuda"):
    model = model.to(device)
    model.eval()
    preds = None
    coords = None
    gt = None
    with torch.no_grad():
        for patch, position, exp, adj, *_, center in tqdm(loader):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            pred = model(patch, position, adj)[0]
            preds = pred.squeeze().cpu().numpy()
            coords = center.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
    adata = ad.AnnData(preds)
    adata.obsm["spatial"] = coords
    adata_gt = ad.AnnData(gt)
    adata_gt.obsm["spatial"] = coords
    return adata, adata_gt


def cluster(adata, label):
    idx = label != "undetermined"
    tmp = adata[idx]
    l = label[idx]
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm["X_pca"])
    p = kmeans.labels_.astype(str)
    lbl = np.full(len(adata), str(len(set(l))))
    lbl[idx] = p
    adata.obs["kmeans"] = lbl
    return p, round(ari_score(p, l), 3)


def get_R(data1, data2, dim=1, func=pearsonr):
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            r, pv = func(adata1[:, g], adata2[:, g])
        else:
            r, pv = func(adata1[g, :], adata2[g, :])
        r1.append(r)
        p1.append(pv)
    return np.array(r1), np.array(p1)

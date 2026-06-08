from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import anndata as ad
from pathlib import Path
from scipy.spatial import distance
from scipy.sparse import csr_matrix


@dataclass
class Phase1Results:

    spot_mse: np.ndarray
    coords: np.ndarray
    adata_path: str
    morans_I: float
    morans_p: float
    

def build_spatial_graph(coords: np.ndarray, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a k-NN graph from 2D coordinates.

    Returns
    -------
    edge_index : (2, E) int array of edges (src, dst)
    edge_weight: (E,) float array of weights
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    # sample: [0.         0.39772745 0.49205896 1.04313463], [ 0  7  5  3]: return top 3 nearest neighbors, and indices of them
    src_list = []
    dst_list = []
    w_list = []

    for i in range(coords.shape[0]):

        for j, dist in zip(indices[i][1:], distances[i][1:]):
            src_list.append(i)
            dst_list.append(j)
            w_list.append(1.0 / (dist + 1e-8))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(w_list, dtype=np.float32)
    return edge_index, edge_weight


def smooth_on_graph(
    values: np.ndarray,       # (N,) or (N, D)
    edge_index: np.ndarray,   # (2, E)
    edge_weight: np.ndarray,  # (E,)
    n_iter: int = 1,
    alpha: float = 0.7,
) -> np.ndarray:
    
    out = values.astype(np.float32)
    N = out.shape[0]

    for _ in range(n_iter):

        accum = np.zeros_like(out)
        weight_sum = np.zeros(N, dtype=np.float32)
        
        for src, dst, w in zip(
            edge_index[0],     # (E,)
            edge_index[1],     # (E,)
            edge_weight,       # (E,)
        ):

            if out.ndim == 1:
                accum[dst] += out[src] * w
            else:
                accum[dst] += out[src, :] * w

            weight_sum[dst] += w

        weight_sum = np.maximum(weight_sum, 1e-8)
        if out.ndim == 1:
            smoothed = accum / weight_sum
        else:
            smoothed = accum / weight_sum[:, None]
        out = (
            alpha * smoothed
            + (1 - alpha) * out
        )
    return out


def expand_mask_one_hop(
    mask: np.ndarray,
    edge_index: np.ndarray,
):
    """
    Expand active spots by one graph hop.
    """

    expanded = mask.copy()

    for src, dst in zip(
        edge_index[0],
        edge_index[1]
    ):
        if mask[src]:
            expanded[dst] = True

    return expanded


def expand_mask_k_hops(
    mask: np.ndarray,
    edge_index: np.ndarray,
    k: int = 1,
):
    expanded = mask.copy()

    for _ in range(k):
        expanded = expand_mask_one_hop(
            expanded,
            edge_index,
        )

    return expanded

def normalise_difficulty(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] with numerical safety."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-10:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-12)
    return mat / row_sum


def _morans_i_from_graph(values: np.ndarray, W: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    if W.shape[0] != W.shape[1] or W.shape[0] != vals.shape[0]:
        raise ValueError("W must be (N, N) with N == len(values)")
    if vals.size < 2:
        return float("nan")

    W_norm = _row_normalize(W)
    z = vals - vals.mean()
    denom = np.sum(z**2)
    if denom <= 1e-12:
        return float("nan")
    numerator = z @ (W_norm @ z)
    return float(numerator / denom)


def prepare_morans_adata(
    n_obs: int,
    coords: Optional[np.ndarray] = None,
    adj: Optional[np.ndarray] = None,
    k_neighbours: int = 8,
) -> ad.AnnData:
    if coords is None and adj is None:
        raise ValueError("Either coords or adj must be provided for Moran's I.")

    adata = ad.AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    if adj is not None:
        adata.obsp["connectivities"] = csr_matrix(adj)
        return adata

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    nbrs = NearestNeighbors(n_neighbors=min(k_neighbours + 1, n_obs)).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    W = np.zeros((n_obs, n_obs), dtype=np.float32)
    for i in range(n_obs):
        for j, dist in zip(indices[i][1:], distances[i][1:]):
            W[i, j] = 1.0 / (dist + 1e-8)
    adata.obsp["connectivities"] = csr_matrix(W)
    return adata


def morans_i_scanpy_from_adata(
    adata: ad.AnnData,
    values: np.ndarray,
    n_perms: Optional[int] = 999,
) -> Tuple[float, float]:
    if "connectivities" not in adata.obsp:
        raise ValueError("Missing connectivities in adata.obsp")
    W = np.asarray(adata.obsp["connectivities"].todense())
    vals = np.asarray(values)
    if vals.ndim > 1:
        vals = vals.mean(axis=1)

    morans_i = _morans_i_from_graph(vals, W)
    if n_perms is None or n_perms <= 0:
        return morans_i, float("nan")

    perm_vals = np.zeros(n_perms, dtype=np.float64)
    for i in range(n_perms):
        perm = np.random.permutation(vals)
        perm_vals[i] = _morans_i_from_graph(perm, W)

    p_val = float((np.sum(perm_vals >= morans_i) + 1.0) / (n_perms + 1.0))
    return morans_i, p_val


def morans_i_scanpy(
    values: np.ndarray,
    coords: Optional[np.ndarray] = None,
    adj: Optional[np.ndarray] = None,
    k_neighbours: int = 8,
    n_perms: Optional[int] = 999,
) -> Tuple[float, float]:
    adata = prepare_morans_adata(
        n_obs=len(values),
        coords=coords,
        adj=adj,
        k_neighbours=k_neighbours,
    )
    return morans_i_scanpy_from_adata(adata, values, n_perms=n_perms)




def move_to_device(x, device: torch.device):
    if isinstance(x, (tuple, list)):
        return tuple(item.to(device) for item in x)
    return x.to(device)


def flatten_indices(idx) -> torch.Tensor: # turn to 1D tensor [[1, 2], [3, 4]] -> [1, 2, 3, 4]
    if torch.is_tensor(idx):
        return idx.view(-1)
    return torch.as_tensor(idx).view(-1)


def _select_tensor_spots(tensor: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    keep = keep.to(tensor.device)
    if tensor.ndim >= 2 and tensor.shape[0] == 1:
        return tensor[:, keep]
    return tensor[keep]


def _select_adj(adj: Optional[torch.Tensor], keep: torch.Tensor) -> Optional[torch.Tensor]:
    if adj is None:
        return None
    keep = keep.to(adj.device)
    if adj.ndim == 3 and adj.shape[0] == 1:
        return adj[:, keep][:, :, keep]
    return adj[keep][:, keep]


def select_active_inputs(x, keep: torch.Tensor):
    if not isinstance(x, (tuple, list)):
        return _select_tensor_spots(x, keep)
    patches, centers, adj = x
    return (
        _select_tensor_spots(patches, keep),
        _select_tensor_spots(centers, keep),
        _select_adj(adj, keep),
    )


def select_active_targets(y: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return _select_tensor_spots(y, keep)


def _resolve_data_root(data_root: Optional[Path]) -> Path:
    if data_root is None:
        return Path.cwd()
    return Path(data_root).resolve()


def _require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} directory: {path}")


def calc_adj(coord, k: int = 8, distance_type: str = "euclidean", prune_tag: str = "NA"):
    """
    Calculate spatial adjacency matrix from X/Y coordinates.
    """
    spatial_matrix = coord
    nodes = spatial_matrix.shape[0]
    adj = torch.zeros((nodes, nodes))
    for i in np.arange(spatial_matrix.shape[0]):
        tmp = spatial_matrix[i, :].reshape(1, -1)
        dist_mat = distance.cdist(tmp, spatial_matrix, distance_type)
        if k == 0:
            k = spatial_matrix.shape[0] - 1
        res = dist_mat.argsort()[: k + 1]
        tmpdist = dist_mat[0, res[0][1 : k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k + 1):
            if prune_tag == "NA":
                adj[i][res[0][j]] = 1.0
            elif prune_tag == "STD":
                if dist_mat[0, res[0][j]] <= boundary:
                    adj[i][res[0][j]] = 1.0
            elif prune_tag == "Grid":
                if dist_mat[0, res[0][j]] <= 2.0:
                    adj[i][res[0][j]] = 1.0
    return adj


    
def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def _make_seeded_generator(seed: int) -> torch.Generator:
    """Pass to DataLoader(generator=...) for reproducible batch order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
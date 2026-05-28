from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import inspect
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import anndata as ad
import scanpy as sc
from pathlib import Path
from scipy.spatial import distance


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
    """
    Graph diffusion / neighbor smoothing.

    Update rule:
        X_new = alpha * A_norm @ X + (1 - alpha) * X

    where:
        A_norm = row-normalized adjacency

    Parameters
    ----------
    values:
        Node features.

        Shape:
            (N,)     -> scalar value per node
            (N, D)   -> D-dimensional feature per node

    edge_index:
        Graph connectivity.

        Shape:
            (2, E)

        edge_index[0] = source nodes
        edge_index[1] = destination nodes

    edge_weight:
        Edge importance.

        Shape:
            (E,)

    n_iter:
        Number of diffusion iterations.

    alpha:
        Diffusion strength.

        alpha=1.0
            pure smoothing

        alpha=0.0
            keep original values

    Returns
    -------
    out:
        Smoothed values

        Shape:
            same as values
    """

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


def normalise_difficulty(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] with numerical safety."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-10:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _as_scalar(value) -> float:
    arr = np.asarray(value)
    return float(arr.ravel()[0])


def _parse_morans_i_output(result) -> Tuple[float, float]:
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        return _as_scalar(result[0]), _as_scalar(result[1])

    if hasattr(result, "columns"):
        if "I" in result.columns:
            i_val = _as_scalar(result["I"].values)
            if "pval" in result.columns:
                return i_val, _as_scalar(result["pval"].values)
            if "p" in result.columns:
                return i_val, _as_scalar(result["p"].values)
            return i_val, float("nan")

    return _as_scalar(result), float("nan")


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
        adata.obsp["connectivities"] = np.asarray(adj)
        return adata

    adata.obsm["spatial"] = np.asarray(coords)
    sc.pp.neighbors(adata, n_neighbors=k_neighbours, use_rep="spatial")
    return adata


def morans_i_scanpy_from_adata(
    adata: ad.AnnData,
    values: np.ndarray,
    n_perms: Optional[int] = 999,
) -> Tuple[float, float]:
    vals = np.asarray(values)
    if vals.ndim == 1:
        vals = vals[:, None]

    params = inspect.signature(sc.metrics.morans_i).parameters
    kwargs = {}
    if "vals" in params:
        kwargs["vals"] = vals
    else:
        adata.X = vals

    if "use_graph" in params and "connectivities" in adata.obsp:
        kwargs["use_graph"] = "connectivities"

    if "n_perms" in params and n_perms is not None:
        kwargs["n_perms"] = n_perms

    result = sc.metrics.morans_i(adata, **kwargs)
    return _parse_morans_i_output(result)


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


def flatten_indices(idx) -> torch.Tensor:
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
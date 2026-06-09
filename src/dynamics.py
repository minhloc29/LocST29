from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hdbscan
from .utils import build_spatial_graph, normalise_difficulty, smooth_on_graph
from .analysis import DifficultyDynamics
from .difficulty_gse import gse_difficulty_from_field


@dataclass
class SpatialDynamicsField:
    """
    Derived maps from the dynamic difficulty field D(x,y,t).

    Attributes
    ----------
    D_field  : (T, N) raw normalized difficulty at each epoch
    D_bar    : (N,) persistent hardness
    dD_dt    : (N,) learning speed (signed slope)
    learning_speed: (N,) absolute learning speed
    volatility: (N,) temporal variance
    T_L      : (N,) learning time (epoch index; T if never learned)
    coords   : (N, 2)
    """
    D_field: np.ndarray
    D_bar: np.ndarray
    dD_dt: np.ndarray
    learning_speed: np.ndarray
    volatility: np.ndarray
    T_L: np.ndarray
    coords: np.ndarray
    
    difficulty_score: np.ndarray
    
    @property
    def T(self) -> int:
        return self.D_field.shape[0]

    @property
    def N(self) -> int: # number of spots
        return self.D_field.shape[1]

    def snapshot(self, epoch: int) -> np.ndarray:
        """Difficulty map at a specific epoch, shape (N,)."""
        return self.D_field[epoch]


def build_dynamics_field(
    dynamics: DifficultyDynamics,
    learn_threshold: float = 0.3,
) -> SpatialDynamicsField:
    """
    Convert epoch-wise MSE into all derived spatial maps.
    """
    T, _ = dynamics.epoch_mse.shape

    all_mse = dynamics.epoch_mse

    global_min = all_mse.min()
    global_max = all_mse.max()

    D_field = (
        all_mse - global_min
    ) / (
        global_max - global_min + 1e-8
    )

    D_bar = D_field.mean(axis=0)

    t_vec = np.arange(T, dtype=np.float64)
    t_c = t_vec - t_vec.mean()
    num = (t_c[:, None] * D_field).sum(axis=0)
    den = (t_c ** 2).sum()
    dD_dt = (num / (den + 1e-10)).astype(np.float32)

    learning_speed = -dD_dt
    volatility = D_field.var(axis=0).astype(np.float32)

    # Learning time: first epoch where difficulty < threshold
    N = D_field.shape[1]
    T_L = np.full(N, T, dtype=np.int32)

    consecutive = 3
    for i in range(N):

        hard = D_field[:, i]
        for t in range(T - consecutive + 1):

            window = hard[t:t+consecutive]
            if np.all(window < learn_threshold):
                T_L[i] = t
                break

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    D_bar_n = normalize(D_bar)
    print(np.percentile(
    D_field.flatten(),
    [1,5,10,25,50]
))

    return SpatialDynamicsField(
        D_field=D_field,
        D_bar=D_bar,
        dD_dt=dD_dt,
        learning_speed=learning_speed,
        volatility=volatility,
        T_L=T_L,
        coords=dynamics.coords,
        difficulty_score=D_bar_n
    )


@dataclass
class TopologyResult:
    """
    Outputs of spatial topology analysis.
    """
    persistent_clusters: np.ndarray
    wave_metric: np.ndarray
    interface_mask: np.ndarray
    interface_threshold: float


def analyse_topology(
    field: SpatialDynamicsField,
    k_neighbours: int = 6,
    dbscan_eps: float = 50.0,
    dbscan_min_samples: int = 5,
    interface_percentile: float = 80.0,
) -> TopologyResult:
    """
    Run all three topology analyses.
    """
    N = field.N
    coords = field.coords

    # 1. Clusters of persistent hardness
    hard_mask = field.difficulty_score > np.median(field.difficulty_score)
    cluster_labels = np.full(N, -1, dtype=np.int32)
    if hard_mask.sum() > dbscan_min_samples:
        hard_coords = coords[hard_mask]
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        sub_labels = db.fit_predict(hard_coords)
        sub_labels_shifted = np.where(sub_labels >= 0, sub_labels, -1)
        cluster_labels[hard_mask] = sub_labels_shifted

    # 2. Learning waves (spatial propagation of speed)
    edge_index, edge_weight = build_spatial_graph(coords, k=k_neighbours)
    smoothed_speed = smooth_on_graph(field.learning_speed, edge_index, edge_weight, n_iter=2)
    wave_metric = (field.learning_speed - smoothed_speed).astype(np.float32)

    # 3. Interface zones of late learning
    T_L_float = field.T_L.astype(np.float32)
    thresh = np.percentile(T_L_float, interface_percentile)
    interface_mask = T_L_float >= thresh

    return TopologyResult(
        persistent_clusters=cluster_labels,
        wave_metric=wave_metric,
        interface_mask=interface_mask,
        interface_threshold=float(thresh),
    )


def summarise_field(field: SpatialDynamicsField) -> Dict[str, float]:
    """Print and return summary statistics of the dynamic field."""
    stats = {
        "mean_D_bar": float(field.D_bar.mean()),
        "frac_persistent_hard": float((field.difficulty_score > 0.75).mean()),
        "mean_learning_speed": float(field.learning_speed.mean()),
        "mean_volatility": float(field.volatility.mean()),
        "mean_T_L_epochs": float(field.T_L.mean()),
        "frac_never_learned": float((field.T_L == field.T).mean()),
    }
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    return stats


def build_difficulty_field(
    dynamics: DifficultyDynamics,
    learn_threshold: float = 0.3,
    k_neighbours: int = 6,
    dbscan_eps: float = 50.0,
    interface_percentile: float = 80.0,
) -> Dict[str, object]:
   
    print("[Phase 3] Building dynamic spatial difficulty field D(x,y,t)...")
    field = build_dynamics_field(dynamics, learn_threshold=learn_threshold)
    field.difficulty_score = gse_difficulty_from_field(
        field, k=6, alpha=0.5
    )
    print("[Phase 3] Summary statistics:")
    stats = summarise_field(field)

    print("[Phase 3] Running spatial topology analysis...")
    topology = analyse_topology(
        field,
        k_neighbours=k_neighbours,
        dbscan_eps=dbscan_eps,
        interface_percentile=interface_percentile,
    )

    n_clusters = len(set(topology.persistent_clusters[topology.persistent_clusters >= 0]))
    print(f"  Persistent-hardness clusters found: {n_clusters}")
    print(
        f"  Interface-zone spots: {topology.interface_mask.sum()} "
        f"({100 * topology.interface_mask.mean():.1f}%)"
    )
    print("[Phase 3] Done.")

    return {"field": field, "topology": topology, "stats": stats}

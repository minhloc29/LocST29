from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from .analysis import DifficultyDynamics
from .difficulty_gse import gse_difficulty_from_field, topological_difficulty_from_data


@dataclass
class SpatialDynamicsField:
   
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
    expression: Optional[np.ndarray] = None,
) -> Dict[str, object]:

    if expression is not None and dynamics.epoch_mse.shape[0] == 0:
        # Expression-based path — no warm-up training data.
        # Build a minimal SpatialDynamicsField directly.
        N = dynamics.N
        coords = dynamics.coords
        print("[Phase 3] Computing difficulty from expression data "
              f"(no warm-up needed)…")

        difficulty_score = topological_difficulty_from_data(
            expression=expression,
            coords=coords,
            k_neighbours=k_neighbours,
        )

        field = SpatialDynamicsField(
            D_field=np.empty((0, N), dtype=np.float32),
            D_bar=np.full(N, np.nan, dtype=np.float32),
            dD_dt=np.full(N, np.nan, dtype=np.float32),
            learning_speed=np.full(N, np.nan, dtype=np.float32),
            volatility=np.full(N, np.nan, dtype=np.float32),
            T_L=np.full(N, 0, dtype=np.int32),
            coords=coords,
            difficulty_score=difficulty_score,
        )

        print(f"  difficulty_score: mean={difficulty_score.mean():.3f}  "
              f"std={difficulty_score.std():.3f}")

        stats = summarise_field(field)
        return {"field": field, "topology": None, "stats": stats}

    else:
        # Original path — dynamics object has real epoch data.
        print("[Phase 3] Building dynamic spatial difficulty field D(x,y,t)...")
        field = build_dynamics_field(dynamics, learn_threshold=learn_threshold)

        if expression is not None:
            field.difficulty_score = topological_difficulty_from_data(
                expression=expression,
                coords=field.coords,
                k_neighbours=k_neighbours,
            )
        else:
            field.difficulty_score = gse_difficulty_from_field(
                field, k=6, alpha=0.5
            )

    print("[Phase 3] Summary statistics:")
    stats = summarise_field(field)
    print("[Phase 3] Done.")

    return {"field": field, "topology": None, "stats": stats}

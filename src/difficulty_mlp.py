"""Difficulty MLP: learn a non-linear mapping from niche difficulty features
(heterogeneity, topology, ambiguity) to a scalar difficulty score.

Rationale
---------
Experiments show that the correlation between difficulty components and
true per-spot prediction error varies across folds/tissues. Hand-tuned linear
weights (alpha, beta, gamma) do not generalise.

Instead, we train a small MLP to predict per-spot prediction error from
the three features, then use the MLP's output as the difficulty score for
curriculum learning. This lets the model adapt to each tissue's specific
structure.

Usage
-----
    # 1. Collect features + targets from a trained model
    features, targets = prepare_training_data(
        expression, coords, niche_labels, pred_error
    )

    # 2. Train the MLP
    mlp = train_difficulty_mlp(features, targets, hidden_dim=32)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MLP architecture
# ---------------------------------------------------------------------------

class DifficultyMLP(nn.Module):
    """Small MLP that maps difficulty features → scalar difficulty score.

    Input features (per spot or per niche):
        - heterogeneity : internal diversity of the niche
        - topology      : boundary energy (Laplacian on niche-expression graph)
        - ambiguity     : closeness to other niche centroids

    Output: scalar difficulty score in [0, 1] (after sigmoid).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # output in [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, input_dim)  →  out : (N, 1)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

@dataclass
class DifficultyFeatures:
    """Container for the 3 difficulty features, targets, and metadata."""
    heterogeneity: np.ndarray   # (N,) or (K,)
    topology: np.ndarray        # (N,) or (K,)
    ambiguity: np.ndarray       # (N,) or (K,)
    targets: np.ndarray         # (N,) per-spot MSE as difficulty proxy
    niche_labels: np.ndarray    # (N,) — only for spot-level features
    is_niche_level: bool = False

    @property
    def N(self) -> int:
        return len(self.targets)

    @property
    def feature_matrix(self) -> np.ndarray:
        """Stack features into (N, 3) array."""
        return np.column_stack([
            self.heterogeneity,
            self.topology,
            self.ambiguity,
        ])


def prepare_training_data(
    expression: np.ndarray,
    coords: np.ndarray,
    niche_labels: np.ndarray,
    per_spot_error: np.ndarray,
    use_niche_level: bool = False,
) -> DifficultyFeatures:
    """Extract difficulty features and align with per-spot prediction error.

    Parameters
    ----------
    expression     : (N, G) log-normalised gene expression
    coords         : (N, 2) spatial coordinates
    niche_labels   : (N,) niche assignment
    per_spot_error : (N,) per-spot MSE from a trained model
    use_niche_level : if True, aggregate to niche-level features

    Returns
    -------
    features : DifficultyFeatures
    """
    from src.niche import niche_heterogeneity, niche_topology_difficulty, niche_ambiguity

    K = int(niche_labels.max()) + 1

    het = niche_heterogeneity(expression, niche_labels)           # (K,)
    topo = niche_topology_difficulty(niche_labels, coords, expression)  # (K,)
    amb_per_spot = niche_ambiguity(expression, niche_labels)      # (N,)
    amb_mean = np.array([amb_per_spot[niche_labels == k].mean() for k in range(K)])

    if use_niche_level:
        # Aggregate targets to niche level (mean error per niche)
        targets = np.array([per_spot_error[niche_labels == k].mean() for k in range(K)])
        return DifficultyFeatures(
            heterogeneity=het,
            topology=topo,
            ambiguity=amb_mean,
            targets=targets,
            niche_labels=niche_labels,
            is_niche_level=True,
        )
    else:
        # Propagate niche-level features to spots, add per-spot ambiguity
        spot_het = np.array([het[lbl] for lbl in niche_labels])
        spot_topo = np.array([topo[lbl] for lbl in niche_labels])
        return DifficultyFeatures(
            heterogeneity=spot_het,
            topology=spot_topo,
            ambiguity=amb_per_spot,  # already per-spot
            targets=per_spot_error,
            niche_labels=niche_labels,
            is_niche_level=False,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class MLPTrainingLog:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_state: Optional[dict] = None


def train_difficulty_mlp(
    features: DifficultyFeatures,
    hidden_dim: int = 32,
    n_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    n_epochs: int = 200,
    val_split: float = 0.2,
    batch_size: int = 64,
    patience: int = 20,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[DifficultyMLP, MLPTrainingLog]:
    """Train the DifficultyMLP on extracted features.

    Parameters
    ----------
    features     : DifficultyFeatures with feature_matrix and targets
    hidden_dim   : width of hidden layers
    n_layers     : number of hidden layers
    dropout      : dropout rate
    learning_rate : Adam learning rate
    weight_decay : L2 regularisation
    n_epochs     : maximum training epochs
    val_split    : fraction of data for validation
    batch_size   : mini-batch size
    patience     : early stopping patience
    device       : torch device
    verbose      : print progress

    Returns
    -------
    mlp   : trained DifficultyMLP (on CPU)
    log   : MLPTrainingLog
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = features.feature_matrix  # (N, 3)
    y = features.targets         # (N,)

    # Normalise features to [0, 1] per dimension
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X = (X - X_min) / X_range

    # Normalise targets to [0, 1]
    y_min, y_max = y.min(), y.max()
    if y_max - y_min > 1e-10:
        y = (y - y_min) / (y_max - y_min)
    else:
        y = np.zeros_like(y)

    N = len(X)
    n_val = max(1, int(N * val_split))
    indices = np.random.RandomState(42).permutation(N)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True
    )

    mlp = DifficultyMLP(
        input_dim=3,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    log = MLPTrainingLog()
    best_epoch = 0

    for epoch in range(n_epochs):
        mlp.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = mlp(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)

        train_loss = epoch_loss / len(train_dataset)

        # Validation
        mlp.eval()
        with torch.no_grad():
            val_pred = mlp(X_val.to(device))
            val_loss = loss_fn(val_pred, y_val.to(device)).item()

        log.train_losses.append(train_loss)
        log.val_losses.append(val_loss)

        if val_loss < log.best_val_loss:
            log.best_val_loss = val_loss
            log.best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            best_epoch = epoch

        if epoch - best_epoch > patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    # Restore best state
    if log.best_state is not None:
        mlp.load_state_dict(log.best_state)
    mlp.to("cpu")

    if verbose:
        print(f"[DiffMLP] Training done. Best val loss: {log.best_val_loss:.6f}")
        # Report Spearman correlation between MLP output and targets on full data
        mlp.eval()
        with torch.no_grad():
            full_X = torch.tensor(X, dtype=torch.float32)
            full_pred = mlp(full_X).squeeze().numpy()
        from scipy.stats import spearmanr
        rho, _ = spearmanr(full_pred, y)
        print(f"[DiffMLP] Spearman ρ (MLP vs target on full data): {rho:.4f}")

    return mlp, log


def mlp_difficulty_from_features(
    mlp: DifficultyMLP,
    features: DifficultyFeatures,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run the trained MLP on difficulty features to get difficulty scores.

    Parameters
    ----------
    mlp      : trained DifficultyMLP
    features : DifficultyFeatures with feature_matrix

    Returns
    -------
    scores : (N,) difficulty scores in [0, 1]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = features.feature_matrix  # (N, 3)

    # Normalise using same scheme as training (recompute from full data)
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range

    mlp.to(device)
    mlp.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
        scores = mlp(X_t).squeeze().cpu().numpy()

    mlp.to("cpu")
    return np.asarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Convenience: end-to-end from trained model + data
# ---------------------------------------------------------------------------

def compute_mlp_difficulty(
    expression: np.ndarray,
    coords: np.ndarray,
    niche_labels: np.ndarray,
    per_spot_error: np.ndarray,
    mlp_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, DifficultyMLP]:
    """One-call: prepare features, train MLP, return difficulty scores.

    Parameters
    ----------
    expression     : (N, G) log-normalised expression
    coords         : (N, 2) spatial coordinates
    niche_labels   : (N,) niche assignment
    per_spot_error : (N,) per-spot MSE from a trained model
    mlp_kwargs     : kwargs passed to train_difficulty_mlp

    Returns
    -------
    scores : (N,) MLP-predicted difficulty scores
    mlp    : trained DifficultyMLP
    """
    if mlp_kwargs is None:
        mlp_kwargs = {}

    features = prepare_training_data(
        expression=expression,
        coords=coords,
        niche_labels=niche_labels,
        per_spot_error=per_spot_error,
        use_niche_level=False,
    )

    mlp, log = train_difficulty_mlp(features, **mlp_kwargs)
    scores = mlp_difficulty_from_features(mlp, features)

    return scores, mlp

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes_grid1 import make_axes_locatable


def _add_colorbar(ax, mappable):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    return plt.colorbar(mappable, cax=cax)


def plot_factor_grid(coords, factors, corrs, output_file):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    configs = [
        ("H", "Cell-type Entropy (H)", "viridis"),
        ("S", "Transcript Sparsity (S)", "plasma"),
        ("M", "Morphology Ambiguity (M)", "cool"),
        ("B", "Boundary Intensity (B)", "hot"),
    ]

    for ax, (key, label, cmap) in zip(axes.flatten(), configs):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=factors[key], cmap=cmap, s=80)
        ax.set_title(f"{label} - corr: {corrs[key]:.3f}", fontsize=11)
        ax.set_aspect("equal")
        _add_colorbar(ax, sc)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_temporal_dynamics(results, testset, output_dir="./difficulty_factor"):
    os.makedirs(output_dir, exist_ok=True)
    coords = testset.loc_dict[testset.names[0]]

    for slide in results:
        persistent = results[slide]["persistent"]
        speed = results[slide]["speed"]
        volatility = results[slide]["volatility"]
        learn_time = results[slide]["learn_time"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plots = [
            (persistent, "Persistent Difficulty", axes[0, 0], "RdYlGn_r"),
            (speed, "Learning Speed", axes[0, 1], "viridis"),
            (volatility, "Difficulty Volatility", axes[1, 0], "plasma"),
            (learn_time, "Learning Time", axes[1, 1], "coolwarm"),
        ]

        for values, title, ax, cmap in plots:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=70, cmap=cmap)
            ax.set_title(title)
            ax.set_aspect("equal")
            _add_colorbar(ax, sc)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{slide}_difficulty_map.png"), dpi=200)
        plt.close(fig)


def plot_difficulty_trajectories(results, output_dir="./difficulty_factor"):
    os.makedirs(output_dir, exist_ok=True)

    for slide in results:
        traj = results[slide]["trajectories"]
        epochs = results[slide]["epochs"]
        persistent = results[slide]["persistent"]

        order = np.argsort(persistent)
        selected = [order[10], order[len(order) // 3], order[2 * len(order) // 3], order[-10]]
        labels = ["Easy", "Medium", "Hard", "Persistent Hard"]

        fig = plt.figure(figsize=(10, 6))
        for idx, label in zip(selected, labels):
            plt.plot(epochs, traj[:, idx], linewidth=3, label=label)

        plt.xlabel("Epoch")
        plt.ylabel("Difficulty")
        plt.title(f"{slide}: Temporal Learning")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{slide}_trajectory.png"), dpi=200)
        plt.close(fig)

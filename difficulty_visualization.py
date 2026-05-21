import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def plot_temporal_dynamics(
    results,
    testset,
    output_dir="./difficulty_factor",
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    coords = (
        testset.loc_dict[
            testset.names[0]
        ]
    )

    for slide in results:

        persistent = (
            results[slide]["persistent"]
        )

        speed = (
            results[slide]["speed"]
        )

        volatility = (
            results[slide]["volatility"]
        )

        learn_time = (
            results[slide]["learn_time"]
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 10)
        )

        plots = [

            (
                persistent,
                "Persistent Difficulty",
                axes[0, 0],
                "RdYlGn_r"
            ),

            (
                speed,
                "Learning Speed",
                axes[0, 1],
                "viridis"
            ),

            (
                volatility,
                "Difficulty Volatility",
                axes[1, 0],
                "plasma"
            ),

            (
                learn_time,
                "Learning Time",
                axes[1, 1],
                "coolwarm"
            ),

        ]

        for values, title, ax, cmap in plots:

            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values,
                s=70,
                cmap=cmap
            )

            ax.set_title(
                title
            )

            ax.set_aspect(
                "equal"
            )

            plt.colorbar(
                sc,
                ax=ax
            )

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                output_dir,
                f"{slide}_difficulty_map.png"
            ),

            dpi=200
        )

        plt.close()




def plot_difficulty_trajectories(
    results,
    output_dir="./difficulty_factor"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for slide in results:

        traj = (
            results[slide]
            ["trajectories"]
        )

        epochs = (
            results[slide]
            ["epochs"]
        )

        persistent = (
            results[slide]
            ["persistent"]
        )

        order = np.argsort(
            persistent
        )

        selected = [

            order[10],

            order[
                len(order)//3
            ],

            order[
                2*len(order)//3
            ],

            order[-10]

        ]

        labels = [

            "Easy",

            "Medium",

            "Hard",

            "Persistent Hard"

        ]

        fig = plt.figure(
            figsize=(10, 6)
        )

        for idx, label in zip(
            selected,
            labels
        ):

            plt.plot(

                epochs,

                traj[:, idx],

                linewidth=3,

                label=label

            )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Difficulty"
        )

        plt.title(
            f"{slide}: Temporal Learning"
        )

        plt.legend()

        plt.grid()

        plt.savefig(

            os.path.join(

                output_dir,

                f"{slide}_trajectory.png"

            ),

            dpi=200

        )

        plt.close()




def compute_factor_correlation(
    results,
    factors
):

    correlations = {}

    for slide in results:

        persistent = (
            results[slide]
            ["persistent"]
        )

        correlations[
            slide
        ] = {}

        for name in factors:

            rho, p = spearmanr(

                factors[name],

                persistent

            )

            correlations[
                slide
            ][name] = {

                "rho": rho,

                "p": p

            }

    return correlations




def print_correlation_table(
    correlations
):

    print()

    print(
        "="*60
    )

    print(
        "Difficulty Correlation"
    )

    print(
        "="*60
    )

    for slide in correlations:

        print()

        print(
            slide
        )

        for factor in correlations[
            slide
        ]:

            v = correlations[
                slide
            ][factor]

            print(

                f"{factor:<10}"

                f"rho={v['rho']:.3f}"

                f" p={v['p']:.4f}"

            )
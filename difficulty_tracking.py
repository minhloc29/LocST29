import numpy as np
import torch
from collections import defaultdict
from scipy.signal import savgol_filter


class DifficultyTracker:

    def __init__(self):

        # epoch → slide → spot_error
        self.history = defaultdict(dict)

    def log_epoch(
        self,
        epoch,
        slide_name,
        pred,
        gt
    ):
        """
        pred: [spots, genes]
        gt:   [spots, genes]
        """

        error = ((pred - gt) ** 2).mean(axis=1)

        self.history[epoch][slide_name] = error

    def get_slide_matrix(self, slide):

        epochs = sorted(self.history.keys())

        matrix = []

        for e in epochs:
            matrix.append(
                self.history[e][slide]
            )

        return epochs, np.array(matrix)

    def compute_metrics(self):

        results = {}

        slides = list(
            next(
                iter(
                    self.history.values()
                )
            ).keys()
        )

        for slide in slides:

            epochs, errors = self.get_slide_matrix(
                slide
            )

            T, N = errors.shape

            # --------------------------------
            # Persistent Difficulty
            # --------------------------------

            tail = max(
                int(T * 0.2),
                5
            )

            persistent = (
                errors[-tail:]
                .mean(0)
            )

            # --------------------------------
            # Learning Speed
            # --------------------------------

            speed = np.zeros(N)

            for i in range(N):

                smooth = savgol_filter(
                    errors[:, i],
                    min(
                        7,
                        T // 2 * 2 + 1
                    ),
                    2
                )

                grad = np.gradient(
                    smooth
                )

                speed[i] = (
                    -grad.mean()
                )

            # --------------------------------
            # Volatility
            # --------------------------------

            volatility = (
                errors.std(0)
            )

            # --------------------------------
            # Learning Time
            # --------------------------------

            learn_time = np.zeros(
                N
            )

            for i in range(N):

                threshold = (
                    persistent[i]
                    * 1.2
                )

                idx = np.where(
                    errors[:, i]
                    < threshold
                )[0]

                if len(idx):

                    learn_time[i] = (
                        epochs[idx[0]]
                    )

                else:

                    learn_time[i] = (
                        epochs[-1]
                    )

            results[slide] = {

                "persistent":
                persistent,

                "speed":
                speed,

                "volatility":
                volatility,

                "learn_time":
                learn_time,

                "trajectories":
                errors,

                "epochs":
                epochs

            }

        return results


def evaluate_epoch(
    model,
    loader,
    device,
    tracker,
    epoch
):

    model.eval()

    with torch.no_grad():

        for batch_id, batch in enumerate(loader):

            (
                patch,
                center,
                exp,
                adj,
                *_
            ) = batch

            patch = patch.to(device)

            center = center.to(device)

            exp = exp.to(device)

            adj = adj.to(device)

            pred, _, _ = model(
                patch,
                center,
                adj.squeeze(0)
            )

            pred = (
                pred
                .squeeze(0)
                .cpu()
                .numpy()
                .T
            )

            gt = (
                exp
                .squeeze(0)
                .cpu()
                .numpy()
                .T
            )

            tracker.log_epoch(
                epoch,
                f"slide_{batch_id}",
                pred,
                gt
            )


def train_with_difficulty_tracking(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device
):

    model.to(device)

    tracker = DifficultyTracker()

    for epoch in range(epochs):

        model.train()

        losses = []

        for batch in train_loader:

            (
                patch,
                center,
                exp,
                adj,
                *_
            ) = batch

            patch = patch.to(device)

            center = center.to(device)

            exp = exp.to(device)

            adj = adj.to(device)

            optimizer.zero_grad()

            pred, _, _ = model(
                patch,
                center,
                adj.squeeze(0)
            )

            loss = criterion(
                pred,
                exp
            )

            loss.backward()

            optimizer.step()

            losses.append(
                loss.item()
            )

        # -----------------------
        # Dynamic logging
        # -----------------------

        if epoch < 30:

            evaluate_epoch(
                model,
                val_loader,
                device,
                tracker,
                epoch
            )

        elif epoch % 5 == 0:

            evaluate_epoch(
                model,
                val_loader,
                device,
                tracker,
                epoch
            )

        print(
            f"Epoch {epoch} "
            f"Loss={np.mean(losses):.4f}"
        )

    return tracker
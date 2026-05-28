import numpy as np

from src.utils.spatial import (
    Phase1Results,
    build_spatial_graph,
    normalise_difficulty,
    smooth_on_graph,
)


def main() -> None:
    rng = np.random.default_rng(7) # generate random number from gauss distribution

    # Mock 2D coordinates
    coords = rng.normal(size=(12, 2)).astype(np.float32)
    edge_index, edge_weight = build_spatial_graph(coords, k=3)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_weight.shape[0]

    # Mock scalar and vector node values
    values_1d = rng.normal(size=(coords.shape[0],)).astype(np.float32) # gauss
    values_2d = rng.normal(size=(coords.shape[0], 4)).astype(np.float32) # gauss

    smoothed_1d = smooth_on_graph(values_1d, edge_index, edge_weight, n_iter=2)
    smoothed_2d = smooth_on_graph(values_2d, edge_index, edge_weight, n_iter=2)
    assert smoothed_1d.shape == values_1d.shape
    assert smoothed_2d.shape == values_2d.shape

    # Normalization behavior
    normalized = normalise_difficulty(values_1d)
    assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)

    flat = np.ones((5,), dtype=np.float32)
    normalized_flat = normalise_difficulty(flat)
    assert np.allclose(normalized_flat, 0.0)

    # Dataclass sanity check
    _ = Phase1Results(
        spot_mse=values_1d,
        coords=coords,
        adata_path="mock/path.h5ad",
        morans_I=0.1,
        morans_p=0.9,
    )

    print(_)
    print("OK: spatial utils mock tests passed.")


if __name__ == "__main__":
    main()

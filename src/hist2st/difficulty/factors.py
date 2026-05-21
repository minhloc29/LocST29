import numpy as np
import cv2


def compute_entropy_factor(exp: np.ndarray, adj: np.ndarray) -> np.ndarray:
    n_spots = exp.shape[0]
    H = np.zeros(n_spots)
    for i in range(n_spots):
        neighbors = adj[i] > 0
        neighbor_exp = exp[neighbors, :]
        mean_exp = np.mean(neighbor_exp, axis=0)
        mean_exp = mean_exp / (np.sum(mean_exp) + 1e-10)
        H[i] = -np.sum(mean_exp * np.log(mean_exp + 1e-10))
    return H


def compute_sparsity_factor(exp: np.ndarray, ratio_threshold: float = 0.3) -> np.ndarray:
    gene_mean = exp.mean(axis=0)
    ratio = exp / (gene_mean + 1e-8)
    return (ratio < ratio_threshold).mean(axis=1)


def compute_morphology_factor(img, coords: np.ndarray, patch_radius: int) -> np.ndarray:
    n_spots = len(coords)
    M = np.zeros(n_spots)
    img_np = img.numpy() if hasattr(img, "numpy") else np.array(img)
    if img_np.max() <= 1.0:
        img_np = img_np * 255

    for i in range(n_spots):
        x, y = coords[i].astype(int)
        patch = img_np[
            max(0, x - patch_radius):min(img_np.shape[0], x + patch_radius),
            max(0, y - patch_radius):min(img_np.shape[1], y + patch_radius),
        ]
        if patch.size == 0:
            continue
        patch_gray = np.mean(patch, axis=2) if patch.ndim == 3 else patch
        patch_gray = (patch_gray / patch_gray.max() * 255) if patch_gray.max() > 0 else patch_gray
        hist, _ = np.histogram(patch_gray.flatten(), bins=32, range=(0, 255))
        hist = hist / (hist.sum() + 1e-10)
        M[i] = -np.sum(hist[hist > 0] * np.log(hist[hist > 0] + 1e-10))
    return M


def compute_boundary_factor(img, coords: np.ndarray, patch_radius: int) -> np.ndarray:
    img_np = img.numpy() if hasattr(img, "numpy") else np.array(img)
    if img_np.max() <= 1.0:
        img_np = img_np * 255
    img_uint8 = np.uint8(np.clip(img_np, 0, 255))
    if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
        img_cv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        img_cv = img_uint8 if img_uint8.ndim == 2 else img_uint8[:, :, 0]
    sobelx = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(sobelx ** 2 + sobely ** 2)

    n_spots = len(coords)
    B = np.zeros(n_spots)
    for i in range(n_spots):
        x, y = coords[i].astype(int)
        patch = gradient[
            max(0, x - patch_radius):min(gradient.shape[0], x + patch_radius),
            max(0, y - patch_radius):min(gradient.shape[1], y + patch_radius),
        ]
        if patch.size == 0:
            continue
        B[i] = (patch > 0).mean()
    return B


def calculate_factors(testset, patch_radius: int):
    sample = testset.names[0]
    coords = testset.center_dict[sample]
    exp = testset.exp_dict[sample]
    img = testset.img_dict[sample]
    adj = np.asarray(testset.adj_dict[sample])

    H = compute_entropy_factor(exp, adj)
    S = compute_sparsity_factor(exp)
    M = compute_morphology_factor(img, coords, patch_radius)
    B = compute_boundary_factor(img, coords, patch_radius)
    return H, S, M, B

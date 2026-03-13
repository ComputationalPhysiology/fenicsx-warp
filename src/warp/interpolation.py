from collections.abc import Callable

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree


def create_rbf_interpolator(
    points_mean: np.ndarray,
    displacements: np.ndarray,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a boundary displacement function using Radial Basis Functions."""
    print(f"Fitting Radial Basis Function (RBF) with '{kernel}' kernel...")
    rbf = RBFInterpolator(points_mean, displacements, kernel=kernel, smoothing=smoothing)

    def boundary_displacement(x: np.ndarray) -> np.ndarray:
        return rbf(x.T).T

    return boundary_displacement


def create_kdtree_interpolator(
    points_mean: np.ndarray, displacements: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a boundary displacement function using Nearest-Neighbor KDTree."""
    print("Fitting KDTree for nearest-neighbor interpolation...")
    tree = KDTree(points_mean)

    def boundary_displacement(x: np.ndarray) -> np.ndarray:
        _, closest_idx = tree.query(x.T)
        return displacements[closest_idx].T

    return boundary_displacement

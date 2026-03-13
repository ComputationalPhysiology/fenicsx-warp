from mpi4py import MPI

import numpy as np
import pytest
from dolfinx.mesh import create_unit_cube

from warp import warp_mesh


@pytest.fixture
def sample_data():
    """
    Creates a simple unit cube mesh and displacement points to use across tests.
    """
    # Create a 3D unit cube mesh mapped across multiple elements
    domain = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

    # We define the 8 corners of the cube as the mean points.
    points_mean = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    # We define a target set of points as a simple rigid translation (+0.1 on all axes).
    # Since it's a rigid shift, it's very easy to predict and check the final geometry.
    points_target = points_mean + 0.1

    return domain, points_mean, points_target


@pytest.mark.parametrize("interp_method", ["kdtree", "rbf"])
@pytest.mark.parametrize("solver_method", ["laplace", "hyperelastic"])
def test_warp_mesh(sample_data, interp_method, solver_method):
    domain, points_mean, points_target = sample_data

    # Copy the original coordinates to verify that the mesh is actually displaced.
    orig_coords = domain.geometry.x.copy()

    # Execute the modular warp pipeline
    warp_mesh(
        domain=domain,
        points_reference=points_mean,
        points_target=points_target,
        interpolation_method=interp_method,
        solver_method=solver_method,
    )

    # Assertion 2: Verify the coordinates have been updated
    assert not np.allclose(orig_coords, domain.geometry.x)

    # Assertion 3: Because our target points strictly shifted everything by +0.1,
    # we expect the mean displacement of the entire mesh to be very close to [0.1, 0.1, 0.1].
    # (The PDE propagates the boundary exactly inwards)
    mean_shift = np.mean(domain.geometry.x - orig_coords, axis=0)
    assert np.allclose(mean_shift[:3], [0.1, 0.1, 0.1], atol=1e-2)


def test_invalid_interp_method(sample_data):
    domain, points_mean, points_target = sample_data
    with pytest.raises(ValueError, match="Invalid interpolation method"):
        warp_mesh(
            domain,
            points_reference=points_mean,
            points_target=points_target,
            interpolation_method="invalid_name",
        )


def test_invalid_solver_method(sample_data):
    domain, points_mean, points_target = sample_data
    with pytest.raises(ValueError, match="Invalid solver method"):
        warp_mesh(
            domain,
            points_reference=points_mean,
            points_target=points_target,
            solver_method="invalid_name",
        )

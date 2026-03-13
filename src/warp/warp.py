import logging

import dolfinx
import numpy as np

from . import interpolation, models

logger = logging.getLogger(__name__)


def _constrain_base_displacement(displacement_func, origin, normal, tol=1e-2):
    """
    Wraps the displacement function to strictly constrain points on the clipping
    plane so that they remain perfectly flat during deformation.
    """
    origin = np.asarray(origin, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    def constrained_func(x):
        # x has shape (3, N)
        u = displacement_func(x)

        # Calculate signed distance from the clipping plane for all points
        dist = np.sum((x - origin[:, None]) * normal[:, None], axis=0)

        # Mask for points lying on the base plane
        is_base = np.abs(dist) < tol

        if np.any(is_base):
            u_base = u[:, is_base]

            # Calculate the displacement component along the normal direction
            u_dot_n = np.sum(u_base * normal[:, None], axis=0)

            # To keep the base perfectly flat but allow it to translate,
            # we replace varying normal displacements with the average normal displacement.
            mean_u_dot_n = np.mean(u_dot_n)

            # Subtract the varying normal component and add the uniform one
            u_base_projected = (
                u_base - (u_dot_n * normal[:, None]) + (mean_u_dot_n * normal[:, None])
            )

            u[:, is_base] = u_base_projected

        return u

    return constrained_func


def compute_base_normal(mesh: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.MeshTags, marker: int):
    base_facets = facet_tags.find(marker)
    base_midpoints = mesh.comm.gather(
        dolfinx.mesh.compute_midpoints(mesh, 2, base_facets),
        root=0,
    )
    bm = np.concatenate(base_midpoints)
    base_centroid = bm.mean(axis=0)
    base_points_centered = bm - base_centroid
    _, _, vh = np.linalg.svd(base_points_centered)
    base_normal = vh[-1, :]
    return base_normal


def get_boundary_conditions(domain, V, displacement_func):
    """Interpolates the displacement function onto the mesh boundary."""
    logger.info("Applying boundary conditions...")
    u_bc_func = dolfinx.fem.Function(V)
    u_bc_func.interpolate(displacement_func)

    # Identify the exterior surface of the mesh
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, exterior_facets)

    return dolfinx.fem.dirichletbc(u_bc_func, boundary_dofs)


def warp_mesh(
    domain: dolfinx.mesh.Mesh,
    points_reference: np.ndarray,
    points_target: np.ndarray,
    interpolation_method: str = "rbf",
    solver_method: str = "hyperelastic",
    clip_origin: tuple[float, float, float] | np.ndarray | None = None,
    clip_normal: tuple[float, float, float] | np.ndarray | None = None,
):
    """
    Warps a given mesh from a mean shape to a target shape using PDE-based deformation.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The computational domain (template mesh) to be warped.
    points_reference : np.ndarray
        The reference boundary points (N, 3).
    points_target : np.ndarray
        The target boundary points (N, 3).
    interpolation_method : str, optional
        Interpolation method ('rbf' or 'kdtree'), by default 'rbf'.
    solver_method : str, optional
        PDE solver method ('hyperelastic' or 'laplace'), by default 'hyperelastic'.

    Notes
    -----
    The input domain geometry is updated in-place.
    """
    displacements = points_target - points_reference

    # 1. Select and Create Interpolator
    if interpolation_method == "rbf":
        displacement_func = interpolation.create_rbf_interpolator(points_reference, displacements)
    elif interpolation_method == "kdtree":
        displacement_func = interpolation.create_kdtree_interpolator(
            points_reference, displacements
        )
    else:
        raise ValueError("Invalid interpolation method. Choose 'rbf' or 'kdtree'.")

    # Optional: Constrain the base to remain flat if clipping params are provided
    if clip_origin is not None and clip_normal is not None:
        logger.info("Applying flat base constraints...")
        displacement_func = _constrain_base_displacement(
            displacement_func, clip_origin, clip_normal
        )

    # 2. Set Up Function Space and Boundary Conditions
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    # bc = get_boundary_conditions(domain, V, displacement_func)
    bc = get_boundary_conditions(
        domain,
        V,
        displacement_func,
    )
    # 3. Solve PDE
    if solver_method == "hyperelastic":
        u_solution = models.solve_hyperelastic(domain, V, bc)
    elif solver_method == "laplace":
        u_solution = models.solve_laplace(domain, V, bc)
    else:
        raise ValueError("Invalid solver method. Choose 'hyperelastic' or 'laplace'.")

    # 4. Apply Warp
    logger.info("Applying warp to mesh coordinates...")
    domain.geometry.x[:, :3] += u_solution.x.array.reshape(-1, 3)

import dolfinx
import numpy as np

from . import interpolation, models


def get_boundary_conditions(domain, V, displacement_func):
    """Interpolates the displacement function onto the mesh boundary."""
    print("Applying boundary conditions...")
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

    # 2. Set Up Function Space and Boundary Conditions
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    bc = get_boundary_conditions(domain, V, displacement_func)

    # 3. Solve PDE
    if solver_method == "hyperelastic":
        u_solution = models.solve_hyperelastic(domain, V, bc)
    elif solver_method == "laplace":
        u_solution = models.solve_laplace(domain, V, bc)
    else:
        raise ValueError("Invalid solver method. Choose 'hyperelastic' or 'laplace'.")

    # 4. Apply Warp
    print("Applying warp to mesh coordinates...")
    domain.geometry.x[:, :3] += u_solution.x.array.reshape(-1, 3)

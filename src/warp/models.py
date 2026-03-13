import dolfinx
import dolfinx.fem.petsc
import ufl


def solve_laplace(
    domain: dolfinx.mesh.Mesh, V: dolfinx.fem.FunctionSpace, bc: dolfinx.fem.DirichletBC
) -> dolfinx.fem.Function:
    """
    Solves the linear Laplace equation for mesh warping.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The computational domain (template mesh).
    V : dolfinx.fem.FunctionSpace
        The vector function space defined on the domain.
    bc : dolfinx.fem.DirichletBC
        The Dirichlet boundary condition prescribing the displacement.

    Returns
    -------
    dolfinx.fem.Function
        The computed displacement field.
    """
    print("Setting up Linear Laplace PDE...")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (
        ufl.inner(dolfinx.fem.Constant(domain, dolfinx.default_scalar_type((0.0, 0.0, 0.0))), v)
        * ufl.dx
    )

    print("Solving linear Laplace warping...")
    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="warp_laplace_",
    )
    u_solution = problem.solve()

    return u_solution


def solve_hyperelastic(domain, V, bc, E=1.0e4, nu=0.45):
    """
    Solves a non-linear Neo-Hookean hyperelastic model to prevent element inversion.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The computational domain (template mesh).
    V : dolfinx.fem.FunctionSpace
        The vector function space defined on the domain.
    bc : dolfinx.fem.DirichletBC
        The Dirichlet boundary condition prescribing the displacement.
    E : float, optional
        Young's modulus of the hyperelastic material, by default 1.0e4.
    nu : float, optional
        Poisson's ratio of the hyperelastic material, by default 0.45.

    Returns
    -------
    dolfinx.fem.Function
        The computed displacement field.
    """
    print("Setting up Non-Linear Hyperelastic PDE...")
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    J = ufl.variable(ufl.det(F))

    # Material Parameters (Stiff rubber to resist volume loss)
    mu = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(E / (2.0 * (1.0 + nu))))
    lmbda = dolfinx.fem.Constant(
        domain, dolfinx.default_scalar_type(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    )

    # Neo-Hookean Strain Energy
    Ic = ufl.variable(ufl.tr(C))
    psi = (mu / 2) * (Ic - d) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
    Pi = psi * ufl.dx

    # Define the Non-Linear Residual
    residual = ufl.derivative(Pi, u, v)

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual, u, bcs=[bc], petsc_options_prefix="warp_hyperelastic_"
    )

    print("Solving non-linear hyperelastic warping...")
    return problem.solve()

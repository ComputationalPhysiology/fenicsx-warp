import logging

from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl

logger = logging.getLogger(__name__)


class ConvergenceError(RuntimeError):
    """Raised when the PETSc solver of a warp does not converge."""


def _reason_name(reasons, reason: int) -> str:
    names = {v: k for k, v in vars(reasons).items() if isinstance(v, int) and not k.startswith("_")}
    return names.get(reason, str(reason))


def check_convergence(solver) -> None:
    """Raise a :class:`ConvergenceError` unless the PETSc solver (SNES or KSP) converged.

    Parameters
    ----------
    solver : petsc4py.PETSc.SNES or petsc4py.PETSc.KSP
        The solver after its solve.

    Raises
    ------
    ConvergenceError
        If the converged reason is not positive.
    """
    reason = solver.getConvergedReason()
    if reason > 0:
        return
    if isinstance(solver, PETSc.SNES):
        kind, name = "SNES", _reason_name(PETSc.SNES.ConvergedReason, reason)
    else:
        kind, name = "KSP", _reason_name(PETSc.KSP.ConvergedReason, reason)
    raise ConvergenceError(
        f"{kind} ({solver.getOptionsPrefix()}) did not converge: {name} ({reason}) "
        f"after {solver.getIterationNumber()} iterations"
    )


def solve_laplace(
    domain: dolfinx.mesh.Mesh,
    V: dolfinx.fem.FunctionSpace,
    bc: dolfinx.fem.DirichletBC,
    petsc_options: dict | None = None,
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
    petsc_options : dict, optional
        PETSc options for the linear solver (KSP), merged over the default
        ``{"ksp_type": "preonly", "pc_type": "lu"}``. Global PETSc options with the
        prefix ``warp_laplace_`` are read as well.

    Returns
    -------
    dolfinx.fem.Function
        The computed displacement field.

    Raises
    ------
    ConvergenceError
        If the linear solver does not converge.
    """
    logger.info("Setting up Linear Laplace PDE...")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (
        ufl.inner(dolfinx.fem.Constant(domain, dolfinx.default_scalar_type((0.0, 0.0, 0.0))), v)
        * ufl.dx
    )

    logger.info("Solving linear Laplace warping...")
    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", **(petsc_options or {})},
        petsc_options_prefix="warp_laplace_",
    )
    u_solution = problem.solve()
    check_convergence(problem.solver)
    assert isinstance(u_solution, dolfinx.fem.Function)  # a single form gives a single function

    return u_solution


def solve_hyperelastic(domain, V, bc, E=1.0e4, nu=0.45, petsc_options: dict | None = None):
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
    petsc_options : dict, optional
        PETSc options for the nonlinear solver (SNES) and its linear solver, e.g.
        ``{"snes_linesearch_type": "basic", "ksp_type": "preonly", "pc_type": "lu"}``.
        By default PETSc's defaults (Newton with a backtracking line search). Global
        PETSc options with the prefix ``warp_hyperelastic_`` are read as well.

    Returns
    -------
    dolfinx.fem.Function
        The computed displacement field.

    Raises
    ------
    ConvergenceError
        If the nonlinear solver does not converge.
    """
    logger.info("Setting up Non-Linear Hyperelastic PDE...")
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

    # Passing a dict (even an empty one) makes dolfinx call setFromOptions,
    # so global options with the prefix are read too
    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual,
        u,
        bcs=[bc],
        petsc_options_prefix="warp_hyperelastic_",
        petsc_options=dict(petsc_options or {}),
    )

    logger.info("Solving non-linear hyperelastic warping...")
    u_solution = problem.solve()
    check_convergence(problem.solver)
    return u_solution

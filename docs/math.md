# Mathematical Theory of Template Warping

This document provides a detailed mathematical background for the fenicsx-warp library, explaining how volumetric meshes are robustly morphed between different configurations of a Statistical Shape Model (SSM).

## Geometric Foundation

The fundamental assumption of this library is the existence of a fixed point correspondence. Given a template (mean) configuration and a target configuration, every boundary node $i$ in the template corresponds exactly to node $i$ in the target.

Let $\mathbf{X}_{ref} \in \mathbb{R}^{N \times 3}$ be the reference coordinates of $N$ landmarks (from the cardiac atlas) and $\mathbf{X}_{tar} \in \mathbb{R}^{N \times 3}$ be the target coordinates. The discrete boundary displacement field $\mathbf{u}_b$ is defined as:

$$\mathbf{u}_b^i = \mathbf{X}_{tar}^i - \mathbf{X}_{ref}^i, \quad i = 1, \dots, N$$

## Boundary Interpolation

Since the volumetric mesh boundary nodes $\mathbf{x} \in \partial\Omega$ do not necessarily coincide with the atlas landmarks, we must interpolate the discrete displacements $\mathbf{u}_b$ to create a continuous boundary condition $\mathbf{g}(\mathbf{x})$.

### Radial Basis Functions (RBF)

We utilize Thin Plate Splines (TPS) to ensure maximum smoothness ($C^1$ continuity). The RBF interpolant takes the form:

$$\mathbf{s}(\mathbf{x}) = \sum_{i=1}^N \mathbf{w}_i \phi(||\mathbf{x} - \mathbf{X}_{ref}^i||) + \mathbf{P}(\mathbf{x})$$

where $\phi(r) = r^2 \ln(r)$ for TPS. The weights $\mathbf{w}_i$ and polynomial terms $\mathbf{P}$ are found by solving a dense linear system such that $\mathbf{s}(\mathbf{X}_{ref}^i) = \mathbf{u}_b^i$.

## Governing Equations for Volumetric Deformation

To propagate the boundary displacement $\mathbf{g}(\mathbf{x})$ into the interior $\Omega$, we solve a partial differential equation (PDE) for the displacement field $\mathbf{u}(\mathbf{x})$.

### Linear Laplace Model

A fast approach is to solve the vector Laplace equation:

$$\nabla^2 \mathbf{u} = 0 \quad \text{in } \Omega$$

$$\mathbf{u} = \mathbf{g}(\mathbf{x}) \quad \text{on } \partial\Omega$$

This is equivalent to minimizing the Dirichlet energy $\int_{\Omega} ||\nabla \mathbf{u}||^2 d\Omega$. While computationally efficient, it treats components independently and does not prevent element inversion.

### Non-Linear Hyperelastic Model (Neo-Hookean)

To guarantee mesh validity under extreme PCA modes, we treat the mesh as a physical hyperelastic body. We seek the displacement field that minimizes the total potential energy:

$$\Pi(\mathbf{u}) = \int_{\Omega} \psi(\mathbf{F}) d\Omega - \int_{\partial\Omega} \mathbf{t} \cdot \mathbf{u} ds$$

where $\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}$ is the deformation gradient. We use a compressible Neo-Hookean strain energy density $\psi$:

$$\psi = \frac{\mu}{2} (I_c - 3) - \mu \ln(J) + \frac{\lambda}{2} (\ln(J))^2$$

where:

$J = \det(\mathbf{F})$ is the volumetric Jacobian.

$I_c = \text{tr}(\mathbf{F}^T \mathbf{F})$ is the first invariant of the right Cauchy-Green tensor.

$\mu, \lambda$ are Lamé parameters.

Element Inversion Penalty: As an element's volume approaches zero ($J \to 0$), the terms $-\mu \ln(J)$ and $\frac{\lambda}{2}(\ln(J))^2$ approach infinity. This provides a mathematical barrier that prevents elements from inverting, even under high-strain warping.

## Numerical Implementation

The problem is solved using the Finite Element Method in FEniCSx. For the non-linear model, we use a Newton-Raphson scheme to find $\mathbf{u}$ such that the residual $L(\mathbf{u}; \mathbf{v}) = \delta \Pi = 0$:

$$\int_{\Omega} \mathbf{P}(\mathbf{F}) : \nabla \mathbf{v} d\Omega = 0 \quad \forall \mathbf{v} \in \mathcal{V}$$

where $\mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}$ is the first Piola-Kirchhoff stress tensor.

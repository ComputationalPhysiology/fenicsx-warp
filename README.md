# FEniCSx-Warp

`fenicsx-warp` is a specialized Python library designed for robust volumetric mesh generation of cardiac geometries using Template Warping.

Traditional mesh generation for Statistical Shape Models (SSMs) often fails at extreme PCA modes due to surface self-intersections. This library solves that problem by taking a high-quality template mesh (the mean shape) and mathematically warping it to target geometries using partial differential equations (PDEs) in FEniCSx.


## Install

```
python3 -m pip install fenicsx-warp
```

## Quick Start
See example in the [documentation](https://computationalphysiology.github.io/fenicsx-warp).

## Solver options and convergence
`warp_mesh(..., petsc_options={...})` passes PETSc options to the solver: the Newton solver (SNES)
of the hyperelastic warp, or the linear solver (KSP) of the Laplace warp. Global PETSc options with
the prefix `warp_hyperelastic_` or `warp_laplace_` are read as well. If the solver does not converge,
`warp_mesh` raises `warp.ConvergenceError` and leaves the mesh unchanged.

```python
import warp

try:
    warp.warp_mesh(mesh, points_mean, points_target,
                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
except warp.ConvergenceError as e:
    print(e)  # e.g. "SNES (warp_hyperelastic_) did not converge: DIVERGED_LINE_SEARCH (-6) ..."
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

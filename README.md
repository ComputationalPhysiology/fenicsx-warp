# FEniCSx-Warp

`fenicsx-warp` is a specialized Python library designed for robust volumetric mesh generation of cardiac geometries using Template Warping.

Traditional mesh generation for Statistical Shape Models (SSMs) often fails at extreme PCA modes due to surface self-intersections. This library solves that problem by taking a high-quality template mesh (the mean shape) and mathematically warping it to target geometries using partial differential equations (PDEs) in FEniCSx.


## Install
```
python3 -m pip install git+https://github.com/ComputationalPhysiology/fenicsx-warp
```

## Quick Start
See example in the [documentation](https://computationalphysiology.github.io/fenicsx-warp).

## License
This project is licensed under the MIT License - see the LICENSE file for details.

# # Cardiac Mesh Warping Demo
# This demo demonstrates how to warp a volumetric template mesh (representing a mean heart shape)
# to a target configuration defined by a specific Principal Component Analysis (PCA) mode
# from the UK Biobank cardiac atlas.

# Workflow:
# 1. Generate surfaces and a volumetric mesh for the population mean shape.
# 2. Extract point clouds for the mean (reference) and the target mode.
# 3. Use FEniCSx and the `fenicsx-warp` library to solve a non-linear hyperelastic
#    PDE to morph the volume smoothly.
# 4. Visualize the reference wireframe vs. the warped volume.

import logging
from pathlib import Path

from mpi4py import MPI

import dolfinx
import ukb.atlas  # UK Biobank atlas utilities
import ukb.mesh  # UK Biobank mesh generation
import ukb.surface  # UK Biobank surface extraction

import warp  # The fenicsx-warp library

# Try to import PyVista for 3D visualization
try:
    import pyvista

    has_pyvista = True
except ImportError:
    has_pyvista = False

# ## Setup Directories and Data
# We define a cache directory for the atlas file and a folder for the template mesh.

logging.basicConfig(level=logging.INFO)
cache_dir = Path.home() / ".ukb"
atlas_file = cache_dir / "UKBRVLV.h5"
mean_folder = Path("data_mean")
mean_folder.mkdir(exist_ok=True)

# ## 2. Generate the Mean Template
# First, we extract the surfaces for the mean shape (mode -1, std 0.0).
# This provides the anatomical landmarks used for the reference configuration.

print("Extracting mean surfaces...")
ukb.surface.main(
    folder=mean_folder,
    case="ED",  # End-Diastole
    mode=-1,  # Mean shape
    std=0.0,
    cache_dir=cache_dir,
)

# Generate the volumetric 3D mesh from the mean surfaces using Gmsh.
# This template will be warped to create all subsequent shapes.

print("Generating volumetric template mesh...")
ukb.mesh.main(
    folder=mean_folder,
    case="ED",
    char_length_max=5.0,
    char_length_min=5.0,
    verbose=False,
    clipped=False,
)

# ### Extract Point Correspondence
# We extract the landmark coordinates for the Reference (mean) and Target shapes.
# These two arrays have exact point-to-point correspondence.

print("Generating atlas point clouds...")
points_mean = ukb.atlas.generate_points(filename=atlas_file, mode=-1, std=0.0).ED

# For the target, we use Mode 1 (capturing overall size/mass) at 2.0 standard deviations.

mode = 1
std = 2.0
points_target = ukb.atlas.generate_points(filename=atlas_file, mode=mode, std=std).ED

# ## Load Mesh into FEniCSx
# We load the Gmsh (.msh) file into a Dolfinx domain.

comm = MPI.COMM_WORLD
msh = dolfinx.io.gmsh.read_from_msh(mean_folder / "ED.msh", comm, 0, gdim=3)
domain = msh.mesh
facet_tags = msh.facet_tags

# Save the template for comparison

with dolfinx.io.XDMFFile(domain.comm, "template_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags, domain.geometry)

# ### Execute Warping
# We configure the warping parameters:
# - `rbf`: Radial Basis Functions ensure a smooth boundary displacement field.
# - `hyperelastic`: Uses a Neo-Hookean model to prevent element inversion.

interpolation_method = "rbf"
solver_method = "hyperelastic"

print(f"Starting warp using {interpolation_method} interpolation and {solver_method} solver...")
warp.warp_mesh(
    domain=domain,
    points_reference=points_mean,
    points_target=points_target,
    interpolation_method=interpolation_method,
    solver_method=solver_method,
)

# ## Export Result
# The warped mesh is saved as an XDMF file. All meshes generated this way
# share the exact same node and element indexing (Isomorphism).

output_filename = f"warped_mode{mode}_{std}_{solver_method}_{interpolation_method}.xdmf"
print(f"Saving results to {output_filename}...")
with dolfinx.io.XDMFFile(domain.comm, output_filename, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags, domain.geometry)

# ## Visualization

if has_pyvista:
    print("Launching PyVista visualization...")
    plotter = pyvista.Plotter(title="Mesh Warping Comparison")

    # Plot the original template as a red wireframe
    vtk_mesh_template = dolfinx.plot.vtk_mesh(
        domain, domain.topology.dim
    )  # domain is updated in-place, reload if comparing
    # Note: To show original wireframe simultaneously, one would need
    # to load the saved template_mesh.xdmf
    with dolfinx.io.XDMFFile(comm, "template_mesh.xdmf", "r") as xdmf:
        template_domain = xdmf.read_mesh()

    vtk_template = dolfinx.plot.vtk_mesh(template_domain, template_domain.topology.dim)
    grid_template = pyvista.UnstructuredGrid(*vtk_template)
    plotter.add_mesh(
        grid_template, color="red", style="wireframe", opacity=0.3, label="Mean Template"
    )

    # Plot the warped mesh as a solid volume
    vtk_warped = dolfinx.plot.vtk_mesh(domain, domain.topology.dim)
    grid_warped = pyvista.UnstructuredGrid(*vtk_warped)
    plotter.add_mesh(grid_warped, color="teal", label=f"Warped (Mode {mode}, std {std})")

    plotter.add_legend()
    plotter.view_xy()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(output_filename.replace(".xdmf", ".png"))

print("Demo completed successfully.")

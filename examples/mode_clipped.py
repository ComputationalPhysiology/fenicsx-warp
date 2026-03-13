# # Cardiac Mesh Warping Demo (Clipped Version)


# This demo demonstrates how to warp a clipped volumetric template mesh
# (representing a mean heart shape with a flat base) to a target configuration
# defined by a specific Principal Component Analysis (PCA) mode from the
# UK Biobank cardiac atlas.

# Workflow:
# 1. Generate complete surfaces for the population mean shape.
# 2. Clip the surfaces using a mathematical plane to remove the atria/outflow tracts,
#    creating a flat basal plane.
# 3. Generate a volumetric tetrahedral mesh from the clipped surfaces.
# 4. Extract the full (unclipped) point clouds for the mean (reference) and target modes.
# 5. Use FEniCSx and the `fenicsx-warp` library to morph the volume smoothly.
#    Critically, we apply a geometric constraint to ensure the clipped base
#    remains perfectly flat despite the deformation.
# 6. Visualize the reference wireframe vs. the warped volume.


from pathlib import Path

from mpi4py import MPI

import dolfinx
import ukb.atlas  # UK Biobank atlas utilities
import ukb.clip  # UK Biobank clipping utilities
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

# We define a cache directory for the atlas HDF5 file and a folder
# for the generated template mesh and intermediate surfaces.

cache_dir = Path.home() / ".ukb"
atlas_file = cache_dir / "UKBRVLV.h5"
mean_folder = Path("data_mean_clipped")
mean_folder.mkdir(exist_ok=True)


# ## Generate the Mean Template
# Step 2a: Extract the raw, unclipped surfaces for the mean shape (mode -1, std 0.0).
# This provides the baseline anatomical landmarks.

print("Extracting mean surfaces...")
ukb.surface.main(
    folder=mean_folder,
    case="ED",  # End-Diastole
    mode=-1,  # Mean shape
    std=0.0,
    cache_dir=cache_dir,
)

# ### Clip the surfaces.
# We define a specific origin and normal vector to slice off the top of the heart.

clip_origin = ukb.clip.default_origin()
clip_normal = ukb.clip.default_normal()

print("Clipping surfaces to create a flat base...")
ukb.clip.main(
    folder=mean_folder,
    case="ED",
    origin_x=clip_origin[0],
    origin_y=clip_origin[1],
    origin_z=clip_origin[2],
    normal_x=clip_normal[0],
    normal_y=clip_normal[1],
    normal_z=clip_normal[2],
)

# ### Generate the volumetric 3D mesh from the clipped surfaces using Gmsh.
# The 'clipped=True' flag ensures Gmsh creates a closed volume by adding a flat
# surface at the clipping plane.

print("Generating clipped volumetric template mesh...")
ukb.mesh.main(
    folder=mean_folder,
    case="ED",
    char_length_max=5.0,
    char_length_min=5.0,
    verbose=False,
    clipped=True,
)

# ## Extract Point Correspondence

# We extract the complete landmark coordinates for the Reference (mean) and Target shapes.
# Note: Even though our mesh is clipped, we use the FULL point clouds here.
# The RBF interpolator will use the full point cloud to build a global deformation field.

print("Generating atlas point clouds...")
points_mean = ukb.atlas.generate_points(filename=atlas_file, mode=-1, std=0.0).ED

# For the target, we use Mode 1 (e.g., overall size/mass) at 3.0 standard deviations.
# A high standard deviation makes the deformation highly pronounced.

mode = 1
std = 3.0
points_target = ukb.atlas.generate_points(filename=atlas_file, mode=mode, std=std).ED


# ## Load Mesh into FEniCSx
# We load the clipped Gmsh (.msh) file into a distributed Dolfinx domain.
comm = MPI.COMM_WORLD
msh = dolfinx.io.gmsh.read_from_msh(mean_folder / "ED_clipped.msh", comm, 0, gdim=3)
domain = msh.mesh
facet_tags = msh.facet_tags

# Save the original template mesh to disk for later comparison.
with dolfinx.io.XDMFFile(domain.comm, "template_mesh_clipped.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags, domain.geometry)


# ## Execute Warping with Flat-Base Constraint
# We configure the warping parameters:
# - `rbf`: Radial Basis Functions ensure a globally smooth boundary displacement field.
# - `hyperelastic`: Uses a Neo-Hookean model to prevent elements from turning inside-out.
interpolation_method = "rbf"
solver_method = "hyperelastic"

print(f"Starting warp using {interpolation_method} interpolation and {solver_method} solver...")

# **Crucial Step for Clipped Meshes**:
# Because the global RBF interpolation field would naturally warp the flat base out
# of its plane (to follow the unclipped shape), we pass `clip_origin` and `clip_normal`.
# The warper will geometrically identify the boundary nodes lying on this plane and
# mathematically project their normal displacements to the *average* normal displacement.
# This ensures the base remains a perfectly flat plane while still allowing it to
# translate longitudinally or contract radially.

warp.warp_mesh(
    domain=domain,
    points_reference=points_mean,
    points_target=points_target,
    interpolation_method=interpolation_method,
    solver_method=solver_method,
    clip_origin=clip_origin,
    clip_normal=clip_normal,
)


# ## Export Result
# The warped mesh is saved. Because the mesh was warped in-place, the connectivity
# (nodes and elements) matches the template mesh perfectly (Isomorphism).

output_filename = f"warped_mode{mode}_{std}_{solver_method}_{interpolation_method}_clipped.xdmf"
print(f"Saving results to {output_filename}...")
with dolfinx.io.XDMFFile(domain.comm, output_filename, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags, domain.geometry)

# ## Visualization

if has_pyvista:
    print("Launching PyVista visualization...")
    plotter = pyvista.Plotter(title="Clipped Mesh Warping Comparison")

    # Load the original unwarped template we saved earlier
    with dolfinx.io.XDMFFile(comm, "template_mesh_clipped.xdmf", "r") as xdmf:
        template_domain = xdmf.read_mesh()

    # Plot the original template as a red, semi-transparent wireframe
    vtk_template = dolfinx.plot.vtk_mesh(template_domain, template_domain.topology.dim)
    grid_template = pyvista.UnstructuredGrid(*vtk_template)
    plotter.add_mesh(
        grid_template,
        color="red",
        style="wireframe",
        opacity=0.3,
        label="Mean Template (Flat Base)",
    )

    # Plot the newly warped mesh as a solid teal volume
    vtk_warped = dolfinx.plot.vtk_mesh(domain, domain.topology.dim)
    grid_warped = pyvista.UnstructuredGrid(*vtk_warped)
    plotter.add_mesh(grid_warped, color="teal", label=f"Warped (Mode {mode}, std {std})")

    plotter.add_legend()
    plotter.view_xy()

    # Display interactively or save as screenshot if running headless
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(output_filename.replace(".xdmf", ".png"))

print("Demo completed successfully.")

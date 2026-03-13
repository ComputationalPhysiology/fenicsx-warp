from pathlib import Path

from mpi4py import MPI

import dolfinx
import ukb.atlas  # pip install ukb-atlas
import ukb.mesh  # pip install ukb-atlas
import ukb.surface  # pip install ukb-atlas

import warp

# Create mean mesh
cache_dir = Path.home() / ".ukb"
atlas_file = cache_dir / "UKBRVLV.h5"
mean_folder = Path("data_mean")
mean_folder.mkdir(exist_ok=True)

ukb.surface.main(
    folder=mean_folder,
    case="ED",
    mode=-1,
    std=0.0,
    cache_dir=cache_dir,
)

ukb.mesh.main(
    folder=mean_folder,
    case="ED",
    char_length_max=5.0,
    char_length_min=5.0,
    verbose=False,
    clipped=False,
)

points_mean = ukb.atlas.generate_points(filename=atlas_file, mode=-1, std=0.0).ED
mode = 1
std = 2.0
points_target = ukb.atlas.generate_points(filename=atlas_file, mode=mode, std=std).ED

comm = MPI.COMM_WORLD
msh = dolfinx.io.gmsh.read_from_msh(mean_folder / "ED.msh", MPI.COMM_WORLD, 0, gdim=3)

domain = msh.mesh
facet_tags = msh.facet_tags

with dolfinx.io.XDMFFile(domain.comm, "template_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags, domain.geometry)

interpolation_method = "rbf"
solver_method = "hyperelastic"

warp.warp_mesh(
    domain=domain,
    points_reference=points_mean,
    points_target=points_target,
    interpolation_method=interpolation_method,
    solver_method=solver_method,
)


with dolfinx.io.XDMFFile(
    domain.comm, f"warped_mode{mode}_{std}_{solver_method}_{interpolation_method}.xdmf", "w"
) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags, domain.geometry)

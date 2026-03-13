import logging
from pathlib import Path

from mpi4py import MPI

import dolfinx
import ukb.atlas  # pip install ukb-atlas
import ukb.mesh  # pip install ukb-atlas
import ukb.surface  # pip install ukb-atlas

import warp

logging.basicConfig(level=logging.INFO)
# Create mean mesh
burns_path = Path("BioBank_EDES_200.mat")
mean_folder = Path("data_mean_burns")
mean_folder.mkdir(exist_ok=True)

ukb.surface.main(
    folder=mean_folder,
    case="ED",
    mode=-1,
    std=0.0,
    burns_path=burns_path,
)

ukb.mesh.main(
    folder=mean_folder,
    case="ED",
    char_length_max=5.0,
    char_length_min=5.0,
    verbose=False,
    clipped=False,
)

points_mean = ukb.atlas.generate_points_burns(filename=burns_path, mode=-1, std=0.0).ED

pc_scores = [
    0.524396147,
    -1.980220282,
    1.734850838,
    1.121269184,
    -0.327393264,
    -0.215172643,
    -1.900164259,
    0.74459319,
    -1.063954452,
    -0.615966305,
    -0.939961167,
    -0.217399598,
    0.728223858,
    -0.195969312,
    1.418891817,
    -0.605124636,
    2.327259055,
    0.807671764,
    0.227093642,
    0.228008102,
    0.030184821,
    1.740475635,
    2.049635063,
    0.145756787,
    0.586155106,
]

points_target = ukb.atlas.generate_points_burns(filename=burns_path, score=pc_scores).ED

comm = MPI.COMM_WORLD
msh = dolfinx.io.gmsh.read_from_msh(mean_folder / "ED.msh", MPI.COMM_WORLD, 0, gdim=3)

domain = msh.mesh
facet_tags = msh.facet_tags

with dolfinx.io.XDMFFile(domain.comm, "template_mesh_burns.xdmf", "w") as xdmf:
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
    domain.comm, f"warped_pca_{solver_method}_{interpolation_method}_burns.xdmf", "w"
) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags, domain.geometry)

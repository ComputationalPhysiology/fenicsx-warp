from importlib.metadata import metadata

from . import interpolation, models, warp
from .warp import warp_mesh

meta = metadata("fenicsx-warp")
__version__ = meta["Version"]
__author__ = meta["Author-email"]
__license__ = meta["license-expression"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


__all__ = ["interpolation", "models", "warp", "warp_mesh"]

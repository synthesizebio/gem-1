"""
Public entry-point for the text_to_rna package.

The package bundles pretrained models plus helpers that are reused by the
inference scripts.  Keeping imports here prevents inference code from having
to reach into deep submodules just to instantiate the core model.
"""

# --- Back-compat shim: allow "import constants", "import data", etc. ---
import importlib, sys as _sys

# List any submodules that legacy code imports without the package prefix
for _name in ("constants", "data", "metrics", "vocab", "gene_order", "scalers"):
    try:
        _sys.modules.setdefault(_name, importlib.import_module(f"{__name__}.{_name}"))
    except Exception:
        pass

from importlib import resources

from . import data, metrics, scalers, vocab
from .model.base_model import SynthesizeBioModel

__all__ = [
    "data",
    "metrics",
    "scalers",
    "vocab",
    "SynthesizeBioModel",
    "package_path",
]


def package_path(*path_segments: str) -> str:
    """Return an absolute path for a resource shipped inside the package."""
    return str(resources.files(__package__).joinpath(*path_segments))

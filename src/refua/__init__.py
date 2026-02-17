from importlib.metadata import version as _distribution_version
from pathlib import Path
import tomllib

from refua.api import download_assets
from refua.boltz.api import Boltz2
from refua.boltzgen.api import BoltzGen
from refua.chem import (
    MolProperties,
    SM,
    SmallMolecule,
    available_mol_properties,
    available_mol_property_groups,
    register_mol_property,
)
from refua.protein import (
    ProteinProperties,
    available_protein_properties,
    available_protein_property_groups,
    protein_property_specs,
    register_protein_property,
)
from refua.unified import (
    AntibodyBinders,
    Binder,
    BinderDesigns,
    Complex,
    DNA,
    RefuaEnv,
    Protein,
    RNA,
    antibody_framework_specs,
)


def _read_version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    if not version:
        return None
    return str(version)


def _resolve_version() -> str:
    local_version = _read_version_from_pyproject()
    if local_version is not None:
        return local_version
    return _distribution_version("refua")


__version__ = _resolve_version()

__all__ = [
    "Boltz2",
    "BoltzGen",
    "AntibodyBinders",
    "Binder",
    "BinderDesigns",
    "Complex",
    "DNA",
    "MolProperties",
    "ProteinProperties",
    "Protein",
    "RefuaEnv",
    "RNA",
    "SM",
    "SmallMolecule",
    "available_mol_properties",
    "available_mol_property_groups",
    "available_protein_properties",
    "available_protein_property_groups",
    "protein_property_specs",
    "download_assets",
    "register_mol_property",
    "register_protein_property",
    "antibody_framework_specs",
]

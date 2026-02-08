from importlib.metadata import PackageNotFoundError, version

from refua.api import download_assets
from refua.chem import (
    MolProperties,
    SM,
    SmallMolecule,
    available_mol_properties,
    available_mol_property_groups,
    register_mol_property,
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
from refua.boltz.api import Boltz2
from refua.boltzgen.api import BoltzGen

try:  # noqa: SIM105
    __version__ = version("refua")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = [
    "Boltz2",
    "BoltzGen",
    "AntibodyBinders",
    "Binder",
    "BinderDesigns",
    "Complex",
    "DNA",
    "MolProperties",
    "Protein",
    "RefuaEnv",
    "RNA",
    "SM",
    "SmallMolecule",
    "available_mol_properties",
    "available_mol_property_groups",
    "download_assets",
    "register_mol_property",
    "antibody_framework_specs",
]

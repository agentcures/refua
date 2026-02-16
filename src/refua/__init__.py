from importlib.metadata import PackageNotFoundError, version

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

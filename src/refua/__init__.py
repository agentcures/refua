from importlib import import_module
from importlib.metadata import version as _distribution_version
from pathlib import Path
import tomllib


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

_EXPORTS = {
    "Boltz2": ("refua.boltz.api", "Boltz2"),
    "BoltzGen": ("refua.boltzgen.api", "BoltzGen"),
    "AntibodyBinders": ("refua.unified", "AntibodyBinders"),
    "Binder": ("refua.unified", "Binder"),
    "BinderDesigns": ("refua.unified", "BinderDesigns"),
    "Complex": ("refua.unified", "Complex"),
    "DNA": ("refua.unified", "DNA"),
    "MolProperties": ("refua.chem", "MolProperties"),
    "ProteinProperties": ("refua.protein", "ProteinProperties"),
    "Protein": ("refua.unified", "Protein"),
    "RefuaEnv": ("refua.unified", "RefuaEnv"),
    "RNA": ("refua.unified", "RNA"),
    "SM": ("refua.chem", "SM"),
    "SmallMolecule": ("refua.chem", "SmallMolecule"),
    "available_mol_properties": ("refua.chem", "available_mol_properties"),
    "available_mol_property_groups": ("refua.chem", "available_mol_property_groups"),
    "available_protein_properties": (
        "refua.protein",
        "available_protein_properties",
    ),
    "available_protein_property_groups": (
        "refua.protein",
        "available_protein_property_groups",
    ),
    "protein_property_specs": ("refua.protein", "protein_property_specs"),
    "download_assets": ("refua.api", "download_assets"),
    "register_mol_property": ("refua.chem", "register_mol_property"),
    "register_protein_property": ("refua.protein", "register_protein_property"),
    "antibody_framework_specs": ("refua.unified", "antibody_framework_specs"),
}

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


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})

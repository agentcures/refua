"""Small-molecule property helpers built on RDKit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.rdchem import Mol

PropertyValue = float | int | tuple[float | int, ...] | None
PropertyFn = Callable[[Mol], PropertyValue]


@dataclass(frozen=True, slots=True)
class MolPropertySpec:
    """RDKit-backed property definition."""

    name: str
    fn: PropertyFn
    description: str
    groups: tuple[str, ...] = ()


_PROPERTY_REGISTRY: dict[str, MolPropertySpec] = {}
_GROUP_REGISTRY: dict[str, list[str]] = {}

_ALIASES: dict[str, str] = {
    "mw": "mol_wt",
    "molwt": "mol_wt",
    "molecular_weight": "mol_wt",
    "exact_mw": "exact_mol_wt",
    "exactmolwt": "exact_mol_wt",
    "logp": "mol_log_p",
    "clogp": "mol_log_p",
    "alogp": "mol_log_p",
    "hbd": "num_h_donors",
    "hba": "num_h_acceptors",
    "rotb": "num_rotatable_bonds",
    "rotatable_bonds": "num_rotatable_bonds",
    "rings": "ring_count",
    "aromatic_rings": "num_aromatic_rings",
    "heavy_atoms": "heavy_atom_count",
    "hetero_atoms": "num_heteroatoms",
    "h_erg": "herg",
}

_BASIC_PROPERTIES = {
    "mol_wt",
    "exact_mol_wt",
    "mol_log_p",
    "tpsa",
    "num_h_donors",
    "num_h_acceptors",
    "num_rotatable_bonds",
    "ring_count",
    "num_aromatic_rings",
    "heavy_atom_count",
    "num_heteroatoms",
    "fraction_csp3",
    "formal_charge",
}

_LIPINSKI_PROPERTIES = {
    "mol_wt",
    "mol_log_p",
    "num_h_donors",
    "num_h_acceptors",
    "num_rotatable_bonds",
    "tpsa",
}

_DRUGLIKE_PROPERTIES = {
    "fraction_csp3",
    "qed",
}


def _to_snake(name: str) -> str:
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def _normalize_name(name: str) -> str:
    key = _to_snake(name)
    return _ALIASES.get(key, key)


def _register_property(spec: MolPropertySpec, *, allow_existing: bool = False) -> None:
    if spec.name in _PROPERTY_REGISTRY:
        if allow_existing:
            return
        msg = f"Property already registered: {spec.name}"
        raise ValueError(msg)
    _PROPERTY_REGISTRY[spec.name] = spec
    for group in spec.groups:
        _GROUP_REGISTRY.setdefault(group, []).append(spec.name)


def register_mol_property(
    name: str,
    fn: PropertyFn,
    *,
    description: str,
    groups: Iterable[str] = (),
) -> None:
    """Register a new small-molecule property."""
    normalized = _normalize_name(name)
    if normalized in _PROPERTY_REGISTRY:
        msg = f"Property already registered: {normalized}"
        raise ValueError(msg)
    spec = MolPropertySpec(
        name=normalized,
        fn=fn,
        description=description,
        groups=tuple(groups),
    )
    _register_property(spec)


def available_mol_properties() -> tuple[str, ...]:
    """Return the available property names."""
    return tuple(_PROPERTY_REGISTRY)


def available_mol_property_groups() -> tuple[str, ...]:
    """Return the available property groups."""
    return tuple(_GROUP_REGISTRY)


def mol_property_specs() -> Mapping[str, MolPropertySpec]:
    """Return property specs keyed by name."""
    return dict(_PROPERTY_REGISTRY)


def _normalize_value(value: PropertyValue) -> PropertyValue:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, (list, tuple)) and hasattr(value, "item"):
        casted = value.item()
        if isinstance(casted, (int, float)):
            return casted
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(v) for v in value)  # type: ignore[arg-type]
    return float(value)


def _descriptor_description(name: str, fn: PropertyFn) -> str:
    doc = getattr(fn, "__doc__", None)
    if doc:
        return doc.strip().splitlines()[0]
    return f"RDKit descriptor {name}."


def _register_rdkit_descriptors() -> None:
    for name, fn in Descriptors.descList:
        normalized = _normalize_name(name)
        groups = ["rdkit"]
        if normalized in _BASIC_PROPERTIES:
            groups.append("basic")
        if normalized in _LIPINSKI_PROPERTIES:
            groups.append("lipinski")
        if normalized in _DRUGLIKE_PROPERTIES:
            groups.append("druglike")
        spec = MolPropertySpec(
            name=normalized,
            fn=fn,
            description=_descriptor_description(name, fn),
            groups=tuple(groups),
        )
        _register_property(spec, allow_existing=True)


_register_rdkit_descriptors()

_EXTRA_PROPERTIES = (
    MolPropertySpec(
        name="formal_charge",
        fn=Chem.GetFormalCharge,
        description="Formal charge.",
        groups=("basic",),
    ),
    MolPropertySpec(
        name="qed",
        fn=QED.qed,
        description="QED drug-likeness score.",
        groups=("druglike",),
    ),
)

for _spec in _EXTRA_PROPERTIES:
    _register_property(_spec, allow_existing=True)


def _admet_profile_from_mol(mol: Mol) -> dict[str, Any]:
    from refua.admet import admet_profile  # noqa: PLC0415

    smiles = Chem.MolToSmiles(mol, canonical=True)
    return admet_profile(smiles, copy_result=False)


def _admet_prediction_fn(task_id: str) -> PropertyFn:
    def _fn(mol: Mol) -> PropertyValue:
        profile = _admet_profile_from_mol(mol)
        predictions = profile.get("predictions")
        if isinstance(predictions, dict):
            return predictions.get(task_id)
        return None

    return _fn


def _admet_summary_fn(summary_key: str) -> PropertyFn:
    def _fn(mol: Mol) -> PropertyValue:
        profile = _admet_profile_from_mol(mol)
        value = profile.get(summary_key)
        if isinstance(value, (float, int)):
            return value
        return None

    return _fn


def _register_admet_properties() -> None:
    from refua.admet import ENDPOINTS  # noqa: PLC0415

    for endpoint in ENDPOINTS:
        name = _normalize_name(endpoint.task_id)
        description = endpoint.description
        groups = ("admet", endpoint.category, "optional")
        spec = MolPropertySpec(
            name=name,
            fn=_admet_prediction_fn(endpoint.task_id),
            description=description,
            groups=groups,
        )
        _register_property(spec)

    for name, description in (
        ("admet_score", "Overall ADMET score."),
        ("safety_score", "Safety-focused score derived from ADMET predictions."),
        ("adme_score", "ADME-focused score derived from predictions."),
    ):
        spec = MolPropertySpec(
            name=name,
            fn=_admet_summary_fn(name),
            description=description,
            groups=("admet", "summary", "optional"),
        )
        _register_property(spec)


_register_admet_properties()


def SM(
    smiles_or_mol: str | Mol,
    *,
    lazy: bool = True,
    sanitize: bool = True,
) -> MolProperties:
    """Create a fluent property builder from a SMILES string or Mol."""
    if isinstance(smiles_or_mol, Mol):
        return MolProperties(smiles_or_mol, lazy=lazy)
    return MolProperties.from_smiles(smiles_or_mol, lazy=lazy, sanitize=sanitize)


class SmallMolecule:
    """Small-molecule wrapper for RDKit-backed utilities."""

    def __init__(self, mol: Mol, *, name: str | None = None) -> None:
        if mol is None:
            raise ValueError("mol must be an RDKit Mol.")
        self.mol = mol
        self.name = name

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        *,
        name: str | None = None,
        sanitize: bool = True,
    ) -> SmallMolecule:
        """Create a SmallMolecule from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        if name is not None:
            mol.SetProp("_Name", name)
        return cls(mol, name=name)

    @classmethod
    def from_mol(cls, mol: Mol, *, name: str | None = None) -> SmallMolecule:
        """Create a SmallMolecule from an RDKit Mol."""
        return cls(mol, name=name)

    def properties(self, *, lazy: bool = True) -> MolProperties:
        """Create a fluent property builder for this molecule."""
        return MolProperties(self.mol, lazy=lazy)


class MolProperties:
    """Fluent small-molecule property builder."""

    def __init__(self, mol: Mol, *, lazy: bool = True) -> None:
        if mol is None:
            raise ValueError("mol must be an RDKit Mol.")
        self._mol = mol
        self._lazy = lazy
        self._values: dict[str, PropertyValue] = {}
        if not lazy:
            self.to_dict()

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        *,
        lazy: bool = True,
        sanitize: bool = True,
    ) -> MolProperties:
        """Create a property builder from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return cls(mol, lazy=lazy)

    @property
    def lazy(self) -> bool:
        """Return whether this builder computes lazily."""
        return self._lazy

    @property
    def mol(self) -> Mol:
        """Return the underlying RDKit Mol."""
        return self._mol

    def get(self, name: str) -> PropertyValue:
        """Retrieve a computed property value."""
        normalized = _normalize_name(name)
        if normalized not in _PROPERTY_REGISTRY:
            msg = f"Unknown property: {name}."
            raise KeyError(msg)
        return self._compute(normalized)

    def to_dict(
        self,
        *,
        groups: Iterable[str] | None = None,
        include_optional: bool = False,
    ) -> dict[str, PropertyValue]:
        """Compute registered properties and return their values.

        Parameters
        ----------
        groups : Iterable[str], optional
            Restrict computation to properties that are in any of these groups.
        include_optional : bool, optional
            When ``groups`` is None, include optional properties (default False).
        """
        group_filter = {group.lower() for group in groups} if groups else None
        results: dict[str, PropertyValue] = {}
        for name, spec in _PROPERTY_REGISTRY.items():
            if group_filter is not None:
                if not group_filter.intersection(spec.groups):
                    continue
            elif not include_optional and "optional" in spec.groups:
                continue
            results[name] = self._compute(name)
        return results

    def admet_profile(
        self,
        *,
        max_new_tokens: int = 8,
        include_scoring: bool = True,
        model_variant: str = "9b-chat",
    ) -> dict[str, Any]:
        """Return model-based ADMET predictions for this molecule."""
        from refua.admet import admet_profile  # noqa: PLC0415

        smiles = Chem.MolToSmiles(self._mol, canonical=True)
        return admet_profile(
            smiles,
            model_variant=model_variant,
            max_new_tokens=max_new_tokens,
            include_scoring=include_scoring,
        )

    def __getitem__(self, name: str) -> PropertyValue:
        return self.get(name)

    def __getattr__(self, name: str) -> Callable[[], PropertyValue]:
        normalized = _normalize_name(name)
        if normalized in _PROPERTY_REGISTRY:
            def _call() -> PropertyValue:
                return self._compute(normalized)

            return _call
        msg = f"{type(self).__name__!s} has no attribute {name!r}"
        raise AttributeError(msg)

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *_PROPERTY_REGISTRY})

    def _compute(self, name: str) -> PropertyValue:
        if name in self._values:
            return self._values[name]
        spec = _PROPERTY_REGISTRY[name]
        value = _normalize_value(spec.fn(self._mol))
        self._values[name] = value
        return value

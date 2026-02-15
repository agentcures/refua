"""Protein property helpers built on sequence analysis."""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from Bio.SeqUtils.ProtParam import ProteinAnalysis

ProteinPropertyValue = float | int | tuple[float | int, ...] | None
ProteinPropertyFn = Callable[[str], ProteinPropertyValue]


@dataclass(frozen=True, slots=True)
class ProteinPropertySpec:
    """Sequence-backed property definition."""

    name: str
    fn: ProteinPropertyFn
    description: str
    groups: tuple[str, ...] = ()


_PROPERTY_REGISTRY: dict[str, ProteinPropertySpec] = {}
_GROUP_REGISTRY: dict[str, list[str]] = {}

_ALIASES: dict[str, str] = {
    "mw": "molecular_weight",
    "molecular_weight_da": "molecular_weight",
    "seq_len": "length",
    "length_aa": "length",
    "pi": "isoelectric_point",
    "p_i": "isoelectric_point",
    "charge": "net_charge_ph_7_4",
    "net_charge": "net_charge_ph_7_4",
    "instability": "instability_index",
    "stable": "is_stable",
    "extinction_reduced": "extinction_coefficient_reduced",
    "extinction_oxidized": "extinction_coefficient_oxidized",
}

_AA_THREE_LETTER: dict[str, str] = {
    "A": "ala",
    "C": "cys",
    "D": "asp",
    "E": "glu",
    "F": "phe",
    "G": "gly",
    "H": "his",
    "I": "ile",
    "K": "lys",
    "L": "leu",
    "M": "met",
    "N": "asn",
    "P": "pro",
    "Q": "gln",
    "R": "arg",
    "S": "ser",
    "T": "thr",
    "V": "val",
    "W": "trp",
    "Y": "tyr",
}

for _aa_code, _aa_name in _AA_THREE_LETTER.items():
    _ALIASES[f"{_aa_code.lower()}_count"] = f"count_{_aa_name}"
    _ALIASES[f"aa_{_aa_code.lower()}_count"] = f"count_{_aa_name}"
    _ALIASES[f"{_aa_code.lower()}_fraction"] = f"fraction_{_aa_name}"
    _ALIASES[f"aa_{_aa_code.lower()}_fraction"] = f"fraction_{_aa_name}"

_CANONICAL_AA: tuple[str, ...] = tuple(_AA_THREE_LETTER)
_CANONICAL_AA_SET = frozenset(_CANONICAL_AA)

_HYDROPATHY_KD: dict[str, float] = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

_HYDROPHOBIC = frozenset({"A", "V", "I", "L", "M", "F", "W", "Y"})
_POLAR = frozenset({"S", "T", "N", "Q", "C", "Y", "D", "E", "H", "K", "R"})
_CHARGED = frozenset({"D", "E", "H", "K", "R"})
_POSITIVE = frozenset({"H", "K", "R"})
_NEGATIVE = frozenset({"D", "E"})
_TINY = frozenset({"A", "C", "G", "S", "T"})
_SMALL = frozenset({"A", "C", "D", "G", "N", "P", "S", "T", "V"})
_SULFUR = frozenset({"C", "M"})


def _to_snake(name: str) -> str:
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def _normalize_name(name: str) -> str:
    key = _to_snake(name)
    return _ALIASES.get(key, key)


def _register_property(
    spec: ProteinPropertySpec,
    *,
    allow_existing: bool = False,
) -> None:
    if spec.name in _PROPERTY_REGISTRY:
        if allow_existing:
            return
        msg = f"Property already registered: {spec.name}"
        raise ValueError(msg)
    _PROPERTY_REGISTRY[spec.name] = spec
    for group in spec.groups:
        _GROUP_REGISTRY.setdefault(group, []).append(spec.name)


def register_protein_property(
    name: str,
    fn: ProteinPropertyFn,
    *,
    description: str,
    groups: Iterable[str] = (),
) -> None:
    """Register a new protein property."""
    normalized = _normalize_name(name)
    if normalized in _PROPERTY_REGISTRY:
        msg = f"Property already registered: {normalized}"
        raise ValueError(msg)
    _register_property(
        ProteinPropertySpec(
            name=normalized,
            fn=fn,
            description=description,
            groups=tuple(groups),
        ),
    )


def available_protein_properties() -> tuple[str, ...]:
    """Return available protein property names."""
    return tuple(_PROPERTY_REGISTRY)


def available_protein_property_groups() -> tuple[str, ...]:
    """Return available protein property groups."""
    return tuple(_GROUP_REGISTRY)


def protein_property_specs() -> Mapping[str, ProteinPropertySpec]:
    """Return property specs keyed by name."""
    return dict(_PROPERTY_REGISTRY)


def _normalize_value(value: ProteinPropertyValue) -> ProteinPropertyValue:
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


def _prepare_sequence(sequence: str, *, sanitize: bool) -> str:
    candidate = sequence
    if sanitize:
        candidate = re.sub(r"\s+", "", candidate).upper()
    if not candidate:
        raise ValueError("Protein sequence cannot be empty.")
    invalid = sorted(
        {residue for residue in candidate if residue not in _CANONICAL_AA_SET},
    )
    if invalid:
        msg = (
            "Protein sequence contains non-canonical residues. "
            f"Unsupported: {', '.join(invalid)}"
        )
        raise ValueError(msg)
    return candidate


def _fraction_of(
    fractions: Mapping[str, float],
    residues: Iterable[str],
) -> float:
    return float(sum(fractions[residue] for residue in residues))


@functools.lru_cache(maxsize=4096)
def _sequence_metrics(sequence: str) -> dict[str, ProteinPropertyValue]:
    analysis = ProteinAnalysis(sequence)
    length = len(sequence)
    counts = analysis.count_amino_acids()
    fractions = {
        residue: float(counts.get(residue, 0)) / float(length)
        for residue in _CANONICAL_AA
    }

    helix, turn, sheet = analysis.secondary_structure_fraction()
    ext_reduced, ext_oxidized = analysis.molar_extinction_coefficient()
    flexibility_window = tuple(float(value) for value in analysis.flexibility())
    if flexibility_window:
        flexibility_mean = float(sum(flexibility_window) / len(flexibility_window))
        flexibility_min = float(min(flexibility_window))
        flexibility_max = float(max(flexibility_window))
    else:
        flexibility_mean = None
        flexibility_min = None
        flexibility_max = None

    shannon_entropy = -sum(
        fraction * math.log2(fraction)
        for fraction in fractions.values()
        if fraction > 0.0
    )
    instability_index = float(analysis.instability_index())
    aliphatic_index = 100.0 * (
        fractions["A"] + 2.9 * fractions["V"] + 3.9 * (fractions["I"] + fractions["L"])
    )
    hydropathy_kd = sum(
        _HYDROPATHY_KD[residue] * fractions[residue] for residue in _CANONICAL_AA
    )

    metrics: dict[str, ProteinPropertyValue] = {
        "length": length,
        "molecular_weight": float(analysis.molecular_weight()),
        "aromaticity": float(analysis.aromaticity()),
        "instability_index": instability_index,
        "is_stable": int(instability_index < 40.0),
        "isoelectric_point": float(analysis.isoelectric_point()),
        "gravy": float(analysis.gravy()),
        "helix_fraction": float(helix),
        "turn_fraction": float(turn),
        "sheet_fraction": float(sheet),
        "extinction_coefficient_reduced": int(ext_reduced),
        "extinction_coefficient_oxidized": int(ext_oxidized),
        "net_charge_ph_5_5": float(analysis.charge_at_pH(5.5)),
        "net_charge_ph_7_4": float(analysis.charge_at_pH(7.4)),
        "net_charge_ph_9_0": float(analysis.charge_at_pH(9.0)),
        "aliphatic_index": float(aliphatic_index),
        "shannon_entropy": float(shannon_entropy),
        "hydropathy_kyte_doolittle": float(hydropathy_kd),
        "hydrophobic_residue_fraction": _fraction_of(fractions, _HYDROPHOBIC),
        "polar_residue_fraction": _fraction_of(fractions, _POLAR),
        "nonpolar_residue_fraction": 1.0 - _fraction_of(fractions, _POLAR),
        "charged_residue_fraction": _fraction_of(fractions, _CHARGED),
        "positive_residue_fraction": _fraction_of(fractions, _POSITIVE),
        "negative_residue_fraction": _fraction_of(fractions, _NEGATIVE),
        "tiny_residue_fraction": _fraction_of(fractions, _TINY),
        "small_residue_fraction": _fraction_of(fractions, _SMALL),
        "sulfur_residue_fraction": _fraction_of(fractions, _SULFUR),
        "glycine_fraction": fractions["G"],
        "proline_fraction": fractions["P"],
        "cysteine_fraction": fractions["C"],
        "flexibility_mean": flexibility_mean,
        "flexibility_min": flexibility_min,
        "flexibility_max": flexibility_max,
    }
    for residue, name in _AA_THREE_LETTER.items():
        metrics[f"count_{name}"] = int(counts.get(residue, 0))
        metrics[f"fraction_{name}"] = fractions[residue]
    return metrics


def _metric_fn(metric_name: str) -> ProteinPropertyFn:
    def _fn(sequence: str) -> ProteinPropertyValue:
        metrics = _sequence_metrics(sequence)
        return metrics.get(metric_name)

    return _fn


def _register_default_properties() -> None:
    for name, description, groups in (
        ("length", "Sequence length in residues.", ("basic",)),
        ("molecular_weight", "Estimated molecular weight in Daltons.", ("basic",)),
        (
            "isoelectric_point",
            "Estimated isoelectric point (pI).",
            ("basic", "charge"),
        ),
        ("net_charge_ph_5_5", "Estimated net charge at pH 5.5.", ("charge",)),
        (
            "net_charge_ph_7_4",
            "Estimated net charge at pH 7.4.",
            ("basic", "charge"),
        ),
        ("net_charge_ph_9_0", "Estimated net charge at pH 9.0.", ("charge",)),
        ("aromaticity", "Fraction of aromatic residues.", ("basic", "composition")),
        (
            "instability_index",
            "Instability index (<40 is generally stable).",
            ("basic", "biophysical"),
        ),
        (
            "is_stable",
            "1 if instability index is below 40, else 0.",
            ("basic", "biophysical"),
        ),
        ("gravy", "Grand average of hydropathy (GRAVY).", ("basic", "biophysical")),
        (
            "hydropathy_kyte_doolittle",
            "Average Kyte-Doolittle hydropathy score.",
            ("biophysical",),
        ),
        (
            "aliphatic_index",
            "Aliphatic index (relative volume occupied by aliphatic side chains).",
            ("biophysical",),
        ),
        (
            "shannon_entropy",
            "Shannon entropy of amino-acid composition.",
            ("biophysical", "composition"),
        ),
        (
            "helix_fraction",
            "Predicted fraction in alpha-helical secondary structure.",
            ("secondary_structure",),
        ),
        (
            "turn_fraction",
            "Predicted fraction in beta-turn secondary structure.",
            ("secondary_structure",),
        ),
        (
            "sheet_fraction",
            "Predicted fraction in beta-sheet secondary structure.",
            ("secondary_structure",),
        ),
        (
            "extinction_coefficient_reduced",
            "Molar extinction coefficient assuming reduced cysteines.",
            ("absorbance",),
        ),
        (
            "extinction_coefficient_oxidized",
            "Molar extinction coefficient assuming cystines from disulfides.",
            ("absorbance",),
        ),
        (
            "flexibility_mean",
            "Average local flexibility score (sliding-window estimate).",
            ("flexibility",),
        ),
        ("flexibility_min", "Minimum local flexibility score.", ("flexibility",)),
        ("flexibility_max", "Maximum local flexibility score.", ("flexibility",)),
        (
            "hydrophobic_residue_fraction",
            "Fraction of hydrophobic residues (A,V,I,L,M,F,W,Y).",
            ("composition",),
        ),
        (
            "polar_residue_fraction",
            "Fraction of polar residues (including charged residues).",
            ("composition",),
        ),
        (
            "nonpolar_residue_fraction",
            "Fraction of non-polar residues.",
            ("composition",),
        ),
        (
            "charged_residue_fraction",
            "Fraction of charged residues (D,E,H,K,R).",
            ("composition",),
        ),
        (
            "positive_residue_fraction",
            "Fraction of positively charged residues (H,K,R).",
            ("composition", "charge"),
        ),
        (
            "negative_residue_fraction",
            "Fraction of negatively charged residues (D,E).",
            ("composition", "charge"),
        ),
        (
            "tiny_residue_fraction",
            "Fraction of tiny residues (A,C,G,S,T).",
            ("composition",),
        ),
        (
            "small_residue_fraction",
            "Fraction of small residues (A,C,D,G,N,P,S,T,V).",
            ("composition",),
        ),
        (
            "sulfur_residue_fraction",
            "Fraction of sulfur-containing residues (C,M).",
            ("composition",),
        ),
        ("glycine_fraction", "Fraction of glycine residues.", ("composition",)),
        ("proline_fraction", "Fraction of proline residues.", ("composition",)),
        ("cysteine_fraction", "Fraction of cysteine residues.", ("composition",)),
    ):
        _register_property(
            ProteinPropertySpec(
                name=name,
                fn=_metric_fn(name),
                description=description,
                groups=groups,
            ),
            allow_existing=True,
        )

    for residue, name in _AA_THREE_LETTER.items():
        _register_property(
            ProteinPropertySpec(
                name=f"count_{name}",
                fn=_metric_fn(f"count_{name}"),
                description=f"Residue count for {residue} ({name.upper()}).",
                groups=("composition", "aa_count"),
            ),
            allow_existing=True,
        )
        _register_property(
            ProteinPropertySpec(
                name=f"fraction_{name}",
                fn=_metric_fn(f"fraction_{name}"),
                description=f"Residue fraction for {residue} ({name.upper()}).",
                groups=("composition", "aa_fraction"),
            ),
            allow_existing=True,
        )


_register_default_properties()


class ProteinProperties:
    """Fluent protein property builder."""

    def __init__(
        self,
        sequence: str,
        *,
        lazy: bool = True,
        sanitize: bool = True,
    ) -> None:
        self._sequence = _prepare_sequence(sequence, sanitize=sanitize)
        self._lazy = lazy
        self._values: dict[str, ProteinPropertyValue] = {}
        if not lazy:
            self.to_dict()

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        *,
        lazy: bool = True,
        sanitize: bool = True,
    ) -> ProteinProperties:
        """Create a property builder from an amino-acid sequence."""
        return cls(sequence, lazy=lazy, sanitize=sanitize)

    @property
    def lazy(self) -> bool:
        """Return whether this builder computes lazily."""
        return self._lazy

    @property
    def sequence(self) -> str:
        """Return the normalized protein sequence."""
        return self._sequence

    def get(self, name: str) -> ProteinPropertyValue:
        """Retrieve a computed protein property value."""
        normalized = _normalize_name(name)
        if normalized not in _PROPERTY_REGISTRY:
            msg = f"Unknown property: {name}."
            raise KeyError(msg)
        return self._compute(normalized)

    def to_dict(
        self,
        *,
        groups: Iterable[str] | None = None,
    ) -> dict[str, ProteinPropertyValue]:
        """Compute registered properties and return their values."""
        group_filter = {group.lower() for group in groups} if groups else None
        results: dict[str, ProteinPropertyValue] = {}
        for name, spec in _PROPERTY_REGISTRY.items():
            if group_filter is not None and not group_filter.intersection(spec.groups):
                continue
            results[name] = self._compute(name)
        return results

    def __getitem__(self, name: str) -> ProteinPropertyValue:
        return self.get(name)

    def __getattr__(self, name: str) -> Callable[[], ProteinPropertyValue]:
        normalized = _normalize_name(name)
        if normalized in _PROPERTY_REGISTRY:

            def _call() -> ProteinPropertyValue:
                return self._compute(normalized)

            return _call
        msg = f"{type(self).__name__!s} has no attribute {name!r}"
        raise AttributeError(msg)

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *_PROPERTY_REGISTRY})

    def _compute(self, name: str) -> ProteinPropertyValue:
        if name in self._values:
            return self._values[name]
        spec = _PROPERTY_REGISTRY[name]
        value = _normalize_value(spec.fn(self._sequence))
        self._values[name] = value
        return value

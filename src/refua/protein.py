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
    "ncpr": "net_charge_per_residue_ph_7_4",
    "fcr": "fraction_charged_residues",
    "kappa": "charge_patterning_kappa",
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
_HYDROPHILIC = frozenset({"D", "E", "K", "R", "H", "N", "Q", "S", "T"})
_TINY = frozenset({"A", "C", "G", "S", "T"})
_SMALL = frozenset({"A", "C", "D", "G", "N", "P", "S", "T", "V"})
_SULFUR = frozenset({"C", "M"})
_PEPTIDE_HYDROPHOBIC = frozenset({"F", "I", "L", "V", "W", "Y"})
_CHARGE_NUMERIC: dict[str, int] = {"D": -1, "E": -1, "H": 1, "K": 1, "R": 1}
_DISORDER_PROMOTING = frozenset({"A", "D", "E", "G", "K", "P", "Q", "R", "S"})
_ORDER_PROMOTING = frozenset({"C", "F", "I", "L", "N", "V", "W", "Y"})

_BOMAN_RESIDUE_PROPENSITY: dict[str, float] = {
    "A": 0.17,
    "C": 0.24,
    "D": 3.0,
    "E": 2.68,
    "F": -2.98,
    "G": 0.01,
    "H": 2.06,
    "I": -3.1,
    "K": 2.71,
    "L": -2.05,
    "M": -1.1,
    "N": 2.05,
    "P": -0.66,
    "Q": 2.36,
    "R": 2.58,
    "S": 0.84,
    "T": 0.52,
    "V": -1.5,
    "W": -3.65,
    "Y": -2.33,
}

_HYDROPHOBIC_MOMENT_WINDOW = 11
_HYDROPHOBIC_MOMENT_ANGLE_DEGREES = 100.0
_LOW_COMPLEXITY_WINDOW = 12
_LOW_COMPLEXITY_ENTROPY_THRESHOLD = 2.2

_ANTIBODY_MOTIFS: dict[str, re.Pattern[str]] = {
    "deamidation_high": re.compile(r"N[GS]"),
    "deamidation_medium": re.compile(r"N[AHNT]"),
    "deamidation_low": re.compile(r"[STK]N"),
    "n_glycosylation": re.compile(r"N[^P][ST]"),
    "aspartate_isomerization": re.compile(r"D[DGHST]"),
    "fragmentation_high": re.compile(r"DP"),
    "fragmentation_medium": re.compile(r"TS"),
    "integrin_binding": re.compile(r"GPR|RGD|RYD|LDV|DGE|KGD|NGR"),
    "polyreactive": re.compile(r"GGG|GG|RR|VG|VVV|WWW|YY|W.W"),
    "aggregation_patch": re.compile(r"FHW"),
    "viscosity_patch": re.compile(r"HYF|HWH"),
}

_PEPTIDE_MOTIFS: dict[str, re.Pattern[str]] = {
    "deamidation_hotspot": re.compile(r"N[GSQA]"),
    "aspartate_cleavage": re.compile(r"D[PGS]"),
    "n_terminal_cyclization": re.compile(r"^[QN]"),
    "dpp4_cleavage": re.compile(r"^[PX]?[AP]"),
    "hydrophobic_patch": re.compile(r"[FILVWY]{3,}"),
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


def _count_motif_matches(sequence: str, pattern: re.Pattern[str]) -> int:
    return sum(1 for _ in pattern.finditer(sequence))


def _max_consecutive_identical(sequence: str) -> int:
    if not sequence:
        return 0
    max_run = 1
    current_run = 1
    for idx in range(1, len(sequence)):
        if sequence[idx] == sequence[idx - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def _max_consecutive_in_set(sequence: str, residues: frozenset[str]) -> int:
    max_run = 0
    current_run = 0
    for residue in sequence:
        if residue in residues:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _max_consecutive_true(mask: Iterable[bool]) -> int:
    max_run = 0
    current_run = 0
    for value in mask:
        if value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _antibody_unpaired_cysteine_count(sequence: str) -> int:
    cysteine_positions = [idx for idx, residue in enumerate(sequence) if residue == "C"]
    paired: set[int] = set()
    for idx in range(len(cysteine_positions) - 1):
        left = cysteine_positions[idx]
        right = cysteine_positions[idx + 1]
        if abs(right - left) in (1, 2):
            paired.update({left, right})
    return sum(1 for idx in cysteine_positions if idx not in paired)


def _linear_unpaired_cysteine_count(sequence: str) -> int:
    cysteine_count = sequence.count("C")
    if cysteine_count % 2 == 1:
        return cysteine_count
    return 0


def _cyclic_internal_unpaired_cysteine_count(sequence: str) -> int:
    internal_cysteines = [
        idx
        for idx, residue in enumerate(sequence)
        if residue == "C" and idx not in {0, len(sequence) - 1}
    ]
    if len(internal_cysteines) % 2 == 1:
        return len(internal_cysteines)
    return 0


def _charge_patterning_kappa(sequence: str) -> float:
    charged_residues = [residue for residue in sequence if residue in _CHARGE_NUMERIC]
    charged_count = len(charged_residues)
    if charged_count < 2:
        return 0.0

    positive_count = sum(
        1 for residue in charged_residues if _CHARGE_NUMERIC[residue] > 0
    )
    negative_count = charged_count - positive_count
    if positive_count == 0 or negative_count == 0:
        return 1.0

    same_charge_adjacent = sum(
        1
        for idx in range(1, charged_count)
        if _CHARGE_NUMERIC[charged_residues[idx]]
        == _CHARGE_NUMERIC[charged_residues[idx - 1]]
    )
    observed_same_charge_fraction = same_charge_adjacent / float(charged_count - 1)
    expected_same_charge_fraction = (
        positive_count * (positive_count - 1) + negative_count * (negative_count - 1)
    ) / float(charged_count * (charged_count - 1))
    denominator = 1.0 - expected_same_charge_fraction
    if denominator <= 0.0:
        return 0.0

    kappa = (
        observed_same_charge_fraction - expected_same_charge_fraction
    ) / denominator
    return float(min(1.0, max(0.0, kappa)))


def _hydrophobic_moment(segment: str, *, angle_degrees: float) -> float:
    if not segment:
        return 0.0
    angle = math.radians(angle_degrees)
    x_component = 0.0
    y_component = 0.0
    for idx, residue in enumerate(segment):
        theta = idx * angle
        hydropathy = _HYDROPATHY_KD[residue]
        x_component += hydropathy * math.cos(theta)
        y_component += hydropathy * math.sin(theta)
    return float(math.hypot(x_component, y_component) / len(segment))


def _hydrophobic_moment_profile(
    sequence: str,
    *,
    window: int = _HYDROPHOBIC_MOMENT_WINDOW,
    angle_degrees: float = _HYDROPHOBIC_MOMENT_ANGLE_DEGREES,
) -> tuple[float, float]:
    if not sequence:
        return 0.0, 0.0
    if len(sequence) < window:
        single = _hydrophobic_moment(sequence, angle_degrees=angle_degrees)
        return single, single

    moments = [
        _hydrophobic_moment(
            sequence[start : start + window],
            angle_degrees=angle_degrees,
        )
        for start in range(len(sequence) - window + 1)
    ]
    return float(sum(moments) / len(moments)), float(max(moments))


def _window_shannon_entropy(segment: str) -> float:
    counts: dict[str, int] = {}
    for residue in segment:
        counts[residue] = counts.get(residue, 0) + 1
    length = len(segment)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
        if count > 0
    )


def _low_complexity_profile(
    sequence: str,
    *,
    window: int = _LOW_COMPLEXITY_WINDOW,
    entropy_threshold: float = _LOW_COMPLEXITY_ENTROPY_THRESHOLD,
) -> tuple[float, int]:
    if not sequence:
        return 0.0, 0

    if len(sequence) < window:
        low_complexity = _window_shannon_entropy(sequence) <= entropy_threshold
        if low_complexity:
            return 1.0, len(sequence)
        return 0.0, 0

    mask = [False] * len(sequence)
    for start in range(len(sequence) - window + 1):
        segment = sequence[start : start + window]
        if _window_shannon_entropy(segment) <= entropy_threshold:
            for idx in range(start, start + window):
                mask[idx] = True

    low_complexity_fraction = sum(mask) / float(len(sequence))
    max_low_complexity_run = _max_consecutive_true(mask)
    return float(low_complexity_fraction), max_low_complexity_run


def _boman_index(sequence: str) -> float:
    if not sequence:
        return 0.0
    score = sum(_BOMAN_RESIDUE_PROPENSITY[residue] for residue in sequence)
    return float(score / len(sequence))


def _friendly_property_doc(
    *,
    requested_name: str,
    canonical_name: str,
    description: str,
    groups: tuple[str, ...],
) -> str:
    """Build a user-facing docstring for dynamic protein property accessors."""
    summary = description.strip()
    if not summary:
        summary = f"Value for the `{canonical_name}` protein property."
    if summary[-1] not in ".!?":
        summary = f"{summary}."

    doc = [
        f"Compute the `{canonical_name}` protein property.",
        "",
        summary,
    ]
    if _to_snake(requested_name) != canonical_name:
        doc.extend(
            [
                "",
                f"This method name is an alias of `{canonical_name}`.",
            ]
        )
    if groups:
        doc.extend(
            [
                "",
                f"Groups: {', '.join(groups)}.",
            ]
        )
    doc.extend(
        [
            "",
            "Returns",
            "-------",
            "ProteinPropertyValue",
            "    Computed value for this sequence, or ``None`` when unavailable.",
        ]
    )
    return "\n".join(doc)


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
    molecular_weight = float(analysis.molecular_weight())
    aliphatic_index = 100.0 * (
        fractions["A"] + 2.9 * fractions["V"] + 3.9 * (fractions["I"] + fractions["L"])
    )
    hydropathy_kd = sum(
        _HYDROPATHY_KD[residue] * fractions[residue] for residue in _CANONICAL_AA
    )
    net_charge_ph_5_5 = float(analysis.charge_at_pH(5.5))
    net_charge_ph_7_4 = float(analysis.charge_at_pH(7.4))
    net_charge_ph_9_0 = float(analysis.charge_at_pH(9.0))
    net_charge_per_residue_ph_7_4 = net_charge_ph_7_4 / float(length)
    charged_residue_fraction = _fraction_of(fractions, _CHARGED)
    fraction_charged_residues = charged_residue_fraction
    charge_patterning_kappa = _charge_patterning_kappa(sequence)
    extinction_per_molecular_weight_reduced = float(ext_reduced) / molecular_weight
    extinction_per_molecular_weight_oxidized = float(ext_oxidized) / molecular_weight
    hydrophobic_moment_mean, hydrophobic_moment_max = _hydrophobic_moment_profile(
        sequence,
    )
    low_complexity_fraction, max_low_complexity_run = _low_complexity_profile(sequence)
    disorder_promoting_fraction = _fraction_of(fractions, _DISORDER_PROMOTING)
    order_promoting_fraction = _fraction_of(fractions, _ORDER_PROMOTING)
    boman_index = _boman_index(sequence)
    deamidation_high_risk_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["deamidation_high"],
    )
    deamidation_medium_risk_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["deamidation_medium"],
    )
    deamidation_low_risk_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["deamidation_low"],
    )
    n_glycosylation_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["n_glycosylation"],
    )
    aspartate_isomerization_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["aspartate_isomerization"],
    )
    aspartate_fragmentation_high_risk_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["fragmentation_high"],
    )
    aspartate_fragmentation_medium_risk_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["fragmentation_medium"],
    )
    methionine_oxidation_motif_count = sequence.count("M")
    tryptophan_oxidation_motif_count = sequence.count("W")
    integrin_binding_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["integrin_binding"],
    )
    polyreactive_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["polyreactive"],
    )
    aggregation_patch_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["aggregation_patch"],
    )
    viscosity_patch_motif_count = _count_motif_matches(
        sequence,
        _ANTIBODY_MOTIFS["viscosity_patch"],
    )
    unpaired_cysteine_count = _antibody_unpaired_cysteine_count(sequence)

    antibody_liability_motif_count = (
        deamidation_high_risk_motif_count
        + deamidation_medium_risk_motif_count
        + deamidation_low_risk_motif_count
        + n_glycosylation_motif_count
        + aspartate_isomerization_motif_count
        + aspartate_fragmentation_high_risk_motif_count
        + aspartate_fragmentation_medium_risk_motif_count
        + methionine_oxidation_motif_count
        + tryptophan_oxidation_motif_count
        + integrin_binding_motif_count
        + polyreactive_motif_count
        + aggregation_patch_motif_count
        + viscosity_patch_motif_count
        + unpaired_cysteine_count
    )
    antibody_liability_score = (
        deamidation_high_risk_motif_count * 10
        + deamidation_medium_risk_motif_count * 5
        + deamidation_low_risk_motif_count * 1
        + n_glycosylation_motif_count * 5
        + aspartate_isomerization_motif_count * 10
        + aspartate_fragmentation_high_risk_motif_count * 10
        + aspartate_fragmentation_medium_risk_motif_count * 5
        + methionine_oxidation_motif_count * 5
        + tryptophan_oxidation_motif_count * 10
        + integrin_binding_motif_count * 10
        + polyreactive_motif_count * 5
        + aggregation_patch_motif_count * 5
        + viscosity_patch_motif_count * 5
        + unpaired_cysteine_count * 10
    )

    peptide_deamidation_hotspot_count = _count_motif_matches(
        sequence,
        _PEPTIDE_MOTIFS["deamidation_hotspot"],
    )
    peptide_aspartate_cleavage_motif_count = _count_motif_matches(
        sequence,
        _PEPTIDE_MOTIFS["aspartate_cleavage"],
    )
    peptide_n_terminal_cyclization_risk = int(
        bool(_PEPTIDE_MOTIFS["n_terminal_cyclization"].search(sequence)),
    )
    peptide_trypsin_cleavage_site_count = sum(
        1 for residue in sequence[:-1] if residue in {"K", "R"}
    )
    peptide_dpp4_cleavage_motif_present = int(
        bool(_PEPTIDE_MOTIFS["dpp4_cleavage"].search(sequence)),
    )
    peptide_hydrophobic_patch_count = _count_motif_matches(
        sequence,
        _PEPTIDE_MOTIFS["hydrophobic_patch"],
    )
    peptide_hydrophilic_residue_fraction = _fraction_of(fractions, _HYDROPHILIC)
    peptide_max_consecutive_identical_residues = _max_consecutive_identical(sequence)
    peptide_max_consecutive_hydrophobic_residues = _max_consecutive_in_set(
        sequence,
        _PEPTIDE_HYDROPHOBIC,
    )
    peptide_linear_unpaired_cysteine_count = _linear_unpaired_cysteine_count(sequence)
    peptide_cyclic_internal_unpaired_cysteine_count = (
        _cyclic_internal_unpaired_cysteine_count(sequence)
    )
    peptide_low_hydrophilic_flag = int(peptide_hydrophilic_residue_fraction < 0.4)
    peptide_consecutive_identical_flag = int(
        peptide_max_consecutive_identical_residues > 1,
    )
    peptide_long_hydrophobic_run_flag = int(
        peptide_max_consecutive_hydrophobic_residues > 4,
    )
    peptide_linear_liability_score = (
        peptide_deamidation_hotspot_count * 10
        + peptide_aspartate_cleavage_motif_count * 10
        + peptide_n_terminal_cyclization_risk * 5
        + peptide_trypsin_cleavage_site_count * 10
        + peptide_dpp4_cleavage_motif_present * 5
        + methionine_oxidation_motif_count * 5
        + tryptophan_oxidation_motif_count * 10
        + peptide_hydrophobic_patch_count * 5
        + peptide_linear_unpaired_cysteine_count * 10
    )
    peptide_cyclic_liability_score = (
        peptide_deamidation_hotspot_count * 10
        + peptide_aspartate_cleavage_motif_count * 10
        + peptide_trypsin_cleavage_site_count * 10
        + methionine_oxidation_motif_count * 5
        + tryptophan_oxidation_motif_count * 10
        + peptide_hydrophobic_patch_count * 5
        + peptide_cyclic_internal_unpaired_cysteine_count * 10
        + peptide_low_hydrophilic_flag * 7
        + peptide_consecutive_identical_flag * 7
        + peptide_long_hydrophobic_run_flag * 7
    )

    metrics: dict[str, ProteinPropertyValue] = {
        "length": length,
        "molecular_weight": molecular_weight,
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
        "extinction_per_molecular_weight_reduced": extinction_per_molecular_weight_reduced,
        "extinction_per_molecular_weight_oxidized": (
            extinction_per_molecular_weight_oxidized
        ),
        "net_charge_ph_5_5": net_charge_ph_5_5,
        "net_charge_ph_7_4": net_charge_ph_7_4,
        "net_charge_ph_9_0": net_charge_ph_9_0,
        "net_charge_per_residue_ph_7_4": net_charge_per_residue_ph_7_4,
        "aliphatic_index": float(aliphatic_index),
        "shannon_entropy": float(shannon_entropy),
        "hydropathy_kyte_doolittle": float(hydropathy_kd),
        "hydrophobic_moment_mean": hydrophobic_moment_mean,
        "hydrophobic_moment_max": hydrophobic_moment_max,
        "hydrophobic_residue_fraction": _fraction_of(fractions, _HYDROPHOBIC),
        "polar_residue_fraction": _fraction_of(fractions, _POLAR),
        "nonpolar_residue_fraction": 1.0 - _fraction_of(fractions, _POLAR),
        "charged_residue_fraction": charged_residue_fraction,
        "fraction_charged_residues": fraction_charged_residues,
        "charge_patterning_kappa": charge_patterning_kappa,
        "positive_residue_fraction": _fraction_of(fractions, _POSITIVE),
        "negative_residue_fraction": _fraction_of(fractions, _NEGATIVE),
        "disorder_promoting_fraction": disorder_promoting_fraction,
        "order_promoting_fraction": order_promoting_fraction,
        "tiny_residue_fraction": _fraction_of(fractions, _TINY),
        "small_residue_fraction": _fraction_of(fractions, _SMALL),
        "sulfur_residue_fraction": _fraction_of(fractions, _SULFUR),
        "glycine_fraction": fractions["G"],
        "proline_fraction": fractions["P"],
        "cysteine_fraction": fractions["C"],
        "low_complexity_fraction": low_complexity_fraction,
        "max_low_complexity_run": max_low_complexity_run,
        "boman_index": boman_index,
        "deamidation_high_risk_motif_count": deamidation_high_risk_motif_count,
        "deamidation_medium_risk_motif_count": deamidation_medium_risk_motif_count,
        "deamidation_low_risk_motif_count": deamidation_low_risk_motif_count,
        "n_glycosylation_motif_count": n_glycosylation_motif_count,
        "aspartate_isomerization_motif_count": aspartate_isomerization_motif_count,
        "aspartate_fragmentation_high_risk_motif_count": (
            aspartate_fragmentation_high_risk_motif_count
        ),
        "aspartate_fragmentation_medium_risk_motif_count": (
            aspartate_fragmentation_medium_risk_motif_count
        ),
        "methionine_oxidation_motif_count": methionine_oxidation_motif_count,
        "tryptophan_oxidation_motif_count": tryptophan_oxidation_motif_count,
        "integrin_binding_motif_count": integrin_binding_motif_count,
        "polyreactive_motif_count": polyreactive_motif_count,
        "aggregation_patch_motif_count": aggregation_patch_motif_count,
        "viscosity_patch_motif_count": viscosity_patch_motif_count,
        "unpaired_cysteine_count": unpaired_cysteine_count,
        "antibody_liability_motif_count": antibody_liability_motif_count,
        "antibody_liability_score": antibody_liability_score,
        "peptide_deamidation_hotspot_count": peptide_deamidation_hotspot_count,
        "peptide_aspartate_cleavage_motif_count": (
            peptide_aspartate_cleavage_motif_count
        ),
        "peptide_n_terminal_cyclization_risk": peptide_n_terminal_cyclization_risk,
        "peptide_trypsin_cleavage_site_count": peptide_trypsin_cleavage_site_count,
        "peptide_dpp4_cleavage_motif_present": peptide_dpp4_cleavage_motif_present,
        "peptide_hydrophobic_patch_count": peptide_hydrophobic_patch_count,
        "peptide_hydrophilic_residue_fraction": peptide_hydrophilic_residue_fraction,
        "peptide_max_consecutive_identical_residues": (
            peptide_max_consecutive_identical_residues
        ),
        "peptide_max_consecutive_hydrophobic_residues": (
            peptide_max_consecutive_hydrophobic_residues
        ),
        "peptide_linear_unpaired_cysteine_count": (
            peptide_linear_unpaired_cysteine_count
        ),
        "peptide_cyclic_internal_unpaired_cysteine_count": (
            peptide_cyclic_internal_unpaired_cysteine_count
        ),
        "peptide_low_hydrophilic_flag": peptide_low_hydrophilic_flag,
        "peptide_consecutive_identical_flag": peptide_consecutive_identical_flag,
        "peptide_long_hydrophobic_run_flag": peptide_long_hydrophobic_run_flag,
        "peptide_linear_liability_score": peptide_linear_liability_score,
        "peptide_cyclic_liability_score": peptide_cyclic_liability_score,
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
        (
            "net_charge_per_residue_ph_7_4",
            "Estimated net charge at pH 7.4 normalized by sequence length (NCPR).",
            ("charge", "composition", "idr"),
        ),
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
            "hydrophobic_moment_mean",
            "Mean windowed hydrophobic moment (11 aa, 100-degree rotation).",
            ("biophysical", "amphipathicity"),
        ),
        (
            "hydrophobic_moment_max",
            "Maximum windowed hydrophobic moment (11 aa, 100-degree rotation).",
            ("biophysical", "amphipathicity"),
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
            "extinction_per_molecular_weight_reduced",
            "Reduced-cysteine extinction coefficient normalized by molecular weight.",
            ("absorbance",),
        ),
        (
            "extinction_per_molecular_weight_oxidized",
            "Oxidized-cysteine extinction coefficient normalized by molecular weight.",
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
            "fraction_charged_residues",
            "Fraction of charged residues (FCR; D,E,H,K,R).",
            ("composition", "charge", "idr"),
        ),
        (
            "charge_patterning_kappa",
            "Normalized charge clustering score over charged residues (0 mixed, 1 segregated).",
            ("composition", "charge", "idr"),
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
        (
            "disorder_promoting_fraction",
            "Fraction of residues in a disorder-promoting set (A,D,E,G,K,P,Q,R,S).",
            ("composition", "idr"),
        ),
        (
            "order_promoting_fraction",
            "Fraction of residues in an order-promoting set (C,F,I,L,N,V,W,Y).",
            ("composition", "idr"),
        ),
        (
            "low_complexity_fraction",
            "Fraction of residues in low-complexity windows (entropy <= 2.2 over 12 aa).",
            ("composition", "low_complexity", "idr"),
        ),
        (
            "max_low_complexity_run",
            "Longest contiguous run of residues in low-complexity windows.",
            ("composition", "low_complexity", "idr"),
        ),
        (
            "boman_index",
            "Boman index proxy from residue binding propensity values.",
            ("biophysical", "developability", "peptide"),
        ),
        (
            "deamidation_high_risk_motif_count",
            "Count of high-risk deamidation motifs matching N[GS].",
            ("developability", "liability", "antibody"),
        ),
        (
            "deamidation_medium_risk_motif_count",
            "Count of medium-risk deamidation motifs matching N[AHNT].",
            ("developability", "liability", "antibody"),
        ),
        (
            "deamidation_low_risk_motif_count",
            "Count of low-risk deamidation motifs matching [STK]N.",
            ("developability", "liability", "antibody"),
        ),
        (
            "n_glycosylation_motif_count",
            "Count of canonical N-glycosylation sequons matching N[^P][ST].",
            ("developability", "liability", "antibody"),
        ),
        (
            "aspartate_isomerization_motif_count",
            "Count of aspartate isomerization-prone motifs matching D[DGHST].",
            ("developability", "liability", "antibody"),
        ),
        (
            "aspartate_fragmentation_high_risk_motif_count",
            "Count of high-risk fragmentation motifs matching DP.",
            ("developability", "liability", "antibody"),
        ),
        (
            "aspartate_fragmentation_medium_risk_motif_count",
            "Count of medium-risk fragmentation motifs matching TS.",
            ("developability", "liability", "antibody"),
        ),
        (
            "methionine_oxidation_motif_count",
            "Count of methionine oxidation-prone sites (M).",
            ("developability", "liability", "antibody", "peptide"),
        ),
        (
            "tryptophan_oxidation_motif_count",
            "Count of tryptophan oxidation-prone sites (W).",
            ("developability", "liability", "antibody", "peptide"),
        ),
        (
            "integrin_binding_motif_count",
            "Count of integrin-binding motif hits (e.g., RGD, LDV, NGR).",
            ("developability", "liability", "antibody"),
        ),
        (
            "polyreactive_motif_count",
            "Count of sequence patterns associated with polyreactivity risk.",
            ("developability", "liability", "antibody"),
        ),
        (
            "aggregation_patch_motif_count",
            "Count of motif matches associated with aggregation patch risk (FHW).",
            ("developability", "liability", "antibody"),
        ),
        (
            "viscosity_patch_motif_count",
            "Count of motif matches associated with viscosity patch risk (HYF/HWH).",
            ("developability", "liability", "antibody"),
        ),
        (
            "unpaired_cysteine_count",
            "Approximate count of cysteines lacking a nearby putative pair.",
            ("developability", "liability", "antibody", "peptide"),
        ),
        (
            "antibody_liability_motif_count",
            "Total count across antibody liability motifs and unpaired cysteines.",
            ("developability", "liability", "antibody"),
        ),
        (
            "antibody_liability_score",
            "Weighted antibody liability score from sequence motif counts.",
            ("developability", "liability", "antibody"),
        ),
        (
            "peptide_deamidation_hotspot_count",
            "Count of peptide deamidation hotspots matching N[GSQA].",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_aspartate_cleavage_motif_count",
            "Count of peptide acidic cleavage motifs matching D[PGS].",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_n_terminal_cyclization_risk",
            "1 when the sequence starts with Q or N, indicating N-terminal cyclization risk.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_trypsin_cleavage_site_count",
            "Count of internal trypsin cleavage sites (K/R not at C-terminus).",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_dpp4_cleavage_motif_present",
            "1 when an N-terminal DPP4 cleavage motif (^[PX]?[AP]) is present.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_hydrophobic_patch_count",
            "Count of hydrophobic patches with 3+ consecutive FILVWY residues.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_hydrophilic_residue_fraction",
            "Fraction of hydrophilic residues (D,E,K,R,H,N,Q,S,T).",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_max_consecutive_identical_residues",
            "Longest run of consecutive identical residues.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_max_consecutive_hydrophobic_residues",
            "Longest run of consecutive hydrophobic residues in FILVWY.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_linear_unpaired_cysteine_count",
            "For linear peptides with odd cysteine count, flags all cysteines as potentially unpaired.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_cyclic_internal_unpaired_cysteine_count",
            "For cyclic peptides, potential unpaired cysteine count among internal cysteines only.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_low_hydrophilic_flag",
            "1 when hydrophilic residue fraction is below 0.40.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_consecutive_identical_flag",
            "1 when any consecutive identical run is longer than one residue.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_long_hydrophobic_run_flag",
            "1 when the longest FILVWY run exceeds four residues.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_linear_liability_score",
            "Weighted linear-peptide liability score from motif and composition flags.",
            ("developability", "liability", "peptide"),
        ),
        (
            "peptide_cyclic_liability_score",
            "Weighted cyclic-peptide liability score with cyclic-specific penalties.",
            ("developability", "liability", "peptide"),
        ),
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
        """Whether property values are computed on demand."""
        return self._lazy

    @property
    def sequence(self) -> str:
        """Normalized amino-acid sequence used for all calculations."""
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
            spec = _PROPERTY_REGISTRY[normalized]

            def _call() -> ProteinPropertyValue:
                return self._compute(normalized)

            _call.__doc__ = _friendly_property_doc(
                requested_name=name,
                canonical_name=normalized,
                description=spec.description,
                groups=spec.groups,
            )
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

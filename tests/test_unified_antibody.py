from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from refua import (
    AntibodyBinders,
    Binder,
    BinderDesigns,
    Complex,
    Protein,
    antibody_framework_specs,
)
from refua.unified import FoldResult


def test_antibody_framework_specs_default_windows() -> None:
    heavy_spec, light_spec = antibody_framework_specs()
    assert "CAAS12W" in heavy_spec
    assert "WV10" in heavy_spec
    assert "C14W" in heavy_spec
    assert "CRAS10W" in light_spec
    assert "IY9A" in light_spec
    assert "C9F" in light_spec


def test_binder_template_values_render_spec() -> None:
    binder = Binder("A{x}C{y}", template_values={"x": 8, "y": 6}, ids="P")
    assert binder.sequence == "A8C6"


def test_binder_template_values_missing_placeholder_raises() -> None:
    binder = Binder("A{x}C{y}", template_values={"x": 8}, ids="P")
    with pytest.raises(ValueError, match="missing value for placeholder"):
        binder.sequence_spec()


def test_binder_designs_antibody_returns_coherent_pair() -> None:
    heavy_lengths = (13, 11, 15)
    light_lengths = (9, 8, 10)
    expected_heavy, expected_light = antibody_framework_specs(
        heavy_cdr_lengths=heavy_lengths,
        light_cdr_lengths=light_lengths,
    )

    pair = BinderDesigns.antibody(
        heavy_cdr_lengths=heavy_lengths,
        light_cdr_lengths=light_lengths,
        heavy_id="VH",
        light_id="VL",
    )
    assert isinstance(pair, AntibodyBinders)
    assert tuple(pair) == pair.as_tuple()
    assert pair.heavy.ids == "VH"
    assert pair.light.ids == "VL"
    assert pair.heavy.sequence == expected_heavy
    assert pair.light.sequence == expected_light


def test_binder_designs_peptide_presets() -> None:
    linear = BinderDesigns.peptide(length=14, ids="P")
    disulfide = BinderDesigns.disulfide_peptide(segment_lengths=(10, 6, 3), ids="Q")

    assert isinstance(linear, Binder)
    assert linear.ids == "P"
    assert linear.sequence == "14"
    assert linear.cyclic is False

    assert isinstance(disulfide, Binder)
    assert disulfide.ids == "Q"
    assert disulfide.sequence == "10C6C3"
    assert disulfide.cyclic is True


def test_binder_designs_disulfide_peptide_validates_segments() -> None:
    with pytest.raises(ValueError, match="must have exactly three values"):
        BinderDesigns.disulfide_peptide(segment_lengths=(8, 5))
    with pytest.raises(ValueError, match="must be >= 1"):
        BinderDesigns.disulfide_peptide(segment_lengths=(8, 0, 5))


def test_explicit_antibody_complex_builds_default_entities() -> None:
    antigen = Protein(
        "ACDEFGHIKLMNPQRSTVWY" * 12,
        ids="A",
        binding_types={"binding": "30..80"},
    )
    pair = BinderDesigns.antibody()
    design = Complex([antigen, *pair], name="antibody_design")
    antigen, heavy, light = design.entities
    assert isinstance(antigen, Protein)
    assert isinstance(heavy, Binder)
    assert isinstance(light, Binder)

    assert antigen.ids == "A"
    assert antigen.binding_types == {"binding": "30..80"}
    assert heavy.ids == "H"
    assert light.ids == "L"
    expected_heavy, expected_light = antibody_framework_specs()
    assert heavy.sequence == expected_heavy
    assert light.sequence == expected_light


def test_standard_complex_antibody_binders_match_framework_specs() -> None:
    antigen_sequence = "ACDEFGHIKLMNPQRSTVWY" * 12
    heavy_lengths = (12, 10, 14)
    light_lengths = (10, 9, 9)

    explicit = Complex(
        [
            Protein(antigen_sequence, ids="A", binding_types={"binding": "30..80"}),
            *BinderDesigns.antibody(
                heavy_cdr_lengths=heavy_lengths,
                light_cdr_lengths=light_lengths,
                heavy_id="H",
                light_id="L",
            ),
        ],
        name="antibody_design",
    )
    explicit_antigen, explicit_heavy, explicit_light = explicit.entities
    expected_heavy, expected_light = antibody_framework_specs(
        heavy_cdr_lengths=heavy_lengths,
        light_cdr_lengths=light_lengths,
    )

    assert isinstance(explicit_antigen, Protein)
    assert isinstance(explicit_heavy, Binder)
    assert isinstance(explicit_light, Binder)

    assert explicit_antigen.binding_types == {"binding": "30..80"}
    assert explicit_heavy.sequence == expected_heavy
    assert explicit_light.sequence == expected_light


def test_explicit_antibody_design_keeps_explicit_chain_ids() -> None:
    antigen = Protein("M" * 60, ids="X")
    pair = BinderDesigns.antibody(heavy_id="VH", light_id="VL")
    design = Complex([antigen, *pair], name="antibody_design")
    resolved_antigen, resolved_heavy, resolved_light = design.entities
    assert isinstance(resolved_antigen, Protein)
    assert isinstance(resolved_heavy, Binder)
    assert isinstance(resolved_light, Binder)
    assert resolved_antigen.ids == "X"
    assert resolved_heavy.ids == "VH"
    assert resolved_light.ids == "VL"


def test_fold_result_chain_design_summary_uses_trace_tokens() -> None:
    token_dtype = np.dtype(
        [
            ("asym_id", np.int32),
            ("res_type", np.int32),
            ("res_name", "U4"),
            ("design_mask", np.bool_),
            ("binding_type", np.int32),
            ("mol_type", np.int8),
        ]
    )
    tokens = np.array(
        [
            (0, 14, "MET", False, 0, 0),
            (0, 15, "SER", False, 1, 0),
            (0, 16, "THR", False, 2, 0),
            (1, 2, "ALA", True, 0, 0),
            (1, 4, "ASN", True, 0, 0),
            (2, 5, "ASP", True, 0, 0),
            (2, 6, "CYS", True, 0, 0),
        ],
        dtype=token_dtype,
    )
    chain_dtype = np.dtype([("name", "U5"), ("asym_id", np.int32)])
    chains = np.array([("A", 0), ("H", 1), ("L", 2)], dtype=chain_dtype)

    trace = SimpleNamespace(
        tokenized=SimpleNamespace(tokens=tokens),
        target=SimpleNamespace(structure=SimpleNamespace(chains=chains)),
    )
    result = FoldResult(
        backend="boltzgen",
        design=trace,
        binder_sequences={"H": "12", "L": "11"},
    )

    summary = {row["chain_id"]: row for row in result.chain_design_summary()}
    assert summary["A"]["sequence"] == "MST"
    assert summary["A"]["design_residue_count"] == 0
    assert summary["A"]["binding_residue_count"] == 1
    assert summary["A"]["not_binding_residue_count"] == 1
    assert summary["H"]["sequence"] == "AN"
    assert summary["H"]["design_residue_count"] == 2
    assert summary["L"]["sequence"] == "DC"
    assert summary["L"]["design_residue_count"] == 2
    assert result.binder_specs == {"H": "12", "L": "11"}

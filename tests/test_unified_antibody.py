from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from refua import Binder, Complex, Protein, antibody_framework_specs
from refua.unified import FoldResult


def test_antibody_framework_specs_default_windows() -> None:
    heavy_spec, light_spec = antibody_framework_specs()
    assert "CAAS12W" in heavy_spec
    assert "WV10" in heavy_spec
    assert "C14W" in heavy_spec
    assert "CRAS10W" in light_spec
    assert "IY9A" in light_spec
    assert "C9F" in light_spec


def test_complex_antibody_design_builds_default_entities() -> None:
    design = Complex.antibody_design(
        "ACDEFGHIKLMNPQRSTVWY" * 12,
        epitope="30..80",
    )
    antigen, heavy, light = design.entities
    assert isinstance(antigen, Protein)
    assert isinstance(heavy, Binder)
    assert isinstance(light, Binder)

    assert antigen.ids == "A"
    assert antigen.binding_types == {"binding": "30..80"}
    assert heavy.ids == "H"
    assert light.ids == "L"


def test_complex_antibody_design_keeps_explicit_chain_ids() -> None:
    antigen = Protein("M" * 60, ids="X")
    heavy = Binder("8C6", ids="VH")
    light = Binder("7C5", ids="VL")
    design = Complex.antibody_design(
        antigen,
        heavy=heavy,
        light=light,
    )
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

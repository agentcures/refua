import sys
from types import ModuleType

import numpy as np
import pytest

from refua.api import _download_hf_artifact
from refua.boltz.api import Pipeline, Spec
from refua.boltz.data.types import MSA
from refua.boltzgen.api import _propagate_design_chain_mask, _resolve_moldir
from refua.boltzgen.api import Spec as DesignSpec


def test_in_memory_api_smoke():
    spec = Spec("ligand_only").ligand("L", smiles="CCO")
    pipe = Pipeline(version=2)

    trace = pipe.prepare(spec)
    assert trace.tokenized.tokens.size > 0
    assert trace.chain_map == {"L": 0}

    trace = pipe.featurize(trace)
    features = trace.features
    assert features is not None
    assert "token_pad_mask" in features
    assert "token_bonds" in features


def test_boltzgen_spec_rejects_duplicate_chain_ids() -> None:
    spec = DesignSpec("duplicate_ids")
    spec.protein("A", "ACDE")
    spec.ligand("A", smiles="CCO")

    with pytest.raises(ValueError, match="Duplicate chain id A"):
        spec.to_schema()


def test_boltzgen_total_length_rejects_inverted_bounds() -> None:
    spec = DesignSpec("bad_total_length").protein("A", "ACDE")

    with pytest.raises(ValueError, match="minimum must be <= maximum"):
        spec.total_length(10, 4)


def test_download_hf_artifact_does_not_swallow_auth_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LocalTokenNotFoundError(Exception):
        pass

    fake_hf = ModuleType("huggingface_hub")

    def fake_download(*, local_files_only: bool = False, **_kwargs: object) -> str:
        if local_files_only:
            raise LocalTokenNotFoundError("token required")
        raise AssertionError("remote download should not be attempted")

    fake_hf.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(LocalTokenNotFoundError, match="token required"):
        _download_hf_artifact(
            "huggingface:org/repo:file.txt",
            repo_type="model",
            cache_dir=None,
            token=None,
            skip_existing=True,
        )


def test_resolve_moldir_reports_auth_failures_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LocalTokenNotFoundError(Exception):
        pass

    fake_hf = ModuleType("huggingface_hub")

    def fake_download(*_args: object, **_kwargs: object) -> str:
        raise LocalTokenNotFoundError("token required")

    fake_hf.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(RuntimeError, match="authentication"):
        _resolve_moldir(
            "huggingface:org/repo:mols.zip",
            auto_download=True,
        )


def test_boltz_spec_deduplicates_equivalent_msa_content() -> None:
    msa_a = MSA(
        sequences=np.array(["ACDE"], dtype="<U4"),
        deletions=np.zeros((1, 4), dtype=np.int64),
        residues=np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    msa_b = MSA(
        sequences=np.array(["ACDE"], dtype="<U4"),
        deletions=np.zeros((1, 4), dtype=np.int64),
        residues=np.array([[0, 1, 2, 3]], dtype=np.int64),
    )

    schema = Spec("shared_msa").protein("A", "ACDE", msa=msa_a).protein(
        "B",
        "ACDE",
        msa=msa_b,
    ).to_schema()

    msa_ids = [entry["protein"]["msa"] for entry in schema["sequences"]]
    assert msa_ids == ["in_memory:0", "in_memory:0"]


def test_boltz_spec_rejects_conflicting_msa_content_for_same_sequence() -> None:
    msa_a = MSA(
        sequences=np.array(["ACDE"], dtype="<U4"),
        deletions=np.zeros((1, 4), dtype=np.int64),
        residues=np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    msa_b = MSA(
        sequences=np.array(["ACDE"], dtype="<U4"),
        deletions=np.zeros((1, 4), dtype=np.int64),
        residues=np.array([[3, 2, 1, 0]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="equivalent MSA content"):
        Spec("conflicting_msa").protein("A", "ACDE", msa=msa_a).protein(
            "B",
            "ACDE",
            msa=msa_b,
        ).to_schema()


def test_boltzgen_design_mask_propagates_across_chain_bond_graph() -> None:
    asym_id = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    bonds = np.array(
        [
            (1, 2, 1),
            (3, 4, 1),
        ],
        dtype=np.int64,
    )
    design_mask = np.array([True, True, False, False, False, False])

    propagated = _propagate_design_chain_mask(asym_id, bonds, design_mask)

    assert propagated.tolist() == [True, True, True, True, True, True]

import sys
from types import ModuleType

import pytest

from refua.api import _download_hf_artifact
from refua.boltz.api import Pipeline, Spec
from refua.boltzgen.api import _resolve_moldir
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

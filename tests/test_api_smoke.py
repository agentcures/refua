import pytest

from refua.boltz.api import Pipeline, Spec
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

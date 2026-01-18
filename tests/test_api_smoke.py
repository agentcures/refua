from refua.boltz.api import Pipeline, Spec


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

from refua import SM


def test_small_molecule_properties_lazy():
    props = SM("CCO", lazy=True)
    mol_wt = props.mol_wt()
    assert mol_wt > 40
    values = props.to_dict()
    assert values["mol_wt"] == mol_wt
    assert props.logp() == values["mol_log_p"]


def test_small_molecule_properties_eager():
    props = SM("CCO", lazy=False)
    values = props.to_dict()
    assert values["num_h_donors"] >= 0
    assert props.hbd() == values["num_h_donors"]

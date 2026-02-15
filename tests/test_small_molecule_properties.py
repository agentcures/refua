from refua import available_mol_property_groups, SM


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


def test_small_molecule_medchem_properties():
    props = SM("CCO", lazy=True)
    values = props.to_dict(groups=["medchem"])

    expected_keys = (
        "pains_alert_count",
        "brenk_alert_count",
        "nih_alert_count",
        "zinc_alert_count",
        "medchem_alert_count",
        "medchem_pass",
    )
    assert set(expected_keys).issubset(values)

    count_keys = expected_keys[:4]
    for key in count_keys:
        count = values[key]
        if count is not None:
            assert isinstance(count, int)
            assert count >= 0

    if all(values[key] is not None for key in count_keys):
        expected_total = sum(int(values[key]) for key in count_keys)
        assert values["medchem_alert_count"] == expected_total
        assert values["medchem_pass"] == int(expected_total == 0)
    else:
        assert values["medchem_alert_count"] is None
        assert values["medchem_pass"] is None

    assert props.pains() == values["pains_alert_count"]
    assert props.passes_medchem_filters() == values["medchem_pass"]


def test_small_molecule_medchem_group_is_registered():
    assert "medchem" in available_mol_property_groups()

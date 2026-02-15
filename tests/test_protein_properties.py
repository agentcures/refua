from refua import Protein, ProteinProperties, available_protein_property_groups


def test_protein_properties_lazy():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    protein = Protein(sequence)

    assert protein.length() == len(sequence)
    assert protein.pi() == protein.isoelectric_point()

    values = protein.to_dict(groups=["basic"])
    assert values["length"] == len(sequence)
    assert values["molecular_weight"] is not None
    assert values["instability_index"] is not None


def test_protein_properties_eager():
    sequence = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQQLAGKYTPEEIRNVLSTLQKAD"
    props = ProteinProperties.from_sequence(sequence, lazy=False)

    values = props.to_dict(groups=["composition"])
    assert 0.0 <= float(values["hydrophobic_residue_fraction"]) <= 1.0
    assert 0.0 <= float(values["polar_residue_fraction"]) <= 1.0
    assert 0.0 <= float(values["charged_residue_fraction"]) <= 1.0
    assert int(values["count_met"]) >= 0
    assert 0.0 <= float(values["fraction_met"]) <= 1.0


def test_protein_properties_aa_composition_invariants():
    sequence = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQQLAGKYTPEEIRNVLSTLQKAD"
    protein = Protein(sequence)

    count_values = protein.to_dict(groups=["aa_count"])
    fraction_values = protein.to_dict(groups=["aa_fraction"])

    total_count = sum(int(value) for value in count_values.values())
    total_fraction = sum(float(value) for value in fraction_values.values())
    assert total_count == len(sequence)
    assert abs(total_fraction - 1.0) < 1e-9


def test_protein_properties_groups_and_unified_protein():
    assert "basic" in available_protein_property_groups()
    assert "aa_fraction" in available_protein_property_groups()

    protein = Protein("ACDEFGHIKLMNPQRSTVWY", ids="A")
    assert protein.length() == 20
    assert protein.charge() == protein.net_charge_ph_7_4()

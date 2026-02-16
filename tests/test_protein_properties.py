from refua import (
    Protein,
    ProteinProperties,
    available_protein_properties,
    available_protein_property_groups,
    protein_property_specs,
)


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


def test_protein_property_method_docstrings_are_friendly():
    protein = Protein("ACDEFGHIKLMNPQRSTVWY")

    for name in available_protein_properties():
        doc = getattr(protein, name).__doc__
        assert doc is not None
        assert doc.startswith(f"Compute the `{name}` protein property.")
        assert "Returns" in doc

    alias_doc = protein.pi.__doc__
    assert alias_doc is not None
    assert "alias of `isoelectric_point`" in alias_doc


def test_protein_developability_properties_are_available_as_methods():
    groups = available_protein_property_groups()
    assert "developability" in groups
    assert "liability" in groups
    assert "antibody" in groups
    assert "peptide" in groups

    specs = protein_property_specs()
    assert "antibody_liability_score" in specs
    assert specs["antibody_liability_score"].description.startswith("Weighted antibody")

    antibody = Protein("MNGSDPAFHWHYFRGDAC")
    developability_values = antibody.to_dict(groups=["developability"])
    assert "antibody_liability_score" in developability_values
    assert int(antibody.deamidation_high_risk_motif_count()) >= 1
    assert int(antibody.aggregation_patch_motif_count()) >= 1
    assert int(antibody.viscosity_patch_motif_count()) >= 1
    assert int(antibody.integrin_binding_motif_count()) >= 1
    assert int(antibody.unpaired_cysteine_count()) == 1
    assert int(antibody.antibody_liability_score()) > 0

    linear_peptide = Protein("APMNGSDPKRVVVWC")
    assert int(linear_peptide.peptide_deamidation_hotspot_count()) >= 1
    assert int(linear_peptide.peptide_aspartate_cleavage_motif_count()) >= 1
    assert int(linear_peptide.peptide_dpp4_cleavage_motif_present()) == 1
    assert int(linear_peptide.peptide_trypsin_cleavage_site_count()) >= 2
    assert int(linear_peptide.peptide_hydrophobic_patch_count()) >= 1
    assert int(linear_peptide.peptide_linear_unpaired_cysteine_count()) == 1
    assert int(linear_peptide.peptide_linear_liability_score()) > 0

    cyclic_peptide = Protein("CAVVVVVCC")
    assert int(cyclic_peptide.peptide_cyclic_internal_unpaired_cysteine_count()) == 1
    assert int(cyclic_peptide.peptide_low_hydrophilic_flag()) == 1
    assert int(cyclic_peptide.peptide_consecutive_identical_flag()) == 1
    assert int(cyclic_peptide.peptide_long_hydrophobic_run_flag()) == 1
    assert int(cyclic_peptide.peptide_cyclic_liability_score()) > 0

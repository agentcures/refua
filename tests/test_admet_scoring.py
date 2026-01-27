from refua.admet import AdmetScorer, ENDPOINTS


CRITICAL_TASKS = (
    "hERG",
    "AMES",
    "DILI",
    "Carcinogens_Lagunin",
    "ClinTox",
    "Tox21_SR_p53",
    "Tox21_SR_MMP",
)


def _baseline_predictions(value: float = 0.5) -> dict[str, float]:
    return {endpoint.task_id: value for endpoint in ENDPOINTS}


def test_admet_score_sanity():
    scorer = AdmetScorer()
    predictions_good = _baseline_predictions(0.5)
    for task in CRITICAL_TASKS:
        predictions_good[task] = 0.0

    predictions_bad = dict(predictions_good)
    for task in CRITICAL_TASKS:
        predictions_bad[task] = 1.0

    rdkit_good = {
        "mol_wt": 350.0,
        "exact_mol_wt": 350.0,
        "mol_log_p": 2.5,
        "tpsa": 80.0,
        "num_h_donors": 1.0,
        "num_h_acceptors": 6.0,
        "num_rotatable_bonds": 5.0,
        "ring_count": 2.0,
        "num_aromatic_rings": 2.0,
        "heavy_atom_count": 25.0,
        "num_heteroatoms": 6.0,
        "fraction_csp3": 0.4,
        "formal_charge": 0.0,
        "qed": 0.8,
    }
    rdkit_bad = {
        "mol_wt": 850.0,
        "exact_mol_wt": 850.0,
        "mol_log_p": 6.5,
        "tpsa": 180.0,
        "num_h_donors": 8.0,
        "num_h_acceptors": 16.0,
        "num_rotatable_bonds": 20.0,
        "ring_count": 10.0,
        "num_aromatic_rings": 8.0,
        "heavy_atom_count": 70.0,
        "num_heteroatoms": 20.0,
        "fraction_csp3": 0.05,
        "formal_charge": 4.0,
        "qed": 0.1,
    }

    result_good = scorer.analyze_profile(predictions_good, rdkit_metrics=rdkit_good)
    result_bad = scorer.analyze_profile(predictions_bad, rdkit_metrics=rdkit_good)
    result_rdkit_bad = scorer.analyze_profile(
        predictions_good,
        rdkit_metrics=rdkit_bad,
    )

    assert 0.0 <= result_good["admet_score"] <= 1.0
    assert 0.0 <= result_good["safety_score"] <= 1.0
    assert 0.0 <= result_good["adme_score"] <= 1.0
    assert 0.0 <= result_good["rdkit_score"] <= 1.0
    assert result_bad["admet_score"] < result_good["admet_score"]
    assert result_rdkit_bad["admet_score"] < result_good["admet_score"]

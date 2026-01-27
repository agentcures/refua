"""ADMET prediction and scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import copy
import functools
import json
import math
import re
import warnings
from typing import Any

AnalysisResult = dict[str, Any]

DEFAULT_MODEL_VARIANT = "9b-chat"
_MODEL_ID_TEMPLATE = "google/txgemma-{variant}"
_PROMPT_PLACEHOLDER = "{Drug SMILES}"


@dataclass(frozen=True, slots=True)
class AdmetEndpoint:
    """Metadata describing an ADMET prediction endpoint."""

    task_id: str
    display_name: str
    description: str
    category: str


ENDPOINTS: tuple[AdmetEndpoint, ...] = (
    AdmetEndpoint(
        task_id="Pgp_Broccatelli",
        display_name="P-glycoprotein Substrate",
        description=(
            "Predicts whether a compound is a P-glycoprotein substrate that may be "
            "effluxed and show reduced absorption or brain exposure."
        ),
        category="distribution",
    ),
    AdmetEndpoint(
        task_id="Bioavailability_Ma",
        display_name="Oral Bioavailability",
        description=(
            "Estimates the likelihood of achieving adequate oral bioavailability "
            "after dosing."
        ),
        category="absorption",
    ),
    AdmetEndpoint(
        task_id="BBB_Martins",
        display_name="Blood-Brain Barrier Penetration",
        description="Estimates the probability a compound penetrates the blood-brain barrier.",
        category="distribution",
    ),
    AdmetEndpoint(
        task_id="HIA_Hou",
        display_name="Human Intestinal Absorption",
        description="Predicts the likelihood of human intestinal absorption.",
        category="absorption",
    ),
    AdmetEndpoint(
        task_id="Caco2_Wang",
        display_name="Caco-2 Permeability",
        description=(
            "Predicts Caco-2 cell permeability as a proxy for passive intestinal "
            "transport."
        ),
        category="absorption",
    ),
    AdmetEndpoint(
        task_id="PAMPA_NCATS",
        display_name="Passive Permeability",
        description=(
            "Predicts passive permeability in the PAMPA assay as an indicator of "
            "membrane diffusion."
        ),
        category="absorption",
    ),
    AdmetEndpoint(
        task_id="Solubility_AqSolDB",
        display_name="Aqueous Solubility",
        description=(
            "Estimates aqueous solubility to gauge formulation and absorption risk."
        ),
        category="absorption",
    ),
    AdmetEndpoint(
        task_id="VDss_Lombardo",
        display_name="Volume of Distribution (VDss)",
        description=(
            "Predicts volume of distribution at steady state, reflecting tissue "
            "partitioning."
        ),
        category="distribution",
    ),
    AdmetEndpoint(
        task_id="PPBR_AZ",
        display_name="Plasma Protein Binding",
        description=(
            "Predicts plasma protein binding fraction to indicate free drug exposure."
        ),
        category="distribution",
    ),
    AdmetEndpoint(
        task_id="CYP2D6_Veith",
        display_name="CYP2D6 Inhibition",
        description=(
            "Predicts CYP2D6 inhibition liability that can drive drug-drug interactions."
        ),
        category="metabolism",
    ),
    AdmetEndpoint(
        task_id="CYP3A4_Veith",
        display_name="CYP3A4 Inhibition",
        description=(
            "Predicts CYP3A4 inhibition liability that can drive drug-drug interactions."
        ),
        category="metabolism",
    ),
    AdmetEndpoint(
        task_id="CYP2C9_Veith",
        display_name="CYP2C9 Inhibition",
        description=(
            "Predicts CYP2C9 inhibition liability that can drive drug-drug interactions."
        ),
        category="metabolism",
    ),
    AdmetEndpoint(
        task_id="CYP2C19_Veith",
        display_name="CYP2C19 Inhibition",
        description=(
            "Predicts CYP2C19 inhibition liability that can drive drug-drug interactions."
        ),
        category="metabolism",
    ),
    AdmetEndpoint(
        task_id="CYP1A2_Veith",
        display_name="CYP1A2 Inhibition",
        description=(
            "Predicts CYP1A2 inhibition liability that can drive drug-drug interactions."
        ),
        category="metabolism",
    ),
    AdmetEndpoint(
        task_id="CYP2D6_Substrate_CarbonMangels",
        display_name="CYP2D6 Substrate",
        description=(
            "Predicts whether a compound is a CYP2D6 substrate, influencing clearance "
            "and interaction risk."
        ),
        category="metabolism",
    ),
    AdmetEndpoint(
        task_id="CYP3A4_Substrate_CarbonMangels",
        display_name="CYP3A4 Substrate",
        description=(
            "Predicts whether a compound is a CYP3A4 substrate, influencing clearance "
            "and interaction risk."
        ),
        category="metabolism",
    ),
    AdmetEndpoint(
        task_id="CYP2C9_Substrate_CarbonMangels",
        display_name="CYP2C9 Substrate",
        description=(
            "Predicts whether a compound is a CYP2C9 substrate, influencing clearance "
            "and interaction risk."
        ),
        category="metabolism",
    ),
    AdmetEndpoint(
        task_id="hERG",
        display_name="hERG Cardiotoxicity",
        description=(
            "Predicts hERG channel inhibition risk associated with QT prolongation and "
            "cardiotoxicity."
        ),
        category="safety_toxicity",
    ),
    AdmetEndpoint(
        task_id="AMES",
        display_name="Ames Mutagenicity",
        description="Predicts Ames mutagenicity potential as an indicator of genotoxic risk.",
        category="safety_toxicity",
    ),
    AdmetEndpoint(
        task_id="DILI",
        display_name="Drug-Induced Liver Injury",
        description=(
            "Predicts risk of drug-induced liver injury based on learned toxicity "
            "patterns."
        ),
        category="safety_toxicity",
    ),
    AdmetEndpoint(
        task_id="Skin_Reaction",
        display_name="Skin Sensitization",
        description="Predicts potential for skin sensitization or allergic response.",
        category="safety_toxicity",
    ),
    AdmetEndpoint(
        task_id="Carcinogens_Lagunin",
        display_name="Carcinogenicity",
        description="Predicts carcinogenicity potential based on chemical structure.",
        category="safety_toxicity",
    ),
    AdmetEndpoint(
        task_id="ClinTox",
        display_name="Clinical Toxicity",
        description=(
            "Predicts likelihood of clinical toxicity and trial failures due to safety "
            "issues."
        ),
        category="safety_toxicity",
    ),
    AdmetEndpoint(
        task_id="Tox21_NR_AhR",
        display_name="Aryl Hydrocarbon Receptor",
        description=(
            "Predicts activation of the aryl hydrocarbon receptor (AhR) signaling "
            "pathway."
        ),
        category="endocrine",
    ),
    AdmetEndpoint(
        task_id="Tox21_NR_Aromatase",
        display_name="Aromatase Inhibition",
        description=(
            "Predicts inhibition of aromatase, a key enzyme in estrogen biosynthesis."
        ),
        category="endocrine",
    ),
    AdmetEndpoint(
        task_id="Tox21_NR_AR",
        display_name="Androgen Receptor",
        description="Predicts activation of the androgen receptor signaling pathway.",
        category="endocrine",
    ),
    AdmetEndpoint(
        task_id="Tox21_NR_AR_LBD",
        display_name="Androgen Receptor LBD",
        description=(
            "Predicts binding or activation at the androgen receptor ligand binding "
            "domain."
        ),
        category="endocrine",
    ),
    AdmetEndpoint(
        task_id="Tox21_NR_ER",
        display_name="Estrogen Receptor",
        description="Predicts activation of the estrogen receptor signaling pathway.",
        category="endocrine",
    ),
    AdmetEndpoint(
        task_id="Tox21_NR_ER_LBD",
        display_name="Estrogen Receptor LBD",
        description=(
            "Predicts binding or activation at the estrogen receptor ligand binding "
            "domain."
        ),
        category="endocrine",
    ),
    AdmetEndpoint(
        task_id="Tox21_NR_PPAR_gamma",
        display_name="PPAR-gamma",
        description=(
            "Predicts activation of PPAR-gamma, a nuclear receptor involved in lipid "
            "and glucose regulation."
        ),
        category="endocrine",
    ),
    AdmetEndpoint(
        task_id="Tox21_SR_ARE",
        display_name="Antioxidant Response",
        description=(
            "Predicts activation of antioxidant response element (ARE) stress signaling."
        ),
        category="cellular_stress",
    ),
    AdmetEndpoint(
        task_id="Tox21_SR_ATAD5",
        display_name="DNA Damage Response",
        description=(
            "Predicts activation of ATAD5-mediated DNA damage response signaling."
        ),
        category="cellular_stress",
    ),
    AdmetEndpoint(
        task_id="Tox21_SR_HSE",
        display_name="Heat Shock Response",
        description="Predicts activation of heat shock response signaling.",
        category="cellular_stress",
    ),
    AdmetEndpoint(
        task_id="Tox21_SR_MMP",
        display_name="Mitochondrial Toxicity",
        description=(
            "Predicts mitochondrial membrane potential disruption indicating "
            "mitochondrial stress."
        ),
        category="cellular_stress",
    ),
    AdmetEndpoint(
        task_id="Tox21_SR_p53",
        display_name="p53 Activation",
        description=(
            "Predicts p53 pathway activation associated with DNA damage response."
        ),
        category="cellular_stress",
    ),
    AdmetEndpoint(
        task_id="Half_Life_Obach",
        display_name="Half-Life",
        description="Predicts in vivo half-life as a proxy for systemic exposure duration.",
        category="pharmacokinetics",
    ),
    AdmetEndpoint(
        task_id="Clearance_Hepatocyte_AZ",
        display_name="Hepatocyte Clearance",
        description=(
            "Predicts intrinsic clearance in hepatocytes, reflecting metabolic turnover."
        ),
        category="pharmacokinetics",
    ),
    AdmetEndpoint(
        task_id="Clearance_Microsome_AZ",
        display_name="Microsomal Clearance",
        description=(
            "Predicts intrinsic clearance in liver microsomes, reflecting metabolic "
            "turnover."
        ),
        category="pharmacokinetics",
    ),
    AdmetEndpoint(
        task_id="LD50_Zhu",
        display_name="LD50 (Acute Toxicity)",
        description="Predicts acute toxicity LD50 to estimate lethal dose in animal models.",
        category="safety_toxicity",
    ),
    AdmetEndpoint(
        task_id="Lipophilicity_AstraZeneca",
        display_name="Lipophilicity",
        description=(
            "Predicts lipophilicity (logP/logD) to inform permeability, solubility, and "
            "distribution."
        ),
        category="absorption",
    ),
)

ENDPOINT_BY_ID: dict[str, AdmetEndpoint] = {
    endpoint.task_id: endpoint for endpoint in ENDPOINTS
}
DEFAULT_ENDPOINTS: tuple[str, ...] = tuple(endpoint.task_id for endpoint in ENDPOINTS)


def get_endpoint(task_id: str) -> AdmetEndpoint | None:
    """Return endpoint metadata for a task id, if available."""
    return ENDPOINT_BY_ID.get(task_id)


def get_endpoint_description(
    task_id: str,
    *,
    default: str | None = None,
) -> str | None:
    """Return the endpoint description for a task id."""
    endpoint = get_endpoint(task_id)
    return endpoint.description if endpoint is not None else default


def _require_hf_hub_download() -> Any:
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError(
            "ADMET prediction requires the optional dependency "
            "'huggingface_hub'. Install refua[admet] or huggingface_hub."
        ) from exc
    return hf_hub_download


def _require_transformers() -> tuple[Any, Any, Any, Any]:
    try:
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )
    except Exception as exc:
        raise RuntimeError(
            "ADMET prediction requires the optional dependency 'transformers'. "
            "Install refua[admet] or transformers."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


def _require_rdkit() -> tuple[Any, Any, Any]:
    try:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem import Descriptors, QED  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError(
            "ADMET scoring requires RDKit. Install refua with RDKit available."
        ) from exc
    return Chem, Descriptors, QED


def _default_device_map() -> str | None:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    if find_spec("accelerate") is None:
        return None

    return "auto"


def _clamp_unit(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _looks_like_percent(text: str, end_index: int) -> bool:
    trailing = text[end_index:]
    stripped = trailing.lstrip()
    if stripped.startswith("%"):
        return True
    lowered = stripped.lower()
    return lowered.startswith("percent") or lowered.startswith("percentage")


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_probability(value: float | int | None) -> float:
    val = _safe_float(value)
    if val is None:
        return 0.5
    if val < 0.0:
        return 0.0
    if 0.0 <= val <= 1.0:
        return val
    if 1.0 < val <= 100.0:
        return val / 100.0
    if 100.0 < val <= 1000.0:
        return val / 1000.0
    return _clamp_unit(val)


def _score_range(
    value: float | int | None,
    *,
    hard_low: float,
    ideal_low: float,
    ideal_high: float,
    hard_high: float,
) -> float:
    val = _safe_float(value)
    if val is None:
        return 0.5
    if val <= hard_low or val >= hard_high:
        return 0.0
    if ideal_low <= val <= ideal_high:
        return 1.0
    if val < ideal_low:
        span = ideal_low - hard_low
        return (val - hard_low) / span if span > 0.0 else 0.0
    span = hard_high - ideal_high
    return (hard_high - val) / span if span > 0.0 else 0.0


def _score_max(value: float | int | None, *, ideal_max: float, hard_max: float) -> float:
    val = _safe_float(value)
    if val is None:
        return 0.5
    if val <= ideal_max:
        return 1.0
    if val >= hard_max:
        return 0.0
    span = hard_max - ideal_max
    return (hard_max - val) / span if span > 0.0 else 0.0


def _score_abs_max(
    value: float | int | None,
    *,
    ideal_abs: float,
    hard_abs: float,
    floor: float = 0.0,
) -> float:
    val = _safe_float(value)
    if val is None:
        return 0.5
    abs_val = abs(val)
    if abs_val <= ideal_abs:
        return 1.0
    if abs_val >= hard_abs:
        return floor
    span = hard_abs - ideal_abs
    if span <= 0.0:
        return floor
    scaled = (hard_abs - abs_val) / span
    return floor + (1.0 - floor) * scaled


def _strip_prompt(generated_text: str, prompt: str) -> str:
    if prompt and generated_text.startswith(prompt):
        return generated_text[len(prompt) :]
    return generated_text


def _extract_prediction(generated_text: str, prompt: str) -> float:
    text = _strip_prompt(generated_text, prompt).strip()

    if not text:
        return 0.0

    number_pattern = r"[-+]?(?:\d*\.?\d+)"
    prompt_ranges: list[tuple[int, int]] = []
    if prompt:
        start = 0
        while True:
            idx = text.find(prompt, start)
            if idx == -1:
                break
            prompt_ranges.append((idx, idx + len(prompt)))
            start = idx + len(prompt)

    for match in re.finditer(number_pattern, text):
        start_idx, end_idx = match.span()
        if any(
            prompt_start <= start_idx <= end_idx <= prompt_end
            for prompt_start, prompt_end in prompt_ranges
        ):
            continue
        try:
            value = float(match.group())
        except ValueError:
            continue
        if _looks_like_percent(text, end_idx):
            value /= 100.0
        return value

    lowered = text.lower()
    if "(a)" in lowered or lowered.strip().startswith("a"):
        return 0.0
    if "(b)" in lowered or lowered.strip().startswith("b"):
        return 1.0
    if any(word in lowered for word in ("no", "negative", "inactive", "false")):
        return 0.0
    if any(word in lowered for word in ("yes", "positive", "active", "true")):
        return 1.0

    return 0.0


def _compute_rdkit_metrics(smiles: str) -> dict[str, float]:
    Chem, Descriptors, QED = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mol_wt": float(Descriptors.MolWt(mol)),
        "exact_mol_wt": float(Descriptors.ExactMolWt(mol)),
        "mol_log_p": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "num_h_donors": float(Descriptors.NumHDonors(mol)),
        "num_h_acceptors": float(Descriptors.NumHAcceptors(mol)),
        "num_rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
        "ring_count": float(Descriptors.RingCount(mol)),
        "num_aromatic_rings": float(Descriptors.NumAromaticRings(mol)),
        "heavy_atom_count": float(Descriptors.HeavyAtomCount(mol)),
        "num_heteroatoms": float(Descriptors.NumHeteroatoms(mol)),
        "fraction_csp3": float(Descriptors.FractionCSP3(mol)),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
        "qed": float(QED.qed(mol)),
    }


class AdmetPredictor:
    """Model-backed ADMET predictor."""

    def __init__(
        self,
        *,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        task_ids: tuple[str, ...] | None = None,
        device_map: str | None = None,
    ) -> None:
        self.model_variant = model_variant
        self.model_id = _MODEL_ID_TEMPLATE.format(variant=model_variant)
        self._prompt_token = _PROMPT_PLACEHOLDER

        hf_hub_download = _require_hf_hub_download()
        prompts_path = hf_hub_download(
            repo_id=self.model_id,
            filename="tdc_prompts.json",
        )
        with open(prompts_path, "r", encoding="utf-8") as handle:
            prompts: dict[str, str] = json.load(handle)

        requested = task_ids if task_ids is not None else DEFAULT_ENDPOINTS
        available = [task for task in requested if task in prompts]
        missing = [task for task in requested if task not in prompts]
        if missing:
            warnings.warn(
                "Some ADMET prompts are missing; skipping endpoints: "
                + ", ".join(missing),
                stacklevel=2,
            )
        if not available:
            raise ValueError(
                "No valid ADMET prompts were found for the requested endpoints."
            )

        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline = (
            _require_transformers()
        )

        self._prompts = prompts
        self._task_ids = tuple(available)
        self.missing_task_ids = tuple(missing)
        self._device_map = device_map if device_map is not None else _default_device_map()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = self._load_model(
            AutoModelForCausalLM=AutoModelForCausalLM,
            BitsAndBytesConfig=BitsAndBytesConfig,
        )
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return the configured endpoint ids."""
        return self._task_ids

    def _load_model(self, *, AutoModelForCausalLM: Any, BitsAndBytesConfig: Any) -> Any:
        if self.model_variant == "2b-predict":
            return AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self._device_map,
            )

        quant_config = None
        try:
            if self._device_map is not None:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            quant_config = None

        if quant_config is not None:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quant_config,
                    device_map=self._device_map,
                )
            except (ImportError, ValueError, RuntimeError):
                pass

        return AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self._device_map,
        )

    def predict(
        self,
        smiles: str,
        *,
        max_new_tokens: int = 8,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Run model predictions for the configured endpoints."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        predictions: dict[str, float] = {}
        raw_outputs: dict[str, str] = {}

        for task_id in self._task_ids:
            prompt = self._prompts[task_id].replace(self._prompt_token, smiles)
            output = self._pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[
                0
            ]
            generated = output.get("generated_text", "")
            raw_outputs[task_id] = generated
            predictions[task_id] = _extract_prediction(generated, prompt)

        return predictions, raw_outputs

    def analyze(
        self,
        smiles: str,
        *,
        max_new_tokens: int = 8,
        include_scoring: bool = True,
    ) -> AnalysisResult:
        """Return predictions and optional scoring for a SMILES string."""
        predictions, raw_outputs = self.predict(
            smiles,
            max_new_tokens=max_new_tokens,
        )
        result: AnalysisResult = {
            "smiles": smiles,
            "predictions": predictions,
            "raw_outputs": raw_outputs,
            "missing_tasks": list(self.missing_task_ids),
        }
        if include_scoring:
            rdkit_metrics = _compute_rdkit_metrics(smiles)
            scorer = AdmetScorer()
            result.update(
                scorer.analyze_profile(predictions, rdkit_metrics=rdkit_metrics)
            )
        return result


class AdmetScorer:
    """Scoring utilities for ADMET predictions."""

    def __init__(self) -> None:
        self._task_scorers: dict[str, Any] = {
            "Pgp_Broccatelli": self._score_pgp,
            "Bioavailability_Ma": self._score_bioavailability,
            "BBB_Martins": self._score_bbb,
            "HIA_Hou": self._score_hia,
            "Caco2_Wang": self._score_caco2,
            "PAMPA_NCATS": self._score_pampa,
            "Solubility_AqSolDB": self._score_solubility,
            "VDss_Lombardo": self._score_vdss,
            "PPBR_AZ": self._score_ppbr,
            "CYP2D6_Veith": self._score_cyp_inhibition,
            "CYP3A4_Veith": self._score_cyp_inhibition,
            "CYP2C9_Veith": self._score_cyp_inhibition,
            "CYP2C19_Veith": self._score_cyp_inhibition,
            "CYP1A2_Veith": self._score_cyp_inhibition,
            "CYP2D6_Substrate_CarbonMangels": self._score_cyp_substrate,
            "CYP3A4_Substrate_CarbonMangels": self._score_cyp_substrate,
            "CYP2C9_Substrate_CarbonMangels": self._score_cyp_substrate,
            "hERG": self._score_herg,
            "AMES": self._score_ames,
            "DILI": self._score_dili,
            "Skin_Reaction": self._score_skin,
            "Carcinogens_Lagunin": self._score_carcinogenicity,
            "ClinTox": self._score_clintox,
            "Tox21_NR_AhR": self._score_tox21_nr,
            "Tox21_NR_Aromatase": self._score_tox21_nr,
            "Tox21_NR_AR": self._score_tox21_nr,
            "Tox21_NR_AR_LBD": self._score_tox21_nr,
            "Tox21_NR_ER": self._score_tox21_nr,
            "Tox21_NR_ER_LBD": self._score_tox21_nr,
            "Tox21_NR_PPAR_gamma": self._score_tox21_nr,
            "Tox21_SR_ARE": self._score_tox21_sr_general,
            "Tox21_SR_ATAD5": self._score_tox21_sr_general,
            "Tox21_SR_HSE": self._score_tox21_sr_general,
            "Tox21_SR_MMP": self._score_tox21_sr_critical,
            "Tox21_SR_p53": self._score_tox21_sr_critical,
            "Half_Life_Obach": self._score_half_life,
            "Clearance_Hepatocyte_AZ": self._score_clearance,
            "Clearance_Microsome_AZ": self._score_clearance,
            "LD50_Zhu": self._score_ld50,
            "Lipophilicity_AstraZeneca": self._score_lipophilicity,
        }

        self._task_weights: dict[str, float] = {
            "hERG": 3.0,
            "AMES": 3.0,
            "DILI": 2.5,
            "Carcinogens_Lagunin": 3.0,
            "ClinTox": 2.5,
            "Tox21_SR_p53": 2.0,
            "Tox21_SR_MMP": 2.0,
            "Bioavailability_Ma": 2.0,
            "HIA_Hou": 1.8,
            "Caco2_Wang": 1.4,
            "Solubility_AqSolDB": 1.5,
            "PAMPA_NCATS": 1.2,
            "VDss_Lombardo": 1.0,
            "PPBR_AZ": 0.9,
            "CYP3A4_Veith": 1.8,
            "CYP2D6_Veith": 1.5,
            "CYP2C9_Veith": 1.3,
            "CYP2C19_Veith": 1.4,
            "CYP1A2_Veith": 1.1,
            "CYP3A4_Substrate_CarbonMangels": 1.0,
            "CYP2D6_Substrate_CarbonMangels": 0.8,
            "CYP2C9_Substrate_CarbonMangels": 0.8,
            "BBB_Martins": 1.0,
            "Pgp_Broccatelli": 1.0,
            "Tox21_NR_AR": 1.2,
            "Tox21_NR_ER": 1.2,
            "Tox21_NR_Aromatase": 1.0,
            "Tox21_NR_AR_LBD": 1.0,
            "Tox21_NR_ER_LBD": 1.0,
            "Tox21_NR_AhR": 0.8,
            "Tox21_NR_PPAR_gamma": 0.8,
            "Tox21_SR_ARE": 0.8,
            "Tox21_SR_ATAD5": 0.8,
            "Tox21_SR_HSE": 0.6,
            "Skin_Reaction": 0.8,
            "Half_Life_Obach": 1.5,
            "Clearance_Hepatocyte_AZ": 1.3,
            "Clearance_Microsome_AZ": 1.2,
            "LD50_Zhu": 2.5,
            "Lipophilicity_AstraZeneca": 1.4,
        }

        self._rdkit_weights: dict[str, float] = {
            "mol_wt": 1.4,
            "exact_mol_wt": 0.4,
            "mol_log_p": 1.5,
            "tpsa": 1.2,
            "num_h_donors": 0.7,
            "num_h_acceptors": 0.7,
            "num_rotatable_bonds": 0.6,
            "ring_count": 0.5,
            "num_aromatic_rings": 0.6,
            "heavy_atom_count": 0.3,
            "num_heteroatoms": 0.3,
            "fraction_csp3": 0.7,
            "formal_charge": 0.5,
            "qed": 2.0,
        }

        self._safety_tasks = tuple(
            endpoint.task_id
            for endpoint in ENDPOINTS
            if endpoint.category in {"safety_toxicity", "endocrine", "cellular_stress"}
        )
        self._admet_tasks = tuple(
            endpoint.task_id
            for endpoint in ENDPOINTS
            if endpoint.category
            in {"absorption", "distribution", "metabolism", "pharmacokinetics"}
        )
        self._safety_weights = {
            task: self._task_weights[task]
            for task in self._safety_tasks
            if task in self._task_weights
        }
        self._admet_weights = {
            task: self._task_weights[task]
            for task in self._admet_tasks
            if task in self._task_weights
        }
        self._critical_tasks = (
            "hERG",
            "AMES",
            "DILI",
            "Carcinogens_Lagunin",
            "ClinTox",
            "Tox21_SR_p53",
            "Tox21_SR_MMP",
        )

    def score_prediction(self, task: str, prediction: float) -> float:
        """Score a single prediction for an endpoint."""
        scorer = self._task_scorers.get(task)
        if scorer is None:
            return 0.5 if prediction is None else _clamp_unit(prediction)
        try:
            return scorer(prediction)
        except (TypeError, ValueError):
            return 0.0

    def score_profile(
        self,
        predictions: dict[str, float],
        *,
        rdkit_metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Score a full prediction profile."""
        scores: dict[str, float] = {}
        for task, value in predictions.items():
            scores[f"score_{task}"] = self.score_prediction(task, value)
        if rdkit_metrics:
            scores.update(self._score_rdkit_metrics(rdkit_metrics))
        scores["score_admet"] = self._combined_score(scores)
        return scores

    def analyze_profile(
        self,
        predictions: dict[str, float],
        *,
        rdkit_metrics: dict[str, float] | None = None,
    ) -> AnalysisResult:
        """Analyze predictions and return scores with flags."""
        scores = self.score_profile(predictions, rdkit_metrics=rdkit_metrics)

        red_flags = self._collect_flags(
            scores,
            tasks=("hERG", "AMES", "DILI", "Carcinogens_Lagunin", "ClinTox"),
            threshold=0.5,
        )
        yellow_flags = self._collect_flags(
            scores,
            tasks=(
                "CYP3A4_Veith",
                "CYP2D6_Veith",
                "CYP2C19_Veith",
                "CYP1A2_Veith",
                "Bioavailability_Ma",
            ),
            threshold=0.6,
        )

        safety_score = self._safety_score(scores)
        adme_score = self._admet_score(scores)
        rdkit_score = self._rdkit_score(scores) if rdkit_metrics else None
        admet_score = scores["score_admet"]
        assessment = self._assessment(
            admet_score,
            safety_score,
            adme_score,
            red_flags,
            yellow_flags,
            scores,
        )
        result: AnalysisResult = {
            "scores": scores,
            "admet_score": admet_score,
            "assessment": assessment,
            "red_flags": red_flags,
            "yellow_flags": yellow_flags,
            "num_predictions": len(predictions),
            "safety_score": safety_score,
            "adme_score": adme_score,
        }
        if rdkit_metrics is not None:
            result["rdkit_metrics"] = rdkit_metrics
            result["rdkit_score"] = rdkit_score
        return result

    def _collect_flags(
        self,
        scores: dict[str, float],
        *,
        tasks: tuple[str, ...],
        threshold: float,
    ) -> list[str]:
        flags: list[str] = []
        for task in tasks:
            score = scores.get(f"score_{task}")
            if score is not None and score < threshold:
                flags.append(task)
        return flags

    def _weighted_average(
        self,
        scores: dict[str, float],
        weights: dict[str, float],
        *,
        prefix: str,
    ) -> float | None:
        num, den = 0.0, 0.0
        for key, weight in weights.items():
            score = scores.get(f"{prefix}{key}")
            if score is None or (isinstance(score, float) and math.isnan(score)):
                continue
            num += weight * score
            den += weight
        if den <= 0.0:
            return None
        return num / den

    def _rdkit_score(self, scores: dict[str, float]) -> float:
        value = self._weighted_average(
            scores,
            self._rdkit_weights,
            prefix="score_rdkit_",
        )
        return value if value is not None else 0.5

    def _score_prob_good(self, value: float, *, floor: float = 0.0) -> float:
        prob = _as_probability(value)
        return floor + (1.0 - floor) * prob

    def _score_prob_bad(self, value: float, *, floor: float = 0.0) -> float:
        prob = _as_probability(value)
        return floor + (1.0 - floor) * (1.0 - prob)

    def _score_rdkit_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        scorers = {
            "mol_wt": lambda v: _score_range(
                v,
                hard_low=120.0,
                ideal_low=200.0,
                ideal_high=500.0,
                hard_high=700.0,
            ),
            "exact_mol_wt": lambda v: _score_range(
                v,
                hard_low=120.0,
                ideal_low=200.0,
                ideal_high=500.0,
                hard_high=700.0,
            ),
            "mol_log_p": lambda v: _score_range(
                v,
                hard_low=-0.5,
                ideal_low=1.0,
                ideal_high=3.5,
                hard_high=5.5,
            ),
            "tpsa": lambda v: _score_range(
                v,
                hard_low=20.0,
                ideal_low=40.0,
                ideal_high=110.0,
                hard_high=140.0,
            ),
            "num_h_donors": lambda v: _score_max(v, ideal_max=3.0, hard_max=7.0),
            "num_h_acceptors": lambda v: _score_max(v, ideal_max=8.0, hard_max=12.0),
            "num_rotatable_bonds": lambda v: _score_max(
                v,
                ideal_max=7.0,
                hard_max=15.0,
            ),
            "ring_count": lambda v: _score_range(
                v,
                hard_low=0.0,
                ideal_low=1.0,
                ideal_high=4.0,
                hard_high=7.0,
            ),
            "num_aromatic_rings": lambda v: _score_max(
                v,
                ideal_max=2.0,
                hard_max=4.0,
            ),
            "heavy_atom_count": lambda v: _score_range(
                v,
                hard_low=10.0,
                ideal_low=20.0,
                ideal_high=55.0,
                hard_high=80.0,
            ),
            "num_heteroatoms": lambda v: _score_range(
                v,
                hard_low=0.0,
                ideal_low=2.0,
                ideal_high=10.0,
                hard_high=18.0,
            ),
            "fraction_csp3": lambda v: _score_range(
                v,
                hard_low=0.1,
                ideal_low=0.3,
                ideal_high=0.6,
                hard_high=0.9,
            ),
            "formal_charge": lambda v: _score_abs_max(
                v,
                ideal_abs=1.0,
                hard_abs=3.0,
                floor=0.1,
            ),
            "qed": lambda v: _clamp_unit(_safe_float(v) or 0.0),
        }

        scores: dict[str, float] = {}
        for name, scorer in scorers.items():
            if name in metrics:
                scores[f"score_rdkit_{name}"] = scorer(metrics[name])
        return scores

    def _safety_penalty(self, scores: dict[str, float]) -> float:
        penalty = 1.0
        for task in self._critical_tasks:
            score = scores.get(f"score_{task}")
            if score is None:
                continue
            if score < 0.15:
                penalty *= 0.5
            elif score < 0.3:
                penalty *= 0.7
        return penalty

    def _combined_score(self, scores: dict[str, float]) -> float:
        admet_component = self._weighted_average(
            scores,
            self._task_weights,
            prefix="score_",
        )
        if admet_component is None:
            admet_component = 0.5

        rdkit_component = self._weighted_average(
            scores,
            self._rdkit_weights,
            prefix="score_rdkit_",
        )
        if rdkit_component is None:
            combined = admet_component
        else:
            combined = 0.7 * admet_component + 0.3 * rdkit_component

        combined *= self._safety_penalty(scores)
        return _clamp_unit(combined)

    def _safety_score(self, scores: dict[str, float]) -> float:
        value = self._weighted_average(
            scores,
            self._safety_weights,
            prefix="score_",
        )
        return value if value is not None else 0.5

    def _admet_score(self, scores: dict[str, float]) -> float:
        value = self._weighted_average(
            scores,
            self._admet_weights,
            prefix="score_",
        )
        return value if value is not None else 0.5

    def _assessment(
        self,
        admet_score: float,
        safety_score: float,
        adme_score: float,
        red_flags: list[str],
        yellow_flags: list[str],
        scores: dict[str, float],
    ) -> str:
        if admet_score >= 0.85:
            tier = "Outstanding drug-like profile"
        elif admet_score >= 0.75:
            tier = "Strong drug-like profile"
        elif admet_score >= 0.65:
            tier = "Good drug-like profile"
        elif admet_score >= 0.55:
            tier = "Acceptable drug-like profile"
        elif admet_score >= 0.4:
            tier = "Suboptimal drug-like profile"
        else:
            tier = "Poor drug-like profile"

        notes: list[str] = []
        if safety_score >= 0.8:
            notes.append("excellent safety margins")
        elif safety_score >= 0.7:
            notes.append("favorable safety margins")
        elif safety_score < 0.5:
            notes.append("concerning safety signals")

        if adme_score >= 0.8:
            notes.append("strong ADME properties")
        elif adme_score >= 0.7:
            notes.append("good pharmacokinetics")
        elif adme_score < 0.5:
            notes.append("challenging pharmacokinetics")

        strengths = []
        for task, label in (
            ("Bioavailability_Ma", "high oral bioavailability"),
            ("HIA_Hou", "strong intestinal absorption"),
            ("Caco2_Wang", "solid permeability"),
            ("Solubility_AqSolDB", "favorable solubility"),
            ("PPBR_AZ", "balanced protein binding"),
        ):
            score = scores.get(f"score_{task}")
            if score is not None and score > 0.8:
                strengths.append(label)
        if len(strengths) >= 2:
            notes.append(f"{strengths[0]} and {strengths[1]}")
        elif len(strengths) == 1:
            notes.append(strengths[0])

        if red_flags:
            notes.append("critical safety flags present")
        elif yellow_flags:
            notes.append("moderate liabilities to review")

        if notes:
            return f"{tier} with {', '.join(notes)}"
        return tier

    def _score_pgp(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.2)

    def _score_bioavailability(self, value: float) -> float:
        return self._score_prob_good(value)

    def _score_bbb(self, value: float) -> float:
        return self._score_prob_good(value, floor=0.5)

    def _score_hia(self, value: float) -> float:
        return self._score_prob_good(value, floor=0.1)

    def _score_cyp_inhibition(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.05)

    def _score_cyp_substrate(self, value: float) -> float:
        prob = _as_probability(value)
        return _clamp_unit(0.85 - 0.25 * prob)

    def _score_herg(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.05)

    def _score_ames(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.02)

    def _score_dili(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.05)

    def _score_skin(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.2)

    def _score_carcinogenicity(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.01)

    def _score_clintox(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.05)

    def _score_tox21_nr(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.2)

    def _score_tox21_sr_general(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.2)

    def _score_tox21_sr_critical(self, value: float) -> float:
        return self._score_prob_bad(value, floor=0.1)

    def _score_pampa(self, value: float) -> float:
        return self._score_prob_good(value, floor=0.4)

    def _score_solubility(self, value: float) -> float:
        val = _safe_float(value)
        if val is None:
            return 0.5
        if -1.0 <= val <= 1.0:
            return 0.9
        if -2.0 <= val < -1.0 or 1.0 < val <= 2.0:
            return 0.7
        if -3.0 <= val < -2.0 or 2.0 < val <= 3.0:
            return 0.4
        return 0.1

    def _score_half_life(self, value: float) -> float:
        val = _safe_float(value)
        if val is None:
            return 0.5
        if val > 1.0:
            try:
                val = math.log10(val)
            except ValueError:
                return 0.4
        if 0.3 <= val <= 0.6:
            return 0.9
        if 0.2 <= val < 0.3 or 0.6 < val <= 0.7:
            return 0.75
        if 0.1 <= val < 0.2 or 0.7 < val <= 0.85:
            return 0.6
        return 0.4

    def _score_lipophilicity(self, value: float) -> float:
        val = _safe_float(value)
        if val is None:
            return 0.5
        if 0.0 <= val <= 1.0:
            if 0.35 <= val <= 0.65:
                return 0.9
            if 0.25 <= val < 0.35 or 0.65 < val <= 0.75:
                return 0.7
            if 0.15 <= val < 0.25 or 0.75 < val <= 0.85:
                return 0.5
            return 0.3
        return _score_range(
            val,
            hard_low=-0.5,
            ideal_low=1.0,
            ideal_high=3.5,
            hard_high=5.5,
        )

    def _score_caco2(self, value: float) -> float:
        val = _safe_float(value)
        if val is None:
            return 0.5
        if 0.0 <= val <= 1.0:
            return _clamp_unit(val)
        if val >= -5.2:
            return 0.9
        if val >= -5.5:
            return 0.75
        if val >= -5.8:
            return 0.55
        if val >= -6.2:
            return 0.35
        return 0.2

    def _score_vdss(self, value: float) -> float:
        val = _safe_float(value)
        if val is None:
            return 0.5

        candidates: list[float] = []
        if val > 0:
            candidates.append(val)
        if -3.0 <= val <= 3.0:
            try:
                candidates.append(10**val)
            except OverflowError:
                pass

        viable = [c for c in candidates if 0.05 <= c <= 25]
        if viable:
            target = min(viable, key=lambda c: abs(math.log10(c) - math.log10(1.0)))
        elif candidates:
            target = max(0.05, min(abs(candidates[0]), 25.0))
        else:
            return 0.4

        if 0.5 <= target <= 3.0:
            return 0.9
        if 0.3 <= target < 0.5 or 3.0 < target <= 6.0:
            return 0.75
        if 0.1 <= target < 0.3 or 6.0 < target <= 10.0:
            return 0.5
        return 0.3

    def _score_ppbr(self, value: float) -> float:
        val = _safe_float(value)
        if val is None:
            return 0.5
        binding = val * 100.0 if 0.0 <= val <= 1.0 else val
        if not 0.0 <= binding <= 100.0:
            return 0.3
        if 10.0 <= binding <= 90.0:
            return 0.85
        if 5.0 <= binding < 10.0 or 90.0 < binding <= 95.0:
            return 0.65
        if 1.0 <= binding < 5.0 or 95.0 < binding <= 99.0:
            return 0.45
        return 0.25

    def _score_clearance(self, value: float) -> float:
        val = _safe_float(value)
        if val is None:
            return 0.5
        if 0.0 <= val <= 1.0:
            return 1.0 - _clamp_unit(val)
        return _score_max(val, ideal_max=15.0, hard_max=40.0)

    def _score_ld50(self, value: float) -> float:
        val = _safe_float(value)
        if val is None:
            return 0.5
        if 0.0 <= val <= 1.0:
            return _clamp_unit(val)
        if val >= 2000.0:
            return 0.95
        if val >= 500.0:
            return 0.8
        if val >= 200.0:
            return 0.6
        if val >= 50.0:
            return 0.4
        return 0.2


@functools.lru_cache(maxsize=2)
def _predictor_for_variant(model_variant: str) -> AdmetPredictor:
    return AdmetPredictor(model_variant=model_variant)


@functools.lru_cache(maxsize=128)
def _cached_profile(
    smiles: str,
    model_variant: str,
    max_new_tokens: int,
    include_scoring: bool,
) -> AnalysisResult:
    predictor = _predictor_for_variant(model_variant)
    return predictor.analyze(
        smiles,
        max_new_tokens=max_new_tokens,
        include_scoring=include_scoring,
    )


def admet_profile(
    smiles: str,
    *,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    max_new_tokens: int = 8,
    include_scoring: bool = True,
    copy_result: bool = True,
) -> AnalysisResult:
    """Return ADMET predictions and optional scoring for a SMILES string."""
    profile = _cached_profile(
        smiles,
        model_variant,
        max_new_tokens,
        include_scoring,
    )
    if not copy_result:
        return profile
    return copy.deepcopy(profile)

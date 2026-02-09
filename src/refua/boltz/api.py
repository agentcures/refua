from __future__ import annotations

import io
import importlib.util
import os
import pickle
import tempfile
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from boltz.data import const
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals
from boltz.data.parse.a3m import _parse_a3m
from boltz.data.parse.schema import (
    get_template_records_from_matching,
    get_template_records_from_search,
    parse_boltz_schema,
)
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Coords,
    Input,
    Interface,
    Structure,
    StructureV2,
    Target,
    TemplateInfo,
    Tokenized,
)
from boltz.data.write.mmcif import to_mmcif
from boltz.model.models.boltz2 import Boltz2 as Boltz2Model
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from torch import Tensor

from refua.api import (
    AtomRef,
    Bond,
    ChainIds,
    TokenRef,
    as_atom_ref,
    as_token_ref,
    normalize_chain_ids,
)

_CUEQUIVARIANCE_MODULE = "cuequivariance_torch"


def _cuequivariance_available() -> bool:
    return importlib.util.find_spec(_CUEQUIVARIANCE_MODULE) is not None


def _is_cuequivariance_import_error(exc: ModuleNotFoundError) -> bool:
    missing_name = getattr(exc, "name", "") or ""
    if missing_name.startswith("cuequivariance"):
        return True
    return "cuequivariance" in str(exc).lower()


def msa_from_a3m(
    text: str,
    taxonomy: Mapping[str, str] | None = None,
    max_seqs: int | None = None,
) -> MSA:
    """Parse A3M content from a string into an MSA object."""
    return _parse_a3m(io.StringIO(text), taxonomy, max_seqs)


def bcif_bytes_from_mmcif(mmcif_text: str) -> bytes:
    """Convert an mmCIF string into in-memory BCIF bytes."""
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError("gemmi is required for BCIF output.") from exc

    if hasattr(gemmi.cif, "read_string"):
        doc = gemmi.cif.read_string(mmcif_text)
    else:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                tmp.write(mmcif_text.encode("utf-8"))
                tmp.flush()
                tmp_path = tmp.name
            doc = gemmi.cif.read(tmp_path)
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    for attr in ("as_binary", "as_binary_string"):
        if hasattr(doc, attr):
            data = getattr(doc, attr)()
            return (
                data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8")
            )

    if hasattr(doc, "write_binary"):
        buffer = io.BytesIO()
        doc.write_binary(buffer)
        return buffer.getvalue()

    if hasattr(doc, "write_file"):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".bcif", delete=False) as tmp:
                tmp_path = tmp.name
            doc.write_file(tmp_path)
            return Path(tmp_path).read_bytes()
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    raise RuntimeError("BCIF output is not supported by this gemmi build.")


def bcif_bytes_from_structure(
    structure: Structure,
    *,
    plddts: Any | None = None,
    boltz2: bool = False,
) -> bytes:
    """Serialize a structure to BCIF bytes using gemmi."""
    from boltz.data.write.mmcif import to_mmcif

    mmcif_text = to_mmcif(structure, plddts=plddts, boltz2=boltz2)
    return bcif_bytes_from_mmcif(mmcif_text)


def _normalize_ids(ids: ChainIds) -> tuple[str, ...]:
    return normalize_chain_ids(ids)


def _normalize_modifications(
    modifications: Iterable[Modification | tuple[int, str]],
) -> tuple[Modification, ...]:
    normalized = []
    for mod in modifications:
        if isinstance(mod, Modification):
            normalized.append(mod)
        else:
            position, ccd = mod
            normalized.append(Modification(position=position, ccd=ccd))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class Modification:
    """Single residue modification."""

    position: int
    ccd: str

    def __post_init__(self) -> None:
        if self.position < 1:
            msg = "Modification positions are 1-indexed."
            raise ValueError(msg)

    def to_schema(self) -> dict[str, int | str]:
        return {"position": self.position, "ccd": self.ccd}


def _as_atom_ref(ref: AtomRef | Sequence[str | int]) -> AtomRef:
    return as_atom_ref(ref)


def _as_token_ref(ref: TokenRef | Sequence[str | int]) -> TokenRef:
    return as_token_ref(ref)


@dataclass(frozen=True, slots=True)
class Pocket:
    """Pocket constraint."""

    binder: str
    contacts: tuple[TokenRef, ...]
    max_distance: float = 6.0
    force: bool = False

    def to_schema(self) -> dict[str, dict[str, Any]]:
        return {
            "pocket": {
                "binder": self.binder,
                "contacts": [c.to_schema() for c in self.contacts],
                "max_distance": self.max_distance,
                "force": self.force,
            }
        }


@dataclass(frozen=True, slots=True)
class Contact:
    """Contact constraint."""

    token1: TokenRef
    token2: TokenRef
    max_distance: float = 6.0
    force: bool = False

    def to_schema(self) -> dict[str, dict[str, Any]]:
        return {
            "contact": {
                "token1": self.token1.to_schema(),
                "token2": self.token2.to_schema(),
                "max_distance": self.max_distance,
                "force": self.force,
            }
        }


@dataclass(frozen=True, slots=True)
class Affinity:
    """Affinity prediction request."""

    binder: str

    def to_schema(self) -> dict[str, dict[str, str]]:
        return {"affinity": {"binder": self.binder}}


@dataclass(frozen=True, slots=True)
class Protein:
    """Protein chain specification."""

    ids: tuple[str, ...]
    sequence: str
    modifications: tuple[Modification, ...] = ()
    msa: MSA | None = None
    cyclic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", _normalize_ids(self.ids))
        object.__setattr__(
            self, "modifications", _normalize_modifications(self.modifications)
        )

    def to_schema(self, msa_id: str | int) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "id": list(self.ids) if len(self.ids) > 1 else self.ids[0],
            "sequence": self.sequence,
            "msa": msa_id,
        }
        if self.modifications:
            payload["modifications"] = [mod.to_schema() for mod in self.modifications]
        if self.cyclic:
            payload["cyclic"] = True
        return {"protein": payload}


@dataclass(frozen=True, slots=True)
class DNA:
    """DNA chain specification."""

    ids: tuple[str, ...]
    sequence: str
    modifications: tuple[Modification, ...] = ()
    cyclic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", _normalize_ids(self.ids))
        object.__setattr__(
            self, "modifications", _normalize_modifications(self.modifications)
        )

    def to_schema(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "id": list(self.ids) if len(self.ids) > 1 else self.ids[0],
            "sequence": self.sequence,
        }
        if self.modifications:
            payload["modifications"] = [mod.to_schema() for mod in self.modifications]
        if self.cyclic:
            payload["cyclic"] = True
        return {"dna": payload}


@dataclass(frozen=True, slots=True)
class RNA:
    """RNA chain specification."""

    ids: tuple[str, ...]
    sequence: str
    modifications: tuple[Modification, ...] = ()
    cyclic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", _normalize_ids(self.ids))
        object.__setattr__(
            self, "modifications", _normalize_modifications(self.modifications)
        )

    def to_schema(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "id": list(self.ids) if len(self.ids) > 1 else self.ids[0],
            "sequence": self.sequence,
        }
        if self.modifications:
            payload["modifications"] = [mod.to_schema() for mod in self.modifications]
        if self.cyclic:
            payload["cyclic"] = True
        return {"rna": payload}


@dataclass(frozen=True, slots=True)
class Ligand:
    """Ligand specification."""

    ids: tuple[str, ...]
    ccd: tuple[str, ...] | None = None
    smiles: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", _normalize_ids(self.ids))
        if (self.ccd is None) == (self.smiles is None):
            msg = "Ligand requires exactly one of ccd or smiles."
            raise ValueError(msg)
        if self.ccd is not None:
            if isinstance(self.ccd, str):
                normalized = (self.ccd,)
            else:
                normalized = tuple(self.ccd)
            object.__setattr__(self, "ccd", normalized)

    def to_schema(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "id": list(self.ids) if len(self.ids) > 1 else self.ids[0],
        }
        if self.ccd is not None:
            payload["ccd"] = list(self.ccd) if len(self.ccd) > 1 else self.ccd[0]
        else:
            payload["smiles"] = self.smiles
        return {"ligand": payload}


Chain = Union[Protein, DNA, RNA, Ligand]


@dataclass(frozen=True, slots=True)
class Template:
    """In-memory template specification."""

    name: str
    structure: StructureV2
    sequences: Mapping[str, str]
    chain_ids: Sequence[str] | None = None
    template_chain_ids: Sequence[str] | None = None
    force: bool = False
    threshold: float | None = None

    def records(self, query_sequences: Mapping[str, str]) -> list[TemplateInfo]:
        if self.force and self.threshold is None:
            msg = f"Template {self.name} requires threshold when force is enabled."
            raise ValueError(msg)

        chain_ids = (
            list(self.chain_ids)
            if self.chain_ids is not None
            else list(query_sequences)
        )
        template_chain_ids = (
            list(self.template_chain_ids)
            if self.template_chain_ids is not None
            else list(self.sequences)
        )

        if not chain_ids:
            msg = f"Template {self.name} has no query chains to match."
            raise ValueError(msg)
        if not template_chain_ids:
            msg = f"Template {self.name} has no template chains to match."
            raise ValueError(msg)

        for chain_id in chain_ids:
            if chain_id not in query_sequences:
                msg = f"Template {self.name} chain {chain_id} is not a protein chain."
                raise ValueError(msg)
        for chain_id in template_chain_ids:
            if chain_id not in self.sequences:
                msg = f"Template {self.name} missing sequence for chain {chain_id}."
                raise ValueError(msg)

        matched = (
            self.chain_ids is not None
            and self.template_chain_ids is not None
            and len(chain_ids) == len(template_chain_ids)
            and len(chain_ids) > 0
        )
        threshold = self.threshold if self.force else float("inf")

        if matched:
            return get_template_records_from_matching(
                template_id=self.name,
                chain_ids=chain_ids,
                sequences=query_sequences,
                template_chain_ids=template_chain_ids,
                template_sequences=self.sequences,
                force=self.force,
                threshold=threshold,
            )

        return get_template_records_from_search(
            template_id=self.name,
            chain_ids=chain_ids,
            sequences=query_sequences,
            template_chain_ids=template_chain_ids,
            template_sequences=self.sequences,
            force=self.force,
            threshold=threshold,
        )


@dataclass
class Spec:
    """In-memory model input specification for a molecular complex."""

    name: str
    sequences: list[Chain] = field(default_factory=list)
    constraints: list[Bond | Pocket | Contact] = field(default_factory=list)
    templates: list[Template] = field(default_factory=list)
    affinity: Affinity | None = None

    @property
    def chains(self) -> list[Chain]:
        """Alias for sequences using biological chain terminology."""
        return self.sequences

    @chains.setter
    def chains(self, value: list[Chain]) -> None:
        self.sequences = value

    def add(self, *chains: Chain) -> Spec:
        self.sequences.extend(chains)
        return self

    def add_chain(self, *chains: Chain) -> Spec:
        """Add chains to the complex."""
        return self.add(*chains)

    def protein_chain(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        msa: MSA | None = None,
        cyclic: bool = False,
    ) -> Spec:
        """Add a protein chain to the complex."""
        return self.protein(
            ids,
            sequence,
            modifications=modifications,
            msa=msa,
            cyclic=cyclic,
        )

    def protein(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        msa: MSA | None = None,
        cyclic: bool = False,
    ) -> Spec:
        self.sequences.append(
            Protein(
                ids=ids,
                sequence=sequence,
                modifications=tuple(modifications),
                msa=msa,
                cyclic=cyclic,
            )
        )
        return self

    def dna_chain(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        cyclic: bool = False,
    ) -> Spec:
        """Add a DNA chain to the complex."""
        return self.dna(
            ids,
            sequence,
            modifications=modifications,
            cyclic=cyclic,
        )

    def dna(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        cyclic: bool = False,
    ) -> Spec:
        self.sequences.append(
            DNA(
                ids=ids,
                sequence=sequence,
                modifications=tuple(modifications),
                cyclic=cyclic,
            )
        )
        return self

    def rna_chain(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        cyclic: bool = False,
    ) -> Spec:
        """Add an RNA chain to the complex."""
        return self.rna(
            ids,
            sequence,
            modifications=modifications,
            cyclic=cyclic,
        )

    def rna(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        cyclic: bool = False,
    ) -> Spec:
        self.sequences.append(
            RNA(
                ids=ids,
                sequence=sequence,
                modifications=tuple(modifications),
                cyclic=cyclic,
            )
        )
        return self

    def ligand_chain(
        self,
        ids: ChainIds,
        *,
        ccd: str | Sequence[str] | None = None,
        smiles: str | None = None,
    ) -> Spec:
        """Add a ligand chain to the complex."""
        return self.ligand(ids, ccd=ccd, smiles=smiles)

    def ligand(
        self,
        ids: ChainIds,
        *,
        ccd: str | Sequence[str] | None = None,
        smiles: str | None = None,
    ) -> Spec:
        self.sequences.append(Ligand(ids=ids, ccd=ccd, smiles=smiles))
        return self

    def bond(
        self,
        atom1: AtomRef | Sequence[str | int],
        atom2: AtomRef | Sequence[str | int],
    ) -> Spec:
        self.constraints.append(Bond(_as_atom_ref(atom1), _as_atom_ref(atom2)))
        return self

    def pocket(
        self,
        binder: str,
        contacts: Iterable[TokenRef | Sequence[str | int]],
        *,
        max_distance: float = 6.0,
        force: bool = False,
    ) -> Spec:
        contact_refs = tuple(_as_token_ref(c) for c in contacts)
        self.constraints.append(
            Pocket(
                binder=binder,
                contacts=contact_refs,
                max_distance=max_distance,
                force=force,
            )
        )
        return self

    def contact(
        self,
        token1: TokenRef | Sequence[str | int],
        token2: TokenRef | Sequence[str | int],
        *,
        max_distance: float = 6.0,
        force: bool = False,
    ) -> Spec:
        self.constraints.append(
            Contact(
                token1=_as_token_ref(token1),
                token2=_as_token_ref(token2),
                max_distance=max_distance,
                force=force,
            )
        )
        return self

    def add_template(self, template: Template) -> Spec:
        self.templates.append(template)
        return self

    def request_affinity(self, binder: str) -> Spec:
        self.affinity = Affinity(binder=binder)
        return self

    def msa_by_chain(self) -> dict[str, MSA]:
        msa_map: dict[str, MSA] = {}
        for chain in self.sequences:
            if isinstance(chain, Protein) and chain.msa is not None:
                for chain_id in chain.ids:
                    msa_map[chain_id] = chain.msa
        return msa_map

    def requires_components(self) -> bool:
        for chain in self.sequences:
            if isinstance(chain, (Protein, DNA, RNA)):
                return True
            if isinstance(chain, Ligand) and chain.ccd is not None:
                return True
        return False

    def protein_sequences(self) -> dict[str, str]:
        sequences: dict[str, str] = {}
        for chain in self.sequences:
            if isinstance(chain, Protein):
                for chain_id in chain.ids:
                    sequences[chain_id] = chain.sequence
        return sequences

    def to_schema(self) -> dict[str, Any]:
        self._validate()
        msa_ids: dict[int, str] = {}
        msa_counter = 0

        entries: list[dict[str, Any]] = []
        for chain in self.sequences:
            if isinstance(chain, Protein):
                msa_id: str | int = -1
                if chain.msa is not None:
                    key = id(chain.msa)
                    if key not in msa_ids:
                        msa_ids[key] = f"in_memory:{msa_counter}"
                        msa_counter += 1
                    msa_id = msa_ids[key]
                entries.append(chain.to_schema(msa_id))
            else:
                entries.append(chain.to_schema())

        schema: dict[str, Any] = {"version": 1, "sequences": entries}
        if self.constraints:
            schema["constraints"] = [c.to_schema() for c in self.constraints]
        if self.affinity is not None:
            schema["properties"] = [self.affinity.to_schema()]
        return schema

    def _validate(self) -> None:
        if not self.sequences:
            raise ValueError("Spec requires at least one chain.")

        chain_ids: set[str] = set()
        seq_to_msa: dict[str, MSA | None] = {}
        ligand_ids: set[str] = set()

        for chain in self.sequences:
            for chain_id in chain.ids:
                if chain_id in chain_ids:
                    msg = f"Duplicate chain id {chain_id} in spec."
                    raise ValueError(msg)
                chain_ids.add(chain_id)

            if isinstance(chain, Protein):
                if (
                    chain.sequence in seq_to_msa
                    and seq_to_msa[chain.sequence] is not chain.msa
                ):
                    msg = "Proteins with identical sequences must share the same MSA object."
                    raise ValueError(msg)
                seq_to_msa[chain.sequence] = chain.msa
            if isinstance(chain, Ligand):
                ligand_ids.update(chain.ids)

        for constraint in self.constraints:
            if isinstance(constraint, Bond):
                if (
                    constraint.atom1.chain not in chain_ids
                    or constraint.atom2.chain not in chain_ids
                ):
                    msg = "Bond constraint refers to unknown chain id."
                    raise ValueError(msg)
            elif isinstance(constraint, Pocket):
                if constraint.binder not in chain_ids:
                    msg = "Pocket constraint refers to unknown binder chain id."
                    raise ValueError(msg)
                for contact in constraint.contacts:
                    if contact.chain not in chain_ids:
                        msg = "Pocket constraint refers to unknown contact chain id."
                        raise ValueError(msg)
            elif isinstance(constraint, Contact):
                if (
                    constraint.token1.chain not in chain_ids
                    or constraint.token2.chain not in chain_ids
                ):
                    msg = "Contact constraint refers to unknown chain id."
                    raise ValueError(msg)

        if self.affinity is not None and self.affinity.binder not in ligand_ids:
            msg = "Affinity binder must reference a ligand chain id."
            raise ValueError(msg)


class Complex(Spec):
    """Biological complex specification (fluent alias of Spec)."""


@dataclass(frozen=True)
class Trace:
    """Captured pipeline state."""

    spec: Spec
    target: Target
    input: Input
    tokenized: Tokenized
    chain_map: Mapping[str, int]
    features: Mapping[str, Any] | None = None


class Pipeline:
    """In-memory pipeline for tokenization and featurization."""

    def __init__(
        self,
        *,
        version: int | str = 2,
        components: Mapping[str, Mol] | None = None,
        molecules: Mapping[str, Mol] | None = None,
        tokenizer: BoltzTokenizer | Boltz2Tokenizer | None = None,
        featurizer: BoltzFeaturizer | Boltz2Featurizer | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        version_value = int(version)
        if version_value not in (1, 2):
            raise ValueError("Pipeline version must be 1 or 2.")

        self.version = version_value
        self.components = dict(components or {})
        self.molecules = dict(molecules or self.components)
        self.tokenizer = tokenizer or (
            Boltz2Tokenizer() if self.version == 2 else BoltzTokenizer()
        )
        self.featurizer = featurizer or (
            Boltz2Featurizer() if self.version == 2 else BoltzFeaturizer()
        )
        self.rng = rng or np.random.default_rng(42)

    def build_target(
        self,
        spec: Spec,
        *,
        components: Mapping[str, Mol] | None = None,
    ) -> Target:
        ccd = dict(components or self.components)
        if not ccd and spec.requires_components():
            raise ValueError("components must include CCD molecules for parsing.")

        target = parse_boltz_schema(
            name=spec.name,
            schema=spec.to_schema(),
            ccd=ccd,
            mol_dir=None,
            boltz_2=self.version == 2,
        )
        return self._apply_templates(spec, target)

    def build_input(
        self,
        spec: Spec,
        target: Target,
        *,
        msa: Mapping[str, MSA] | None = None,
    ) -> Input:
        msa_by_chain = dict(msa or spec.msa_by_chain())
        chain_map = self._chain_name_to_id(target)
        msa_by_id: dict[int, MSA] = {}
        for chain_id, chain_msa in msa_by_chain.items():
            if chain_id not in chain_map:
                msg = f"MSA provided for unknown chain id {chain_id}."
                raise ValueError(msg)
            msa_by_id[chain_map[chain_id]] = chain_msa

        return Input(
            structure=target.structure,
            msa=msa_by_id,
            record=target.record,
            residue_constraints=target.residue_constraints,
            templates=target.templates,
            extra_mols=target.extra_mols,
        )

    def tokenize(self, input_data: Input) -> Tokenized:
        return self.tokenizer.tokenize(input_data)

    def prepare(
        self,
        spec: Spec,
        *,
        components: Mapping[str, Mol] | None = None,
        msa: Mapping[str, MSA] | None = None,
    ) -> Trace:
        target = self.build_target(spec, components=components)
        input_data = self.build_input(spec, target, msa=msa)
        tokenized = self.tokenize(input_data)
        chain_map = self._chain_name_to_id(target)
        return Trace(
            spec=spec,
            target=target,
            input=input_data,
            tokenized=tokenized,
            chain_map=chain_map,
        )

    def featurize(
        self,
        data: Trace | Tokenized,
        *,
        molecules: Mapping[str, Mol] | None = None,
        max_seqs: int = const.max_msa_seqs,
        random: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> Trace | Mapping[str, Any]:
        if isinstance(data, Trace):
            features = self._featurize(
                data.tokenized,
                molecules=molecules,
                max_seqs=max_seqs,
                random=random,
                **kwargs,
            )
            return replace(data, features=features)

        return self._featurize(
            data,
            molecules=molecules,
            max_seqs=max_seqs,
            random=random,
            **kwargs,
        )

    def predict(self, model: Any, features: Mapping[str, Any], **kwargs: Any) -> Any:
        return model(features, **kwargs)

    def _featurize(
        self,
        tokenized: Tokenized,
        *,
        molecules: Mapping[str, Mol] | None,
        max_seqs: int,
        random: np.random.Generator | None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        if self.version == 1:
            options = tokenized.record.inference_options if tokenized.record else None
            if options and options.pocket_constraints:
                binder, pocket = options.pocket_constraints[0][:2]
            else:
                binder, pocket = None, None

            call_kwargs = {
                "training": False,
                "max_atoms": None,
                "max_tokens": None,
                "max_seqs": max_seqs,
                "pad_to_max_seqs": False,
                "symmetries": {},
                "compute_symmetries": False,
                "inference_binder": binder,
                "inference_pocket": pocket,
                "compute_constraint_features": binder is not None
                and pocket is not None,
            }
            call_kwargs.update(kwargs)
            features = self.featurizer.process(tokenized, **call_kwargs)
            features["record"] = tokenized.record
            return features

        molecules_map = dict(molecules or self.molecules)
        if tokenized.extra_mols:
            molecules_map.update(tokenized.extra_mols)

        missing = set(tokenized.tokens["res_name"].tolist()) - set(molecules_map.keys())
        if missing:
            msg = f"Missing molecules for residues: {sorted(missing)}."
            raise ValueError(msg)

        options = tokenized.record.inference_options if tokenized.record else None
        pocket_constraints = options.pocket_constraints if options else None
        contact_constraints = options.contact_constraints if options else None
        compute_constraints = bool(pocket_constraints or contact_constraints)
        compute_affinity = bool(tokenized.record and tokenized.record.affinity)

        call_kwargs = {
            "molecules": molecules_map,
            "random": random or self.rng,
            "training": False,
            "max_atoms": None,
            "max_tokens": None,
            "max_seqs": max_seqs,
            "pad_to_max_seqs": False,
            "single_sequence_prop": 0.0,
            "compute_frames": True,
            "inference_pocket_constraints": pocket_constraints,
            "inference_contact_constraints": contact_constraints,
            "compute_constraint_features": compute_constraints,
            "compute_affinity": compute_affinity,
        }
        call_kwargs.update(kwargs)
        features = self.featurizer.process(tokenized, **call_kwargs)
        features["record"] = tokenized.record
        return features

    def _apply_templates(self, spec: Spec, target: Target) -> Target:
        if not spec.templates:
            return target
        if self.version != 2:
            raise ValueError("Templates are only supported in Boltz-2.")

        query_sequences = spec.protein_sequences()
        template_records: list[TemplateInfo] = []
        template_structures: dict[str, StructureV2] = {}

        for template in spec.templates:
            if template.name in template_structures:
                msg = f"Duplicate template name {template.name}."
                raise ValueError(msg)
            template_records.extend(template.records(query_sequences))
            template_structures[template.name] = template.structure

        record = replace(target.record, templates=template_records)
        return replace(target, record=record, templates=template_structures)

    @staticmethod
    def _chain_name_to_id(target: Target) -> dict[str, int]:
        return {
            str(chain["name"]): int(chain["asym_id"])
            for chain in target.structure.chains
        }


_DEFAULT_STRUCTURE_PREDICT_ARGS = {
    "recycling_steps": 3,
    "sampling_steps": 30,
    "diffusion_samples": 1,
    "max_parallel_samples": None,
}

_DEFAULT_AFFINITY_PREDICT_ARGS = {
    "recycling_steps": 3,
    "sampling_steps": 30,
    "diffusion_samples": 1,
    "max_parallel_samples": None,
}

_DEFAULT_PAIRFORMER_ARGS = {
    "num_blocks": 64,
    "num_heads": 16,
    "dropout": 0.0,
    "activation_checkpointing": False,
    "offload_to_cpu": False,
    "v2": True,
}

_DEFAULT_MSA_ARGS = {
    "msa_s": 64,
    "msa_blocks": 4,
    "msa_dropout": 0.0,
    "z_dropout": 0.0,
    "use_paired_feature": True,
    "pairwise_head_width": 32,
    "pairwise_num_heads": 4,
    "activation_checkpointing": False,
    "offload_to_cpu": False,
    "subsample_msa": False,
    "num_subsampled_msa": 1024,
}

_DEFAULT_DIFFUSION_PROCESS_ARGS = {
    "gamma_0": 0.8,
    "gamma_min": 1.0,
    "noise_scale": 1.003,
    "rho": 7,
    "step_scale": 1.5,
    "sigma_min": 0.0001,
    "sigma_max": 160.0,
    "sigma_data": 16.0,
    "P_mean": -1.2,
    "P_std": 1.5,
    "coordinate_augmentation": True,
    "alignment_reverse_diff": True,
    "synchronize_sigmas": True,
}

_DEFAULT_STEERING_ARGS = {
    "fk_steering": False,
    "num_particles": 3,
    "fk_lambda": 4.0,
    "fk_resampling_interval": 3,
    "physical_guidance_update": False,
    "contact_guidance_update": True,
    "num_gd_steps": 20,
}

_DEFAULT_AFFINITY_STEERING_ARGS = {
    **_DEFAULT_STEERING_ARGS,
}

_BATCH_SKIP_KEYS = {
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
    "record",
    "affinity_mw",
}


def _default_cache_dir() -> Path:
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved = Path(env_cache).expanduser().resolve()
        if not resolved.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return resolved
    return Path("~/.boltz").expanduser()


def _normalize_steering_args(
    steering_args: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(_DEFAULT_STEERING_ARGS)
    if steering_args:
        merged.update(steering_args)
    return merged


def _batchify(features: Mapping[str, Any]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in features.items():
        if key in _BATCH_SKIP_KEYS:
            batch[key] = [value]
        elif isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    for key, value in batch.items():
        if key in _BATCH_SKIP_KEYS:
            continue
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def _mmcif_to_bcif(mmcif_text: str) -> bytes:
    return bcif_bytes_from_mmcif(mmcif_text)


def _select_best_prediction(
    out: Mapping[str, Tensor],
) -> tuple[int, float | None]:
    scores = None
    complex_plddt = out.get("complex_plddt")
    iptm = out.get("iptm")
    ptm = out.get("ptm")

    if complex_plddt is not None and (iptm is not None or ptm is not None):
        if iptm is not None and not torch.allclose(iptm, torch.zeros_like(iptm)):
            metric = iptm
        else:
            metric = ptm

        if metric is not None:
            scores = (4 * complex_plddt + metric) / 5

    if scores is None:
        return 0, None

    best_idx = int(torch.argmax(scores).item())
    best_score = float(scores[best_idx].item())
    return best_idx, best_score


def _apply_predicted_coords(structure: StructureV2, coords: np.ndarray) -> StructureV2:
    structure = structure.remove_invalid_chains()
    atoms = structure.atoms.copy()
    residues = structure.residues.copy()
    atoms["coords"] = coords
    atoms["is_present"] = True
    residues["is_present"] = True
    interfaces = np.array([], dtype=Interface)
    coord_rows = np.array([(x,) for x in coords], dtype=Coords)
    return replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
        coords=coord_rows,
    )


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def _ligand_chain_ids(spec: Spec) -> list[str]:
    ligand_ids: list[str] = []
    for chain in spec.sequences:
        if isinstance(chain, Ligand):
            ligand_ids.extend(chain.ids)
    return ligand_ids


@dataclass(frozen=True, slots=True)
class StructurePrediction:
    """Predicted structure with optional confidence metadata."""

    structure: StructureV2
    plddt: Tensor | None = None
    confidence_score: float | None = None


@dataclass(frozen=True, slots=True)
class AffinityPrediction:
    """Affinity prediction output."""

    ic50: float
    binding_probability: float
    ic50_1: float | None = None
    binding_probability_1: float | None = None
    ic50_2: float | None = None
    binding_probability_2: float | None = None

    @property
    def pred_value(self) -> float:
        return self.ic50

    @property
    def probability_binary(self) -> float:
        return self.binding_probability

    @property
    def pred_value1(self) -> float | None:
        return self.ic50_1

    @property
    def probability_binary1(self) -> float | None:
        return self.binding_probability_1

    @property
    def pred_value2(self) -> float | None:
        return self.ic50_2

    @property
    def probability_binary2(self) -> float | None:
        return self.binding_probability_2


class FoldComplex:
    """Builder for simple in-memory Boltz2 predictions."""

    def __init__(self, model: Boltz2, name: str = "complex") -> None:
        self._model = model
        self._spec = Spec(name)
        self._structure_prediction: StructurePrediction | None = None
        self._dirty = True

    @property
    def spec(self) -> Spec:
        return self._spec

    def protein(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        msa: MSA | None = None,
        cyclic: bool = False,
    ) -> FoldComplex:
        self._spec.protein(
            ids,
            sequence,
            modifications=modifications,
            msa=msa,
            cyclic=cyclic,
        )
        self._dirty = True
        return self

    def dna(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        cyclic: bool = False,
    ) -> FoldComplex:
        self._spec.dna(ids, sequence, modifications=modifications, cyclic=cyclic)
        self._dirty = True
        return self

    def rna(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        modifications: Iterable[Modification | tuple[int, str]] = (),
        cyclic: bool = False,
    ) -> FoldComplex:
        self._spec.rna(ids, sequence, modifications=modifications, cyclic=cyclic)
        self._dirty = True
        return self

    def ligand(
        self,
        ids: ChainIds,
        smiles_or_ccd: str | None = None,
        *,
        ccd: str | Sequence[str] | None = None,
        smiles: str | None = None,
    ) -> FoldComplex:
        if smiles_or_ccd is not None:
            if ccd is not None or smiles is not None:
                raise ValueError("Provide smiles_or_ccd or ccd/smiles, not both.")
            smiles = smiles_or_ccd
        self._spec.ligand(ids, ccd=ccd, smiles=smiles)
        self._dirty = True
        return self

    def bond(
        self,
        atom1: AtomRef | Sequence[str | int],
        atom2: AtomRef | Sequence[str | int],
    ) -> FoldComplex:
        self._spec.bond(atom1, atom2)
        self._dirty = True
        return self

    def pocket(
        self,
        binder: str,
        contacts: Iterable[TokenRef | Sequence[str | int]],
        *,
        max_distance: float = 6.0,
        force: bool = False,
    ) -> FoldComplex:
        self._spec.pocket(
            binder,
            contacts,
            max_distance=max_distance,
            force=force,
        )
        self._dirty = True
        return self

    def contact(
        self,
        token1: TokenRef | Sequence[str | int],
        token2: TokenRef | Sequence[str | int],
        *,
        max_distance: float = 6.0,
        force: bool = False,
    ) -> FoldComplex:
        self._spec.contact(
            token1,
            token2,
            max_distance=max_distance,
            force=force,
        )
        self._dirty = True
        return self

    def add_template(self, template: Template) -> FoldComplex:
        self._spec.add_template(template)
        self._dirty = True
        return self

    def request_affinity(self, binder: str) -> FoldComplex:
        self._spec.request_affinity(binder)
        return self

    def fold(self, **predict_overrides: Any) -> StructurePrediction:
        """Run a structure prediction and return the in-memory structure."""
        return self._predict_structure(**predict_overrides)

    def to_mmcif(self, **predict_overrides: Any) -> str:
        prediction = self._predict_structure(**predict_overrides)
        return to_mmcif(
            prediction.structure,
            plddts=prediction.plddt,
            boltz2=True,
        )

    def to_bcif(self, **predict_overrides: Any) -> bytes:
        mmcif_text = self.to_mmcif(**predict_overrides)
        return _mmcif_to_bcif(mmcif_text)

    def get_affinity(
        self,
        binder: str | None = None,
        *,
        use_structure_prediction: bool = False,
        crop_affinity: bool = False,
        override_method: str | None = None,
        **predict_overrides: Any,
    ) -> AffinityPrediction:
        affinity_binder = binder
        if affinity_binder is None and self._spec.affinity is not None:
            affinity_binder = self._spec.affinity.binder

        if affinity_binder is None:
            ligand_ids = _ligand_chain_ids(self._spec)
            if len(ligand_ids) != 1:
                msg = (
                    "Affinity requires exactly one ligand chain or an explicit binder."
                )
                raise ValueError(msg)
            affinity_binder = ligand_ids[0]

        if self._spec.affinity and self._spec.affinity.binder == affinity_binder:
            affinity_spec = self._spec
        else:
            affinity_spec = replace(
                self._spec,
                affinity=Affinity(binder=affinity_binder),
            )

        structure_prediction = None
        if use_structure_prediction:
            structure_prediction = self._predict_structure()

        target = self._model.pipeline.build_target(affinity_spec)
        if structure_prediction is not None:
            target = replace(target, structure=structure_prediction.structure)
        input_data = self._model.pipeline.build_input(affinity_spec, target)
        tokenized = self._model.pipeline.tokenize(input_data)
        if crop_affinity:
            tokenized = AffinityCropper().crop(
                tokenized,
                max_tokens=256,
                max_atoms=2048,
            )
        features = self._model.pipeline.featurize(
            tokenized,
            override_method=override_method,
            random=np.random.default_rng(42),
        )
        batch = _batchify(features)
        batch = _move_batch_to_device(batch, self._model.device)
        args = self._model._resolve_predict_args(  # noqa: SLF001
            predict_overrides, affinity=True
        )
        out = self._model._run_model(  # noqa: SLF001
            self._model._get_affinity_model(),  # noqa: SLF001
            batch,
            **args,
        )
        return self._affinity_from_output(out)

    def _predict_structure(self, **predict_overrides: Any) -> StructurePrediction:
        if (
            not predict_overrides
            and not self._dirty
            and self._structure_prediction is not None
        ):
            return self._structure_prediction

        spec = self._spec
        if spec.affinity is not None:
            spec = replace(spec, affinity=None)

        trace = self._model.pipeline.prepare(spec)
        trace = self._model.pipeline.featurize(
            trace,
            random=np.random.default_rng(42),
        )
        batch = _batchify(trace.features)
        batch = _move_batch_to_device(batch, self._model.device)
        args = self._model._resolve_predict_args(  # noqa: SLF001
            predict_overrides, affinity=False
        )
        out = self._model._run_model(self._model.model, batch, **args)  # noqa: SLF001

        best_idx, best_score = _select_best_prediction(out)
        coords = out["sample_atom_coords"][best_idx].detach().cpu().numpy()
        atom_pad_mask = batch["atom_pad_mask"]
        if isinstance(atom_pad_mask, torch.Tensor):
            atom_pad_mask = atom_pad_mask.detach().cpu().numpy()
        if atom_pad_mask.ndim > 1:
            atom_pad_mask = atom_pad_mask[0]
        coords = coords[atom_pad_mask.astype(bool)]

        structure = _apply_predicted_coords(trace.target.structure, coords)
        plddt = None
        if "plddt" in out:
            plddt = out["plddt"][best_idx].detach().cpu()

        prediction = StructurePrediction(
            structure=structure,
            plddt=plddt,
            confidence_score=best_score,
        )

        if not predict_overrides:
            self._structure_prediction = prediction
            self._dirty = False

        return prediction

    @staticmethod
    def _affinity_from_output(out: Mapping[str, Tensor]) -> AffinityPrediction:
        ic50 = _to_float(out["affinity_pred_value"])
        binding_probability = _to_float(out["affinity_probability_binary"])
        ic50_1 = (
            _to_float(out["affinity_pred_value1"])
            if "affinity_pred_value1" in out
            else None
        )
        binding_probability_1 = (
            _to_float(out["affinity_probability_binary1"])
            if "affinity_probability_binary1" in out
            else None
        )
        ic50_2 = (
            _to_float(out["affinity_pred_value2"])
            if "affinity_pred_value2" in out
            else None
        )
        binding_probability_2 = (
            _to_float(out["affinity_probability_binary2"])
            if "affinity_probability_binary2" in out
            else None
        )
        return AffinityPrediction(
            ic50=ic50,
            binding_probability=binding_probability,
            ic50_1=ic50_1,
            binding_probability_1=binding_probability_1,
            ic50_2=ic50_2,
            binding_probability_2=binding_probability_2,
        )


class Boltz2:
    """Simple in-memory Boltz2 wrapper with sensible defaults."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        affinity_checkpoint: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        components: Mapping[str, Mol] | None = None,
        molecules: Mapping[str, Mol] | None = None,
        auto_download: bool = True,
        use_kernels: bool = True,
        affinity_mw_correction: bool = True,
        predict_args: Mapping[str, Any] | None = None,
        affinity_predict_args: Mapping[str, Any] | None = None,
    ) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        )
        self.cache_dir = self.cache_dir.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_download = auto_download
        self.use_kernels = use_kernels
        if self.use_kernels and not _cuequivariance_available():
            self._disable_kernels(
                "cuEquivariance kernels requested but cuequivariance_torch is not installed."
            )
        self.affinity_mw_correction = affinity_mw_correction

        self.predict_args = dict(_DEFAULT_STRUCTURE_PREDICT_ARGS)
        if predict_args:
            unknown = set(predict_args) - set(self.predict_args)
            if unknown:
                raise ValueError(f"Unknown predict args: {sorted(unknown)}")
            self.predict_args.update(predict_args)

        self.affinity_predict_args = dict(_DEFAULT_AFFINITY_PREDICT_ARGS)
        if affinity_predict_args:
            unknown = set(affinity_predict_args) - set(self.affinity_predict_args)
            if unknown:
                raise ValueError(f"Unknown affinity predict args: {sorted(unknown)}")
            self.affinity_predict_args.update(affinity_predict_args)

        self.mol_dir = self.cache_dir / "mols"
        default_checkpoint = self.cache_dir / "boltz2_conf.ckpt"
        default_affinity_checkpoint = self.cache_dir / "boltz2_aff.ckpt"

        checkpoint_path = (
            Path(checkpoint) if checkpoint is not None else default_checkpoint
        )
        affinity_checkpoint_path = (
            Path(affinity_checkpoint)
            if affinity_checkpoint is not None
            else default_affinity_checkpoint
        )

        needs_mols = components is None
        if self.auto_download and (
            (needs_mols and not self.mol_dir.exists()) or not checkpoint_path.exists()
        ):
            from boltz.main import download_boltz2

            download_boltz2(self.cache_dir)

        if components is None:
            if not self.mol_dir.exists():
                msg = f"Missing CCD molecules at {self.mol_dir}."
                raise FileNotFoundError(msg)
            Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
            components = load_canonicals(str(self.mol_dir))
        if molecules is None:
            molecules = dict(components)

        if not checkpoint_path.exists():
            msg = f"Missing checkpoint at {checkpoint_path}."
            raise FileNotFoundError(msg)

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pipeline = Pipeline(version=2, components=components, molecules=molecules)

        self.model = self._load_model_checkpoint(
            checkpoint_path,
            affinity=False,
        )
        self.model.steering_args = _normalize_steering_args(self.model.steering_args)
        self.model.eval()
        self.model.to(self.device)

        self._affinity_checkpoint = affinity_checkpoint_path
        self._affinity_model: Boltz2Model | None = None

        torch.set_float32_matmul_precision("highest")

    def _disable_kernels(
        self,
        reason: str,
        *,
        model: Boltz2Model | None = None,
    ) -> None:
        should_warn = self.use_kernels
        self.use_kernels = False
        changed_model = False
        for candidate in (
            model,
            getattr(self, "model", None),
            getattr(self, "_affinity_model", None),
        ):
            if candidate is None:
                continue
            if hasattr(candidate, "use_kernels"):
                if bool(getattr(candidate, "use_kernels", False)):
                    changed_model = True
                candidate.use_kernels = False

        if not should_warn and not changed_model:
            return

        warnings.warn(
            f"{reason} Falling back to use_kernels=False.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _model_load_kwargs(self, *, affinity: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "map_location": "cpu",
            "diffusion_process_args": dict(_DEFAULT_DIFFUSION_PROCESS_ARGS),
            "pairformer_args": dict(_DEFAULT_PAIRFORMER_ARGS),
            "msa_args": dict(_DEFAULT_MSA_ARGS),
            "steering_args": dict(
                _DEFAULT_AFFINITY_STEERING_ARGS if affinity else _DEFAULT_STEERING_ARGS
            ),
            "ema": False,
        }
        if affinity:
            kwargs["affinity_mw_correction"] = self.affinity_mw_correction
        else:
            kwargs["use_kernels"] = self.use_kernels
        return kwargs

    def _load_model_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        affinity: bool,
    ) -> Boltz2Model:
        kwargs = self._model_load_kwargs(affinity=affinity)
        safe_globals = None
        try:
            from omegaconf import DictConfig, ListConfig
            from omegaconf.base import ContainerMetadata
        except ImportError:
            safe_globals = None
        else:
            safe_globals = [DictConfig, ListConfig, ContainerMetadata]

        def _load(weights_only: bool | None = None) -> Boltz2Model:
            if weights_only is None:
                return Boltz2Model.load_from_checkpoint(checkpoint_path, **kwargs)
            return Boltz2Model.load_from_checkpoint(
                checkpoint_path,
                weights_only=weights_only,
                **kwargs,
            )

        if safe_globals and hasattr(torch.serialization, "safe_globals"):
            try:
                with torch.serialization.safe_globals(safe_globals):
                    return _load()
            except pickle.UnpicklingError:
                return _load(weights_only=False)
        if safe_globals and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(safe_globals)
        try:
            return _load()
        except pickle.UnpicklingError:
            return _load(weights_only=False)

    def fold_complex(self, name: str = "complex") -> FoldComplex:
        return FoldComplex(self, name=name)

    def _resolve_predict_args(
        self,
        overrides: Mapping[str, Any],
        *,
        affinity: bool,
    ) -> dict[str, Any]:
        base = self.affinity_predict_args if affinity else self.predict_args
        args = dict(base)
        for key, value in overrides.items():
            if key not in args:
                raise ValueError(f"Unknown predict arg: {key}")
            args[key] = value
        return args

    def _run_model(
        self,
        model: Boltz2Model,
        batch: Mapping[str, Any],
        *,
        recycling_steps: int,
        sampling_steps: int,
        diffusion_samples: int,
        max_parallel_samples: int | None,
    ) -> dict[str, Tensor]:
        predict_kwargs = dict(
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            max_parallel_samples=max_parallel_samples,
            run_confidence_sequentially=True,
        )
        with torch.inference_mode():
            try:
                return model(batch, **predict_kwargs)
            except ModuleNotFoundError as exc:
                model_uses_kernels = bool(getattr(model, "use_kernels", False))
                if (
                    self.use_kernels or model_uses_kernels
                ) and _is_cuequivariance_import_error(exc):
                    self._disable_kernels(
                        f"Kernel dependency import failed ({exc}).",
                        model=model,
                    )
                    return model(batch, **predict_kwargs)
                raise

    def _get_affinity_model(self) -> Boltz2Model:
        if self._affinity_model is not None:
            return self._affinity_model

        if self.auto_download and not self._affinity_checkpoint.exists():
            from boltz.main import download_boltz2

            download_boltz2(self.cache_dir)

        if not self._affinity_checkpoint.exists():
            msg = f"Missing affinity checkpoint at {self._affinity_checkpoint}."
            raise FileNotFoundError(msg)

        affinity_model = self._load_model_checkpoint(
            self._affinity_checkpoint,
            affinity=True,
        )
        affinity_model.steering_args = _normalize_steering_args(
            affinity_model.steering_args
        )
        if not self.use_kernels and hasattr(affinity_model, "use_kernels"):
            affinity_model.use_kernels = False
        affinity_model.eval()
        affinity_model.to(self.device)
        self._affinity_model = affinity_model
        return affinity_model

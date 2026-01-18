from __future__ import annotations

"""In-memory API for BoltzGen design parsing and feature preparation."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from boltzgen.data import const
from boltzgen.data.data import Input, Target, Tokenized
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals, load_molecules
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.template.features import load_dummy_templates
from boltzgen.data.tokenize.tokenizer import Tokenizer
from rdkit.Chem.rdchem import Mol

from refua.api import (
    AtomRef,
    Bond,
    ChainIds,
    as_atom_ref,
    normalize_chain_ids,
)

_DEFAULT_MOLDIR_ARTIFACT = "huggingface:boltzgen/inference-data:mols.zip"


def _default_moldir() -> str:
    return os.environ.get("BOLTZGEN_MOLDIR", _DEFAULT_MOLDIR_ARTIFACT)


def _resolve_moldir(
    moldir: str | Path,
    *,
    auto_download: bool,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    force_download: bool = False,
) -> Path:
    if isinstance(moldir, Path):
        resolved = moldir.expanduser()
    else:
        moldir_str = str(moldir)
        if moldir_str.startswith("huggingface:"):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is required to resolve default moldir."
                ) from exc
            try:
                _, repo_id, filename = moldir_str.split(":")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid artifact: {moldir_str}. Expected format: huggingface:<repo_id>:<filename>"
                ) from exc
            try:
                resolved_path = hf_hub_download(
                    repo_id,
                    filename,
                    repo_type="dataset",
                    library_name="boltzgen",
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    token=token,
                    force_download=force_download,
                    local_files_only=not auto_download,
                )
            except Exception as exc:
                raise FileNotFoundError(
                    "Default moldir not found. Set BOLTZGEN_MOLDIR or pass mol_dir explicitly."
                ) from exc
            resolved = Path(resolved_path)
        else:
            resolved = Path(moldir_str).expanduser()

    if not resolved.exists():
        raise FileNotFoundError(f"moldir not found: {resolved}")
    return resolved


def _format_ids(ids: ChainIds) -> str | list[str]:
    """Normalize chain ids into the schema format."""
    normalized = normalize_chain_ids(ids)
    if len(normalized) == 1:
        return normalized[0]
    return list(normalized)


@dataclass(frozen=True, slots=True)
class Protein:
    """Protein chain specification."""

    ids: ChainIds
    sequence: str
    binding_types: str | Mapping[str, Any] | None = None
    secondary_structure: str | Mapping[str, Any] | None = None
    cyclic: bool = False
    msa: str | int | None = None

    def to_schema(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "id": _format_ids(self.ids),
            "sequence": self.sequence,
        }
        if self.binding_types is not None:
            payload["binding_types"] = self.binding_types
        if self.secondary_structure is not None:
            payload["secondary_structure"] = self.secondary_structure
        if self.cyclic:
            payload["cyclic"] = True
        if self.msa is not None:
            payload["msa"] = self.msa
        return {"protein": payload}


@dataclass(frozen=True, slots=True)
class DNA:
    """DNA chain specification."""

    ids: ChainIds
    sequence: str
    binding_types: str | Mapping[str, Any] | None = None
    cyclic: bool = False

    def to_schema(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "id": _format_ids(self.ids),
            "sequence": self.sequence,
        }
        if self.binding_types is not None:
            payload["binding_types"] = self.binding_types
        if self.cyclic:
            payload["cyclic"] = True
        return {"dna": payload}


@dataclass(frozen=True, slots=True)
class RNA:
    """RNA chain specification."""

    ids: ChainIds
    sequence: str
    binding_types: str | Mapping[str, Any] | None = None
    cyclic: bool = False

    def to_schema(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "id": _format_ids(self.ids),
            "sequence": self.sequence,
        }
        if self.binding_types is not None:
            payload["binding_types"] = self.binding_types
        if self.cyclic:
            payload["cyclic"] = True
        return {"rna": payload}


@dataclass(frozen=True, slots=True)
class Ligand:
    """Ligand specification."""

    ids: ChainIds
    ccd: str | Sequence[str] | None = None
    smiles: str | None = None
    binding_types: str | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.ccd is None) == (self.smiles is None):
            msg = "Ligand requires exactly one of ccd or smiles."
            raise ValueError(msg)

    def to_schema(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {"id": _format_ids(self.ids)}
        if self.ccd is not None:
            if isinstance(self.ccd, str):
                payload["ccd"] = self.ccd
            else:
                payload["ccd"] = list(self.ccd)
        else:
            payload["smiles"] = self.smiles
        if self.binding_types is not None:
            payload["binding_types"] = self.binding_types
        return {"ligand": payload}


@dataclass(frozen=True, slots=True)
class File:
    """File-backed structure specification."""

    path: str | Path
    include: str | Sequence[Mapping[str, Any]] | None = None
    exclude: Sequence[Mapping[str, Any]] | None = None
    include_proximity: Sequence[Mapping[str, Any]] | None = None
    binding_types: Sequence[Mapping[str, Any]] | None = None
    structure_groups: Sequence[Mapping[str, Any]] | None = None
    design: Sequence[Mapping[str, Any]] | None = None
    not_design: Sequence[Mapping[str, Any]] | None = None
    secondary_structure: Sequence[Mapping[str, Any]] | None = None
    design_insertions: Sequence[Mapping[str, Any]] | None = None
    fuse: str | None = None
    msa: str | int | None = None
    use_assembly: bool | None = None
    reset_res_index: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_schema(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {"path": str(self.path)}
        if self.include is not None:
            payload["include"] = self.include
        if self.exclude is not None:
            payload["exclude"] = list(self.exclude)
        if self.include_proximity is not None:
            payload["include_proximity"] = list(self.include_proximity)
        if self.binding_types is not None:
            payload["binding_types"] = list(self.binding_types)
        if self.structure_groups is not None:
            payload["structure_groups"] = list(self.structure_groups)
        if self.design is not None:
            payload["design"] = list(self.design)
        if self.not_design is not None:
            payload["not_design"] = list(self.not_design)
        if self.secondary_structure is not None:
            payload["secondary_structure"] = list(self.secondary_structure)
        if self.design_insertions is not None:
            payload["design_insertions"] = list(self.design_insertions)
        if self.fuse is not None:
            payload["fuse"] = self.fuse
        if self.msa is not None:
            payload["msa"] = self.msa
        if self.use_assembly is not None:
            payload["use_assembly"] = self.use_assembly
        if self.reset_res_index is not None:
            payload["reset_res_index"] = self.reset_res_index
        payload.update(self.extra)
        return {"file": payload}


@dataclass(frozen=True, slots=True)
class RawEntity:
    """Raw entity schema passthrough."""

    payload: Mapping[str, Any]

    def to_schema(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class TotalLength:
    """Total sequence length constraint."""

    min: int
    max: int

    def to_schema(self) -> dict[str, dict[str, int]]:
        return {"total_len": {"min": self.min, "max": self.max}}


@dataclass
class Spec:
    """In-memory BoltzGen design specification."""

    name: str
    entities: list[Protein | DNA | RNA | Ligand | File | RawEntity] = field(
        default_factory=list
    )
    constraints: list[Bond | TotalLength] = field(default_factory=list)
    base_dir: Path | None = None

    def add(self, *entities: Protein | DNA | RNA | Ligand | File | RawEntity) -> Spec:
        self.entities.extend(entities)
        return self

    def protein(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        binding_types: str | Mapping[str, Any] | None = None,
        secondary_structure: str | Mapping[str, Any] | None = None,
        cyclic: bool = False,
        msa: str | int | None = None,
    ) -> Spec:
        """Add a protein entity to the spec."""
        self.entities.append(
            Protein(
                ids=ids,
                sequence=sequence,
                binding_types=binding_types,
                secondary_structure=secondary_structure,
                cyclic=cyclic,
                msa=msa,
            )
        )
        return self

    def dna(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        binding_types: str | Mapping[str, Any] | None = None,
        cyclic: bool = False,
    ) -> Spec:
        """Add a DNA entity to the spec."""
        self.entities.append(
            DNA(ids=ids, sequence=sequence, binding_types=binding_types, cyclic=cyclic)
        )
        return self

    def rna(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        binding_types: str | Mapping[str, Any] | None = None,
        cyclic: bool = False,
    ) -> Spec:
        """Add an RNA entity to the spec."""
        self.entities.append(
            RNA(ids=ids, sequence=sequence, binding_types=binding_types, cyclic=cyclic)
        )
        return self

    def ligand(
        self,
        ids: ChainIds,
        *,
        ccd: str | Sequence[str] | None = None,
        smiles: str | None = None,
        binding_types: str | Mapping[str, Any] | None = None,
    ) -> Spec:
        """Add a ligand entity to the spec."""
        self.entities.append(
            Ligand(ids=ids, ccd=ccd, smiles=smiles, binding_types=binding_types)
        )
        return self

    def file(
        self,
        path: str | Path,
        *,
        include: str | Sequence[Mapping[str, Any]] | None = None,
        exclude: Sequence[Mapping[str, Any]] | None = None,
        include_proximity: Sequence[Mapping[str, Any]] | None = None,
        binding_types: Sequence[Mapping[str, Any]] | None = None,
        structure_groups: Sequence[Mapping[str, Any]] | None = None,
        design: Sequence[Mapping[str, Any]] | None = None,
        not_design: Sequence[Mapping[str, Any]] | None = None,
        secondary_structure: Sequence[Mapping[str, Any]] | None = None,
        design_insertions: Sequence[Mapping[str, Any]] | None = None,
        fuse: str | None = None,
        msa: str | int | None = None,
        use_assembly: bool | None = None,
        reset_res_index: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Spec:
        """Add a file-backed entity to the spec."""
        self.entities.append(
            File(
                path=path,
                include=include,
                exclude=exclude,
                include_proximity=include_proximity,
                binding_types=binding_types,
                structure_groups=structure_groups,
                design=design,
                not_design=not_design,
                secondary_structure=secondary_structure,
                design_insertions=design_insertions,
                fuse=fuse,
                msa=msa,
                use_assembly=use_assembly,
                reset_res_index=reset_res_index,
                extra=extra or {},
            )
        )
        return self

    def raw_entity(self, payload: Mapping[str, Any]) -> Spec:
        """Append a raw entity mapping without validation."""
        self.entities.append(RawEntity(payload=payload))
        return self

    def bond(
        self,
        atom1: AtomRef | Sequence[str | int],
        atom2: AtomRef | Sequence[str | int],
    ) -> Spec:
        """Add a covalent bond constraint."""
        self.constraints.append(Bond(as_atom_ref(atom1), as_atom_ref(atom2)))
        return self

    def total_length(self, minimum: int, maximum: int) -> Spec:
        """Add a total length constraint across entities."""
        self.constraints.append(TotalLength(min=minimum, max=maximum))
        return self

    def to_schema(self) -> dict[str, Any]:
        """Return a BoltzGen-compatible schema dictionary."""
        if not self.entities:
            raise ValueError("Spec requires at least one entity.")

        schema: dict[str, Any] = {
            "entities": [entity.to_schema() for entity in self.entities]
        }
        if self.constraints:
            schema["constraints"] = [constraint.to_schema() for constraint in self.constraints]
        return schema


@dataclass(frozen=True)
class Trace:
    """Captured pipeline state."""

    spec: Spec
    target: Target
    tokenized: Tokenized
    input: Input
    chain_design_mask: np.ndarray | None = None
    ss_type: np.ndarray | None = None
    features: Mapping[str, Any] | None = None


class Pipeline:
    """In-memory pipeline for parsing and feature preparation."""

    def __init__(
        self,
        *,
        mol_dir: str | Path | None = None,
        molecules: Mapping[str, Mol] | None = None,
        canonicals: Mapping[str, Mol] | None = None,
        tokenizer: Tokenizer | None = None,
        featurizer: Featurizer | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.mol_dir = Path(mol_dir) if mol_dir is not None else None
        self.molecules = dict(molecules or {})
        if canonicals is not None:
            self.canonicals = dict(canonicals)
        elif self.mol_dir is not None:
            self.canonicals = load_canonicals(str(self.mol_dir))
        else:
            self.canonicals = {}

        parser_dir = self.mol_dir or Path()
        self.parser = YamlDesignParser(mol_dir=parser_dir)
        self.tokenizer = tokenizer or Tokenizer()
        self.featurizer = featurizer or Featurizer()
        self.rng = rng or np.random.default_rng(42)

    def build_target(
        self,
        spec: Spec,
        *,
        mols: Mapping[str, Mol] | None = None,
        mol_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
    ) -> Target:
        """Parse a Spec into a Target using the BoltzGen schema parser."""
        mol_dir_value = Path(mol_dir) if mol_dir is not None else self.mol_dir
        mol_dir_value = mol_dir_value or Path()
        mols_map = dict(self.molecules)
        if mols is not None:
            mols_map.update(mols)

        base_path = Path(base_dir) if base_dir is not None else spec.base_dir
        return self.parser.parse_boltzgen_schema(
            spec.name,
            spec.to_schema(),
            mols_map,
            mol_dir_value,
            base_file_path=base_path,
        )

    def tokenize(self, target: Target) -> tuple[Tokenized, np.ndarray | None, np.ndarray | None]:
        """Tokenize a Target and derive design/secondary-structure masks."""
        tokenized = self.tokenizer.tokenize(target.structure)
        if target.design_info is None:
            return tokenized, None, None

        token_to_res = tokenized.token_to_res
        design_info = target.design_info
        tokenized.tokens["design_mask"] = design_info.res_design_mask[token_to_res]
        tokenized.tokens["binding_type"] = design_info.res_binding_type[token_to_res]
        tokenized.tokens["structure_group"] = design_info.res_structure_groups[token_to_res]

        chain_design_mask = tokenized.tokens["design_mask"].astype(bool)
        asym_id = tokenized.tokens["asym_id"]
        while True:
            design_chains = np.unique(asym_id[chain_design_mask])
            chain_propagated = np.isin(asym_id, design_chains)
            for i, j, _ in tokenized.bonds:
                if chain_propagated[i] or chain_propagated[j]:
                    chain_propagated[i] = True
                    chain_propagated[j] = True
            if np.equal(chain_propagated, chain_design_mask).all():
                break
            chain_design_mask = chain_propagated.astype(bool)

        ss_type = design_info.res_ss_types[token_to_res]
        return tokenized, chain_design_mask, ss_type

    def build_input(self, target: Target, tokenized: Tokenized) -> Input:
        """Build the featurizer Input wrapper from tokenized data."""
        return Input(
            tokens=tokenized.tokens,
            bonds=tokenized.bonds,
            token_to_res=tokenized.token_to_res,
            structure=target.structure,
            msa={},
            templates=target.templates,
            record=target.record,
        )

    def prepare(
        self,
        spec: Spec,
        *,
        mols: Mapping[str, Mol] | None = None,
        mol_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
    ) -> Trace:
        """Parse a Spec to Target/Tokenized/Input and return a Trace."""
        target = self.build_target(spec, mols=mols, mol_dir=mol_dir, base_dir=base_dir)
        tokenized, chain_design_mask, ss_type = self.tokenize(target)
        input_data = self.build_input(target, tokenized)
        return Trace(
            spec=spec,
            target=target,
            tokenized=tokenized,
            input=input_data,
            chain_design_mask=chain_design_mask,
            ss_type=ss_type,
        )

    def featurize(
        self,
        data: Trace | Input,
        *,
        molecules: Mapping[str, Mol] | None = None,
        mol_dir: str | Path | None = None,
        random: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> Trace | Mapping[str, Any]:
        """Featurize a Trace or Input to model-ready tensors."""
        if isinstance(data, Trace):
            features = self._featurize(
                data.input,
                data.tokenized,
                data.target,
                chain_design_mask=data.chain_design_mask,
                ss_type=data.ss_type,
                molecules=molecules,
                mol_dir=mol_dir,
                random=random,
                **kwargs,
            )
            return replace(data, features=features)

        return self._featurize(
            data,
            None,
            None,
            chain_design_mask=None,
            ss_type=None,
            molecules=molecules,
            mol_dir=mol_dir,
            random=random,
            **kwargs,
        )

    def _resolve_molecules(
        self,
        tokenized: Tokenized,
        target: Target | None,
        *,
        molecules: Mapping[str, Mol] | None,
        mol_dir: str | Path | None,
    ) -> dict[str, Mol]:
        mol_dir_value = Path(mol_dir) if mol_dir is not None else self.mol_dir
        molecules_map: dict[str, Mol] = {}
        molecules_map.update(self.canonicals)
        molecules_map.update(self.molecules)
        if molecules is not None:
            molecules_map.update(molecules)
        if target is not None and target.extra_mols:
            molecules_map.update(target.extra_mols)

        missing = set(tokenized.tokens["res_name"].tolist()) - set(molecules_map.keys())
        if missing:
            if mol_dir_value is None:
                raise ValueError(f"Missing molecules for residues: {sorted(missing)}.")
            molecules_map.update(load_molecules(str(mol_dir_value), sorted(missing)))
            missing = set(tokenized.tokens["res_name"].tolist()) - set(molecules_map.keys())
            if missing:
                raise ValueError(f"Missing molecules for residues: {sorted(missing)}.")
        return molecules_map

    def _featurize(
        self,
        input_data: Input,
        tokenized: Tokenized | None,
        target: Target | None,
        *,
        chain_design_mask: np.ndarray | None,
        ss_type: np.ndarray | None,
        molecules: Mapping[str, Mol] | None,
        mol_dir: str | Path | None,
        random: np.random.Generator | None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        molecules_map = self._resolve_molecules(
            tokenized or Tokenized(
                tokens=input_data.tokens,
                bonds=input_data.bonds,
                structure=input_data.structure,
                token_to_res=input_data.token_to_res,
            ),
            target,
            molecules=molecules,
            mol_dir=mol_dir,
        )

        call_kwargs = {
            "molecules": molecules_map,
            "random": random or self.rng,
            "training": False,
            "max_seqs": 1,
            "backbone_only": False,
            "atom14": False,
            "atom37": False,
            "design": True,
            "override_method": "X-RAY DIFFRACTION",
            "compute_affinity": False,
            "disulfide_prob": 1.0,
            "disulfide_on": False,
        }
        call_kwargs.update(kwargs)
        features = self.featurizer.process(input_data, **call_kwargs)

        if ss_type is not None:
            features["ss_type"] = torch.from_numpy(ss_type).to(features["ss_type"])
            mask = ss_type != const.ss_type_ids["UNSPECIFIED"]
            features["design_ss_mask"][mask] = 1

        if chain_design_mask is not None:
            features["chain_design_mask"] = torch.from_numpy(chain_design_mask)

        if "template_restype" not in features:
            template_features = load_dummy_templates(
                tdim=1, num_tokens=len(features["res_type"])
            )
            features.update(template_features)

        if target is not None and target.record is not None:
            features["id"] = target.record.id

        return features


class DesignSpec:
    """Fluent BoltzGen design builder."""

    def __init__(
        self,
        model: BoltzGen,
        name: str = "design",
        *,
        base_dir: str | Path | None = None,
    ) -> None:
        self._model = model
        base_path = Path(base_dir).expanduser().resolve() if base_dir is not None else None
        self._spec = Spec(name=name, base_dir=base_path)

    @property
    def spec(self) -> Spec:
        return self._spec

    def protein(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        binding_types: str | Mapping[str, Any] | None = None,
        secondary_structure: str | Mapping[str, Any] | None = None,
        cyclic: bool = False,
        msa: str | int | None = None,
    ) -> DesignSpec:
        self._spec.protein(
            ids,
            sequence,
            binding_types=binding_types,
            secondary_structure=secondary_structure,
            cyclic=cyclic,
            msa=msa,
        )
        return self

    def dna(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        binding_types: str | Mapping[str, Any] | None = None,
        cyclic: bool = False,
    ) -> DesignSpec:
        self._spec.dna(
            ids,
            sequence,
            binding_types=binding_types,
            cyclic=cyclic,
        )
        return self

    def rna(
        self,
        ids: ChainIds,
        sequence: str,
        *,
        binding_types: str | Mapping[str, Any] | None = None,
        cyclic: bool = False,
    ) -> DesignSpec:
        self._spec.rna(
            ids,
            sequence,
            binding_types=binding_types,
            cyclic=cyclic,
        )
        return self

    def ligand(
        self,
        ids: ChainIds,
        smiles_or_ccd: str | None = None,
        *,
        ccd: str | Sequence[str] | None = None,
        smiles: str | None = None,
        binding_types: str | Mapping[str, Any] | None = None,
    ) -> DesignSpec:
        if smiles_or_ccd is not None:
            if ccd is not None or smiles is not None:
                raise ValueError("Provide smiles_or_ccd or ccd/smiles, not both.")
            smiles = smiles_or_ccd
        self._spec.ligand(ids, ccd=ccd, smiles=smiles, binding_types=binding_types)
        return self

    def file(
        self,
        path: str | Path,
        *,
        include: str | Sequence[Mapping[str, Any]] | None = None,
        exclude: Sequence[Mapping[str, Any]] | None = None,
        include_proximity: Sequence[Mapping[str, Any]] | None = None,
        binding_types: Sequence[Mapping[str, Any]] | None = None,
        structure_groups: Sequence[Mapping[str, Any]] | None = None,
        design: Sequence[Mapping[str, Any]] | None = None,
        not_design: Sequence[Mapping[str, Any]] | None = None,
        secondary_structure: Sequence[Mapping[str, Any]] | None = None,
        design_insertions: Sequence[Mapping[str, Any]] | None = None,
        fuse: str | None = None,
        msa: str | int | None = None,
        use_assembly: bool | None = None,
        reset_res_index: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> DesignSpec:
        self._spec.file(
            path,
            include=include,
            exclude=exclude,
            include_proximity=include_proximity,
            binding_types=binding_types,
            structure_groups=structure_groups,
            design=design,
            not_design=not_design,
            secondary_structure=secondary_structure,
            design_insertions=design_insertions,
            fuse=fuse,
            msa=msa,
            use_assembly=use_assembly,
            reset_res_index=reset_res_index,
            extra=extra,
        )
        return self

    def raw_entity(self, payload: Mapping[str, Any]) -> DesignSpec:
        self._spec.raw_entity(payload)
        return self

    def bond(
        self,
        atom1: AtomRef | Sequence[str | int],
        atom2: AtomRef | Sequence[str | int],
    ) -> DesignSpec:
        self._spec.bond(atom1, atom2)
        return self

    def total_length(self, minimum: int, maximum: int) -> DesignSpec:
        self._spec.total_length(minimum, maximum)
        return self

    def prepare(
        self,
        *,
        mols: Mapping[str, Mol] | None = None,
        mol_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
    ) -> Trace:
        return self._model.pipeline.prepare(
            self._spec,
            mols=mols,
            mol_dir=mol_dir,
            base_dir=base_dir,
        )

    def to_features(
        self,
        *,
        mols: Mapping[str, Mol] | None = None,
        mol_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
        molecules: Mapping[str, Mol] | None = None,
        random: np.random.Generator | None = None,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> Trace | Mapping[str, Any]:
        """Prepare model-ready input features from the design spec."""
        trace = self.prepare(mols=mols, mol_dir=mol_dir, base_dir=base_dir)
        trace = self._model.pipeline.featurize(
            trace,
            molecules=molecules,
            mol_dir=mol_dir,
            random=random,
            **kwargs,
        )
        if return_trace:
            return trace
        if trace.features is None:
            raise RuntimeError("Feature preparation returned no features.")
        return trace.features

    def featurize(
        self,
        *,
        mols: Mapping[str, Mol] | None = None,
        mol_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
        molecules: Mapping[str, Mol] | None = None,
        random: np.random.Generator | None = None,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> Trace | Mapping[str, Any]:
        """Deprecated alias for to_features()."""
        return self.to_features(
            mols=mols,
            mol_dir=mol_dir,
            base_dir=base_dir,
            molecules=molecules,
            random=random,
            return_trace=return_trace,
            **kwargs,
        )


class BoltzGen:
    """Simple wrapper for building BoltzGen design specs."""

    def __init__(
        self,
        *,
        mol_dir: str | Path | None = None,
        molecules: Mapping[str, Mol] | None = None,
        canonicals: Mapping[str, Mol] | None = None,
        tokenizer: Tokenizer | None = None,
        featurizer: Featurizer | None = None,
        rng: np.random.Generator | None = None,
        auto_download: bool = True,
        cache_dir: str | Path | None = None,
        token: str | None = None,
        force_download: bool = False,
    ) -> None:
        resolved_mol_dir: str | Path | None = mol_dir
        if resolved_mol_dir is None:
            resolved_mol_dir = _default_moldir()
        resolved_mol_dir = _resolve_moldir(
            resolved_mol_dir,
            auto_download=auto_download,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
        self.pipeline = Pipeline(
            mol_dir=resolved_mol_dir,
            molecules=molecules,
            canonicals=canonicals,
            tokenizer=tokenizer,
            featurizer=featurizer,
            rng=rng,
        )

    def design(
        self,
        name: str = "design",
        *,
        base_dir: str | Path | None = None,
    ) -> DesignSpec:
        return DesignSpec(self, name=name, base_dir=base_dir)

from __future__ import annotations

"""Unified API bridging Boltz2 folding and BoltzGen design inputs."""

import gc
import string
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import TracebackType
from typing import Any

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from refua.api import AtomRef, ChainIds, TokenRef, normalize_chain_ids
from refua.boltz.api import (
    AffinityPrediction,
    Boltz2,
    Modification,
    StructurePrediction,
    bcif_bytes_from_structure,
)
from refua.boltzgen.api import BoltzGen, Trace as BoltzGenTrace
from refua.chem import MolProperties, SmallMolecule

_BINDING_TYPE_BINDING = 1
_BINDING_TYPE_NOT_BINDING = 2
_CANONICAL_TOKEN_LETTERS = "ARNDCQEGHILKMFPSTWYV"

_MOLECULE_TYPE_LABEL = {
    0: "protein",
    1: "dna",
    2: "rna",
    3: "ligand",
}

_RESNAME_TO_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
    "DA": "A",
    "DG": "G",
    "DC": "C",
    "DT": "T",
    "DN": "N",
}

_DEFAULT_HEAVY_CDR_LENGTHS = (12, 10, 14)
_DEFAULT_LIGHT_CDR_LENGTHS = (10, 9, 9)
_DEFAULT_ANTIBODY_HEAVY_FRAMEWORK = (
    "QVQLVESGGGLVQPGGSLRLSCAAS{h1}WYRQAPGKEREWV{h2}"
    "ISSGGSTYYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC{h3}WGQGTLVTVSS"
)
_DEFAULT_ANTIBODY_LIGHT_FRAMEWORK = (
    "DIQMTQSPSSLSASVGDRVTITCRAS{l1}WYQQKPGKAPKLLIY{l2}"
    "ASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYC{l3}FGGGTKVEIK"
)


def antibody_framework_specs(
    *,
    heavy_cdr_lengths: tuple[int, int, int] = _DEFAULT_HEAVY_CDR_LENGTHS,
    light_cdr_lengths: tuple[int, int, int] = _DEFAULT_LIGHT_CDR_LENGTHS,
) -> tuple[str, str]:
    """Build default heavy/light antibody specs with designable CDRs.

    Parameters
    ----------
    heavy_cdr_lengths : tuple[int, int, int], optional
        CDR-H1/H2/H3 design lengths. Defaults to (12, 10, 14).
    light_cdr_lengths : tuple[int, int, int], optional
        CDR-L1/L2/L3 design lengths. Defaults to (10, 9, 9).

    Returns
    -------
    tuple[str, str]
        Heavy and light sequence-spec strings for ``Binder``.
    """
    h1, h2, h3 = _coerce_cdr_lengths(heavy_cdr_lengths, label="heavy")
    l1, l2, l3 = _coerce_cdr_lengths(light_cdr_lengths, label="light")
    heavy_spec = _DEFAULT_ANTIBODY_HEAVY_FRAMEWORK.format(h1=h1, h2=h2, h3=h3)
    light_spec = _DEFAULT_ANTIBODY_LIGHT_FRAMEWORK.format(l1=l1, l2=l2, l3=l3)
    return heavy_spec, light_spec


@dataclass(frozen=True, slots=True)
class Protein:
    """Minimal protein chain spec shared across Boltz2/BoltzGen."""

    sequence: str
    ids: ChainIds | None = None
    modifications: Sequence[Modification | tuple[int, str]] = ()
    msa: object | None = None
    binding_types: str | Mapping[str, Any] | None = None
    secondary_structure: str | Mapping[str, Any] | None = None
    cyclic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "modifications", tuple(self.modifications))


@dataclass(frozen=True, slots=True)
class DNA:
    """DNA chain spec for Boltz2 folding."""

    sequence: str
    ids: ChainIds | None = None
    modifications: Sequence[Modification | tuple[int, str]] = ()
    cyclic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "modifications", tuple(self.modifications))


@dataclass(frozen=True, slots=True)
class RNA:
    """RNA chain spec for Boltz2 folding."""

    sequence: str
    ids: ChainIds | None = None
    modifications: Sequence[Modification | tuple[int, str]] = ()
    cyclic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "modifications", tuple(self.modifications))


@dataclass(frozen=True, slots=True)
class Binder:
    """Binder placeholder representing a designed molecule.

    Use Binder(...) with a sequence spec (e.g., ``"10C6C3"``), length, or a
    template-style spec plus ``template_values`` to render dynamic placeholders.
    """

    spec: str | int | None = None
    length: int | None = None
    template_values: Mapping[str, Any] | None = None
    ids: ChainIds | None = None
    binding_types: str | Mapping[str, Any] | None = None
    secondary_structure: str | Mapping[str, Any] | None = None
    cyclic: bool = False

    def __post_init__(self) -> None:
        if self.spec is not None and self.length is not None:
            raise ValueError("Binder accepts either spec or length, not both.")
        if self.spec is None and self.length is None:
            raise ValueError("Binder requires spec or length.")
        if self.template_values is not None:
            if not isinstance(self.spec, str):
                raise ValueError("Binder template_values requires a string spec.")
            object.__setattr__(self, "template_values", dict(self.template_values))
        if isinstance(self.spec, int):
            object.__setattr__(self, "length", self.spec)
            object.__setattr__(self, "spec", None)

    def sequence_spec(self) -> str:
        """Return the BoltzGen sequence spec string."""
        if self.length is not None:
            if self.length < 1:
                raise ValueError("Binder length must be >= 1.")
            return str(self.length)
        if not isinstance(self.spec, str) or not self.spec:
            raise ValueError("Binder spec must be a non-empty string or int length.")
        if self.template_values is None:
            return self.spec
        try:
            rendered_spec = self.spec.format(**self.template_values)
        except KeyError as exc:
            missing = str(exc.args[0])
            raise ValueError(
                f"Binder spec template missing value for placeholder '{missing}'.",
            ) from exc
        except (ValueError, IndexError) as exc:
            raise ValueError("Binder spec template formatting failed.") from exc
        if not rendered_spec:
            raise ValueError("Binder spec template rendered an empty string.")
        return rendered_spec

    @property
    def sequence(self) -> str:
        """Return the design sequence spec string."""
        return self.sequence_spec()


@dataclass(frozen=True, slots=True)
class DesignFile:
    """File-backed design entity for BoltzGen inputs."""

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


@dataclass(frozen=True, slots=True)
class BondConstraint:
    atom1: AtomRef | Sequence[str | int]
    atom2: AtomRef | Sequence[str | int]


@dataclass(frozen=True, slots=True)
class PocketConstraint:
    binder: str | object
    contacts: tuple[TokenRef | Sequence[str | int], ...]
    max_distance: float
    force: bool


@dataclass(frozen=True, slots=True)
class ContactConstraint:
    token1: TokenRef | Sequence[str | int]
    token2: TokenRef | Sequence[str | int]
    max_distance: float
    force: bool


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Unified fold output with optional Boltz2 structure or BoltzGen design trace."""

    backend: str
    structure: StructurePrediction | None = None
    design: BoltzGenTrace | None = None
    affinity: AffinityPrediction | None = None
    chain_ids: tuple[tuple[str, ...], ...] = ()
    binder_sequences: Mapping[str, str] = field(default_factory=dict)
    ligand_rdkit: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    ligand_admet: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def features(self) -> Mapping[str, Any] | None:
        if self.design is None:
            return None
        return self.design.features

    @property
    def binder_specs(self) -> Mapping[str, str]:
        """Return binder sequence-spec strings keyed by chain id.

        Notes
        -----
        These values are design specs (for example ``"12C8"`` or framework+CDR
        templates), not finalized generated antibody sequences.
        """
        return self.binder_sequences

    def chain_design_summary(self) -> tuple[Mapping[str, Any], ...]:
        """Summarize per-chain design masks and sequences from BoltzGen trace data."""
        if self.design is None:
            return ()

        tokenized = getattr(self.design, "tokenized", None)
        target = getattr(self.design, "target", None)
        structure = getattr(target, "structure", None)
        tokens = getattr(tokenized, "tokens", None)
        chains = getattr(structure, "chains", None)

        if tokens is None or chains is None:
            return ()

        token_names = getattr(getattr(tokens, "dtype", None), "names", ()) or ()
        chain_names = getattr(getattr(chains, "dtype", None), "names", ()) or ()
        token_fields: set[str] = {str(name) for name in token_names}
        chain_fields: set[str] = {str(name) for name in chain_names}
        if not {"asym_id", "res_type"}.issubset(token_fields):
            return ()
        if not {"asym_id", "name"}.issubset(chain_fields):
            return ()

        chain_by_asym = {int(chain["asym_id"]): str(chain["name"]) for chain in chains}
        summary: dict[str, dict[str, Any]] = {}

        for token in tokens:
            asym_id = int(token["asym_id"])
            chain_id = chain_by_asym.get(asym_id, f"asym{asym_id}")
            if chain_id not in summary:
                summary[chain_id] = {
                    "chain_id": chain_id,
                    "molecule_type": "unknown",
                    "residue_count": 0,
                    "design_residue_count": 0,
                    "binding_residue_count": 0,
                    "not_binding_residue_count": 0,
                    "sequence": "",
                }
            entry = summary[chain_id]

            residue = _token_residue_symbol(token, token_fields)
            if residue:
                entry["residue_count"] += 1
                entry["sequence"] += residue

            if "design_mask" in token_fields and bool(token["design_mask"]):
                entry["design_residue_count"] += 1
            if "binding_type" in token_fields:
                binding_type = int(token["binding_type"])
                if binding_type == _BINDING_TYPE_BINDING:
                    entry["binding_residue_count"] += 1
                elif binding_type == _BINDING_TYPE_NOT_BINDING:
                    entry["not_binding_residue_count"] += 1
            if entry["molecule_type"] == "unknown" and "mol_type" in token_fields:
                entry["molecule_type"] = _MOLECULE_TYPE_LABEL.get(
                    int(token["mol_type"]),
                    "unknown",
                )

        return tuple(summary.values())

    def to_mmcif(self) -> str:
        if self.structure is None:
            raise ValueError("No Boltz2 structure available.")
        from refua.boltz.data.write.mmcif import to_mmcif  # noqa: PLC0415

        return to_mmcif(
            self.structure.structure,
            plddts=self.structure.plddt,
            boltz2=True,
        )

    def to_bcif(self) -> bytes:
        if self.structure is None:
            raise ValueError("No Boltz2 structure available.")
        return bcif_bytes_from_structure(
            self.structure.structure,
            plddts=self.structure.plddt,
            boltz2=True,
        )


Entity = Protein | DNA | RNA | Binder | DesignFile | MolProperties | SmallMolecule | Mol

_REFUA_ENV: ContextVar[RefuaEnv | None] = ContextVar(
    "refua_model_context",
    default=None,
)


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc)
    if "CUDA out of memory" in message or "CUDA error: out of memory" in message:
        return True
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _ModelProxy:
    def __init__(self, context: RefuaEnv, key: str) -> None:
        self._context = context
        self._key = key

    def __getattr__(self, name: str) -> Any:
        model = self._context._resolve(self._key)  # noqa: SLF001
        return getattr(model, name)


class RefuaEnv:
    """Context manager that lazily loads and caches shared models.

    Models load on first use and may evict the last used model on CUDA OOM.
    When auto_run is True, Complex.fold will automatically compute affinity
    (when unambiguous) and ligand RDKit/TxGemma analyses.
    """

    def __init__(
        self,
        *,
        boltz_factory: Callable[[], Boltz2] | None = None,
        boltzgen_factory: Callable[[], BoltzGen] | None = None,
        auto_run: bool = False,
    ) -> None:
        self._factories: dict[str, Callable[[], object]] = {
            "boltz": boltz_factory or Boltz2,
            "boltzgen": boltzgen_factory or BoltzGen,
        }
        self._instances: dict[str, object] = {}
        self._usage: OrderedDict[str, None] = OrderedDict()
        self._closed = False
        self._token: Token[RefuaEnv | None] | None = None
        self._boltz_proxy = _ModelProxy(self, "boltz")
        self._boltzgen_proxy = _ModelProxy(self, "boltzgen")
        self.auto_run = auto_run

    @property
    def boltz(self) -> _ModelProxy:
        """Return a lazy Boltz2 proxy."""
        return self._boltz_proxy

    @property
    def boltzgen(self) -> _ModelProxy:
        """Return a lazy BoltzGen proxy."""
        return self._boltzgen_proxy

    def close(self) -> None:
        """Unload any cached models and release GPU memory."""
        if self._closed:
            return
        self._closed = True
        for key in list(self._instances):
            self._unload(key)
        self._instances.clear()
        self._usage.clear()

    def __enter__(self) -> RefuaEnv:
        self._token = _REFUA_ENV.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        finally:
            if self._token is not None:
                _REFUA_ENV.reset(self._token)
                self._token = None

    def _resolve(self, key: str) -> object:
        if self._closed:
            raise RuntimeError("Model context is closed.")
        if key in self._instances:
            self._touch(key)
            return self._instances[key]
        return self._load(key)

    def _load(self, key: str) -> object:
        factory = self._factories[key]
        while True:
            try:
                instance = factory()
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise
                if not self._evict_last_used():
                    raise
                continue
            self._instances[key] = instance
            self._touch(key)
            return instance

    def _touch(self, key: str) -> None:
        if key in self._usage:
            self._usage.move_to_end(key)
        else:
            self._usage[key] = None

    def _evict_last_used(self) -> bool:
        if not self._usage:
            return False
        key, _ = self._usage.popitem(last=True)
        self._unload(key)
        return True

    def _unload(self, key: str) -> None:
        instance = self._instances.pop(key, None)
        if instance is None:
            return
        unload = getattr(instance, "unload", None)
        if callable(unload):
            unload()
        _release_cuda_memory()


class Complex:
    """Unified complex that routes to Boltz2, BoltzGen, or both."""

    def __init__(
        self,
        entities: Sequence[Entity] | None = None,
        *,
        name: str = "complex",
        base_dir: str | Path | None = None,
        boltz: Boltz2 | None = None,
        boltzgen: BoltzGen | None = None,
    ) -> None:
        self.entities: list[Entity] = list(entities or [])
        self.name = name
        self.base_dir = (
            Path(base_dir).expanduser().resolve() if base_dir is not None else None
        )
        self._boltz = boltz
        self._boltzgen = boltzgen
        self._constraints: list[
            BondConstraint | PocketConstraint | ContactConstraint
        ] = []
        self._affinity_requested = False
        self._affinity_binder: str | object | None = None
        self.last_fold: FoldResult | None = None
        self.last_structure: StructurePrediction | None = None
        self.last_design: BoltzGenTrace | None = None
        self.last_affinity: AffinityPrediction | None = None
        self.ligand_rdkit: dict[str, Mapping[str, Any]] = {}
        self.ligand_admet: dict[str, Mapping[str, Any]] = {}

    @classmethod
    def antibody_design(
        cls,
        antigen: Protein | str,
        *,
        epitope: str | Mapping[str, Any] | None = None,
        not_epitope: str | None = None,
        heavy: Binder | str | int | None = None,
        light: Binder | str | int | None = None,
        heavy_cdr_lengths: tuple[int, int, int] = _DEFAULT_HEAVY_CDR_LENGTHS,
        light_cdr_lengths: tuple[int, int, int] = _DEFAULT_LIGHT_CDR_LENGTHS,
        antigen_id: ChainIds = "A",
        heavy_id: ChainIds = "H",
        light_id: ChainIds = "L",
        name: str = "antibody_design",
        base_dir: str | Path | None = None,
        boltz: Boltz2 | None = None,
        boltzgen: BoltzGen | None = None,
    ) -> Complex:
        """Create an antibody design complex with antigen, heavy, and light chains.

        Parameters
        ----------
        antigen : Protein or str
            Antigen target as a ``Protein`` entity or raw amino-acid sequence.
        epitope : str or Mapping[str, Any], optional
            Antigen binding mask for BoltzGen. String inputs map to
            ``{"binding": epitope}``.
        not_epitope : str, optional
            Optional non-binding antigen range merged into ``epitope`` as
            ``{"not_binding": ...}``.
        heavy : Binder or str or int, optional
            Heavy-chain binder spec. If omitted, a default human framework with
            designable CDRs is generated from ``heavy_cdr_lengths``.
        light : Binder or str or int, optional
            Light-chain binder spec. If omitted, a default human framework with
            designable CDRs is generated from ``light_cdr_lengths``.
        heavy_cdr_lengths : tuple[int, int, int], optional
            Default CDR-H1/H2/H3 lengths when ``heavy`` is omitted.
        light_cdr_lengths : tuple[int, int, int], optional
            Default CDR-L1/L2/L3 lengths when ``light`` is omitted.
        antigen_id : ChainIds, optional
            Antigen chain id(s) when not already set on ``antigen``.
        heavy_id : ChainIds, optional
            Heavy-chain id when not already set on ``heavy``.
        light_id : ChainIds, optional
            Light-chain id when not already set on ``light``.
        """
        epitope_binding_types = _normalize_epitope_binding_types(
            epitope=epitope,
            not_epitope=not_epitope,
        )
        antigen_entity = _coerce_antigen_entity(
            antigen,
            antigen_id=antigen_id,
            epitope_binding_types=epitope_binding_types,
        )

        h1, h2, h3 = _coerce_cdr_lengths(heavy_cdr_lengths, label="heavy")
        l1, l2, l3 = _coerce_cdr_lengths(light_cdr_lengths, label="light")
        default_heavy = Binder(
            spec=_DEFAULT_ANTIBODY_HEAVY_FRAMEWORK,
            template_values={"h1": h1, "h2": h2, "h3": h3},
        )
        default_light = Binder(
            spec=_DEFAULT_ANTIBODY_LIGHT_FRAMEWORK,
            template_values={"l1": l1, "l2": l2, "l3": l3},
        )
        heavy_entity = _coerce_antibody_binder(
            heavy,
            default_binder=default_heavy,
            chain_ids=heavy_id,
            label="heavy",
        )
        light_entity = _coerce_antibody_binder(
            light,
            default_binder=default_light,
            chain_ids=light_id,
            label="light",
        )

        return cls(
            [antigen_entity, heavy_entity, light_entity],
            name=name,
            base_dir=base_dir,
            boltz=boltz,
            boltzgen=boltzgen,
        )

    def _resolve_boltz_model(self, boltz: Boltz2 | None) -> Boltz2 | _ModelProxy:
        if boltz is not None:
            return boltz
        if self._boltz is not None:
            return self._boltz
        context = _REFUA_ENV.get()
        if context is not None:
            return context.boltz
        return Boltz2()

    def _resolve_boltzgen_model(
        self,
        boltzgen: BoltzGen | None,
    ) -> BoltzGen | _ModelProxy:
        if boltzgen is not None:
            return boltzgen
        if self._boltzgen is not None:
            return self._boltzgen
        context = _REFUA_ENV.get()
        if context is not None:
            return context.boltzgen
        return BoltzGen()

    def add(self, *entities: Entity) -> Complex:
        """Append entities to the complex."""
        self.entities.extend(entities)
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
    ) -> Complex:
        """Attach a file-backed design entity for BoltzGen."""
        self.entities.append(
            DesignFile(
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
        if self.base_dir is None:
            try:
                self.base_dir = Path(path).expanduser().resolve().parent
            except (OSError, RuntimeError):
                pass
        return self

    def bond(
        self,
        atom1: AtomRef | Sequence[str | int],
        atom2: AtomRef | Sequence[str | int],
    ) -> Complex:
        """Add a covalent bond constraint for Boltz2."""
        self._constraints.append(BondConstraint(atom1=atom1, atom2=atom2))
        return self

    def pocket(
        self,
        binder: str | object,
        contacts: Iterable[TokenRef | Sequence[str | int]],
        *,
        max_distance: float = 6.0,
        force: bool = False,
    ) -> Complex:
        """Add a pocket constraint for Boltz2."""
        self._constraints.append(
            PocketConstraint(
                binder=binder,
                contacts=tuple(contacts),
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
    ) -> Complex:
        """Add a contact constraint for Boltz2."""
        self._constraints.append(
            ContactConstraint(
                token1=token1,
                token2=token2,
                max_distance=max_distance,
                force=force,
            )
        )
        return self

    def request_affinity(self, binder: str | object | None = None) -> Complex:
        """Request affinity prediction for a ligand binder."""
        self._affinity_requested = True
        self._affinity_binder = binder
        return self

    def affinity(
        self,
        binder: str | object | None = None,
        *,
        boltz: Boltz2 | None = None,
    ) -> AffinityPrediction:
        """Run an affinity prediction without folding."""
        (
            proteins,
            dnas,
            rnas,
            _,
            ligands,
            _,
            _,
            entity_ids,
            ligand_ids,
        ) = _resolve_entities(self.entities)
        if not proteins and not dnas and not rnas and not ligands:
            raise ValueError(
                "Affinity requires at least one protein, DNA, RNA, or ligand entity."
            )
        if not ligands:
            raise ValueError("Affinity requires at least one ligand entity.")

        boltz_model = self._resolve_boltz_model(boltz)
        fold_complex = _build_boltz_complex(
            boltz_model,
            self.name,
            proteins,
            dnas,
            rnas,
            ligands,
            self._constraints,
            entity_ids,
        )

        affinity_binder = binder
        if affinity_binder is None and self._affinity_requested:
            affinity_binder = self._affinity_binder
        binder_id = _resolve_affinity_binder(affinity_binder, entity_ids, ligand_ids)
        affinity = fold_complex.get_affinity(binder_id)
        context = _REFUA_ENV.get()
        if context is not None and context.auto_run and ligands:
            ligand_rdkit: dict[str, Mapping[str, Any]] = {}
            ligand_admet: dict[str, Mapping[str, Any]] = {}
            for ligand, ids, _ in ligands:
                ligand_id = ids[0]
                props = _mol_properties_from_entity(ligand)
                ligand_rdkit[ligand_id] = props.to_dict()
                ligand_admet[ligand_id] = props.admet_profile()
            self.ligand_rdkit = dict(ligand_rdkit)
            self.ligand_admet = dict(ligand_admet)
        self.last_affinity = affinity
        return affinity

    def fold(
        self,
        *,
        boltz: Boltz2 | None = None,
        boltzgen: BoltzGen | None = None,
        run_boltz: bool | None = None,
        run_boltzgen: bool | None = None,
    ) -> FoldResult:
        (
            proteins,
            dnas,
            rnas,
            binders,
            ligands,
            files,
            chain_ids,
            entity_ids,
            ligand_ids,
        ) = _resolve_entities(self.entities)

        if (
            not proteins
            and not dnas
            and not rnas
            and not binders
            and not ligands
            and not files
        ):
            raise ValueError("Complex requires at least one entity.")

        context = _REFUA_ENV.get()
        auto_run = bool(context and context.auto_run)
        auto_affinity = False
        if auto_run and ligands and not self._affinity_requested:
            if self._affinity_binder is not None or len(ligand_ids) == 1:
                auto_affinity = True

        has_design = bool(binders or files)
        if run_boltzgen is None:
            run_boltzgen = has_design

        if run_boltz is None:
            run_boltz = bool(
                proteins
                or dnas
                or rnas
                or ligands
                or self._constraints
                or self._affinity_requested
            )

        structure: StructurePrediction | None = None
        design: BoltzGenTrace | None = None
        affinity: AffinityPrediction | None = None
        backends: list[str] = []

        if run_boltz:
            if not proteins and not dnas and not rnas and not ligands:
                raise ValueError(
                    "Boltz folding requires at least one protein, DNA, RNA, or ligand."
                )
            boltz_model = self._resolve_boltz_model(boltz)
            fold_complex = _build_boltz_complex(
                boltz_model,
                self.name,
                proteins,
                dnas,
                rnas,
                ligands,
                self._constraints,
                entity_ids,
            )
            affinity_binder_id = None
            if self._affinity_requested or auto_affinity:
                affinity_binder_id = _resolve_affinity_binder(
                    self._affinity_binder,
                    entity_ids,
                    ligand_ids,
                )
                fold_complex.request_affinity(affinity_binder_id)
            structure = fold_complex.fold()
            if self._affinity_requested or auto_affinity:
                affinity = fold_complex.get_affinity(affinity_binder_id)
            backends.append("boltz")

        if run_boltzgen:
            if dnas or rnas:
                raise ValueError("BoltzGen does not support DNA/RNA entities.")
            boltzgen_model = self._resolve_boltzgen_model(boltzgen)
            design_base_dir = self.base_dir
            if design_base_dir is None and files:
                try:
                    design_base_dir = Path(files[0].path).expanduser().resolve().parent
                except (OSError, RuntimeError):
                    design_base_dir = None
            design_builder = boltzgen_model.design(self.name, base_dir=design_base_dir)
            for item in files:
                design_builder.file(
                    item.path,
                    include=item.include,
                    exclude=item.exclude,
                    include_proximity=item.include_proximity,
                    binding_types=item.binding_types,
                    structure_groups=item.structure_groups,
                    design=item.design,
                    not_design=item.not_design,
                    secondary_structure=item.secondary_structure,
                    design_insertions=item.design_insertions,
                    fuse=item.fuse,
                    msa=item.msa,
                    use_assembly=item.use_assembly,
                    reset_res_index=item.reset_res_index,
                    extra=item.extra,
                )
            for protein, ids in proteins:
                design_builder.protein(
                    ids,
                    protein.sequence,
                    binding_types=protein.binding_types,
                    secondary_structure=protein.secondary_structure,
                    cyclic=protein.cyclic,
                    msa=_boltzgen_msa(protein.msa),
                )
            for binder, ids in binders:
                design_builder.protein(
                    ids,
                    binder.sequence_spec(),
                    binding_types=binder.binding_types,
                    secondary_structure=binder.secondary_structure,
                    cyclic=binder.cyclic,
                )
            for _, ids, smiles in ligands:
                design_builder.ligand(ids, smiles=smiles)
            design = design_builder.to_features(return_trace=True)  # type: ignore[assignment]
            backends.append("boltzgen")

        ligand_rdkit: dict[str, Mapping[str, Any]] = {}
        ligand_admet: dict[str, Mapping[str, Any]] = {}
        if auto_run:
            if ligands:
                for ligand, ids, _ in ligands:
                    ligand_id = ids[0]
                    props = _mol_properties_from_entity(ligand)
                    ligand_rdkit[ligand_id] = props.to_dict()
                    ligand_admet[ligand_id] = props.admet_profile()
                self.ligand_rdkit = dict(ligand_rdkit)
                self.ligand_admet = dict(ligand_admet)
            else:
                self.ligand_rdkit = {}
                self.ligand_admet = {}
        else:
            ligand_rdkit = dict(self.ligand_rdkit)
            ligand_admet = dict(self.ligand_admet)

        backend = "+".join(backends) if backends else "none"
        result = FoldResult(
            backend=backend,
            structure=structure,
            design=design,
            affinity=affinity,
            chain_ids=chain_ids,
            binder_sequences=_binder_sequence_map(binders),
            ligand_rdkit=ligand_rdkit,
            ligand_admet=ligand_admet,
        )
        self.last_fold = result
        self.last_structure = structure
        self.last_design = design
        self.last_affinity = affinity
        return result


def _coerce_cdr_lengths(
    lengths: Sequence[int],
    *,
    label: str,
) -> tuple[int, int, int]:
    try:
        normalized = tuple(int(length) for length in lengths)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} CDR lengths must be integers.") from exc

    if len(normalized) != 3:
        raise ValueError(f"{label} CDR lengths must have exactly three values.")
    if any(length < 1 for length in normalized):
        raise ValueError(f"{label} CDR lengths must be >= 1.")
    cdr1, cdr2, cdr3 = normalized
    return cdr1, cdr2, cdr3


def _normalize_epitope_binding_types(
    *,
    epitope: str | Mapping[str, Any] | None,
    not_epitope: str | None,
) -> Mapping[str, Any] | None:
    if epitope is None:
        if not_epitope is not None:
            raise ValueError("not_epitope requires epitope to be set.")
        return None

    if isinstance(epitope, str):
        if not epitope.strip():
            raise ValueError("epitope must be a non-empty range string.")
        payload: dict[str, Any] = {"binding": epitope}
    else:
        payload = dict(epitope)
        if not payload:
            raise ValueError("epitope mapping must not be empty.")

    if not_epitope is not None:
        if "not_binding" in payload and payload["not_binding"] != not_epitope:
            raise ValueError(
                "epitope.not_binding conflicts with not_epitope argument.",
            )
        payload["not_binding"] = not_epitope
    return payload


def _coerce_antigen_entity(
    antigen: Protein | str,
    *,
    antigen_id: ChainIds,
    epitope_binding_types: Mapping[str, Any] | None,
) -> Protein:
    if isinstance(antigen, str):
        if not antigen:
            raise ValueError("antigen sequence must be non-empty.")
        return Protein(
            antigen,
            ids=antigen_id,
            binding_types=epitope_binding_types,
        )
    if not isinstance(antigen, Protein):
        raise TypeError("antigen must be a Protein object or sequence string.")

    result = antigen
    if epitope_binding_types is not None:
        if (
            result.binding_types is not None
            and result.binding_types != epitope_binding_types
        ):
            raise ValueError(
                "Provide epitope via antibody_design() or on Protein.binding_types, not both.",
            )
        result = replace(result, binding_types=epitope_binding_types)
    if result.ids is None:
        result = replace(result, ids=antigen_id)
    return result


def _coerce_antibody_binder(
    binder: Binder | str | int | None,
    *,
    default_binder: Binder,
    chain_ids: ChainIds,
    label: str,
) -> Binder:
    if binder is None:
        result = default_binder
    elif isinstance(binder, Binder):
        result = binder
    elif isinstance(binder, str):
        if not binder:
            raise ValueError(f"{label} binder spec must be non-empty.")
        result = Binder(binder)
    elif isinstance(binder, int):
        result = Binder(length=binder)
    else:
        raise TypeError(
            f"{label} binder must be a Binder, sequence spec string, integer length, or None.",
        )

    if result.ids is None:
        result = replace(result, ids=chain_ids)
    return result


def _token_residue_symbol(
    token: Any,
    token_fields: set[str],
) -> str:
    if "res_name" in token_fields:
        token_name = str(token["res_name"]).strip().upper()
        if not token_name or token_name == "<PAD>":
            return ""
        mapped = _RESNAME_TO_LETTER.get(token_name)
        if mapped is not None:
            return mapped
        if len(token_name) == 1:
            return token_name

    token_id = int(token["res_type"]) if "res_type" in token_fields else -1
    if token_id == 0:
        return ""
    if token_id == 1:
        return "-"
    token_offset = token_id - 2
    if 0 <= token_offset < len(_CANONICAL_TOKEN_LETTERS):
        return _CANONICAL_TOKEN_LETTERS[token_offset]
    return "X"


def _resolve_entities(
    entities: Sequence[Entity],
) -> tuple[
    list[tuple[Protein, tuple[str, ...]]],
    list[tuple[DNA, tuple[str, ...]]],
    list[tuple[RNA, tuple[str, ...]]],
    list[tuple[Binder, tuple[str, ...]]],
    list[tuple[Entity, tuple[str, ...], str]],
    list[DesignFile],
    tuple[tuple[str, ...], ...],
    dict[int, tuple[str, ...]],
    tuple[str, ...],
]:
    proteins: list[tuple[Protein, tuple[str, ...]]] = []
    dnas: list[tuple[DNA, tuple[str, ...]]] = []
    rnas: list[tuple[RNA, tuple[str, ...]]] = []
    binders: list[tuple[Binder, tuple[str, ...]]] = []
    ligands: list[tuple[Entity, tuple[str, ...], str]] = []
    files: list[DesignFile] = []
    chain_ids: list[tuple[str, ...]] = []
    entity_ids: dict[int, tuple[str, ...]] = {}
    used: set[str] = set()

    chain_iter = _chain_id_iter()
    ligand_idx = 1

    for entity in entities:
        if isinstance(entity, Protein):
            ids = _resolve_ids(entity.ids, chain_iter, used)
            _reserve_ids(ids, used)
            proteins.append((entity, ids))
            chain_ids.append(ids)
            entity_ids[id(entity)] = ids
            continue
        if isinstance(entity, DNA):
            ids = _resolve_ids(entity.ids, chain_iter, used)
            _reserve_ids(ids, used)
            dnas.append((entity, ids))
            chain_ids.append(ids)
            entity_ids[id(entity)] = ids
            continue
        if isinstance(entity, RNA):
            ids = _resolve_ids(entity.ids, chain_iter, used)
            _reserve_ids(ids, used)
            rnas.append((entity, ids))
            chain_ids.append(ids)
            entity_ids[id(entity)] = ids
            continue
        if isinstance(entity, Binder):
            ids = _resolve_ids(entity.ids, chain_iter, used)
            _reserve_ids(ids, used)
            binders.append((entity, ids))
            chain_ids.append(ids)
            entity_ids[id(entity)] = ids
            continue
        if isinstance(entity, DesignFile):
            files.append(entity)
            continue
        if _is_sm(entity):
            smiles = _smiles_from_entity(entity)
            while True:
                ligand_id = f"L{ligand_idx}"
                ligand_idx += 1
                if ligand_id not in used:
                    break
            ids = (ligand_id,)
            _reserve_ids(ids, used)
            ligands.append((entity, ids, smiles))
            chain_ids.append(ids)
            entity_ids[id(entity)] = ids
            continue
        raise TypeError(f"Unsupported entity: {type(entity)!r}")

    ligand_ids = tuple(ids[0] for _, ids, _ in ligands)
    return (
        proteins,
        dnas,
        rnas,
        binders,
        ligands,
        files,
        tuple(chain_ids),
        entity_ids,
        ligand_ids,
    )


def _binder_sequence_map(
    binders: Sequence[tuple[Binder, tuple[str, ...]]],
) -> Mapping[str, str]:
    sequences: dict[str, str] = {}
    for binder, ids in binders:
        spec = binder.sequence
        for chain_id in ids:
            sequences[chain_id] = spec
    return sequences


def _boltz_msa(value: object | None) -> object | None:
    if isinstance(value, (str, int)):
        return None
    return value


def _boltzgen_msa(value: object | None) -> str | int | None:
    if isinstance(value, (str, int)):
        return value
    return None


def _resolve_affinity_binder(
    binder: str | object | None,
    entity_ids: Mapping[int, tuple[str, ...]],
    ligand_ids: Sequence[str],
) -> str:
    if binder is None:
        if len(ligand_ids) != 1:
            raise ValueError(
                "Affinity requires exactly one ligand or an explicit binder."
            )
        return ligand_ids[0]
    if isinstance(binder, str):
        return binder
    ids = entity_ids.get(id(binder))
    if ids is None:
        raise ValueError("Affinity binder is not part of the complex.")
    if len(ids) != 1:
        raise ValueError("Affinity binder must map to a single chain id.")
    return ids[0]


def _resolve_binder_id(
    binder: str | object,
    entity_ids: Mapping[int, tuple[str, ...]],
) -> str:
    if isinstance(binder, str):
        return binder
    ids = entity_ids.get(id(binder))
    if ids is None:
        raise ValueError("Binder is not part of the complex.")
    if len(ids) != 1:
        raise ValueError("Binder must map to a single chain id.")
    return ids[0]


def _apply_constraints(
    fold_complex: Any,
    constraints: Sequence[BondConstraint | PocketConstraint | ContactConstraint],
    entity_ids: Mapping[int, tuple[str, ...]],
) -> None:
    for constraint in constraints:
        if isinstance(constraint, BondConstraint):
            fold_complex.bond(constraint.atom1, constraint.atom2)
            continue
        if isinstance(constraint, PocketConstraint):
            binder_id = _resolve_binder_id(constraint.binder, entity_ids)
            fold_complex.pocket(
                binder_id,
                contacts=constraint.contacts,
                max_distance=constraint.max_distance,
                force=constraint.force,
            )
            continue
        if isinstance(constraint, ContactConstraint):
            fold_complex.contact(
                constraint.token1,
                constraint.token2,
                max_distance=constraint.max_distance,
                force=constraint.force,
            )


def _build_boltz_complex(
    boltz_model: Boltz2 | _ModelProxy,
    name: str,
    proteins: Sequence[tuple[Protein, tuple[str, ...]]],
    dnas: Sequence[tuple[DNA, tuple[str, ...]]],
    rnas: Sequence[tuple[RNA, tuple[str, ...]]],
    ligands: Sequence[tuple[Entity, tuple[str, ...], str]],
    constraints: Sequence[BondConstraint | PocketConstraint | ContactConstraint],
    entity_ids: Mapping[int, tuple[str, ...]],
):
    fold_complex = boltz_model.fold_complex(name)
    for protein, ids in proteins:
        fold_complex.protein(
            ids,
            protein.sequence,
            modifications=protein.modifications,
            msa=_boltz_msa(protein.msa),
            cyclic=protein.cyclic,
        )
    for dna, ids in dnas:
        fold_complex.dna(
            ids,
            dna.sequence,
            modifications=dna.modifications,
            cyclic=dna.cyclic,
        )
    for rna, ids in rnas:
        fold_complex.rna(
            ids,
            rna.sequence,
            modifications=rna.modifications,
            cyclic=rna.cyclic,
        )
    for _, ids, smiles in ligands:
        fold_complex.ligand(ids, smiles=smiles)
    if constraints:
        _apply_constraints(fold_complex, constraints, entity_ids)
    return fold_complex


def _reserve_ids(ids: tuple[str, ...], used: set[str]) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate chain ids in entity: {ids}")
    conflicts = set(ids) & used
    if conflicts:
        raise ValueError(f"Duplicate chain ids across entities: {sorted(conflicts)}")
    used.update(ids)


def _resolve_ids(
    ids: ChainIds | None,
    chain_iter: Iterator[str],
    used: set[str],
) -> tuple[str, ...]:
    if ids is None:
        return (_next_chain_id(chain_iter, used),)
    normalized = normalize_chain_ids(ids)
    if not normalized:
        raise ValueError("Chain ids cannot be empty.")
    return normalized


def _next_chain_id(chain_iter: Iterator[str], used: set[str]) -> str:
    for candidate in chain_iter:
        if candidate not in used:
            return candidate
    raise RuntimeError("Unable to allocate a new chain id.")


def _chain_id_iter() -> Iterator[str]:
    for letter in string.ascii_uppercase:
        yield letter
    idx = 1
    while True:
        for letter in string.ascii_uppercase:
            yield f"{letter}{idx}"
        idx += 1


def _is_sm(entity: Entity) -> bool:
    return isinstance(entity, (MolProperties, SmallMolecule, Mol))


def _mol_properties_from_entity(entity: Entity) -> MolProperties:
    if isinstance(entity, MolProperties):
        return entity
    if isinstance(entity, SmallMolecule):
        return entity.properties(lazy=True)
    if isinstance(entity, Mol):
        return MolProperties(entity, lazy=True)
    raise TypeError(f"Unsupported ligand entity: {type(entity)!r}")


def _smiles_from_entity(entity: Entity) -> str:
    if isinstance(entity, MolProperties):
        mol = entity.mol
    elif isinstance(entity, SmallMolecule):
        mol = entity.mol
    elif isinstance(entity, Mol):
        mol = entity
    else:
        raise TypeError(f"Unsupported ligand entity: {type(entity)!r}")
    return Chem.MolToSmiles(mol)

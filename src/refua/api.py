"""Shared API primitives for Refua."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ChainIds = str | Sequence[str]
ResidueId = int | str

_BOLTZGEN_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    (
        "design-diverse",
        "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt",
        "model",
    ),
    (
        "design-adherence",
        "huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt",
        "model",
    ),
    (
        "inverse-fold",
        "huggingface:boltzgen/boltzgen-1:boltzgen1_ifold.ckpt",
        "model",
    ),
    (
        "folding",
        "huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt",
        "model",
    ),
    (
        "affinity",
        "huggingface:boltzgen/boltzgen-1:boltz2_aff.ckpt",
        "model",
    ),
    (
        "moldir",
        "huggingface:boltzgen/inference-data:mols.zip",
        "dataset",
    ),
)


def normalize_chain_ids(ids: ChainIds) -> tuple[str, ...]:
    """Normalize a chain id or iterable of ids into a tuple."""
    if isinstance(ids, str):
        return (ids,)
    return tuple(ids)


@dataclass(frozen=True, slots=True)
class AtomRef:
    """Atom reference for bond constraints."""

    chain: str
    residue: int
    atom: str

    def to_schema(self) -> list[str | int]:
        """Return the schema representation for YAML/JSON inputs."""
        return [self.chain, self.residue, self.atom]


@dataclass(frozen=True, slots=True)
class TokenRef:
    """Token reference for pocket/contact constraints."""

    chain: str
    token: ResidueId

    def to_schema(self) -> list[str | int]:
        """Return the schema representation for YAML/JSON inputs."""
        return [self.chain, self.token]


@dataclass(frozen=True, slots=True)
class Bond:
    """Covalent bond constraint."""

    atom1: AtomRef
    atom2: AtomRef

    def to_schema(self) -> dict[str, dict[str, list[str | int]]]:
        """Return the schema representation for YAML/JSON inputs."""
        return {
            "bond": {
                "atom1": self.atom1.to_schema(),
                "atom2": self.atom2.to_schema(),
            }
        }


def as_atom_ref(ref: AtomRef | Sequence[str | int]) -> AtomRef:
    """Coerce an atom reference tuple/list into an AtomRef."""
    if isinstance(ref, AtomRef):
        return ref
    chain, residue, atom = ref
    return AtomRef(chain=str(chain), residue=int(residue), atom=str(atom))


def as_token_ref(ref: TokenRef | Sequence[str | int]) -> TokenRef:
    """Coerce a token reference tuple/list into a TokenRef."""
    if isinstance(ref, TokenRef):
        return ref
    chain, token = ref
    if isinstance(token, str):
        return TokenRef(chain=str(chain), token=token)
    return TokenRef(chain=str(chain), token=int(token))


def _resolve_boltz_cache_dir(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved = Path(env_cache).expanduser().resolve()
        if not resolved.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return resolved
    return Path("~/.boltz").expanduser()


def _download_hf_artifact(
    artifact: str,
    *,
    repo_type: str,
    cache_dir: Path | None,
    token: str | None,
) -> Path:
    if not artifact.startswith("huggingface:"):
        resolved = Path(artifact).expanduser()
        if not resolved.exists():
            msg = f"Artifact not found: {resolved}"
            raise FileNotFoundError(msg)
        return resolved

    try:
        _, repo_id, filename = artifact.split(":")
    except ValueError as exc:
        msg = (
            f"Invalid artifact: {artifact}. Expected format: "
            "huggingface:<repo_id>:<filename>"
        )
        raise ValueError(msg) from exc

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download BoltzGen assets."
        ) from exc

    resolved_path = hf_hub_download(
        repo_id,
        filename,
        repo_type=repo_type,
        library_name="boltzgen",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
    )
    return Path(resolved_path)


def download_assets(
    *,
    boltz_cache_dir: str | Path | None = None,
    boltzgen_cache_dir: str | Path | None = None,
    download_boltz2: bool = True,
    download_boltzgen: bool = True,
    models_token: str | None = None,
) -> dict[str, dict[str, Path]]:
    """Download Boltz2 and BoltzGen model/molecule assets.

    Parameters
    ----------
    boltz_cache_dir : str or Path, optional
        Directory for Boltz2 assets. Defaults to ~/.boltz or $BOLTZ_CACHE.
    boltzgen_cache_dir : str or Path, optional
        Hugging Face cache directory for BoltzGen assets (defaults to HF cache).
    download_boltz2 : bool, optional
        Whether to download Boltz2 assets. Default: True.
    download_boltzgen : bool, optional
        Whether to download BoltzGen assets. Default: True.
    models_token : str, optional
        Optional Hugging Face token (defaults to $HF_TOKEN if set).

    Returns
    -------
    dict[str, dict[str, Path]]
        Mapping of asset groups to local paths.
    """
    if not (download_boltz2 or download_boltzgen):
        raise ValueError(
            "At least one of download_boltz2 or download_boltzgen must be True."
        )

    results: dict[str, dict[str, Path]] = {}

    if download_boltz2:
        cache_dir = _resolve_boltz_cache_dir(boltz_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        from boltz.main import download_boltz2  # noqa: PLC0415

        download_boltz2(cache_dir)
        results["boltz2"] = {
            "mols": cache_dir / "mols",
            "model": cache_dir / "boltz2_conf.ckpt",
            "affinity_model": cache_dir / "boltz2_aff.ckpt",
        }

    if download_boltzgen:
        token = models_token or os.environ.get("HF_TOKEN")
        cache_dir = (
            Path(boltzgen_cache_dir).expanduser()
            if boltzgen_cache_dir is not None
            else None
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        boltzgen_assets: dict[str, Path] = {}
        for name, artifact, repo_type in _BOLTZGEN_ARTIFACTS:
            boltzgen_assets[name] = _download_hf_artifact(
                artifact,
                repo_type=repo_type,
                cache_dir=cache_dir,
                token=token,
            )
        results["boltzgen"] = boltzgen_assets

    return results

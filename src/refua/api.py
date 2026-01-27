"""Shared API primitives for Refua."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from inspect import signature
from pathlib import Path
from typing import Any

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
_TXGEMMA_MODEL_ID_TEMPLATE = "google/txgemma-{variant}"
_DEFAULT_TXGEMMA_VARIANT = "9b-chat"


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


def _admet_support_available() -> bool:
    return (
        find_spec("transformers") is not None
        and find_spec("huggingface_hub") is not None
    )


def _supports_hf_kwarg(fn: Callable[..., Any], name: str) -> bool:
    try:
        return name in signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _download_hf_artifact(
    artifact: str,
    *,
    repo_type: str,
    cache_dir: Path | None,
    token: str | None,
    skip_existing: bool,
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

    def _download(local_files_only: bool) -> Path:
        kwargs = {
            "repo_id": repo_id,
            "filename": filename,
            "repo_type": repo_type,
            "library_name": "boltzgen",
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "token": token,
        }
        if _supports_hf_kwarg(hf_hub_download, "local_files_only"):
            kwargs["local_files_only"] = local_files_only
        return Path(hf_hub_download(**kwargs))

    if skip_existing:
        try:
            return _download(local_files_only=True)
        except Exception:
            pass

    return _download(local_files_only=False)


def _looks_like_hf_auth_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"GatedRepoError", "LocalTokenNotFoundError", "RepositoryNotFoundError"}:
        return True
    if name in {"HfHubHTTPError", "HTTPError"}:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {401, 403}:
            return True
    message = str(exc).lower()
    return any(
        token in message
        for token in ("unauthorized", "forbidden", "gated repo", "access denied")
    )


def _download_txgemma_repo(
    repo_id: str,
    *,
    cache_dir: Path | None,
    token: str | None,
    skip_existing: bool,
) -> Path:
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download TxGemma assets."
        ) from exc

    def _download(local_files_only: bool) -> Path:
        kwargs = {
            "repo_id": repo_id,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "token": token,
        }
        if _supports_hf_kwarg(snapshot_download, "local_files_only"):
            kwargs["local_files_only"] = local_files_only
        return Path(snapshot_download(**kwargs))

    if skip_existing:
        try:
            return _download(local_files_only=True)
        except Exception:
            pass

    try:
        resolved_path = _download(local_files_only=False)
    except Exception as exc:
        if _looks_like_hf_auth_error(exc):
            msg = (
                f"TxGemma download failed for '{repo_id}'. This usually means Hugging Face "
                "authentication is required. Set HF_TOKEN or pass models_token to "
                "download_assets, ensure you have accepted the model license at "
                f"https://huggingface.co/{repo_id}, and confirm the txgemma_variant is valid."
            )
            raise RuntimeError(msg) from exc
        raise

    return Path(resolved_path)


def download_assets(
    *,
    boltz_cache_dir: str | Path | None = None,
    boltzgen_cache_dir: str | Path | None = None,
    download_boltz2: bool = True,
    download_boltzgen: bool = True,
    download_txgemma: bool | None = None,
    skip_existing: bool = True,
    models_token: str | None = None,
    txgemma_variant: str = _DEFAULT_TXGEMMA_VARIANT,
) -> dict[str, dict[str, Path]]:
    """Download Boltz2, BoltzGen, and TxGemma model/molecule assets.

    Parameters
    ----------
    boltz_cache_dir : str or Path, optional
        Directory for Boltz2 assets. Defaults to ~/.boltz or $BOLTZ_CACHE.
    boltzgen_cache_dir : str or Path, optional
        Hugging Face cache directory for BoltzGen/TxGemma assets (defaults to HF cache).
    download_boltz2 : bool, optional
        Whether to download Boltz2 assets. Default: True.
    download_boltzgen : bool, optional
        Whether to download BoltzGen assets. Default: True.
    download_txgemma : bool, optional
        Whether to download TxGemma ADMET assets. Default: auto (True if ADMET
        dependencies are installed, otherwise False).
    skip_existing : bool, optional
        Whether to skip downloads when assets already exist locally. Default: True.
    models_token : str, optional
        Optional Hugging Face token (defaults to $HF_TOKEN if set).
    txgemma_variant : str, optional
        TxGemma model variant to download (e.g., "9b-chat").

    Returns
    -------
    dict[str, dict[str, Path]]
        Mapping of asset groups to local paths.
    """
    if download_txgemma is None:
        download_txgemma = _admet_support_available()

    if not (download_boltz2 or download_boltzgen or download_txgemma):
        raise ValueError(
            "At least one of download_boltz2, download_boltzgen, or "
            "download_txgemma must be True."
        )

    results: dict[str, dict[str, Path]] = {}

    if download_boltz2:
        cache_dir = _resolve_boltz_cache_dir(boltz_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not (
            skip_existing
            and (cache_dir / "mols").exists()
            and (cache_dir / "boltz2_conf.ckpt").exists()
            and (cache_dir / "boltz2_aff.ckpt").exists()
        ):
            from refua.boltz.main import (  # noqa: PLC0415
                download_boltz2 as _download_boltz2,
            )

            _download_boltz2(cache_dir)
        results["boltz2"] = {
            "mols": cache_dir / "mols",
            "model": cache_dir / "boltz2_conf.ckpt",
            "affinity_model": cache_dir / "boltz2_aff.ckpt",
        }

    token = None
    cache_dir = None
    if download_boltzgen or download_txgemma:
        token = models_token or os.environ.get("HF_TOKEN")
        cache_dir = (
            Path(boltzgen_cache_dir).expanduser()
            if boltzgen_cache_dir is not None
            else None
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    if download_boltzgen:
        boltzgen_assets: dict[str, Path] = {}
        for name, artifact, repo_type in _BOLTZGEN_ARTIFACTS:
            boltzgen_assets[name] = _download_hf_artifact(
                artifact,
                repo_type=repo_type,
                cache_dir=cache_dir,
                token=token,
                skip_existing=skip_existing,
            )
        results["boltzgen"] = boltzgen_assets

    if download_txgemma:
        repo_id = _TXGEMMA_MODEL_ID_TEMPLATE.format(variant=txgemma_variant)
        results["txgemma"] = {
            "model": _download_txgemma_repo(
                repo_id,
                cache_dir=cache_dir,
                token=token,
                skip_existing=skip_existing,
            )
        }

    return results

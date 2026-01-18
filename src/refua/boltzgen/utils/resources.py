"""Helpers for loading boltzgen resources from disk or package data."""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

_RESOURCE_PACKAGE = "refua.boltzgen"
_RESOURCE_DIR = "resources"
_RESOURCE_PREFIXES = (
    "src/refua/boltzgen/resources/",
    "src/boltzgen/resources/",
    "boltzgen/resources/",
    "refua/boltzgen/resources/",
)


def _resource_root() -> resources.Traversable:
    return resources.files(_RESOURCE_PACKAGE) / _RESOURCE_DIR


def _extract_resource_suffix(path_value: str) -> str | None:
    normalized = path_value.replace("\\", "/")
    normalized = normalized.removeprefix("./")
    if normalized.startswith("resources/"):
        return normalized[len("resources/") :]
    for prefix in _RESOURCE_PREFIXES:
        if prefix in normalized:
            return normalized.split(prefix, 1)[1]
    return None


def read_text_from_path_or_resource(path_value: str) -> str:
    """Read text from a filesystem path or a packaged resource path."""
    path = Path(os.path.expandvars(path_value)).expanduser()
    if path.is_file():
        return path.read_text()

    suffix = _extract_resource_suffix(path_value)
    if suffix is None:
        msg = f"File not found: {path_value}"
        raise FileNotFoundError(msg)

    resource = _resource_root() / suffix
    if not resource.is_file():
        msg = f"Resource not found: resources/{suffix}"
        raise FileNotFoundError(msg)

    return resource.read_text()


def read_optional_resource_json(resource_rel: str) -> dict | None:
    """Load JSON from packaged resources, or return None when missing."""
    resource = _resource_root() / resource_rel
    if not resource.is_file():
        return None
    return json.loads(resource.read_text())

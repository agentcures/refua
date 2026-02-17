"""Security guardrails for vulnerable dependency surfaces."""

from __future__ import annotations

import ast
from pathlib import Path


def _contains_bio_entrez_import(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "Bio.Entrez" or alias.name.startswith("Bio.Entrez."):
                    return True
        if isinstance(node, ast.ImportFrom):
            if node.module == "Bio.Entrez" or (
                node.module and node.module.startswith("Bio.Entrez.")
            ):
                return True
            if node.module == "Bio":
                if any(alias.name == "Entrez" for alias in node.names):
                    return True
    return False


def test_source_tree_does_not_import_bio_entrez() -> None:
    """Bio.Entrez XXE advisory mitigation: keep Entrez out of runtime code."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "refua"
    offenders = [
        path for path in src_root.rglob("*.py") if _contains_bio_entrez_import(path)
    ]
    assert offenders == []

"""KRAS G12D + MRTX-1133 folding example using the unified API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from refua import Complex, Protein, SM, download_assets
from refua.boltz.data.mol import load_canonicals, load_molecules


# Sequence from PDB 7RPZ chain A (KRAS G12D).
KRAS_G12D_SEQUENCE = (
    "GMTEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETSLLDILDTAGQEEYSAMRDQYMRTGEGFL"
    "LVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKSDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTL"
    "VREIRKHKEK"
)

# MRTX-1133 ligand CCD from PDB 7RPZ.
MRTX_1133_CCD = "6IC"


def _resolve_cache_dir(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser()
    return Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()


def _load_ligand(cache_dir: Path):
    mol_dir = cache_dir / "mols"
    if not mol_dir.exists():
        download_assets(
            boltz_cache_dir=cache_dir,
            download_boltz2=True,
            download_boltzgen=False,
        )
    components = load_canonicals(str(mol_dir))
    components.update(load_molecules(str(mol_dir), [MRTX_1133_CCD]))
    return SM(components[MRTX_1133_CCD])


def build_complex(args: argparse.Namespace, ligand):
    """Create a KRAS/MRTX-1133 complex and request affinity."""
    complex_spec = Complex(
        [Protein(KRAS_G12D_SEQUENCE, ids=args.protein_id), ligand],
        name=args.name,
    )
    complex_spec.request_affinity(ligand)
    return complex_spec


def _infer_format(path: Path, explicit_format: str | None) -> str:
    if explicit_format:
        return explicit_format
    suffix = path.suffix.lower()
    if suffix == ".bcif":
        return "bcif"
    if suffix in {".cif", ".mmcif"}:
        return "cif"
    return "cif"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Fold KRAS G12D with MRTX-1133 and report affinity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protein-id", default="A", help="Protein chain id.")
    parser.add_argument(
        "--name",
        default="kras_g12d_mrtx1133",
        help="Spec name for the prediction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the predicted structure (.cif/.bcif).",
    )
    parser.add_argument(
        "--format",
        choices=("cif", "bcif"),
        default=None,
        help="Output format (defaults to extension when --output is set).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example, print affinity, and optionally save the structure."""
    args = parse_args()
    cache_dir = _resolve_cache_dir()
    ligand = _load_ligand(cache_dir)
    complex_spec = build_complex(args, ligand)
    if args.output:
        result = complex_spec.fold()
        affinity = result.affinity
    else:
        affinity = complex_spec.affinity(ligand)

    print("Affinity prediction (KRAS G12D + MRTX-1133):")
    if affinity is not None:
        print(f"- ic50: {affinity.ic50}")
        print(f"- binding_probability: {affinity.binding_probability}")
    else:
        print("- affinity: None")

    if args.output:
        output_path = args.output.expanduser()
        output_format = _infer_format(output_path, args.format)
        if output_format == "bcif":
            output_path.write_bytes(result.to_bcif())
        else:
            output_path.write_text(result.to_mmcif(), encoding="utf-8")
        print(f"Saved structure: {output_path} ({output_format})")


if __name__ == "__main__":
    main()

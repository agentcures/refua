"""Minimal antibody design example using the unified API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from refua import Binder, Complex


def _resolve_sequence(explicit: Optional[str], length: int, label: str) -> str:
    """Return an explicit sequence or a numeric design token string.

    Numeric tokens (e.g., "120") denote designed residues.
    """
    if explicit:
        return explicit
    if length <= 0:
        msg = f"{label} length must be positive."
        raise ValueError(msg)
    return str(length)


def build_design(args: argparse.Namespace):
    """Create a design for a simple antibody/antigen setup."""
    antigen_path = Path(args.antigen).expanduser().resolve()
    design = Complex(name=args.name, base_dir=antigen_path.parent).file(
        antigen_path,
        include=[{"chain": {"id": args.antigen_chain}}],
        binding_types=(
            [{"chain": {"id": args.antigen_chain, "binding": args.binding_range}}]
            if args.binding_range
            else None
        ),
    )

    heavy_seq = _resolve_sequence(args.heavy_seq, args.heavy_length, "Heavy chain")
    light_seq = _resolve_sequence(args.light_seq, args.light_length, "Light chain")

    design.add(
        Binder(heavy_seq, ids=args.heavy_id),
        Binder(light_seq, ids=args.light_id),
    )
    return design


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Prepare antibody design inputs with the unified API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--antigen",
        required=True,
        help="Path to the antigen structure file (e.g., CIF).",
    )
    parser.add_argument(
        "--antigen-chain",
        default="A",
        help="Antigen chain id to include in the design target.",
    )
    parser.add_argument(
        "--binding-range",
        default="10..40",
        help="Residue range to mark as binding site (set to '' to omit).",
    )
    parser.add_argument("--heavy-id", default="H", help="Heavy chain id.")
    parser.add_argument("--light-id", default="L", help="Light chain id.")
    parser.add_argument(
        "--heavy-length",
        type=int,
        default=120,
        help="Number of designed residues for the heavy chain.",
    )
    parser.add_argument(
        "--light-length",
        type=int,
        default=110,
        help="Number of designed residues for the light chain.",
    )
    parser.add_argument(
        "--heavy-seq",
        default=None,
        help="Optional explicit heavy chain sequence (overrides --heavy-length).",
    )
    parser.add_argument(
        "--light-seq",
        default=None,
        help="Optional explicit light chain sequence (overrides --light-length).",
    )
    parser.add_argument(
        "--name",
        default="antibody_design",
        help="Spec name for the design job.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief feature summary."""
    args = parse_args()
    design = build_design(args)
    result = design.fold()
    features = result.features

    print("Prepared antibody design inputs:")
    if features is not None:
        print(f"- feature_keys: {sorted(features.keys())}")
    else:
        print("- features: None")


if __name__ == "__main__":
    main()

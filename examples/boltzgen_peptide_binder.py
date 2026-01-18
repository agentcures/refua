"""Peptide binder design example using the simple BoltzGen API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from refua import BoltzGen


def build_design(gen: BoltzGen, args: argparse.Namespace):
    """Create a BoltzGen design for a peptide binder against a target structure."""
    target_path = Path(args.target).expanduser().resolve()
    design = gen.design(args.name, base_dir=target_path.parent).file(
        target_path,
        include=[{"chain": {"id": args.target_chain}}],
        binding_types=(
            [{"chain": {"id": args.target_chain, "binding": args.binding_range}}]
            if args.binding_range
            else None
        ),
    )

    peptide_kwargs: dict[str, Optional[str]] = {}
    if args.peptide_ss:
        peptide_kwargs["secondary_structure"] = args.peptide_ss

    design.protein(
        args.peptide_id,
        args.peptide_spec,
        cyclic=args.peptide_cyclic,
        **peptide_kwargs,
    )
    return design


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Prepare BoltzGen model inputs for a peptide binder design spec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the target structure file (e.g., CIF).",
    )
    parser.add_argument(
        "--target-chain",
        default="A",
        help="Target chain id to include in the design target.",
    )
    parser.add_argument(
        "--binding-range",
        default="10..40",
        help="Residue range to mark as binding site (set to '' to omit).",
    )
    parser.add_argument(
        "--peptide-id",
        default="P",
        help="Peptide chain id.",
    )
    parser.add_argument(
        "--peptide-spec",
        default="12",
        help=(
            "Peptide sequence with optional design tokens, e.g. '10C6C3' or '12'."
        ),
    )
    parser.add_argument(
        "--peptide-cyclic",
        action="store_true",
        help="Mark the peptide as cyclic in the schema.",
    )
    parser.add_argument(
        "--peptide-ss",
        default=None,
        help="Optional secondary structure string (u/l/h/s).",
    )
    parser.add_argument("--name", default="peptide_binder", help="Spec name.")
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief feature summary."""
    args = parse_args()
    gen = BoltzGen()
    design = build_design(gen, args)
    features = design.to_features()

    print("Prepared peptide binder design inputs:")
    print(f"- feature_keys: {sorted(features.keys())}")


if __name__ == "__main__":
    main()

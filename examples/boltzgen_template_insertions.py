"""Template-based design with structure groups and insertions using BoltzGen."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from refua import BoltzGen


def _structure_groups(chain_id: str, primary: str, secondary: str) -> Optional[list[dict]]:
    """Build structure group specs if any ranges are provided."""
    groups = []
    if primary.strip():
        groups.append(
            {"group": {"visibility": 1, "id": chain_id, "res_index": primary}}
        )
    if secondary.strip():
        groups.append(
            {"group": {"visibility": 2, "id": chain_id, "res_index": secondary}}
        )
    return groups or None


def _design_insertions(
    chain_id: str,
    position: Optional[int],
    length_spec: str,
    ss: str,
) -> Optional[list[dict]]:
    """Build design insertion specs if a position is provided."""
    if position is None or position <= 0:
        return None
    insertion: dict[str, object] = {
        "id": chain_id,
        "res_index": position,
        "num_residues": length_spec,
    }
    if ss.strip():
        insertion["secondary_structure"] = ss
    return [{"insertion": insertion}]


def build_design(gen: BoltzGen, args: argparse.Namespace):
    """Create a BoltzGen design with template structure groups and insertions."""
    template_path = Path(args.template).expanduser().resolve()

    structure_groups = _structure_groups(
        args.template_chain,
        args.structure_group_primary,
        args.structure_group_secondary,
    )
    insertion_site = None if args.no_insertions else args.insertion_site
    insertions = _design_insertions(
        args.template_chain,
        insertion_site,
        args.insertion_length,
        args.insertion_ss,
    )

    file_kwargs = {
        "include": [{"chain": {"id": args.template_chain}}],
        "structure_groups": structure_groups,
        "design_insertions": insertions,
    }

    if args.design_range.strip():
        file_kwargs["design"] = [
            {"chain": {"id": args.template_chain, "res_index": args.design_range}}
        ]

    if args.secondary_structure.strip():
        file_kwargs["secondary_structure"] = [
            {
                "chain": {
                    "id": args.template_chain,
                    "helix": args.secondary_structure,
                }
            }
        ]

    design = gen.design(args.name, base_dir=template_path.parent).file(
        template_path, **file_kwargs
    )

    if args.binder_spec.strip():
        binder_kwargs = {}
        if args.binder_ss.strip():
            binder_kwargs["secondary_structure"] = args.binder_ss
        design.protein(
            args.binder_chain,
            args.binder_spec,
            cyclic=args.binder_cyclic,
            **binder_kwargs,
        )
    return design


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Prepare BoltzGen model inputs with structure groups and insertions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to the template structure file (e.g., CIF).",
    )
    parser.add_argument(
        "--template-chain",
        default="A",
        help="Template chain id to include.",
    )
    parser.add_argument(
        "--structure-group-primary",
        default="5..15",
        help="Primary structure group range (visibility=1).",
    )
    parser.add_argument(
        "--structure-group-secondary",
        default="20..25",
        help="Secondary structure group range (visibility=2).",
    )
    parser.add_argument(
        "--design-range",
        default="14..19",
        help="Designable range on the template chain (empty to skip).",
    )
    parser.add_argument(
        "--secondary-structure",
        default="14..17",
        help="Secondary structure range (helix) for the template chain.",
    )
    parser.add_argument(
        "--insertion-site",
        type=int,
        default=12,
        help="Residue index for design insertions.",
    )
    parser.add_argument(
        "--no-insertions",
        action="store_true",
        help="Skip adding design insertions to the template.",
    )
    parser.add_argument(
        "--insertion-length",
        default="2..5",
        help="Number of residues to insert (range or integer string).",
    )
    parser.add_argument(
        "--insertion-ss",
        default="HELIX",
        help="Secondary structure tag for inserted residues.",
    )
    parser.add_argument(
        "--binder-chain",
        default="P",
        help="Optional binder chain id to add.",
    )
    parser.add_argument(
        "--binder-spec",
        default="10C6C3",
        help="Optional binder sequence spec (empty to skip).",
    )
    parser.add_argument(
        "--binder-ss",
        default="",
        help="Optional binder secondary structure string (u/l/h/s).",
    )
    parser.add_argument(
        "--binder-cyclic",
        action="store_true",
        help="Mark the binder as cyclic.",
    )
    parser.add_argument("--name", default="template_insertions", help="Spec name.")
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief feature summary."""
    args = parse_args()
    gen = BoltzGen()
    design = build_design(gen, args)
    features = design.to_features()

    print("Prepared template insertion inputs:")
    print(f"- feature_keys: {sorted(features.keys())}")


if __name__ == "__main__":
    main()

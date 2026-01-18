"""Template-driven design with masks and structure groups using BoltzGen."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from refua import BoltzGen


def _structure_groups(chain_id: str, primary: str, secondary: str, masked: str) -> Optional[list[dict]]:
    """Build structure group specs when ranges are provided."""
    groups = []
    if primary.strip():
        groups.append(
            {"group": {"visibility": 1, "id": chain_id, "res_index": primary}}
        )
    if secondary.strip():
        groups.append(
            {"group": {"visibility": 2, "id": chain_id, "res_index": secondary}}
        )
    if masked.strip():
        groups.append(
            {"group": {"visibility": 0, "id": chain_id, "res_index": masked}}
        )
    return groups or None


def _secondary_structure(chain_id: str, helix: str, sheet: str, loop: str) -> Optional[list[dict]]:
    """Build secondary structure specs for a template chain."""
    spec: dict[str, object] = {"id": chain_id}
    if helix.strip():
        spec["helix"] = helix
    if sheet.strip():
        spec["sheet"] = sheet
    if loop.strip():
        spec["loop"] = loop
    if len(spec) == 1:
        return None
    return [{"chain": spec}]


def build_design(gen: BoltzGen, args: argparse.Namespace):
    """Create a BoltzGen design with template masks and structure groups."""
    template_path = Path(args.template).expanduser().resolve()

    file_kwargs = {
        "include": [{"chain": {"id": args.template_chain}}],
        "structure_groups": _structure_groups(
            args.template_chain,
            args.structure_group_primary,
            args.structure_group_secondary,
            args.structure_group_masked,
        ),
        "secondary_structure": _secondary_structure(
            args.template_chain,
            args.helix_range,
            args.sheet_range,
            args.loop_range,
        ),
    }

    if args.include_range.strip():
        file_kwargs["include"] = [
            {"chain": {"id": args.template_chain, "res_index": args.include_range}}
        ]

    if args.exclude_range.strip():
        file_kwargs["exclude"] = [
            {"chain": {"id": args.template_chain, "res_index": args.exclude_range}}
        ]

    if args.design_range.strip():
        file_kwargs["design"] = [
            {"chain": {"id": args.template_chain, "res_index": args.design_range}}
        ]

    if args.not_design_range.strip():
        file_kwargs["not_design"] = [
            {"chain": {"id": args.template_chain, "res_index": args.not_design_range}}
        ]

    if args.binding_range.strip():
        file_kwargs["binding_types"] = [
            {"chain": {"id": args.template_chain, "binding": args.binding_range}}
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
        description="Prepare BoltzGen model inputs with template design masks.",
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
        "--include-range",
        default="",
        help="Optional include range (e.g., '1..120').",
    )
    parser.add_argument(
        "--exclude-range",
        default="",
        help="Optional exclude range (e.g., '1..5').",
    )
    parser.add_argument(
        "--design-range",
        default="10..30",
        help="Residues to design on the template chain.",
    )
    parser.add_argument(
        "--not-design-range",
        default="1..5",
        help="Residues to lock as not-designed on the template chain.",
    )
    parser.add_argument(
        "--binding-range",
        default="12..18",
        help="Residues to mark as binding sites.",
    )
    parser.add_argument(
        "--structure-group-primary",
        default="6..15",
        help="Primary structure group (visibility=1).",
    )
    parser.add_argument(
        "--structure-group-secondary",
        default="20..25",
        help="Secondary structure group (visibility=2).",
    )
    parser.add_argument(
        "--structure-group-masked",
        default="30..35",
        help="Masked structure group (visibility=0).",
    )
    parser.add_argument(
        "--helix-range",
        default="10..14",
        help="Helix range for secondary structure conditioning.",
    )
    parser.add_argument(
        "--sheet-range",
        default="",
        help="Sheet range for secondary structure conditioning.",
    )
    parser.add_argument(
        "--loop-range",
        default="",
        help="Loop range for secondary structure conditioning.",
    )
    parser.add_argument(
        "--binder-chain",
        default="P",
        help="Optional binder chain id to add.",
    )
    parser.add_argument(
        "--binder-spec",
        default="12",
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
    parser.add_argument("--name", default="template_masks", help="Spec name.")
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief feature summary."""
    args = parse_args()
    gen = BoltzGen()
    design = build_design(gen, args)
    features = design.to_features()

    print("Prepared template mask inputs:")
    print(f"- feature_keys: {sorted(features.keys())}")


if __name__ == "__main__":
    main()

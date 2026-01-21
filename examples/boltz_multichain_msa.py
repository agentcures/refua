"""Multi-chain complex with in-memory MSAs and constraints using the unified API."""

from __future__ import annotations

import argparse
from typing import Tuple

from refua import Complex, Protein, SM
from refua.boltz.api import msa_from_a3m


def _simple_a3m(sequence: str) -> str:
    """Return a minimal deterministic A3M string for a sequence."""
    if not sequence:
        raise ValueError("Sequence must be non-empty.")

    first = "A" if sequence[0] != "A" else "G"
    last = "G" if sequence[-1] != "G" else "A"
    variant_1 = first + sequence[1:]
    variant_2 = sequence[:-1] + last

    return (
        f">query\n{sequence}\n"
        f">variant_1\n{variant_1}\n"
        f">variant_2\n{variant_2}\n"
    )


def _parse_pair(value: str, label: str) -> Tuple[int, int]:
    """Parse two comma-separated integers for contact specs."""
    try:
        left, right = value.split(",", 1)
        return int(left.strip()), int(right.strip())
    except ValueError as exc:
        msg = f"{label} must be two integers separated by a comma."
        raise ValueError(msg) from exc


def build_complex(args: argparse.Namespace):
    """Create a complex with in-memory MSAs and constraints."""
    sequence_c = args.sequence_c.strip()

    msa_a = msa_from_a3m(_simple_a3m(args.sequence_a))
    if args.share_msa and args.sequence_a == args.sequence_b:
        msa_b = msa_a
    else:
        msa_b = msa_from_a3m(_simple_a3m(args.sequence_b))

    complex_spec = Complex(name=args.name).add(
        Protein(args.sequence_a, ids=args.chain_a, msa=msa_a)
    )
    complex_spec.add(Protein(args.sequence_b, ids=args.chain_b, msa=msa_b))

    if sequence_c:
        msa_c = msa_from_a3m(_simple_a3m(sequence_c))
        complex_spec.add(Protein(sequence_c, ids=args.chain_c, msa=msa_c))

    ligand = SM(args.ligand_smiles)
    complex_spec.add(ligand)

    pocket_a, pocket_b = _parse_pair(args.pocket_contacts, "--pocket-contacts")
    complex_spec.pocket(
        ligand,
        contacts=[(args.chain_a, pocket_a), (args.chain_b, pocket_b)],
        max_distance=args.pocket_distance,
        force=args.pocket_force,
    )

    contact_a, contact_b = _parse_pair(args.contact_ab, "--contact-ab")
    complex_spec.contact(
        (args.chain_a, contact_a),
        (args.chain_b, contact_b),
        max_distance=args.contact_distance,
    )

    if sequence_c and args.contact_ac:
        contact_a, contact_c = _parse_pair(args.contact_ac, "--contact-ac")
        complex_spec.contact(
            (args.chain_a, contact_a),
            (args.chain_c, contact_c),
            max_distance=args.contact_distance,
        )

    if args.affinity:
        complex_spec.request_affinity(ligand)

    return complex_spec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Predict a multi-chain complex with MSAs using the unified API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default="multichain_msa", help="Spec name.")
    parser.add_argument("--chain-a", default="A", help="Chain id for protein A.")
    parser.add_argument("--chain-b", default="B", help="Chain id for protein B.")
    parser.add_argument("--chain-c", default="C", help="Chain id for protein C.")
    parser.add_argument(
        "--sequence-a",
        default="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        help="Sequence for protein A.",
    )
    parser.add_argument(
        "--sequence-b",
        default="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        help="Sequence for protein B.",
    )
    parser.add_argument(
        "--sequence-c",
        default="",
        help="Optional sequence for protein C.",
    )
    parser.add_argument(
        "--share-msa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share the same MSA object for A/B if sequences match.",
    )
    parser.add_argument(
        "--ligand-smiles",
        default="CCO",
        help="Ligand SMILES string.",
    )
    parser.add_argument(
        "--pocket-contacts",
        default="8,12",
        help="Residues for pocket contacts on A,B (e.g. '8,12').",
    )
    parser.add_argument(
        "--pocket-distance",
        type=float,
        default=6.0,
        help="Max distance for pocket contacts (Angstrom).",
    )
    parser.add_argument(
        "--pocket-force",
        action="store_true",
        help="Force the pocket constraint to be applied.",
    )
    parser.add_argument(
        "--contact-ab",
        default="10,10",
        help="Residues for A/B contact constraint (e.g. '10,10').",
    )
    parser.add_argument(
        "--contact-ac",
        default="",
        help="Optional residues for A/C contact constraint (e.g. '5,7').",
    )
    parser.add_argument(
        "--contact-distance",
        type=float,
        default=6.0,
        help="Max distance for contact constraints (Angstrom).",
    )
    parser.add_argument(
        "--affinity",
        action="store_true",
        help="Request affinity features for the ligand.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief prediction summary."""
    args = parse_args()
    complex_spec = build_complex(args)
    result = complex_spec.fold()
    mmcif = result.to_mmcif()

    print("Predicted multichain MSA complex:")
    print(f"- mmcif_chars: {len(mmcif)}")
    if args.affinity:
        affinity = result.affinity
        print(f"- ic50: {affinity.ic50}")
        print(f"- binding_probability: {affinity.binding_probability}")


if __name__ == "__main__":
    main()

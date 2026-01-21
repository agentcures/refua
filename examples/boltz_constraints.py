"""Protein-ligand complex with pocket/contact constraints using the unified API."""

from __future__ import annotations

import argparse
from typing import Tuple, Union

from refua import Complex, Protein, SM
from refua.boltz.api import msa_from_a3m


TokenRef = Tuple[str, Union[int, str]]


def _parse_token_ref(value: str) -> TokenRef:
    """Parse a token reference of the form CHAIN:RES into a tuple."""
    try:
        chain, token = value.split(":", 1)
    except ValueError as exc:
        msg = f"Invalid token reference '{value}', expected CHAIN:RES."
        raise ValueError(msg) from exc

    token = token.strip()
    if token.isdigit():
        return (chain.strip(), int(token))
    return (chain.strip(), token)


def _parse_token_refs(values: str) -> list[TokenRef]:
    """Parse a comma-separated list of token references."""
    return [_parse_token_ref(item.strip()) for item in values.split(",") if item.strip()]


def _parse_contact(value: str) -> tuple[TokenRef, TokenRef]:
    """Parse a contact specification of the form CHAIN:RES,CHAIN:RES."""
    parts = _parse_token_refs(value)
    if len(parts) != 2:
        msg = f"Contact '{value}' must specify exactly two tokens."
        raise ValueError(msg)
    return parts[0], parts[1]


def _build_demo_msa(sequence: str) -> str:
    """Return a minimal A3M string suitable for msa_from_a3m()."""
    return f">query\n{sequence}\n>alt\n{sequence}\n"


def build_complex(args: argparse.Namespace):
    """Create a complex with optional constraints and affinity request."""
    msa = None
    if args.use_msa:
        msa = msa_from_a3m(_build_demo_msa(args.protein_sequence))

    complex_spec = Complex(name=args.name).add(
        Protein(args.protein_sequence, ids=args.protein_id, msa=msa)
    )
    if args.partner_sequence:
        complex_spec.add(
            Protein(args.partner_sequence, ids=args.partner_id)
        )

    ligand = SM(args.ligand_smiles)
    complex_spec.add(ligand)

    if args.pocket_contacts:
        contacts = _parse_token_refs(args.pocket_contacts)
        complex_spec.pocket(
            ligand,
            contacts=contacts,
            max_distance=args.pocket_distance,
            force=args.pocket_force,
        )

    if args.contact:
        token1, token2 = _parse_contact(args.contact)
        complex_spec.contact(token1, token2, max_distance=args.contact_distance)

    if args.affinity:
        complex_spec.request_affinity(ligand)

    return complex_spec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Predict a constrained protein-ligand complex with the unified API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default="boltz_constraints", help="Spec name.")
    parser.add_argument("--protein-id", default="A", help="Primary protein chain id.")
    parser.add_argument(
        "--protein-sequence",
        default="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        help="Primary protein sequence.",
    )
    parser.add_argument(
        "--partner-id",
        default="B",
        help="Optional partner protein chain id.",
    )
    parser.add_argument(
        "--partner-sequence",
        default="",
        help="Optional partner protein sequence.",
    )
    parser.add_argument(
        "--ligand-smiles",
        default="CCO",
        help="Ligand SMILES string.",
    )
    parser.add_argument(
        "--use-msa",
        action="store_true",
        help="Attach a tiny in-memory MSA to the primary protein.",
    )
    parser.add_argument(
        "--pocket-contacts",
        default="A:5,A:8",
        help="Comma-separated pocket contacts (CHAIN:RES).",
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
        "--contact",
        default="",
        help="Optional contact constraint (CHAIN:RES,CHAIN:RES).",
    )
    parser.add_argument(
        "--contact-distance",
        type=float,
        default=6.0,
        help="Max distance for contact constraint (Angstrom).",
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

    print("Predicted constrained complex:")
    print(f"- mmcif_chars: {len(mmcif)}")
    if args.affinity:
        affinity = result.affinity
        print(f"- ic50: {affinity.ic50}")
        print(f"- binding_probability: {affinity.binding_probability}")


if __name__ == "__main__":
    main()

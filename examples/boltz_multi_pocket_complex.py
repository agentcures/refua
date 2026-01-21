"""Multi-ligand complex with pocket and contact constraints using the unified API."""

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


def _attach_msa(sequence: str) -> object:
    """Build an in-memory MSA for a sequence."""
    return msa_from_a3m(_simple_a3m(sequence))


def build_complex(args: argparse.Namespace):
    """Create a complex with multiple ligands and constraints."""
    msa_a = None
    msa_b = None
    if args.use_msa:
        msa_a = _attach_msa(args.sequence_a)
        if args.share_msa and args.sequence_a == args.sequence_b:
            msa_b = msa_a
        else:
            msa_b = _attach_msa(args.sequence_b)

    complex_spec = Complex(name=args.name).add(
        Protein(args.sequence_a, ids=args.chain_a, msa=msa_a)
    )
    complex_spec.add(Protein(args.sequence_b, ids=args.chain_b, msa=msa_b))

    if args.sequence_c:
        msa_c = _attach_msa(args.sequence_c) if args.use_msa else None
        complex_spec.add(Protein(args.sequence_c, ids=args.chain_c, msa=msa_c))

    ligand_1 = SM(args.ligand_1_smiles)
    complex_spec.add(ligand_1)
    complex_spec.pocket(
        ligand_1,
        contacts=_parse_token_refs(args.ligand_1_contacts),
        max_distance=args.pocket_distance,
        force=args.pocket_force,
    )

    if args.ligand_2_smiles:
        ligand_2 = SM(args.ligand_2_smiles)
        complex_spec.add(ligand_2)
        if args.ligand_2_contacts:
            complex_spec.pocket(
                ligand_2,
                contacts=_parse_token_refs(args.ligand_2_contacts),
                max_distance=args.pocket_distance,
                force=args.pocket_force,
            )

    for contact in args.contact:
        token1, token2 = _parse_contact(contact)
        complex_spec.contact(token1, token2, max_distance=args.contact_distance)

    if args.affinity:
        complex_spec.request_affinity(ligand_1)

    return complex_spec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Predict a multi-ligand constrained complex with the unified API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default="multi_pocket_complex", help="Spec name.")
    parser.add_argument("--chain-a", default="A", help="Chain id for protein A.")
    parser.add_argument("--chain-b", default="B", help="Chain id for protein B.")
    parser.add_argument("--chain-c", default="C", help="Optional chain id for protein C.")
    parser.add_argument(
        "--sequence-a",
        default="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        help="Sequence for protein A.",
    )
    parser.add_argument(
        "--sequence-b",
        default="GHHHHHHMKTAYIAKQRQISFVKSHF",
        help="Sequence for protein B.",
    )
    parser.add_argument(
        "--sequence-c",
        default="",
        help="Optional sequence for protein C.",
    )
    parser.add_argument(
        "--use-msa",
        action="store_true",
        help="Attach a small in-memory MSA to the protein chains.",
    )
    parser.add_argument(
        "--share-msa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share the same MSA object for A/B if sequences match.",
    )
    parser.add_argument(
        "--ligand-1-smiles",
        default="CCO",
        help="Primary ligand SMILES string.",
    )
    parser.add_argument(
        "--ligand-1-contacts",
        default="A:8,B:12",
        help="Pocket contacts for ligand 1 (CHAIN:RES).",
    )
    parser.add_argument(
        "--ligand-2-smiles",
        default="",
        help="Secondary ligand SMILES string (empty to skip).",
    )
    parser.add_argument(
        "--ligand-2-contacts",
        default="",
        help="Pocket contacts for ligand 2 (CHAIN:RES).",
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
        help="Force pocket constraints to be applied.",
    )
    parser.add_argument(
        "--contact",
        action="append",
        default=[],
        help="Contact constraint (CHAIN:RES,CHAIN:RES). Can be repeated.",
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
        help="Request affinity features for ligand 1.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief prediction summary."""
    args = parse_args()
    complex_spec = build_complex(args)
    result = complex_spec.fold()
    mmcif = result.to_mmcif()

    print("Predicted multi-ligand complex:")
    print(f"- mmcif_chars: {len(mmcif)}")
    if args.affinity:
        affinity = result.affinity
        print(f"- ic50: {affinity.ic50}")
        print(f"- binding_probability: {affinity.binding_probability}")


if __name__ == "__main__":
    main()

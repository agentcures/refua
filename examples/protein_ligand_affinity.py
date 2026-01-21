"""Protein-ligand affinity example using the unified API."""

from __future__ import annotations

import argparse

from refua import Complex, Protein, SM


def build_complex(args: argparse.Namespace):
    """Create a protein-ligand complex for affinity prediction."""
    ligand = SM(args.ligand_smiles)
    complex_spec = Complex(
        [Protein(args.protein_sequence, ids=args.protein_id), ligand],
        name=args.name,
    )
    return complex_spec, ligand


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Run a protein-ligand affinity prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--protein-id", default="A", help="Protein chain id for the complex."
    )
    parser.add_argument(
        "--protein-sequence",
        default="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        help="Protein sequence (single-letter amino acids).",
    )
    parser.add_argument(
        "--ligand-smiles",
        default="CCO",
        help="Ligand SMILES string.",
    )
    parser.add_argument(
        "--name",
        default="protein_ligand_affinity",
        help="Spec name for the affinity request.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief affinity summary."""
    args = parse_args()
    complex_spec, ligand = build_complex(args)
    affinity = complex_spec.affinity(ligand)

    print("Protein-ligand affinity:")
    print(f"- ic50: {affinity.ic50}")
    print(f"- binding_probability: {affinity.binding_probability}")


if __name__ == "__main__":
    main()

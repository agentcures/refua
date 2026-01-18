"""Protein-ligand affinity example using the simple Boltz2 API."""

from __future__ import annotations

import argparse

from refua import Boltz2


def build_complex(model: Boltz2, args: argparse.Namespace):
    """Create a protein-ligand complex for affinity prediction."""
    return (
        model.fold_complex(args.name)
        .protein(args.protein_id, args.protein_sequence)
        .ligand(args.ligand_id, args.ligand_smiles)
        .request_affinity(args.ligand_id)
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description="Run a Boltz2 protein-ligand affinity prediction.",
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
    parser.add_argument("--ligand-id", default="L", help="Ligand chain id.")
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
    model = Boltz2()
    complex_spec = build_complex(model, args)
    affinity = complex_spec.get_affinity()

    print("Protein-ligand affinity:")
    print(f"- ic50: {affinity.ic50}")
    print(f"- binding_probability: {affinity.binding_probability}")


if __name__ == "__main__":
    main()

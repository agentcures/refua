"""Official demo-style protein-ligand pair using the unified API."""

from __future__ import annotations

import argparse

from refua import Complex, Protein, SM


PROTEIN_NAME = "P11229"
LIGAND_NAME = "TS-10004"
PROTEIN_SEQUENCE = (
    "MNTSAPPAVSPNITVLAPGKGPWQVAFIGITTGLLSLATVTGNLLVLISFKVNTELKTVNNYFLLSLACADLIIGTFSMN"
    "LYTTYLLMGHWALGTLACDLWLALDYVASNASVMNLLLISFDRYFSVTRPLSYRAKRTPRRAALMIGLAWLVSFVLWAPA"
    "ILFWQYLVGERTVLAGQCYIQFLSQPIITFGTAMAAFYLPVTVMCTLYWRIYRETENRARELAALQGSETPGKGGGSSSS"
    "SERSQPGAEGSPETPPGRCCRCCRAPRLLQAYSWKEEEEEDEGSMESLTSSEGEEPGSEVVIKMPMVDPEAQAPTKQPPR"
    "SSPNTVKRPTKKGRDRAGKGQKPRGKEQLAKRKTFSLVKEKKAARTLSAILLAFILTWTPYNIMVLVSTFCKDCVPETLW"
    "ELGYWLCYVNSTINPMCYALCNKAFRDTFRLLLLCRWDKRRWRKIPKRPGSVHRTPSRQC"
)
LIGAND_SMILES = "C#CCN1CCC[C@@H]1COC(=O)c1cnn(CC)c1"


def build_complex(args: argparse.Namespace):
    """Create the demo complex and return the ligand handle."""
    ligand = SM(args.ligand_smiles)
    complex_spec = Complex(
        [Protein(args.protein_sequence, ids=args.protein_id), ligand],
        name=args.name,
    )
    return complex_spec, ligand


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description=(
            f"Run the official demo-style {PROTEIN_NAME} "
            f"and {LIGAND_NAME} example."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protein-id", default="A", help="Protein chain id.")
    parser.add_argument(
        "--protein-sequence",
        default=PROTEIN_SEQUENCE,
        help="Protein sequence (single-letter amino acids).",
    )
    parser.add_argument(
        "--ligand-smiles",
        default=LIGAND_SMILES,
        help="Ligand SMILES string.",
    )
    parser.add_argument(
        "--name",
        default="official_demo_pair",
        help="Spec name for the prediction.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief affinity summary."""
    args = parse_args()
    complex_spec, ligand = build_complex(args)
    affinity = complex_spec.affinity(ligand)

    print(f"Affinity prediction ({PROTEIN_NAME}-{LIGAND_NAME}):")
    print(f"- ic50: {affinity.ic50}")
    print(f"- binding_probability: {affinity.binding_probability}")


if __name__ == "__main__":
    main()

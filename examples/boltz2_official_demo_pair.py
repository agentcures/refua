"""Official demo-style protein-ligand pair using the Boltz2 API."""

from __future__ import annotations

import argparse

from refua import Boltz2


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


def build_complex(model: Boltz2, args: argparse.Namespace):
    """Create the demo complex and request affinity."""
    return (
        model.fold_complex(args.name)
        .protein(args.protein_id, args.protein_sequence)
        .ligand(args.ligand_id, args.ligand_smiles)
        .request_affinity(args.ligand_id)
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example."""
    parser = argparse.ArgumentParser(
        description=(
            f"Run the official demo-style Boltz2 {PROTEIN_NAME} "
            f"and {LIGAND_NAME} example."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protein-id", default="A", help="Protein chain id.")
    parser.add_argument("--ligand-id", default="B", help="Ligand chain id.")
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
    parser.add_argument(
        "--use-kernels",
        action="store_true",
        help="Enable cuEquivariance kernels if available.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example and print a brief affinity summary."""
    args = parse_args()
    model = Boltz2(use_kernels=args.use_kernels)
    complex_spec = build_complex(model, args)
    affinity = complex_spec.get_affinity()

    print(f"Affinity prediction ({PROTEIN_NAME}-{LIGAND_NAME}):")
    print(f"- ic50: {affinity.ic50}")
    print(f"- binding_probability: {affinity.binding_probability}")


if __name__ == "__main__":
    main()

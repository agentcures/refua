from refua import Boltz2


def main() -> None:
    model = Boltz2()
    complex_spec = (
        model.fold_complex("simple_complex")
        .protein("A", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
        .ligand("L", "CCO")
    )

    bcif_bytes = complex_spec.to_bcif()
    print("BCIF bytes:", len(bcif_bytes))


if __name__ == "__main__":
    main()

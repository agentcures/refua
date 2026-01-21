from refua import Complex, Protein, SM


def main() -> None:
    complex_spec = Complex(
        [
            Protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", ids="A"),
            SM("CCO"),
        ],
        name="simple_complex",
    )

    result = complex_spec.fold()
    bcif_bytes = result.to_bcif()
    print("BCIF bytes:", len(bcif_bytes))


if __name__ == "__main__":
    main()

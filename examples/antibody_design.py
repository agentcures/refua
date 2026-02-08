"""Antibody-first design example using the unified API."""

from __future__ import annotations

import argparse

from refua import Binder, BinderDesigns, Complex, Protein

RBD_SEQUENCE = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"


def _parse_cdr_lengths(value: str, label: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{label} must contain exactly three comma-separated values.")
    lengths = tuple(int(part) for part in parts)
    if any(length < 1 for length in lengths):
        raise ValueError(f"{label} values must be >= 1.")
    cdr1, cdr2, cdr3 = lengths
    return cdr1, cdr2, cdr3


def build_design(args: argparse.Namespace) -> Complex:
    heavy_cdr_lengths = _parse_cdr_lengths(args.heavy_cdr_lengths, "heavy-cdr-lengths")
    light_cdr_lengths = _parse_cdr_lengths(args.light_cdr_lengths, "light-cdr-lengths")
    epitope = args.epitope.strip()
    if not epitope:
        raise ValueError("epitope must be a non-empty residue range.")
    if args.heavy is not None and not args.heavy.strip():
        raise ValueError("heavy binder spec must be non-empty.")
    if args.light is not None and not args.light.strip():
        raise ValueError("light binder spec must be non-empty.")

    antigen = Protein(
        args.antigen_sequence,
        ids=args.antigen_id,
        binding_types={"binding": epitope},
    )
    pair = BinderDesigns.antibody(
        heavy_cdr_lengths=heavy_cdr_lengths,
        light_cdr_lengths=light_cdr_lengths,
        heavy_id=args.heavy_id,
        light_id=args.light_id,
    )
    heavy = (
        pair.heavy
        if args.heavy is None
        else Binder(args.heavy.strip(), ids=args.heavy_id)
    )
    light = (
        pair.light
        if args.light is None
        else Binder(args.light.strip(), ids=args.light_id)
    )
    return Complex([antigen, heavy, light], name=args.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare antibody design inputs with the unified API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--antigen-sequence",
        default=RBD_SEQUENCE,
        help="Antigen amino-acid sequence.",
    )
    parser.add_argument(
        "--epitope",
        default="118..190",
        help="Antigen residue range to mark as binding hotspot.",
    )
    parser.add_argument(
        "--heavy",
        default=None,
        help="Optional heavy-chain binder spec (overrides --heavy-cdr-lengths).",
    )
    parser.add_argument(
        "--light",
        default=None,
        help="Optional light-chain binder spec (overrides --light-cdr-lengths).",
    )
    parser.add_argument(
        "--heavy-cdr-lengths",
        default="12,10,14",
        help="CDR-H1,H2,H3 design lengths.",
    )
    parser.add_argument(
        "--light-cdr-lengths",
        default="10,9,9",
        help="CDR-L1,L2,L3 design lengths.",
    )
    parser.add_argument("--antigen-id", default="A", help="Antigen chain id.")
    parser.add_argument("--heavy-id", default="H", help="Heavy chain id.")
    parser.add_argument("--light-id", default="L", help="Light chain id.")
    parser.add_argument("--name", default="antibody_design", help="Spec name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = build_design(args)
    result = design.fold()
    features = result.features

    print("Prepared antibody design inputs:")
    print(f"- binder_specs: {result.binder_specs}")
    print(f"- chain_design_summary: {result.chain_design_summary()}")
    if features is not None:
        print(f"- feature_keys: {sorted(features.keys())}")
    else:
        print("- features: None")


if __name__ == "__main__":
    main()

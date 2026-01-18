# Refua

Refua is a general drug discovery ML toolkit that brings together structure prediction, affinity modeling, and generative design in one package. It unifies the Boltz inference stack with the BoltzGen design pipeline so you can move from target definition to candidate generation using a shared, programmatic API.

## Why use Refua?

- **Fluent Python API:** Build specs, run predictions, and post-process results with a readable, composable
  API—no need to stitch together CLI calls, YAML, or intermediate files.
- **No per-call model reload:** Keep a predictor/model instance alive and run many inferences in a loop
  without paying initialization and checkpoint-loading cost each time.
- **All-in-memory workflows:** Prepare inputs, run inference, and perform analysis entirely in memory
  (e.g., passing objects/arrays between steps) without writing temporary files to disk.
- **Improved build:** We are able to support more up to date dependencies. Over time we expect to widen this support.

## Quickstart

Install:

```bash
pip install refua
```

Install with NVIDIA GPU support:

```bash
pip install "refua[cuda]"
```

Boltz-2 API:

```python
from pathlib import Path

from refua import Boltz2

model = Boltz2()

complex_spec = (
    model.fold_complex()
    .protein("A", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
    .ligand("L", "CCO")
)

Path("complex.bcif").write_bytes(complex_spec.to_bcif())
affinity = complex_spec.get_affinity()
print(affinity.ic50, affinity.binding_probability)
```

BoltzGen API:

```python
from refua import BoltzGen

gen = BoltzGen()
design = (
    gen.design("simple_design")
    .protein("A", "12")
    .total_length(8, 20)
)

features = design.to_features()
print(sorted(features.keys())[:6])
```

BoltzGen defaults to the bundled molecule library in the Hugging Face cache. Set `BOLTZGEN_MOLDIR` or pass `mol_dir` to override.

CLI entrypoints are still available:

```bash
boltz --help
boltzgen --help
```

## Documentation

- Boltz docs live in `docs/boltz`.
- BoltzGen docs live in `docs/boltzgen`.
- API reference source lives in `docs/api` (build with `sphinx-build -b html docs/api docs/api/_build/html`).

## Examples

- `examples/antibody_design.py` shows a minimal BoltzGen antibody design spec.
- `examples/protein_ligand_affinity.py` shows a Boltz protein-ligand affinity spec.
- `examples/boltz2_kras_mrtx1133.py` folds KRAS G12D with the MRTX-1133 inhibitor and prints affinity.
- `examples/boltzgen_peptide_binder.py` shows a BoltzGen peptide binder spec with optional cyclic peptides.
- `examples/boltz_constraints.py` shows a Boltz complex with pocket/contact constraints and an optional MSA.
- `examples/boltz_multichain_msa.py` shows multi-chain MSAs with cross-chain constraints.
- `examples/boltzgen_template_insertions.py` shows template structure groups and design insertions.
- `examples/boltz_multi_pocket_complex.py` shows multi-ligand pocket constraints with contacts.
- `examples/boltzgen_template_masks.py` shows template masks with design/not-design ranges and structure groups.

## Notes

This repository consolidates the two stacks into a single build and dependency set. If you need the legacy standalone documentation, it has been preserved under `docs/`.

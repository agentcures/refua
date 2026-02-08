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

Unified API (same Complex flow for ligands or binders):

```python
from pathlib import Path

from refua import Binder, Complex, Protein, SM

target = Protein(
    "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQQLAGKYTPEEIRNVLSTLQKAD",
    ids="A",
)

# Protein + ligand -> Boltz2 structure + affinity
result = (
    Complex([target, SM("Cn1cnc2n(C)c(=O)n(C)c(=O)c12")], name="demo")
    .request_affinity()
    .fold()
)
Path("complex.bcif").write_bytes(result.to_bcif())
print(result.affinity.ic50, result.affinity.binding_probability)

# Protein + binder placeholder -> Boltz2 structure + BoltzGen design inputs
binder = Binder(length=12, ids="P")
result = Complex([target, binder], name="design").fold()
Path("design.bcif").write_bytes(result.to_bcif())
print("binder spec:", binder.sequence)
```

For template-based designs, add `.file(...)` to the same `Complex` before `fold()`.

Antibody design with `BinderDesigns` + `Complex`:

```python
from refua import Binder, BinderDesigns, Complex, Protein

antigen_seq = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQQLAGKYTPEEIRNVLSTLQKAD"
antigen = Protein(antigen_seq, ids="A", binding_types={"binding": "10..30"})
binder_pair = BinderDesigns.antibody(
    heavy_cdr_lengths=(12, 10, 14),
    light_cdr_lengths=(10, 9, 9),
    heavy_id="H",
    light_id="L",
)
design = Complex([antigen, *binder_pair], name="antibody_design")

# Optional explicit overrides
# design = Complex([antigen, Binder("8C6", ids="H"), Binder("7C5", ids="L")], name="antibody_design")

result = design.fold()
print(result.binder_specs)
print(result.chain_design_summary())

# Peptide presets
linear_peptide = BinderDesigns.peptide(length=14, ids="P")
disulfide_peptide = BinderDesigns.disulfide_peptide(
    segment_lengths=(10, 6, 3),
    ids="Q",
)
```

Small molecule properties:

```python
from refua import SM

props = SM("Cn1cnc2n(C)c(=O)n(C)c(=O)c12", lazy=True)
print(props.mol_wt(), props.logp())
print(props.to_dict())
```

Pass `lazy=False` to compute all properties eagerly.

Model-based ADMET predictions (optional; requires `refua[admet]`):

```python
props = SM("Cn1cnc2n(C)c(=O)n(C)c(=O)c12")
profile = props.admet_profile()
print(profile["admet_score"], profile["assessment"])
print(props.herg(), props.ames())
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

- `examples/antibody_design.py` shows antibody setup with `BinderDesigns`.
- `examples/protein_ligand_affinity.py` shows a protein-ligand affinity spec.
- `examples/boltz2_kras_mrtx1133.py` folds KRAS G12D with the MRTX-1133 inhibitor and prints affinity.
- `examples/boltzgen_peptide_binder.py` shows a peptide binder spec with optional cyclic peptides.
- `examples/boltz_constraints.py` shows a complex with pocket/contact constraints and an optional MSA.
- `examples/boltz_multichain_msa.py` shows multi-chain MSAs with cross-chain constraints.
- `examples/boltzgen_template_insertions.py` shows template structure groups and design insertions.
- `examples/boltz_multi_pocket_complex.py` shows multi-ligand pocket constraints with contacts.
- `examples/boltzgen_template_masks.py` shows template masks with design/not-design ranges and structure groups.

## Notes

This repository consolidates the two stacks into a single build and dependency set. If you need the legacy standalone documentation, it has been preserved under `docs/`.

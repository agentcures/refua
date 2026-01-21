# Refua Examples

These examples use the unified `Complex` API with `Protein`, `Binder`, and `SM` to
build specs, run predictions, and prepare model inputs. The unified API routes to
Boltz2 or BoltzGen under the hood, depending on which entities you include.

Boltz2 assets download into the default cache on first run. BoltzGen uses the
bundled molecule library from the Hugging Face cache; set `BOLTZGEN_MOLDIR` to
override.

## Antibody design (Complex -> BoltzGen)

`antibody_design.py` builds a simple antibody specification against a target structure
and prepares design inputs.

Example:

```bash
python examples/antibody_design.py \
  --antigen /path/to/antigen.cif \
  --antigen-chain A \
  --binding-range 10..40
```

## Peptide binder design (Complex -> BoltzGen)

`boltzgen_peptide_binder.py` builds a peptide binder spec against a target structure with optional
cyclic peptides and secondary structure hints.

Example:

```bash
python examples/boltzgen_peptide_binder.py \
  --target /path/to/target.cif \
  --target-chain A \
  --binding-range 10..40 \
  --peptide-spec 10C6C3 \
  --peptide-cyclic
```

## Protein-ligand complex with affinity (Complex -> Boltz2)

`protein_ligand_affinity.py` shows how to create a protein-ligand complex and run an
affinity prediction.

Example:

```bash
python examples/protein_ligand_affinity.py \
  --protein-sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \
  --ligand-smiles CCO
```

## KRAS G12D + MRTX-1133 (Complex -> Boltz2)

`boltz2_kras_mrtx1133.py` folds the KRAS G12D protein with the MRTX-1133 inhibitor, prints the
affinity prediction, and optionally writes a CIF/BCIF.

Example:

```bash
python examples/boltz2_kras_mrtx1133.py --output kras_mrtx1133.bcif
```

## Constrained protein-ligand complex (Complex -> Boltz2)

`boltz_constraints.py` builds a protein-ligand complex with pocket/contact constraints and an optional
toy MSA, then runs structure prediction (and optional affinity).

Example:

```bash
python examples/boltz_constraints.py \
  --protein-sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \
  --ligand-smiles CCO \
  --pocket-contacts A:5,A:8 \
  --contact A:3,A:7 \
  --affinity \
  --use-msa
```

## Multi-chain MSA + constraints (Complex -> Boltz2)

`boltz_multichain_msa.py` builds a multi-chain complex, attaches in-memory MSAs to each chain, and
adds pocket/contact constraints across chains before prediction.

Example:

```bash
python examples/boltz_multichain_msa.py \
  --sequence-a MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \
  --sequence-b MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \
  --share-msa \
  --pocket-contacts 8,12 \
  --contact-ab 10,10 \
  --affinity
```

## Template insertions + structure groups (Complex -> BoltzGen)

`boltzgen_template_insertions.py` uses a template structure and shows structure groups, design
insertions, and a small binder spec.

Example:

```bash
python examples/boltzgen_template_insertions.py \
  --template /path/to/template.cif \
  --template-chain A \
  --structure-group-primary 5..15 \
  --structure-group-secondary 20..25 \
  --insertion-site 12 \
  --insertion-length 2..5 \
  --binder-spec 10C6C3
```

## Multi-ligand pockets (Complex -> Boltz2)

`boltz_multi_pocket_complex.py` shows two ligands with independent pocket constraints plus optional
cross-chain contact constraints.

Example:

```bash
python examples/boltz_multi_pocket_complex.py \
  --ligand-1-smiles CCO \
  --ligand-1-contacts A:8,B:12 \
  --ligand-2-smiles CCN \
  --ligand-2-contacts A:15,B:20 \
  --contact A:10,B:10
```

## Template masks + structure groups (Complex -> BoltzGen)

`boltzgen_template_masks.py` mixes include/exclude masks, design/not-design ranges, structure groups,
and secondary-structure conditioning on a template chain.

Example:

```bash
python examples/boltzgen_template_masks.py \
  --template /path/to/template.cif \
  --template-chain A \
  --design-range 10..30 \
  --not-design-range 1..5 \
  --structure-group-primary 6..15 \
  --structure-group-masked 30..35 \
  --helix-range 10..14
```

# In-Memory API

This API is designed for pure in-memory workflows: no file reads, no file writes, and inputs expressed as Python data structures. It lets you build complexes, attach constraints, run tokenization/featurization, and hand features to a model without touching disk.

## Quickstart

```python
from refua.boltz.api import (
    Spec,
    Pipeline,
    Protein,
    Ligand,
    msa_from_a3m,
)

msa = msa_from_a3m(a3m_text)

spec = (
    Spec("demo")
    .add(Protein("A", "MSEQUENCE", msa=msa))
    .add(Ligand("L", smiles="CC1=CC=CC=C1"))
    .pocket("L", contacts=[("A", 10), ("A", 25)], max_distance=6.0)
)

pipe = Pipeline(version=2, components=ccd_dict, molecules=ccd_dict)
trace = pipe.prepare(spec)
trace = pipe.featurize(trace)

outputs = model(trace.features, recycling_steps=3, diffusion_samples=1)
```

## Core Concepts

`Spec`
- The in-memory description of a complex.
- Holds chains, constraints, templates, and optional affinity requests.

`Pipeline`
- Converts a `Spec` into `Target`, `Input`, `Tokenized`, and model-ready features.
- Keeps everything in memory and never reads or writes files.

`Trace`
- A snapshot of the pipeline with intermediate objects and `chain_map`.
- Use it to inspect the structure or tokens before running a model.

## Chains

Use chain classes directly or via `Spec` helpers:

```python
spec = Spec("complex").add(
    Protein("A", "SEQUENCE", msa=msa),
    DNA(["B", "C"], "ATCG"),
    Ligand("L", ccd="ATP"),
)
```

- `Protein`, `DNA`, `RNA` accept `modifications` (list of `(position, ccd)` pairs).
- `Ligand` accepts either `ccd` or `smiles` (exactly one).

## Constraints

Constraints use simple tuple refs:

```python
spec = (
    Spec("constraints")
    .protein("A", "SEQUENCE")
    .ligand("L", smiles="CCO")
    .bond(("A", 1, "CA"), ("L", 1, "C1"))
    .contact(("A", 10), ("A", 50), max_distance=8.0)
    .pocket("L", contacts=[("A", 10), ("A", 25)])
)
```

## Templates (Boltz-2)

Templates are passed as in-memory structures and sequences:

```python
from refua.boltz.api import Template

template = Template(
    name="tmpl_1",
    structure=template_structure,
    sequences={"A": "TEMPLATESEQ"},
    chain_ids=["A"],
    template_chain_ids=["A"],
    force=True,
    threshold=2.0,
)

spec = Spec("templated").add(Protein("A", "QUERYSEQ")).add_template(template)
```

`Template.structure` must be a `StructureV2`, and `sequences` must map template chain ids to protein sequences. The matching logic is identical to the YAML pipeline, but fully in memory.

## MSA Utilities

You can build `MSA` objects from strings:

```python
from refua.boltz.api import msa_from_a3m

msa = msa_from_a3m(a3m_text, max_seqs=4096)
```

Attach MSAs to proteins directly:

```python
spec = Spec("msa").add(Protein("A", "SEQUENCE", msa=msa))
```

## Deep Pipeline Access

Each stage is available directly:

```python
target = pipe.build_target(spec)
input_data = pipe.build_input(spec, target)
tokenized = pipe.tokenize(input_data)
features = pipe.featurize(tokenized)
```

`Trace.chain_map` exposes the mapping from chain ids (strings) to internal asym ids (ints).

## Molecule Dictionaries

`Pipeline` requires in-memory molecule dictionaries:

- `components`: CCD dictionary used during parsing.
- `molecules`: Molecules used for featurization (defaults to `components`).

For ligands or modified residues, ensure those CCD entries are included in both dictionaries. The in-memory API never loads missing molecules from disk.

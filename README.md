# mdforge

**mdforge** is a Python toolkit for molecular-dynamics–driven force-field
parametrization, liquid-property analysis, and QM-vs-model comparison. Parsers
emit plain numpy arrays, compute kernels never touch a file, and every MD engine
sits behind one interface.

> Pre-alpha: the API may change between releases.

## Install

mdforge is not yet on PyPI — install it from a clone:

```bash
git clone https://github.com/roseane-reis/mdforge.git
cd mdforge
pip install -e .
```

Optional features come as extras (each pulls in only what it needs):

```bash
pip install -e ".[openmm]"     # OpenMM + openmm-ml (MACE-OFF23 / UMA)
pip install -e ".[ml]"         # torch (for MACE etc.)
pip install -e ".[gsd]"        # read HOOMD rigid-body GSD trajectories
pip install -e ".[viz]"        # matplotlib + seaborn + pandas (plots)
pip install -e ".[evaluate]"   # pyyaml (water-model evaluation config)
```

## What's in the box

| Module            | What it does                                                        |
|-------------------|---------------------------------------------------------------------|
| `core`            | Units, I/O, data records (`SpiceMolecule`, `Trajectory`, `BulkProperties`) |
| `formats`         | File parsing: Tinker `txyz`/`arc`/`prm`/analyze, `pdb`, HOOMD `gsd`, CHARMM `dcd`, EPSR g(r) |
| `engine`          | MD engine abstraction (Tinker, OpenMM) behind one interface + a registry seam |
| `qm`              | QM-vs-model energy/force comparison + interaction energies + reports |
| `liquid`          | Liquid-phase property kernels (thermo, transport, structure, stats) |
| `liquid.evaluate` | **Water-model quality evaluation vs experiment** (see below)         |
| `fit`             | Force-field parametrization (HIPPO/AMOEBA via Tinker)               |
| `simulate`        | Simulation orchestration (liquid box, gas, NPT/NVT)                 |
| `data`            | Reference database access (DES370K, NCIA, S101)                     |

## Evaluating a water model

`mdforge.liquid.evaluate` takes a finished water simulation and produces a
**judged, exported report**: it ingests a topology + trajectory, computes the
liquid properties, compares each to experiment at 298.15 K / 1 atm, and assigns a
quality verdict. It is **model- and engine-agnostic** — everything comes from a
config file.

**The quality bar is TIP3P.** For each property:

- within **1 %** of experiment (or its uncertainty) → **excellent**;
- else no worse than **TIP3P's** deviation → **good**;
- else → **bad**.

Structural metrics with no TIP3P baseline (tetrahedral order *q*, H-bonds per
molecule, coordination number, g_OO peak height) are shown but **unrated**. The
overall rating is a weighted score (excellent = 2, good = 1, bad = 0) over the
rated core properties.

### Quickstart

```bash
# 1. copy and edit the sample config (paths, model name, temperature)
cp examples/water.yaml my_water.yaml

# 2. run it
python -m mdforge.liquid.evaluate --config my_water.yaml

# ...or point at a campaign run directory and let it discover the legs:
python -m mdforge.liquid.evaluate --config my_water.yaml --campaign /path/to/run_dir
```

A single NPT trajectory is enough — every property computable from the given
data is produced, and missing legs are simply omitted. Inputs supported:
topology as PDB or Tinker XYZ; trajectory as HOOMD `.gsd` or CHARMM/NAMD `.dcd`;
optional per-frame thermo log as `.npy` or `.csv`. See
[`examples/water.yaml`](examples/water.yaml) for every option.

### From Python

```python
from mdforge.liquid.evaluate import EvalConfig, run_evaluation, build_evaluation_report

config = EvalConfig.from_yaml("my_water.yaml")
result = run_evaluation(config)
art = build_evaluation_report(result, outdir="analysis")
print(art["rating"].overall_label, art["rating"].grade)   # e.g. "bad" 0.25
```

### Outputs (written to `output.dir`)

- `results.json` — every computed number + RDF curves + an `evaluation` block
  (per-property verdicts, deviations, the TIP3P bar, the overall grade, citations).
- `properties_table.{csv,md}` — model vs experiment vs TIP3P vs HIPPO, with
  verdict and threshold columns.
- `REPORT.md` — the judged summary (overall verdict, per-property table, caveats, citations).
- `rdf_gOO.png` — model vs experimental O–O g(r).

Experimental references are packaged with citations (Izadi & Onufriev 2016;
Rackers et al. 2021; Soper 2000) under
`src/mdforge/liquid/evaluate/references/`.

## Supported engines

- **Tinker** — file + subprocess model; analyze/dynamic/minimize/testgrad
- **OpenMM** — Python API; MACE-OFF23 / UMA via openmm-ml

Additional engines can be registered via the plugin seam:

```python
from mdforge.engine import registry
registry.register("my_engine", MyEngineFactory)
```

## License

MIT.

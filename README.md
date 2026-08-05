# pulse-echo-cof-sim

The simulation and analysis code behind the Ultrasonics manuscript on
pulse-echo azimuthal measurement of crystal-orientation fabric (COF) in
ice. Everything the paper claims traces to a script in this repository,
usually with the quoted numbers recorded in that script's docstring.
Negative results are kept deliberately: they are part of the evidence
chain, not clutter.

## Directory map

| path | what it is |
|---|---|
| `sim/` | The solver and production stack, in five role-named groups (below). Read `sim/README.md` first: it is the ground-truth document for what every file is and why it exists. |
| `sim/core/` | The solver and specimen machinery: label-indexed anisotropic FDTD (`fdtd.py`), pseudospectral solver, specimen builder, rotation machinery, bit-equivalence gate (`validate_labels.py`). |
| `sim/model/` | Forward models and inversion: the born ladder, fabric inversion (`fit_fabric.py`, `fit_sweep.py`), gridded inversion and its daemon, the studio ray model. |
| `sim/pipeline/` | Drivers: the resumable sweep engine (`sweep_runner.py`), the original azimuth study, trace lab, arc backprojection, visualisation. |
| `sim/fe_crosscheck/` | The FE arbiter that cross-checks the staircase: mesher, solvers, probes, the five cross-check rounds and their trace archives. |
| `sim/fw_checks/` | Solver validation and acceptance studies: full-wave references and crop/corridor licenses, oblique and pseudospectral validation, anisotropy and convergence checks. |
| `sim/analysis/` | About 75 analysis modules plus their stored `.npz` results, in subdirectories by theme (channel_network, fabric_channel, facet_model, frequency, independent_check, physical_optics, statistics, validation). See `sim/analysis/README.md`. |
| `sim/validation/` | Rotated-staggered-grid and exact reflection-coefficient studies. |
| `sim/ref/` | Clean reference traces (inputs to the paper's figures). |
| `sim/runs/` | Batch run drivers for the GPU campaigns. |
| `sim/figures/` | The paper's figure scripts; the built PDFs live in the paper repository. |
| `sim/results/` | Stored results from the run campaigns. |
| `gui/` | Streamlit app: the studio pipeline plus the FW sweep tab that drives `sim/pipeline/sweep_runner.py`. |
| `out/sweeps/` | Sweep sessions: one directory per sweep with `config.json`, one npz per completed azimuth, and fit results. The analysis layer reads these. |
| `out/tesscache/`, `out/observables/` | Tessellation cache and observable matrices. |

## Environment

Python 3.12 with numpy and scipy. That is the whole requirement for the
analysis layer: every module under `sim/analysis/` reads stored sweeps,
reference traces, or its own stored npz, and runs on CPU with no CUDA
anywhere in sight.

CUDA and cupy are needed only to RUN the solver (`sim/core/fdtd.py`,
`sim/pipeline/sweep_runner.py`, the FE solvers and their probes) and so
to regenerate sweeps or reference traces. `sim/core/specimen.py` uses
the GPU for labelling but falls back to CPU. matplotlib is needed for `sim/figures/`; streamlit
and plotly for `gui/`.

## Reproduce a paper number in three commands

No GPU required. This recomputes the Appendix A direct-arrival
attenuation agreement between the two solvers from the archived arbiter
traces:

```
git clone <this repository>
cd pulse-echo-cof-sim/sim/analysis
python fe_direct_attenuation.py
```

Expected: a table of the five cross-check rounds ending with FDTD
-1.367 dB against FE -1.341 dB (0.027 dB gap) for the 6-15 mm placement,
and agreement spanning 0.014 to 0.068 dB over the three valid rounds.
Any cited module works the same way: run it from its own directory and
compare the printed numbers against the ones recorded in its docstring.

## Where to read next

- `sim/README.md`: the production stack, the solver validation chain,
  the FE arbiter evidence chain, the frame-convention warning, and how
  to run a sweep. This is the authoritative map of `sim/`.
- `sim/analysis/README.md`: the analysis modules and their stored
  results.

## Archive note (2026-08 cleanup)

Before release, 1.1 GB of regenerable FE arbiter mesh npz was moved out
of the working tree to the sibling directory
`pulse-echo-cof-sim_archive/` (same layout as the repo). The ab scripts
rebuild those meshes if rerun; nothing in the analysis layer reads them.
About 30 dead files (superseded experiments) were deleted from the tree;
all of them remain recoverable from git history. Everything the paper
cites, every stored npz a result depends on, and every documented
negative result was kept.

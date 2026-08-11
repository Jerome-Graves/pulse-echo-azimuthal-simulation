# sim/: what every file is and why it exists

The working rule of this codebase: **results live in code, not in
memory**. Every claim (positive or negative) traces to a script here and
usually to numbers recorded in that script's docstring. Files marked
NEGATIVE RESULT are kept deliberately: they document dead ends so they
are not re-attempted, and they are part of the paper's evidence chain.

## Layout (2026-08 restructure)

`core/` holds the solver and specimen machinery (fdtd, rsg, fdtd_ps, specimen, rotation_test, config, errors, validate_labels).
`model/` holds the forward models and inversion (born, forward, acquisition, reconstruct, fit_fabric, fit_sweep, grid_inversion, grid_daemon, ladder).
`pipeline/` holds the drivers (sweep_runner, azimuth_sweep, trace_lab, arc_backproject, viz).
`fe_crosscheck/` holds the FE arbiter, its probes and its trace archives (all fe_* files).
`fw_checks/` holds the solver validation and acceptance studies (fw_*, validate, validate_oblique, validate_ps, aniso_check, aniso_accept, coda_convergence, freq_blips). `analysis/` and `validation/` did not move.

## ⚠ Frame convention (2026-07-29 rotation fix)

`rotation_test.rotated_grid` used to rotate the specimen GEOMETRY by
−rot but the c-AXES by +rot: each sweep azimuth was a different
specimen, not a rigid rotation. Fixed: both now rotate by −rot. Sweeps
carry `axes_convention: "rigid2"` in `config.json`; sweeps WITHOUT the
tag are legacy: `fit_sweep` keeps the old negated-axis reporting for
them, `sweep_runner` refuses to continue a part-done legacy sweep, and
their per-cell/per-realisation results (grid inversion, truth maps,
twin calibration) are unreliable: regenerate before trusting those.
Under rigid2 the fitted axis IS the specimen-frame axis (no negation).
Same-day fixes: grid attribution mirror (probe at +rot, was −rot, in
`grid_inversion`/`grid_daemon`), double-counted contrast in the grid
coda background (now `unit_contrast=True` × √E[R²]), Ricker source
delay (1.2/f0) subtracted in all time→depth mappings, `model/ladder.py` no
longer falls back to contaminated-era FW constants, and the coda shape
calibration records its training sweeps (`fit_sweep` warns loudly on
in-sample refits).

## The production stack (what actually runs)

| file | role |
|---|---|
| `core/fdtd.py` | THE solver. Label-indexed anisotropic velocity-stress FDTD, generated/unrolled CUDA kernels, z-column register pipelines. Any edit must pass `core/validate_labels.py --quick` at `err == 0.0` (bit-identical). Contains three documented kernel negative results (fused step, smem stress tiles, stream overlap) proving the solver sits at the GPU memory wall. |
| `core/specimen.py` | `DiskSpecimen`: Laguerre/Voronoi polycrystal builder (GPU labels with CPU fallback), Watson fabric sampling. |
| `model/born.py` | The simplified-model ladder, rungs 1-10, outcomes in docstrings. `boundary_scatter` (rung 3) is the crowned production forward model. Optional `face_weight="normal"` (exact wall areas from Laguerre seed normals; default `"l1"` bit-identical to before). The suspected mesh-locked azimuthal L1 artefact was REFUTED experimentally 2026-07-29 (mesh-locked 4-fold 0.007 dB, i.e. noise floor; L1 inflation is an azimuth-flat +1.7 dB the free gain absorbs). |
| `model/ladder.py` | Frozen scoring protocol (windows, fabrics, reference levels). |
| `model/fit_fabric.py` | The validated fabric inversion: Watson (axis, κ) + robust gain against azimuthal coda RMS + E1 ToF, Huber loss. Three failed objective designs documented in the docstring. |
| `pipeline/sweep_runner.py` | Resumable azimuth-sweep engine (state = one atomic npz per completed azimuth; STOP flag; progressive bit-reversed azimuth order so any partial sweep is maximally spread). Driven by CLI or the GUI's FW sweep tab. |
| `model/fit_sweep.py` | Adapts `fit_fabric` to any sweep directory (observables recomputed from traces; model envelopes built from the sweep's own specimen/frequency, cached). Since 2026-07-31 stage 1 also uses the **coherent-2θ axis channel** (C2T block: full-band envelope-dB 2θ phase over 30-36 µs; calibration-free, ~2-5° axis accuracy, rigid2 sweeps with ≥48 azimuths only; FD-banding would destroy it, the mechanism is cross-band phase coherence). Per-frequency shape calibrations (`cal_path(f0)`), FD bands 3 (2 MHz) / 8 (5 MHz), heartbeat fit lock. |
| `model/grid_inversion.py` | Batch gridded per-area inversion (rung 1): per-cell slowness + backscatter fields from ToF line integrals and time-resolved coda, regularised linear solves, truth-correlation honesty metrics. |
| `model/grid_daemon.py` | The CONTINUOUS grid fitter: a resumable daemon that re-solves the cell fields every few seconds against whatever data exists, with signed extinction/shadowing in the coda Jacobian, folding in new azimuths and global-fit updates as they land. Stop with a `GRID_STOP` file; state = `grid_state.npz`. |
| `core/config.py`, `model/acquisition.py`, `core/errors.py`, `model/forward.py`, `model/reconstruct.py`, `pipeline/viz.py` | The original studio pipeline (ray model, error budget, per-grain demo reconstruction) used by `gui/app.py`. |

## ⚠ Known numerics artifact: lab-frame 2θ "g(t)" (characterised 2026-07-31)

Sweeps carry a specimen-class-dependent lab-frame 2θ azimuthal artifact
(~1-2 dB; coda band phase ~81°, E1/tail ~112-115° for aligned rigid2
specimens; different phases for isotropic/legacy ones). Mechanism:
beam-referenced staircase scattering + corridor resample phase screen
(the rotated resample is exactly 90°-periodic, so the 2θ can only be
beam-referenced (proved). Resample changes do NOT remove it (tested);
it is additive in the FIELD, so dB templates do not transfer between
specimen classes (three-specimen + isotropic-sweep tests). Mitigations:
per-sweep fit-level 2θ nuisance vector (in fit_sweep), ppw 8 (~3.4×
cost), or same-class template. It is a digital-twin numerics artifact:
the physical rig does not rasterise ice. Evidence: isotropic_seed41_ppw6_calibration sweep +
side studies outside this repository (az_harmonic_decomp / g_fine / threetests npz).

## Solver validation chain (the paper's verification section)

| file | what it proves |
|---|---|
| `core/validate_labels.py` | Bit-equivalence gate (labels vs dense path, full length, `err == 0.0`) + rim-damping recipe (`fluid_damp = 0.02`). THE gate for any `core/fdtd.py` edit. |
| `fw_checks/aniso_check.py`, `fw_checks/aniso_accept.py` | Anisotropic velocities vs group-velocity theory. |
| `fw_checks/coda_convergence.py` | ppw convergence: amplitude work needs ppw ≥ 6 (series flattens there). |
| `core/rotation_test.py` | SUPERSEDED as a staircase validation (2026-07-29): the probe never followed the rotation, so the test was an azimuth sweep in disguise (production-settings rerun: 13.7 dB spread = fabric pattern + glint; the old ~2 dB was the ppw-4 noise floor). Staircase validity rests on the FE arbiter. Still hosts `rotated_grid`, the azimuth mechanism reused by sweeps (rigid since 2026-07-29, see frame-convention note above). ppw6→ppw8 coda rms −2.0 dB recorded (production levels carry ~2 dB discretisation allowance). |
| `fw_checks/validate_oblique.py` | Staircased interface vs Aki-Richards vs tilt angle (error budget: good below ~30°, ×3 at 45°). |
| `fw_checks/fw_crop_validate_ray_model.py` | LICENSES the causal coda crop (0.45× cost, window diff −107 dB re E1 = float noise). Reference implementation for coda-only runs. |
| `fw_checks/fw_e1_corridor_validate_ray_model.py` | LICENSES the E1 beam corridor (0.52× cells, E1 amp −0.06 dB, ToF 0 ns). Corridor coda is NEVER licensed. |
| `fw_checks/fw_e1_ppw5_validate_ray_model.py` | NEGATIVE RESULT: ppw5 E1 leg fails (+2.9 dB amplitude); E1 stays ppw6. |

## The FE arbiter (independent cross-check of the staircase: PASSED)

Chronological evidence chain; each file's docstring records its verdict.

| file | verdict |
|---|---|
| `fe_crosscheck/fe_mesh.py` | Conforming tet mesher straight from the Laguerre planes (no Neper). Netgen optimize on (sliver control). |
| `fe_crosscheck/fe_solver_p1.py` | P1/CST solver. DISQUALIFIED as arbiter: −26.8 dB mesh floor. Its `locate()` (barycentric point placement) is still used everywhere. |
| `fe_crosscheck/fe_solver_p2.py` | TET10 + HRZ. Two documented negative results: HRZ lumping degrades to P1 dispersion; consistent-mass Jacobi provably diverges. `straighten()` (edge-node snap) is still used by the P2+ pipeline. |
| `fe_crosscheck/fe_solver_p2plus.py` | **ML2n15** (Geevers-Mulder-van der Vegt 2018): the working arbiter element. Exact symbolic reference stiffness, true diagonal mass, power-iteration dt, rate-scaled quartic absorber, monopole/dilatation source-receiver. |
| `fe_crosscheck/fe_p2_probe.py`, `fe_crosscheck/fe_p2plus_probe.py`, `fe_crosscheck/fe_p2plus_scaling.py` | Element validation: patch tests (machine precision), two-receiver dispersion, order-of-convergence. The scaling probe exposed the bar harness's constant ~0.03-cycle near-field bias (absolute speeds from that harness carry it; solver-vs-solver differences are clean). |
| `fe_crosscheck/fe_p2_floor.py`, `fe_crosscheck/fe_p2plus_floor.py` | Remesh texture floors. Punchline: floors were solver-independent → not dispersion but box reverberation. |
| `fe_crosscheck/fe_arbiter_round1_baseline.py` … `ab5.py` | The five cross-check rounds. The sequence −19.5 → −19.1 → −22.9 → −35.4 → **−46.2 dB is the finite-element solver's OWN coda level** re its own direct arrival, not a solver-to-solver difference; the finite-difference level over the same rounds is −44.9, −44.9, −44.9, −43.2, −43.2. Round 5 is therefore **−46.2 against −43.2 = PASS, a gap of 2.9 dB** (early window 0.8 dB, and note that gap has the OPPOSITE sign: there the finite-element solver is the louder). Fix per round: (2) padded box + quartic shell, (3) monopole source ends S-wave contamination + **attenuation agreement better than 0.07 dB**, (4) deep src/rec kills wall mirrors, (5) Gaussian-ball source kills numerical S leak. |
| `fe_crosscheck/fe_arbiter_trace_forensics.py` | Bin-by-bin trace forensics used to diagnose every round. Run on any `fe_crosscheck/fe_arbiter_round1_baseline*_traces.npz`. |
| `analysis/fe_direct_attenuation.py` | Recomputes the direct-arrival attenuation from the five archives, the numbers Appendix A quotes. Three valid comparisons over **two** src/rec placements: 6-15 mm gives FDTD −1.367 vs FE −1.341 (0.027 dB); 12-21 mm gives −1.805 vs −1.737 (round 4, 0.068) and −1.805 vs −1.791 (round 5, 0.014). FDTD is the more attenuated in all three. Rounds 1-2 (point force) are 1.05 dB out and are the multipole-mismatch evidence, not agreement. |
| `fe_crosscheck/fe_crosscheck.py` | The original harness (geometry, seeds, FDTD reference runner) the rounds build on. |

## Other reference runners / studies

`fw_checks/fw_reference_2mhz_2mhz.py` (2 MHz reference library),
`fw_checks/fw_reference_5mhz.py` (5 MHz production trace),
`pipeline/azimuth_sweep.py` (the original 6-azimuth study; walk-off
discovery), `fw_checks/freq_blips.py` (boundary-blip frequency study),
`core/fdtd_ps.py` + `fw_checks/validate_ps.py` (pseudospectral solver:
NEGATIVE for coda/amplitudes, licensed for timing only, ~4×).

## Data (`ref/`, `../out/`, mesh npz)

`ref/` holds the clean (decontaminated, fluid_damp 0.02) reference
traces: inputs to the paper's figures; regenerable but keep them. The
FE cross-check trace archives (`fe_arbiter_round1_baseline*_traces.npz`) sit next to
their scripts in `fe_crosscheck/`. `../out/sweeps/<name>/` holds sweep
sessions (config + one npz per azimuth + fit results). The large
`fe_*.npz` arbiter meshes are not versioned (regenerable bulk); the ab
scripts rebuild them if rerun.

## How to run the common things

```bash
# gate any core/fdtd.py change (must print 0.00e+00):
python core/validate_labels.py --quick

# create + run a sweep from the CLI (the GUI FW-sweep tab does the same):
python pipeline/sweep_runner.py --new mysweep --set f0_mhz=5.0 az_stop=360
python pipeline/sweep_runner.py --run mysweep     # stop with a STOP file; rerun = continue

# fit the fabric on whatever a sweep has produced (>= 6 azimuths):
python model/fit_sweep.py mysweep
```

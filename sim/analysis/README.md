# sim/analysis: claim map

Every numbered claim of the paper (main article and supplementary
sections S1 to S17) mapped to the module(s) that produce its numbers and
the stored file that archives them. Sources of truth: the paper's tex,
the module docstrings (each states its own READS/WRITES and verdict),
and the files on disk. Paths are relative to the repo root. Raw sweep
data lives under `out/sweeps/<name>/` (one `az*.npz` per azimuth plus
`config.json`); production tessellations are cached in `out/tesscache/`
(`tess_s<seed>_p8_k-8.npz`, `gen_s<seed>_p8.npz`). See `sim/README.md`
for the production stack and the frame-convention warning.

## Reproduction contract

Modules that gate themselves on stored values. Each one reproduces its
stored/published row through the archived code path before it measures
anything, and stops or flags if the row does not return:

| module | gate |
|---|---|
| `identify_delay_audit.py` | recomputes the stored `ident` rows of `tessellation_replication.npz` to zero difference before step 2; stops otherwise |
| `gate_refinement_audit.py` | rebuilds the published identification estimator from scratch and must return the stored seed-23 ppw8 row exactly |
| `gate_refinement_crosscheck.py` | prints the stored ensemble row beside its own reproduction before any variation |
| `gate_refinement_second_ladder.py` | same contract, seed-11 ladder |
| `gate_timebase_cross.py` | same contract, field-against-clock cross |
| `microstructure_error.py` | scores through `tessellation_replication.py`'s own functions on its 30 common azimuths, so the statistic is the published one by construction |
| `coda_noise_floor.py` | the noiseless row of its table must reproduce the published counts before noise is added |
| `identification_transfer.py` | re-derives every load-bearing count from the archives, not from the reporting code that wrote them |
| `sim/validation/rsg_reproduce.py` | reproduces the numbers in `sim/results/rsg_diagnose.log` and `tilt_table.log` or the adjudication stops |
| `descriptor_spreads.py` | imports `ensemble_stats.tessellation_geometry` rather than restating it, so it cannot drift from the published descriptors |

Repo-wide gates that sit outside this directory: any `sim/core/fdtd.py`
edit must pass `sim/core/validate_labels.py --quick` at `err == 0.0`, and
`sim/model/ladder.py` raises `FileNotFoundError` if the FW reference traces
are missing rather than falling back to the contaminated-era constants.

Consequence for maintenance: an edit to any module above, or to any
module that writes one of the `.npz` archives named below, requires a
rerun that returns the stored values unchanged.

### The centring-convention trap (published = harm)

`shift_null_2d.centre` defines four centrings: `raw`, `col`, `double`,
`harm`. Every published field-correlation and identification number
(main Sec. 4.2, S7, S11) is the `harm` one, the strictest: mean,
azimuthal harmonics k <= 4 at every time sample, and the per-azimuth
level all removed. Laxer centrings return larger correlations (up to
0.29 against the published 0.156 on the development specimen) and place
the true registration first in 8/8 tessellations instead of 6/8. A
reproduction that forgets the centring therefore silently inflates the
result. `centring_convention.py` is the arbiter: it evaluates all four
on the same specimen and attributes every circulating number to its
convention.

## Supplementary sections S1 to S17

| sec | content | module(s) | stored archive |
|---|---|---|---|
| S1 | scattering regime, diffusion limit, facet model constants (32 boundaries in beam, 1194 mm^2, R_rms 1.00 percent, anisotropic factor 1.02, g = 1.12); tab:aniso | `physical_optics/po_src_02_facet_geometry.py` (exact facet polygons), `physical_optics/po_src_03_kirchhoff_pred.py` (Eq. S1 machinery), `physical_optics/po_src_12_exact_reflection.py` (scalar vs full anisotropic R), `facet_census.py` (specular-visibility census), `aniso_table.py` (tab:aniso) | `physical_optics/po_src_geom.npz` (also mirrored at `sim/figures/data/po_src_geom.npz`); grain-size survey and tab:regime are literature-only, no module |
| S2 | four cross-solver pitfalls; FE ladder -19.5 to -46.2 dB vs FD -44.9 to -43.2 dB; attenuation 0.03/0.07/0.01 dB | `sim/fe_crosscheck/fe_arbiter_round1_baseline.py` ... `ab5.py` (the five rounds), `sim/fe_crosscheck/fe_arbiter_trace_forensics.py` (trace forensics), `fe_direct_attenuation.py` (cited; the three attenuation comparisons), `sim/fe_crosscheck/fe_p2plus_scaling.py` (near-field 0.03-cycle bias) | `sim/fe_crosscheck/fe_arbiter_round1_baseline_traces.npz`, `_ab2_` ... `_ab5_traces.npz` |
| S3 | computational cost, tab:cost, 64 GPU hours over 2777 azimuths, ppw scaling exponents | `gpu_cost.py` (cited; file timestamps of `out/sweeps/*/az*.npz` plus `sim/results/*_run.log` and `followup*_run.log`) | no npz; the timestamps and logs are the archive |
| S4 | FE comparison in full (2.9 dB coda, 0.8 dB early window) | same modules as S2 plus `sim/fe_crosscheck/fe_crosscheck.py`, `sim/fe_crosscheck/fe_solver_p2plus.py` | the five `_traces.npz` archives |
| S4 | single interface vs closed-form R (r = 0.966 over five contrasts; 6.7e-4 zero-contrast echo; spreading factor closes it to r = 0.995) | `sim/validation/two_grain_reflection.py`, `sim/validation/exact_rc.py` ... `exact_rc_stroh_verify.py` (exact anisotropic coefficient, Christoffel matching) | `sim/results/two_grain.npz`, `sim/results/exact_rc_psib_sweep.npz` (note: `exact_rc_psib_sweep/3.py` load `bicrystal_val.npz` from `Documents/pulse-echo-analysis-scratch`, outside the repo) |
| S4 | physical-optics absolute anchor (-88.1 dB direct vs -87.79 measured; band truncation 1.1 dB; +-3 dB band; dB-average disagreement 5.5 dB) | `physical_optics/po_src_01_measured_levels.py` ... `po_src_11_closed_form.py` (stepwise; `po_src_10_backwall_speed.py` fixes the 3843 m/s delay mapping, `po_src_11_closed_form.py` the closed form, `po_src_07_sensitivity.py` the uncertainty band, `po_src_08_glint_robustness.py` the glint concentration), `po_band_audit.py` (closes the estimator mismatch against the audited level) | `physical_optics/po_src_meas.npz`, `po_src_pred.npz`, `po_src_final.npz`, `po_src_dense.npz`, `po_src_diag.npz`, `po_src_shape.npz` |
| S5 | choice of reference and estimator; tab:reconcile (-81.09/-85.11/-87.79 dB); pedestal audit | `db_reconcile.py` (rebuilds the table on the audited local in-band estimator) | prints; levels re-derived from `out/sweeps` (`girdle_seed11_ppw6_axis_perp*`, `singlemax_seed11_ppw10_ladder`, `girdle_seed11_ppw10_licensing`) |
| S5 | uniform-orientation control and contrast ladder (tab:ladder, intercept, n = 1.95) | `contrast_ladder.py`; rungs produced by `sim/runs/master_run.py` | sweeps `cs_f000/025/050/075_s11_ppw8`, `girdle_seed11_ppw8_uniform_axis`, `girdle_seed11_ppw6_zerocontrast`, `girdle_seed11_ppw10_zerocontrast`; `sim/results/bicrystal_ladder.npz` (master-run byproduct) |
| S5 | seed-23 ladder repeat (steps -3.72/-2.54 dB) | `eight_pairs_two_ladders.py`, `gate_refinement_audit.py` | sweeps `girdle_seed23_ppw6_ladder`, `girdle_seed23_ppw8_ensemble`, `girdle_seed23_ppw10_ladder` |
| S5 | boundary and absorbing-layer exclusion (pad doubling 0.00 dB, halved damping +1.74 dB) | `sim/runs/domain_absorber_check.py` | `sim/results/domain_absorber.npz` |
| S5 | re-rasterisation between rungs (0.01 dB area change) | `revoxelisation.py` (cited) | `sim/results/revoxelisation.npz` |
| S6 | ensemble estimator, eight-tessellation adequacy, fig:ensemble, subset/interval requirements | `ensemble_stats.py` | `sim/results/ensemble.npz` |
| S6 | grain population, crossing counts, speckle floor, gamma rejection three ways, sub-gates and sub-bands | `grain_population.py` (geometric counts), `speckle_looks.py` (in-band look count), `speckle_scatter.py` (4.58 dB scatter vs floor, model tests), `speckle_scope.py` (scope across fabrics/grids/windows) | `sim/results/ensemble.npz`, `sim/results/coda_levels.npz`, `coda_noise_floor_speckle.npz` |
| S7 | centring convention | `centring_convention.py` (cited) | reads cached fields `t2p_*.npz` in `Documents/pulse-echo-analysis-scratch` (outside repo, with in-repo fallbacks); prints only |
| S7 | gate-integrated scalar, first places, split half, rank-null validation | `tessellation_replication.py` (the central replication archive), `identification_transfer.py` | `tessellation_replication.npz` (here), `sim/results/identification_transfer*.npz` |
| S7 | candidate-set geometry (nearest wrong candidate 2.36 mm vs 0.38-0.50 mm tolerance) | `candidate_geometry.py` (cited), `microstructure_error.py` (cited; the displacement sweep) | `sim/results/candidate_geometry.npz`, `sim/results/microstructure_error.npz` |
| S7 | re-implementation, mirror and surrogate controls | `independent_check/skeptic_final_tally.py` + `null_calibration.py` (shares no code with the predictor), `mirror_control.py`, `surrogates.py` | printed; earlier 44-candidate set |
| S8 | tab:ensemble, fabric main effect -2.86 dB, sign-flip null, split halves, specular seeds 7/17 | `eight_pairs_two_ladders.py` | `sim/results/ensemble.npz`; sweeps `mx_girdle_s*_ppw8`, `mx_single_s*_ppw8`, `girdle_seed11_ppw8_dev`, `singlemax_seed11_ppw8_twin` |
| S8 | wrong-estimator ESS check (1.14 dB), azimuthal autocorrelation | `azimuthal_autocorrelation.py` | `sim/results/coda_levels.npz` |
| S8 | variance decomposition, descriptor correlations (0.23/-0.12/0.48/0.79), descriptor spreads | `ensemble_stats.py`, `crossing_within.py` (the r = 0.79 crossing claim rebuilt within-specimen), `descriptor_spreads.py` (cited; the four spreads) | `sim/results/ensemble.npz`, `sim/results/crossing_within.npz`; the property-by-observable screen is `sample_matrix.py` + `matrix_screen.py` -> `sim/results/sample_matrix*.npz` |
| S9 | fabric-to-coda channel: speed route vs reflection-weighting route, wrong-fabric exchange, ablation table | `predict_field.py` + `_t2_common.py` (prediction variants full/geom/nodirec/flat), `facet_model/field_null_exhaustive.py` (shared-tessellation wrong-fabric cases), `geometry_vs_fabric.py`, `facet_predictors.py` (born_spec/born_iso/geom_only decomposition). S9's text predates the restructure; its gain figures came from the scratch-era t2 studies | cached fields `t2p_*.npz` in `Documents/pulse-echo-analysis-scratch` (outside repo); `facet_model/specimen_build_s11_p8_k-8.npz` |
| S10 | axis null: bulk predictor rank 10/30, 22-observable panel FWER, beam-local predictor, matched-pair type contrast | `shift_null.py` (bulk predictor), `observable_panel.py` + `panel_fwer.py` (the panel; caches in `predcache/pred_*.npz`), `beam_local_windows.py`, `beam_local_admissibility.py`, `beam_local_verdict.py`, `beam_descriptors.py`, `eight_pairs_two_ladders.py` (third test) | `beam_local_windows.npz`, `beam_local_admissibility.npz`, `beam_local_break.npz`, `fabric_axis_windows.npz`, `faxwin_fabric_pred_isotropic_seed41_ppw6_calibration.npz` |
| S10 | late gates past the Fresnel equality point (38-50, 40-48 us), leakage and bulk-velocity contamination budgets | `late_window_mechanism.py`, `late_window_adjudication.py`, `axis_window_adjudication.py` | `late_window_mechanism.npz`, `late_window_adjudication.npz`, `late_window_adjudication_placebo.npz`, `axis_window_adjudication.npz` |
| S11 | window sensitivity: sixteen-gate family, earlier-window comparison, seed-23/seed-11 refinement ladders and the clock adjudication | `gate_refinement_audit.py`, `gate_refinement_crosscheck.py`, `gate_refinement_second_ladder.py`, `gate_timebase_cross.py` (all four cited; all four gated, see contract), `fabric_axis_windows.py`, `median_trace_removal.py` | `gate_refinement_audit.npz`, `gate_refinement_crosscheck.npz`, `gate_refinement_second_ladder.npz`, `gate_timebase_cross.npz`, `median_trace_removal.npz` |
| S12 | time-of-flight axis and concentration recovery, tab:axisrecovery, tab:azcount, girdle branch degeneracy | `tof_axis_recovery.py` (all three reported quantities), `channel_network/edgedge1_axis.py` (adds seed 7, runs the same estimator), `channel_network/edgedge1_selfcheck.py` (concentration-swing flag) | `grain_axes_volumes_s*_k*.npz` (here); fig:tofaz is the independent route, drawn by `sim/figures/fig_tofaz.py` from `fit_result.json` of sweeps `singlemax_seed11_ppw6_rigid2`, `singlemax_seed17_ppw6_kappa8`, `singlemax_seed23_ppw6_heldout_axis` |
| S13 | frequency dependence 2 vs 5 MHz (-6.87 dB, residual ~ f^-1/2) | `frequency_scaling.py`; band-resolved follow-ups in `frequency/freq_atten_bias.py`, `freq_level_ratio.py`, `freq_speckle_stats.py` | `sim/results/frequency_scaling.npz`; sweeps `singlemax_seed7_ppw6_fittest_legacy`, `singlemax_seed7_ppw6_5mhz_production` |
| S14 | circular-shift null validity, effective sample size, block bootstrap | `shift_null.py`, `shift_null_2d.py` (the null machinery), `azimuthal_autocorrelation.py` (correlation length, N_eff), `statistics/block_bootstrap.py`, `statistics/measure_sigma.py`, `harmonic_nulls.py` (why harmonic-amplitude tests need a different null) | `sim/results/coda_levels.npz` |
| S15 | tilt testbed, tab:tilterror, seven-point convergence ladder h^1.15, tab:dimensionless | `sim/validation/tilt_testbed.py` (the construction), `sim/runs/tilt_table_run.py` (the regeneration) | `sim/results/tilt_table.npz` |
| S15 | dispersion figure and the operator-ceiling identification (kh = 2.55 pileup) | `sim/figures/fig_dispersion.py`, `dispersion_coeffs.py`, `operator_ceiling.py` | `sim/results/operator_ceiling.log` |
| S16 | three treatments; anti-aliased and Schoenberg-Muir rows | `sim/validation/tilt_testbed.py` (treatment modes), `sim/runs/tilt_table_run.py` | `sim/results/tilt_table.npz` |
| S16 | rotated-staggered-grid withdrawal: Green's-function probe (7.8 dB direction dependence), seven-tilt diagnosis, operator symbol, reproduction, echo-to-noise | `sim/validation/rsg_green_probe.py` (cited), `rsg_diagnose.py` (cited), `rsg_symbol.py` (cited), `rsg_reproduce.py`, `rsg_twoway_run.py` + `rsg_twoway_report.py`, `tilt_rsg.py` (the RSG solver under test) | `sim/results/rsg_green_*.npz` (12 files), `rsg_diagnose.npz`, `rsg_repro_*.npz` (4), `rsg_twoway_*.npz` (6), `rsg_lattice_*.npz`, `rsg_contrast_*.npz`, plus the `rsg_*.log` files the reproduction gates on |
| S17 | noise sensitivity table (identification intact at 0 dB SNR, chance at -25 dB) | `coda_noise_floor.py` (cited) | `coda_noise_floor.npz` (here) |

## Main-article claims

| where | claim | module(s) | stored archive |
|---|---|---|---|
| Sec. 2.1 | regime classification, Eq. (1) and its constants | as S1 | `physical_optics/po_src_geom.npz` |
| Sec. 2.3 | specimen construction, candidate pool, contrast-ladder design | `sim/core/specimen.py`, `sim/runs/master_run.py` | `out/tesscache/tess_s*_p8_k-8.npz` (48 candidates + ensemble), `gen_s*_p8.npz` |
| Sec. 3.1 | agreement with independent solutions (FE 2.9 dB, single interface, PO anchor 0.3 dB) | as S2/S4 | as S2/S4 |
| Sec. 3.2 | staircase error magnitude and first-order convergence | as S15 | `sim/results/tilt_table.npz` |
| Sec. 3.3 | physical vs numerical partition (uniform-orientation control, f^2 ladder, sub-1-percent residue) | as S5 | as S5 |
| Sec. 3.4 | sampling and ensemble adequacy (0.97 dB spread, N requirements) | as S6 | `sim/results/ensemble.npz` |
| Sec. 4.1 | the return is not a developed speckle field (contrast 1.53, missing nulls, gamma rejected) | `speckle_scatter.py`, `speckle_looks.py`, `speckle_scope.py`, `grain_population.py` | as S6 |
| Sec. 4.2 | identification: ranks 1,1,1,1,2,2,2,6, sum 16, p = 4.6e-10; field r = 0.165 +- 0.100, registration 6/8 at p = 3.6e-8 | `tessellation_replication.py` | `tessellation_replication.npz` |
| Sec. 4.2 (fig:identify caption) | uncorrected-delay variant (four firsts fall to three; seed 41 to rank 47) | `identify_delay_audit.py` (cited in the caption; gated) | `identify_delay_audit.npz` |
| Sec. 4.3 | position not crystallography: lag test (0.156 vs 0.062 at zero offset), 0.5 mm displacement kills first rank, identification switches on along the contrast ladder (f = 0 rank 7/48, uniform orientation 26/48) | `facet_model/field_lag_registration.py` (lag curve), `microstructure_error.py` (displacement), `zero_contrast_facet.py` + `analysis/validation/zero_contrast_sweep.py` (zero-contrast control), `centring_convention.py` (scores the ladder rungs), `identification_transfer.py` | `sim/results/microstructure_error.npz`, `sim/results/identification_transfer*.npz`; sweeps `cs_f*`, `zerocontrast_*`, `zc_s11_*` |
| Sec. 4.3 | matched-pair azimuthal correlation (r = +0.139 vs -0.199, 8! relabelling p = 2.5e-5) | `eight_pairs_two_ladders.py` | `sim/results/ensemble.npz` |
| Sec. 4.4 | fabric type separates (-2.86 dB, bandwidth and decay channels); the axis does not (rank 10/30, panel FWER, beam-local predictor) | as S8 and S10 | as S8 and S10 |
| Sec. 5 (discussion scope) | descriptor spreads, grain-size scope statement | `descriptor_spreads.py`, `channel_network/edgedge3_grain_size_size.py`, `channel_network/edges1to4_followups.py` | `channel_network/edge1.npz`, `edge1_a2_vs_kappa.npz`, `edge23.npz` |

## Figures

Every PDF in the paper's `figures/` directory is generated by the
`sim/figures/fig_<name>.py` module of the same name (e.g. `identify.pdf`
by `fig_identify.py`), sharing `figstyle.py`. `fig_tofaz.py` reads sweep
`fit_result.json` files directly and is deliberately independent of
`tof_axis_recovery.py` (two routes to the same claim).

## Dependencies, all in-repo

The `ringfwi` anisotropy code the stack imports is vendored at
`vendor/ringfwi` (see `vendor/NOTICE.md` for provenance), and every
module resolves its paths relative to its own file location, so the
repository runs from any checkout directory. The t2 cached fields
(`t2p_*.npz`) and `bicrystal_val.npz` live in
`sim/analysis/facet_model`, where `centring_convention.py` and the
figure scripts already look.

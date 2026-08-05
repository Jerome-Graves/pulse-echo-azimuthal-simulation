# DATA MANIFEST

One line per data file (or per parametric family) in this repository: what it
holds, which module wrote it, and what the name decodes to. File names are
LOAD BEARING: analysis modules locate these files by name, so renaming any of
them breaks the readers. Do not rename; extend this manifest instead.

The claim-to-module-to-archive map lives in `sim/analysis/README.md`; the
stack itself is documented in `sim/README.md`. Keys, shapes and dtypes of
every npz are recorded in the inventory this manifest was built from and are
repeated here only where a dimension carries meaning (for example "30
azimuths x 48 candidates").

Two kinds of file appear below and the difference matters for the
reproduction contract in `sim/analysis/README.md`:

* ARCHIVE: the numeric record of a published or supplementary result. Editing
  the producer requires a rerun that returns the stored values unchanged.
* CACHE: regenerable intermediate written on first use; safe to delete, the
  producer rebuilds it (some rebuilds need the GPU).

Producer claims below were spot-verified by grepping each named module for
the filename it writes; overturned claims are noted where they occur, and
files with no in-repo producer are listed honestly in the ORPHANS section at
the end.

## NAMING GRAMMAR

Top-level analysis archives: `<module>.npz` is written by
`sim/analysis/<module>.py` beside itself and archives that module's printed
tables; a `_placebo` suffix is the placebo arm of the same run. Key grammar
inside them is `<quantity>_<window>_<treatment>` (windows 4-16, 10-22, 24-36,
30-42, 38-44, 38-50, 40-48 us; treatment `loo` = leave-one-out median trace
removal; estimators `env`/`tf`; `full`/`drop` = backwall-drop variants).
Likewise `<module>.log` in `sim/results` is the module's transcript, and
`<name>_run.log` is the redirected stdout of `sim/runs/<name>.py`. A
`_partial` npz is a resumable checkpoint superseded on completion.

Sweep directories (`out/sweeps/<name>/`): newer names decode as
`<role>_<fabric>_s<seed>_ppw<n>`; older sweeps are freeform. Role prefixes:
`mx` = tessellation-by-fabric matrix (ensemble member); `cs_f<fff>` =
contrast-scaling rung at fraction f = fff/100; `zc` and `zerocontrast` =
zero-contrast numerical-floor control; `lad` = second (seed 23) resolution
ladder; `ladder_ppw<n>` = seed 11 resolution-ladder rung (trailing `d` =
fluid damping halved to 0.012); `lic` = ppw 10 licensing run; `wk ... k3` =
weak girdle at kappa -3; `clus` = spatially clustered orientations
(spatial_corr 0.6); `blind_singlez` = single maximum with axis normal to the
scan plane; `oos` = out-of-sample axis; `gcheck` = 12-azimuth ppw 8 check;
`isotropic_seed41_ppw6_calibration` = isotropic calibration control; `valcal` = inversion
validation-calibration; `girdle_seed11_ppw6_axis_perp` / `girdle_seed11_ppw6_axis_par` / `girdle_seed11_ppw6_axis_tilt20` = girdle
axis perpendicular / parallel / tilted 20 deg to the probe axis. Fabric
tokens: `girdle` = Watson kappa -8, axis [1,0,0] unless the name says
otherwise; `single`/`singlemax` = kappa 3.93, axis [0.866,0.5,0]. Inside each
directory: `az<ddd>.npz` = one completed azimuth in degrees (written
atomically, so sweeps resume for free), `config.json` = the sweep definition,
`status.json` + `run.log` = runner state, `STOP`/`GRID_STOP` = cooperative
stop flags, `fit_result.json` where `fit_sweep.py` has run.

Tessellation caches (`out/tesscache/`):
`tess_s<seed>_p<ppw>_k<kappa>.npz` = cached rasterised tessellation (int16
label volume + per-grain c-axes + generator points + grid pitch h) at ppw
points per wavelength and Watson concentration kappa (k-8 = girdle, k3.93 =
single maximum); `gen_s<seed>_p<ppw>.npz` = replayed Laguerre generators only
(points + weights); `replay_s<seed>_p<ppw>.npz` = full replayed build state
(points, weights, radii, bisection base/scale, keep mask, bit-exactness
flags). Seeds below 100 are real specimens; 100-139 are the wrong-candidate
distractors of the 48-candidate identification pool.

Analysis-side caches: `grain_axes_volumes_s<seed>_k<kappa>_a<ax>.npz`: seed =
tessellation seed, k<kappa> = Watson concentration, a<ax> = x-component of
the nominal fabric axis to three decimals (a1.000 = axis [1,0,0], a0.866 =
axis [0.866,0.5,0]). `t2p_<sweep>__<tag>.npz`: cached azimuth-by-time
predicted coda field for measured sweep <sweep> under prediction-geometry
build <tag> (gp8 = girdle seed 11 ppw 8, sm8 = single maximum ppw 8, iso6 =
isotropic_seed41_ppw6_calibration ppw 6, wrongseed6/8 = wrong-tessellation control at that ppw); a
deliberate sweep/tag mismatch IS the wrong-fabric or wrong-seed control.
`t4build_s<seed>_p<ppw>_k<kappa>.npz` = cached full specimen build.
`pred_<sweep>.npz` (predcache) = bulk fabric predictor cache keyed by sweep
name; `faxwin_pred_<sweep>.npz` is the same cache written by
`fabric_axis_windows.py`. nfcache `march_s<seed>.npz` / `field_s<seed>.npy` =
per-candidate marched geometry / predicted field; field_ exists only for the
eight true girdle seeds (7, 11, 17, 23, 41, 53, 71, 89).

Physical optics: `po_src_<name>.npz` is the output of the numbered step
`po_src_N<name>.py` of the S4 absolute-anchor chain (meas = step 1, geom =
step 2, pred + diag = step 3, dense = step 5, shape = step 9, final = step
10). Channel network: `edge<N>.npz` archives channel-network edge N
(`edge1_a2k` = the A2-vs-kappa model curve supporting edge 1).

RSG validation family (`sim/results/`):
`rsg_<probe>_<scheme>_<medium>_ppw<N>[_t<tilt>].npz`. probe: `green` =
point-source Green function, `lattice` = per-cell field smoothness, `twoway`
= bicrystal + homogeneous two-way reference, `contrast` = echo vs
misorientation, `repro` = independent reproduction, `diagnose` = seven-tilt
diagnosis. scheme: `ssg` = standard staggered grid (control), `rsg` = rotated
staggered grid (the accused). medium: `iso` = homogeneous isotropic
instrument check, `ani` = the anisotropic testbed. `t<tilt>` = interface tilt
in degrees (rsg_repro uses %.4f: t0.0000, t45.0000). Each probe writes
`<probe>.log` beside its npz; `*_report.log` and `*_audit.log` are read-only
verdict passes over the stored npz.

Full-wave references (`sim/ref/`): `fw_fabric<NN>[b].npz` = 2 MHz ladder
reference with NN percent of c-axes randomised, `b` = second independent draw
of the same percentage; `fw_az<D>.npz` = the same specimen rigidly rotated D
degrees (one probe azimuth per file); `fw_<F>mhz*.npz` = frequency variants
(3 MHz blip check, 5 MHz production trace, 5mhz_e1corridor = cropped-domain
validation). `coda_shape_cal_<F>mhz.npz` = per-frequency learned coda shape
correction; suffix `.HOLD` = active file parked to switch the correction OFF,
`.legacy` = superseded single-source version.

FE cross-check (`sim/fe_crosscheck/`): `fe_arbiter_round1_baseline<r>_traces.npz`, r =
arbiter round (blank = round 1, then 2-5); keys fa/fb = FDTD traces, ea/eb =
P2+ FE traces, A = real random c-axes, B = uniform crystal, dtf/dte = the two
sample steps. Each round changes exactly one confound.

## sim/analysis (result archives and caches beside the modules)

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| tessellation_replication.npz | 13 KB | ARCHIVE | sim/analysis/tessellation_replication.py | The central replication archive: per-tessellation azimuthal-scalar and field correlations, the 8x8 cross-tessellation matrix, and identification ranks of 17 sweeps against 48 candidates on the 30 shared azimuths. Backs main Sec. 4.2 (ranks 1,1,1,1,2,2,2,6, p = 4.6e-10) and supplementary S7. Read by identify_delay_audit.py, coda_noise_floor.py, channel_network/edgedge23_links.py, microstructure_error.py. |
| identify_delay_audit.npz | 29 KB | ARCHIVE | sim/analysis/identify_delay_audit.py | Corrected vs uncorrected source-delay identification: per-sweep r and rank over 48 candidates with and without the T0_SRC backwall-speed correction, the speed-scale sweep defence, and stretch percentages. Backs the fig:identify caption in Sec. 4.2 (four firsts fall to three; seed 41 to rank 47). |
| gate_refinement_audit.npz | 5 KB | ARCHIVE | sim/analysis/gate_refinement_audit.py | Seed 23 refinement-ladder audit: identification across the ppw 6/8/10 ladder under four centrings, the harmonic-gate ladder, and nudged gate windows, rebuilt from scratch against the stored published row. Backs supplementary S11. |
| gate_refinement_crosscheck.npz | 4 KB | ARCHIVE | sim/analysis/gate_refinement_crosscheck.py | Third-path variation of the seed 23 ladder: the base rise re-scored under varied ray march (600/1200 rays, rayseed), azimuth halves, candidate sets and a constant speed; shows the rise does not survive all free choices. Backs S11. |
| gate_refinement_second_ladder.npz | 5 KB | ARCHIVE | sim/analysis/gate_refinement_second_ladder.py | The refinement-ladder contract run on the seed 11 ladder (s11_A/B/C tables, cross-rung scores for both seeds, per-rung levels), showing the ladder direction is not seed 23 specific. Backs S11. |
| gate_timebase_cross.npz | 13 KB | ARCHIVE | sim/analysis/gate_timebase_cross.py | The field-against-clock cross that adjudicates the ladder rise: all three wavefields scored under all three per-rung clocks plus a constant speed, 48 candidates per cell, with permutation distributions of the end-to-end gain. Backs S11. |
| median_trace_removal.npz | 155 KB | ARCHIVE | sim/analysis/median_trace_removal.py | Identification scores for 10 sweeps x 8 windows x 5 envelope variants x 48 candidates, testing whether leave-one-out median trace removal lets the record be read earlier than 10 us, plus the zero-contrast control ranks that disqualify the untreated 4-16 us window. Backs S11. |
| fabric_axis_windows.npz | 77 KB | ARCHIVE | sim/analysis/fabric_axis_windows.py | Sec. 5.2's three fabric-axis tests (level regression, 22-observable panel, matched-pair type contrast) plus per-specimen axis error in degrees, re-measured in three windows with and without median removal. Stored as formatted STRING tables (cells, t1t2, t3, axis), so re-plotting requires parsing. Backs S10. |
| faxwin_fabric_pred_isotropic_seed41_ppw6_calibration.npz | 9 KB | CACHE | sim/analysis/fabric_axis_windows.py | Cached fabric predictor for the isotropic_seed41_ppw6_calibration sweep at all 360 azimuths: E[R^2] in dB (er2) and mean qP speed (vbar) per rotation, kept so a rerun does not rasterise on the GPU. Same content family as predcache/pred_*.npz. |
| beam_local_windows.npz | 30 KB | ARCHIVE | sim/analysis/beam_local_windows.py | Admissibility of the beam-local (column) predictor in the 4-16, 10-22 and 24-36 us windows: harness reproduction of the published r, zero-scattering controls, column-geometry descriptors, orientation-permutation nulls and 12-tessellation first-rank counts. Backs S10. |
| beam_local_admissibility.npz | 28 KB | ARCHIVE | sim/analysis/beam_local_admissibility.py | The admissibility verdict tables for the beam-local predictor: per-window correlations, decoy specimens A/B, geometry-column variants (60/30 azimuths, cylinder/cone), ensemble rank counts and permutation envelopes. Backs S10. |
| beam_local_break.npz | 3 KB | ARCHIVE | sim/analysis/beam_local_break.py | Adversarial separability test of the beam-local predictor: 12-tessellation first-rank counts per window and estimator (env/tf), raw (ens_*) and after regressing the orientation-blind column geometry out of the predictor (ensr_*). Backs S10. |
| axis_window_adjudication.npz | 17 KB | ARCHIVE | sim/analysis/axis_window_adjudication.py | Independent re-implementation adjudicating whether the S10 axis null is gate-specific: fabric-type separation, bandwidth/decay channels, noise floors, per-sweep axis tests and ensemble counts in all three windows, untreated and under median removal, plus Monte Carlo nulls. Backs S10. |
| late_window_mechanism.npz | 101 KB | ARCHIVE | sim/analysis/late_window_mechanism.py | The late-gate test of the Fresnel-vs-grain length-scale mechanism: per-window scale ratios, backwall-leakage imports and drop/only estimator variants for the specimen and its two zero-scattering controls across six gates from 24-36 to 40-48 us, with ensemble counts, paired tests and axis errors. Backs S10 (late gates). |
| late_window_adjudication.npz | 40 KB | ARCHIVE | sim/analysis/late_window_adjudication.py | Clean-room adjudication of the late-window result (sections A-F), stored as formatted STRING tables: harness pass flag, the 44 us insonification geometry, three-way backwall-leakage measurements, control ranks with exact floors, the bulk-velocity confound regression and ensemble rows. Backs S10. |
| late_window_adjudication_placebo.npz | 6 KB | ARCHIVE | sim/analysis/late_window_adjudication.py | The placebo arm of the same run (sections G-H): placebo-window rows (plac) and the window profile (prof), as formatted strings, written beside the main archive. Backs S10. |
| coda_noise_floor.npz | 9 KB | ARCHIVE | sim/analysis/coda_noise_floor.py | Sections A-E of the noise-sensitivity study: field correlation, registration rank and identification rank vs in-band SNR (17 levels x 20 realisations), the noiseless baseline against tessellation_replication.npz, the coda-to-backwall level, the wideband conversion factor and the coherent ring contamination sweep. Backs S17 (identification intact at 0 dB SNR, chance at -25 dB). |
| coda_noise_floor_speckle.npz | 5 KB | ARCHIVE | sim/analysis/coda_noise_floor.py | Section F of the same study: implied gamma shape parameter per gate span vs SNR, showing additive noise can only push the below-unity speckle statistic up. Backs the speckle rejection of S6 and main Sec. 4.1. |
| grain_axes_volumes_s<seed>_k<kappa>_a<ax>.npz (13 files) | 3 KB each | CACHE | sim/analysis/tof_axis_recovery.py (written on first use) | Cached CPU rebuild of one tessellation's volume-weighted orientation data: c-axes (n_grains x 3, about 100-108 grains per seed) and per-grain voxel volumes. Feeds the time-of-flight axis and concentration recovery of S12 (tab:axisrecovery); also read by channel_network/edgedge1_axis.py and late_window_adjudication.py. |
| cross_fabric_coda_rows.npy | 42 KB | ARCHIVE | sim/analysis/cross_fabric.py | Pickled object array (needs allow_pickle=True) of 9 rows (sweep name, solver group, azimuths, measured coda power, predicted sigma_d^2 per azimuth) for the adversarial test that the coda level tracks 10log10(sigma_d^2) 1:1 across fabrics at fixed solver settings. Exploratory audit; not in the paper's claim map. |

## sim/analysis/channel_network

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| edge1.npz | 5 KB | ARCHIVE | sim/analysis/channel_network/edgedge1_axis.py | Per-specimen time-of-flight axis errors (template, oracle, 45-degree template), realised vs recovered Watson kappa and A2 amplitudes for 13 tessellations (adds singlemax_seed7_ppw8_ensemble to the S12 set). Edge 1 of the channel network: does fabric type gate the axis channel. Cited in the Sec. 5 discussion-scope row. |
| edge1_a2_vs_kappa.npz | 63 KB | ARCHIVE | sim/analysis/channel_network/edgedge1_axis.py | Model curve of the two-fold TOF harmonic amplitude abs(A2) over 4000 kappa values with the null-crossing kappa, giving the interval where the axis channel is unusable. Backs the Sec. 5 discussion scope. |
| edge23.npz | 4 KB | ARCHIVE | sim/analysis/channel_network/edgedge23_links.py | Edges 2 and 3 measured link by link on the eight girdle tessellations: orientation-tensor eigenvalues, boundary-weighted misorientation, ray-seen and disc-averaged abs(dv)/2vbar, correlated against measured coda level, identification and field scores. Backs the Sec. 5 discussion scope. |

## sim/analysis/facet_model

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| t2p_<sweep>__<tag>.npz (8 files) | 2.4 MB each | CACHE | sim/analysis/predict_field.py | Cached azimuth-by-time coda power fields for one measured sweep under one prediction geometry: measured envelope power (meas) on tgrid (20-42 us), backwall reference e1, and the four prediction variants p_full, p_geom, p_nodirec, p_flat. Behind the S9 fabric-to-coda channel analysis and the Sec. 4.2/4.3 field correlations; read by centring_convention.py, shift_null_2d.py and sim/figures/fig_codafield.py. NOTE: predict_field.py writes via _t2_common.HERE = sim/analysis, one directory up from where these files now sit; a rerun recreates them there. |
| specimen_build_s11_p8_k-8.npz | 685 KB | CACHE | sim/analysis/facet_model/range_domain_common.py (build_spec cache) | Cached full DiskSpecimen build of the seed 11 girdle tessellation at ppw 8, kappa -8: label volume (416x416x146 int16), c-axes, seed points, Laguerre weights and grid step h. Shared machinery for the S9 wrong-fabric cases. |
| bicrystal_val.npz | 1 KB | ARCHIVE | sim/validation/two_grain_reflection.py (writes beside itself in sim/validation; this copy was homed here in the restructure, per sim/analysis/README.md) | The measured single-interface validation set: a 6x6 res matrix of bicrystal reflection measurements over the contrast ladder, compared against the exact anisotropic reflection coefficient. Backs the S4 single-interface claim (r = 0.966 over five contrasts). NOTE: readers exact_rc_psib_sweep.py and exact_rc_residual_scatter.py still load it from Documents/pulse-echo-analysis-scratch, outside the repo. |

## sim/analysis/nfcache

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| march_s<seed>.npz (48 files: seeds 7-89 true + 100-139 decoys) | 36 MB total | CACHE | sim/analysis/coda_noise_floor.py | Marched candidate geometry for one of the 48 candidate tessellations: concatenated facet-event weights (w) and ranges (s) for the 30 common azimuths with per-azimuth offsets (off); geometry only, no dv. Built once so only the noisy measurement is redrawn per realisation of the S17 noise study. |
| field_s<seed>.npy (8 files, true girdle seeds) | 1.1 MB total | CACHE | sim/analysis/coda_noise_floor.py | The facet-model predicted azimuth-by-time power field of one true tessellation, restricted to the 24-36 us gate (30 azimuths x gate samples): the noiseless predictor side of the S17 noise study. |

## sim/analysis/physical_optics

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| po_src_meas.npz | 2 KB | ARCHIVE | sim/analysis/physical_optics/po_src_01_measured_levels.py | Source-referenced measured levels on the 30 common azimuths of the ppw 10 sweep: coda gate level (coda10), backwall envelope peak (e110) and recorded excitation bang level (bang10). Step 1 of the S4 physical-optics absolute anchor. |
| po_src_geom.npz | 147 KB | ARCHIVE | sim/analysis/physical_optics/po_src_02_facet_geometry.py | Exact grain-boundary facet polygons of the seed 11 girdle specimen from half-space clipping of the Laguerre tessellation: seeds, weights, c-axes, per-facet grain pair, normal, in-plane basis, origin, area, centroid and polygon vertices (536 facets). Backs S1's facet-model constants and main Sec. 2.1's Eq. (1). MASTER copy; sim/figures/data holds a committed duplicate. |
| po_src_pred.npz | 1 KB | ARCHIVE | sim/analysis/physical_optics/po_src_03_kirchhoff_pred.py | Physical-optics predicted coda gate level per azimuth (30 azimuths) from the Kirchhoff integral over the exact facets, referenced to the source amplitude, no fitted constants. Step 3 of the S4 anchor. |
| po_src_diag.npz | 123 KB | ARCHIVE | sim/analysis/physical_optics/po_src_03_kirchhoff_pred.py | Per-azimuth and per-facet diagnostics of the PO integral: visible facet counts (nfac, n6) and the 3873 facet-contribution records (incidence angle, reflection coefficient, arrival time) behind the glint-domination analysis of S4. |
| po_src_dense.npz | 3 KB | ARCHIVE | sim/analysis/physical_optics/po_src_05_sampling_stats.py | The PO level on a dense 180-point (2 degree) azimuth grid, used to quantify the sampling error of the 30-azimuth ensemble mean, the only stable statistic of a glint-dominated level. Step 5 of the S4 anchor. |
| po_src_shape.npz | 260 KB | ARCHIVE | sim/analysis/physical_optics/po_src_09_envelope_shape.py | Azimuth-averaged coda envelope vs delay, predicted (t, po) and measured (tm, me), locating where the predicted energy actually sits after the gate-sensitivity discrepancy. Step 9 of the S4 anchor. |
| po_src_final.npz | 5 KB | ARCHIVE | sim/analysis/physical_optics/po_src_10_backwall_speed.py | The calibrated PO azimuthal level (180 azimuths) using each azimuth's own measured backwall speed (c) instead of the nominal 3850 m/s for the range-to-time mapping. The corrected anchor behind S4's -88.1 dB direct vs -87.79 dB measured. |

## sim/analysis/predcache

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| pred_<sweep>.npz (5 files) | 1-9 KB each | CACHE | sim/analysis/observable_panel.py (fabric_pred cache) | Cached bulk fabric predictor for one sweep: E[R^2] in dB (er2) and mean qP speed (vbar) at each cached rotation, matched on azimuth modulo 180. Ships with the repo because a rebuild rasterises on the GPU. Feeds the S10 22-observable panel via observable_panel.py, panel_fwer.py and fabric_channel/panel_run.py. |
| observable_panel_results.npy | 2.3 MB | ARCHIVE | sim/analysis/fabric_channel/panel_run.py | Pickled dict (needs allow_pickle=True) of the full observable-panel run for four sweeps (girdle_seed11_ppw8_dev, girdle_seed11_ppw6_axis_perp, singlemax_seed11_ppw8_12az_check, isotropic_seed41_ppw6_calibration): per-observable correlations, permutation and cyclic p-values, Holm corrections and family-wise summaries at 20000 permutations. The S10 panel FWER numbers in raw form. |

## sim/fe_crosscheck

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| fe_arbiter_round1_baseline_traces.npz | 63 KB | ARCHIVE | sim/fe_crosscheck/fe_arbiter_round1_baseline.py | Arbiter round 1 evidence: raw receiver traces from the staircased-FDTD vs conforming P2+ FE cross-check on the same 12-grain specimen (fa/fb = FDTD with random A vs uniform B c-axes; ea/eb = the FE pair). Round 1 forensics found -19.7 dB box reverberation in FE B (overdamped shell), motivating round 2. Feeds supplementary S2. |
| fe_arbiter_round2_padded_box_traces.npz | 63 KB | ARCHIVE | sim/fe_crosscheck/fe_arbiter_round2_padded_box.py | Round 2: FE box enlarged with an 8 mm uniform-crystal pad and quartic graded absorber so FE solves the same problem as the FDTD reference; FE traces interpolated to a common 2 ns clock. Same fa/fb/ea/eb layout. Feeds S2. |
| fe_arbiter_round3_source_match_traces.npz | 63 KB | ARCHIVE | sim/fe_crosscheck/fe_arbiter_round3_source_match.py | Round 3: SOURCE-MATCHED shot on the round 2 padded mesh; FE source switched to a centre of dilatation (P only) and receiver to div u, the FE counterparts of the FDTD stress monopole/pressure, because round 2's residual -19 dB floor was S-wave physics the FDTD cannot see. Delivered the 0.03 dB attenuation cross-validation. Feeds S2. |
| fe_arbiter_round4_deep_srcrec_traces.npz | 63 KB | ARCHIVE | sim/fe_crosscheck/fe_arbiter_round4_deep_srcrec.py | Round 4: src/rec moved 6 mm deeper (SRC x = 12 mm, REC x = 21 mm, same 9 mm path) so every x = 0 mirror-path echo arrives after 9.3 us, outside the 4.9-6.2 us scoring window; isolates residual numerical S leakage of the discrete monopole. Feeds S2. |
| fe_arbiter_round5_gauss_ball_traces.npz | 63 KB | ARCHIVE | sim/fe_crosscheck/fe_arbiter_round5_gauss_ball.py | Round 5 (final): Gaussian-ball monopole FE source to kill the numerical S leak identified in round 4, on the shifted geometry. Closes the five-round FE ladder (-19.5 to -46.2 dB) cited in S2. |
| fe_p2_test.npz | 5.3 MB | FIXTURE | sim/fe_crosscheck/fe_mesh.py (mesh_polycrystal, order 2) via a one-off call; no script in the repo passes this filename today | Committed gmsh TET10 test-mesh fixture (68045 nodes, 46402 ten-node tets, per-tet grain ids) of a grain-conforming Voronoi box; the only npz whitelisted in .gitignore. Patch-test mesh: a linear field on it must give exact Gauss-point strains and vanishing interior forces. Read by fe_p2_probe.py and fe_p2plus_probe.py (both skip gracefully if absent). |

## sim/figures/data

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| po_src_geom.npz | 147 KB | ARCHIVE (duplicate) | copy of sim/analysis/physical_optics/po_src_geom.npz (written there by po_src_02_facet_geometry.py) | Committed duplicate of the exact facet geometry so sim/figures/fig_scales.py (and fig_regime.py through it) run from the repository alone. The physical_optics copy is the master; keep the two in sync or regenerate this one from it. |

## sim/ref

NOTE (restructure hazard): every producer and reader of this directory
computes its reference path as `<module_dir>/ref` (fw_reference_2mhz_2mhz.py,
fw_reference_5mhz.py, freq_blips.py, fw_crop_validate_ray_model.py, fw_e1_corridor_validate_ray_model.py,
validate_ps.py in sim/fw_checks; azimuth_sweep.py in sim/pipeline;
fit_sweep.cal_path in sim/model). After the restructure those resolve to
sim/fw_checks/ref, sim/pipeline/ref and sim/model/ref, none of which exist:
sim/ref is currently DISCONNECTED, so a rerun would silently create a new ref
directory and a read would fail until the paths are repointed.

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| fw_fabric{00,25,50,100}.npz (4 files) | 40 KB each | ARCHIVE | sim/fw_checks/fw_reference_2mhz_2mhz.py | Full-wave reference trace for the model-vs-full-wave ladder: one 2 MHz, ppw 6, order 8 trace with NN percent of the c-axes randomised (00 = none, 100 = all). Keys trace/dt/f0 plus E1, level_db and the run seeds; printed levels must reproduce the recorded -28.1/-32.9/-31.6/-43.4 dB rungs. |
| fw_az{17,31,47,62,76,90}.npz (6 files) | 38 KB each | ARCHIVE | sim/pipeline/azimuth_sweep.py | Full-wave coda trace with the whole specimen (grains and c-axes) rotated rot_deg against the fixed probe, one probe azimuth per file. Built to test whether the azimuth MEAN approaches the incoherent model; found az 76 is a 20 dB beam walk-off outlier. Keys trace/dt/f0/rot_deg/level_db/fluid_damp. |
| fw_3mhz.npz | 48 KB | ARCHIVE | sim/fw_checks/freq_blips.py | Single full-wave trace at 3 MHz, ppw 6, for the frequency-blips prediction check between the 2 and 5 MHz anchors. |
| fw_reference_5mhz_fabric00.npz | 79 KB | ARCHIVE | sim/fw_checks/fw_reference_5mhz.py | 5 MHz production-recipe full-wave trace on the fabric00 specimen (about 2.6 h, 8 GB on the RTX 4070); the 5 MHz reference every corridor and crop validation compares against. |
| fw_reference_5mhz_e1corridor.npz | 78 KB | ARCHIVE | sim/fw_checks/fw_e1_corridor_validate_ray_model.py | 5 MHz trace recomputed on the cropped E1-corridor domain (half_w = corridor half width), validating that the cheap corridor crop reproduces the full-domain backwall echo. |
| coda_shape_cal_2mhz.HOLD.npz | 3 KB | ARCHIVE (parked) | sim/model/fit_sweep.py calibration builder (writes coda_shape_cal_<f>mhz.npz atomically) | Twin-learned coda shape correction vs beam-to-fabric-axis angle: psi grid (91 points), corr_db added to the model, source sweep names, kappa/alpha at fit. The active file was parked as .HOLD to switch the calibration OFF (fit_sweep comment, 2026-07-31, kappa identifiability); renaming it back silently re-enables it. Per frequency by design: a 5 MHz fit must never load the 2 MHz curve. |
| coda_shape_cal_2mhz.legacy.npz | 3 KB | ARCHIVE (superseded) | sim/model/fit_sweep.py calibration builder | Superseded single-source version of the same 2 MHz shape correction, kept for provenance. |

## sim/results

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| rsg_green_{rsg,ssg}_{iso,ani}_ppw{4,6,10}.npz (12 files) + rsg_green.log | 3.4 MB total | ARCHIVE | sim/validation/rsg_green_probe.py | Discrete Green's function of each scheme with NO interface: one homogeneous run per scheme, medium and resolution. rec is time-by-receiver traces for 45 receivers tagged ARC (fixed radius, every few degrees in x-z), RAY (four radii per direction, tests 1/r) and NBR (cells transverse to the arc point, tests single-cell readability), with ang/rad per receiver and dt/h/N/vp. |
| rsg_green_report.log | 26 KB | ARCHIVE | sim/validation/rsg_green_report.py | Verdict tables from the Green probe: Q1 is the isotropic arc flat in amplitude and time, Q2 does the field spread as 1/r, Q3 does a single cell mean anything; envelope peak vs gate energy separates gain from waveform distortion. |
| rsg_lattice_{rsg,ssg}_ppw6.npz (2 files) + rsg_lattice.log | 1 MB total | ARCHIVE | sim/validation/rsg_lattice_probe.py | Homogeneous isotropic run with a receiver at EVERY cell along the x axis and the x-z face diagonal (off = integer cell offsets, tag = parity class): a resolved wave decays smoothly as 1/r, a grid-scale 2h mode gives a sawtooth. Measures the modulation depth that would make one-cell readings meaningless on the rotated grid. |
| rsg_twoway_{rsg,ssg}_ppw{6,8,10}.npz (6 files) + rsg_twoway.log | 1.3 MB total | ARCHIVE | sim/validation/rsg_twoway_run.py | The two-way reference archive: per tilt t, bicrystal (bi_t) and homogeneous (ho_t) traces at the source cell plus four beam receivers at 0.25-1.0 of the image range, with dt_t, true and nominal receiver radii and cell indices. Feeds the propagation-injection gain G(theta) and the echo-to-noise ratio; saved after every tilt so a kill loses one pair. |
| rsg_twoway_report.log | 14 KB | ARCHIVE | sim/validation/rsg_twoway_report.py | Decision tables on the two-way batch: echo(theta), error(theta), G(theta), the residual S(theta) left for the interface, the ENR split before and after the gate, and the waveform correlation rho_wf against tilt 0. |
| rsg_ref_audit.log | 9 KB | ARCHIVE | sim/validation/rsg_ref_audit.py | Audit of the 8-11 dB homogeneous tilt swing from the archived rsg_twoway traces: separates (A) real radiated-energy change, (B) waveform-shape change moving the envelope peak, and (C) receiver-cell snapping across the parity classes of the rotated operator's sublattice. |
| rsg_contrast_{rsg,ssg}_ppw6_t0.npz (2 files) + rsg_contrast_probe.log | 2 KB total | ARCHIVE | sim/validation/rsg_contrast_probe.py | Echo amplitude vs far-grain c-axis misorientation psi_b at tilt 0, ppw 6 (psi and amp, 3 points): a genuine specular echo falls with the true reflection coefficient on both schemes, a numerical floor does not. ssg is the control; reuses the stored two-way homogeneous runs at the stored dt. |
| rsg_diagnose.npz + rsg_diagnose.log | 7 KB | ARCHIVE | sim/validation/rsg_diagnose.py | Seven-tilt diagnosis of the rotated grid's 45-degree divergence: per tilt the dt, Courant number, echo amp, homogeneous peak and energy, and late-time growth of both runs. Excludes marginal instability, a contaminated homogeneous reference, and face-vs-body-diagonal orientation as explanations. |
| rsg_repro_ppw{8,10}_t{0.0000,45.0000}.npz (4 files) + rsg_reproduce.log | 111 KB total | ARCHIVE | sim/validation/rsg_reproduce.py | Independent reproduction of the published rotated-grid numbers via the published functions unmodified: one file per (ppw, tilt) with the bi/ho source-cell traces plus gate position, echo amp, ENR, Courant and late-time growth. Filename encodes ppw and tilt in degrees (%.4f). |
| rsg_dt_control.log | 1 KB | ARCHIVE | sim/validation/rsg_dt_control.py | Control run: the homogeneous tilt 0 case rerun at the time steps the other tilts chose; the reference amplitude barely moves, excluding the 2.6 percent Courant variation as the source of the two-way swing. |
| rsg_symbol.log | 9 KB | ARCHIVE | sim/validation/rsg_symbol.py | Analytic symbols of the rotated and standard staggered operators (no GPU): numerical phase velocity vs propagation angle in the x-z plane, asking whether the operator itself predicts the measured direction dependence. |
| bicrystal_ladder.npz | 1 KB | ARCHIVE | sim/runs/master_run.py (step 0) | Bicrystal resolution ladder: mean absolute echo error (dB) vs ppw at the production order; the licensing check run before the sweep programme. Read by gpu_cost.py and fig_tilterror.py. |
| master_run.log, ladder_run.log, followup_run.log, followup2_run.log | 14 KB total | ARCHIVE | redirected stdout of sim/runs/{master_run,ladder_run,followup_run,followup2_run}.py | Transcripts of the four GPU sweep programmes: master (bicrystal ladder + 17 production sweeps), ladder (zero-contrast floors at ppw 6/10 plus girdle ppw 10), followup (resolution ladder on seed 23 + four single maxima), followup2 (blinding geometry, weak girdle kappa -3, and the other claim-closing runs). gpu_cost.py reads per-run durations from them. |
| domain_absorber.npz + domain_absorber.log | 4 KB | ARCHIVE | sim/runs/domain_absorber_check.py | Coda level at six azimuths under the production domain vs a wide-padded domain vs a weakened absorber: shows the domain size and absorber settings do not set the measured coda. Backs the Sec. 4 robustness argument. |
| ensemble.npz | 8 KB | ARCHIVE | sim/analysis/ensemble_stats.py | Audited ensemble statistics at ppw 8 (powers averaged, dB once): per-seed girdle and single-maximum levels and 30-azimuth curves, quadrature error, orientation-vs-geometry variance split, ICC, fabric effect, per-seed areas and crossing counts. Feeds tab:ensemble and fig_ensemble.py. |
| coda_levels.npz | 45 KB | ARCHIVE (ORPHAN) | none in repo (readers exist) | Cached per-sweep azimuth curves, four arrays per sweep name: <sweep>_rot rotation angle, _lv trace level, _e1 backwall echo, _coda coda level, for isotropic_seed41_ppw6_calibration, singlemax_seed11_ppw6_rigid2, girdle_seed11_ppw6_axis_perp(_ppw8), singlemax_seed23_ppw6_heldout_axis, girdle_seed11_ppw6_axis_par and friends. The speckle and autocorrelation input for S6, S8 and S14 (read by azimuthal_autocorrelation.py). |
| exact_rc_psib_sweep.npz | 85 KB | ARCHIVE | sim/validation/exact_rc_psib_sweep.py | Exact anisotropic ice/ice reflection coefficient in dB on a fine psi_b grid (3591 points) with a convergence flag per point; the closed-form anchor the measured bicrystal echo is checked against. |
| two_grain.npz + two_grain.log | 3 KB | ARCHIVE | sim/validation/two_grain_reflection.py | Two back-to-back GPU validations: (A) zero-contrast control measuring the numerical floor, and (B) bicrystal sweep of the c-axis pair checking the measured echo is linear in the exact R with the right slope; res is the 6x6 result table. |
| frequency_scaling.npz | 3 KB | ARCHIVE | sim/analysis/frequency_scaling.py | Like-for-like 2 vs 5 MHz comparison on the same specimen at matched ppw: bang, E1 and coda levels (3 rows) at 40 common azimuths per frequency; coda/bang reads off the scattering regime (f^4 Rayleigh, f^2 stochastic, flat geometric). Backs S13. |
| crossing_within.npz | 15 KB | ARCHIVE | sim/analysis/crossing_within.py | The crossing-count claim rebuilt WITHIN each specimen: per-sweep correlation, shift-null rank and combined p for the paper, column-gate, axis-gate and axis-full crossing variants, the 35-column panel with Holm correction, and the between-specimen comparison alongside. |
| sample_matrix.npz (+ sample_matrix.csv) | 42 KB | ARCHIVE | sim/analysis/sample_matrix.py | LEFT side of the property-by-observable matrix, between-specimen regime: 21 specimens (8 girdle, 8 single-max, 5 controls) by 145 geometric and orientation properties from the cached tessellations alone, with the effectively constant columns named. No wave data read. |
| sample_matrix_azimuth.npz (+ sample_matrix_azimuth.csv) | 355 KB | ARCHIVE | sim/analysis/sample_matrix.py | Within-specimen regime of the same matrix: 21 specimens x 35 beam-direction-dependent properties x 60 azimuths, with within and between spread and the measured 180-degree periodicity r_az180 per column (decides 15 vs 30 distinct shifts for the null). |
| candidate_geometry.npz + candidate_geometry.log | 127 KB | ARCHIVE | sim/analysis/candidate_geometry.py | Geometric separation of the 48 candidate tessellations: boundary-offset, Hausdorff and p95/p99 distances for all 1128 pairs, each specimen's nearest wrong candidate, Monte Carlo shell parameters, and the jitter reference curve. The ratio of nearest-distractor distance to the 0.42-0.575 mm tolerance IS the difficulty of the identification test (Sec. 5.2). |
| microstructure_error.npz + microstructure_error.log | 211 KB | ARCHIVE | sim/analysis/microstructure_error.py | Perturb-the-truth degradation study: the true candidate perturbed and rescored in the 48-line-up, 1008 runs over four families (seed jitter, orientation error, missing or merged grains, weight error) x three scores, plus replay checks; converts the collapse level to mm of boundary displacement. |
| identification_transfer.npz + identification_transfer.log | 82 KB | ARCHIVE | sim/analysis/identification_transfer.py | The identification-transfer adjudication archive: 416 rescored runs including four extra jitter levels at six realisations each, per-run scores and first-place counts, per-sweep foreign score distributions, and the re-derived binomial tails; measures rather than interpolates the collapse threshold. |
| identification_transfer_partial.npz | 30 KB | CACHE (stale) | sim/analysis/identification_transfer.py | Resumable checkpoint of the refined jitter runs (216 scored runs); superseded by identification_transfer.npz, which is complete. The docstring convention says checkpoints are removed on completed runs, so this one is stale but harmless. |
| identification_transfer_geom.npz + identification_transfer_geom.log | 4 KB | ARCHIVE | sim/analysis/identification_transfer.py | Direct area-weighted mean boundary-displacement measurement by shell sampling with closed-form distances, checking the delta = fV/A conversion without the facet-area estimate; 15 rows across jitter levels plus facet areas and volume. |
| identification_transfer_section.npz + identification_transfer_section.log | 11 KB | ARCHIVE | sim/analysis/identification_transfer.py | Section-realism variants: a 7-variant x 8-specimen summary table plus 56 per-run scores, testing what a realistically traced section (position error, missing small grains, merged faint boundaries) does to identification. |
| identification_transfer_refine.log | 1 KB | ARCHIVE | sim/analysis/identification_transfer.py (refine stage) | Short transcript of the refine stage that filled the jitter interval with four levels at six realisations. |
| identification.log | 2 KB | ARCHIVE | redirected stdout of sim/analysis/identification_final.py | Range-domain identification test on full azimuth sets with the specular predictor and wide gate: per sweep the own-candidate correlation, rank of 44, wrong-candidate and shift p, z against the null, and the mirror control. |
| replication_harm.log | 8 KB | ARCHIVE | redirected stdout of sim/analysis/tessellation_replication.py | Replication tests across the eight-girdle line-up: azimuthal-scalar facet-model correlation with harmonics 0/2/4 stripped, geometric vs full predictor, shift-null p per sweep (floor 1/30). |
| noise_audit_check.log | 2 KB | ARCHIVE | sim/analysis/noise_audit_check.py | Noise-sweep thresholds recomputed from the coda_noise_floor.npz archive rather than the report that wrote them, converted into the number of coherent averages a stated single-shot receiver needs. |
| operator_ceiling.log | 2 KB | ARCHIVE | sim/analysis/operator_ceiling.py | Adjudicates the kh = 2.55 spectral pileup between the order-dependent operator ceiling and Mittet's order-independent half-spectral rule, evaluated on the deployed multistart kh_max = 2.0 coefficients. Backs the dispersion subsection of Sec. 4. |
| revoxelisation.npz + revoxelisation.log | 2 KB | ARCHIVE | sim/analysis/revoxelisation.py | Whether refining the grid changes the microstructure it represents: discrete boundary-face area and the x/y/z facet-normal mix at the three ladder resolutions, from the label arrays with no simulation. Backs the convergence argument of Sec. 4. |
| tilt_table.npz + tilt_table.log | 13 KB | ARCHIVE | sim/runs/tilt_table_run.py | Tilted-interface echo-error table for the three interface treatments (naive, aa = anti-aliased field, sm = Schoenberg-Muir) at ppw 6/8/10: per cell the summary error plus per-tilt errors and amplitudes over five tilts; resumable per cell. Read by fig_tilterror.py and gpu_cost.py. |
| tilt_ppw6.log | 1 KB | ARCHIVE | redirected stdout of sim/validation/tilt_rsg.py at ppw 6 | Transcript of the rotated-staggered-grid tilt testbed at ppw 6: echo amplitude and dB vs tilt 0 per tilt with per-run seconds. |

## out/sweeps (full-wave sweep directories, one az<ddd>.npz per completed azimuth)

All sweeps are produced by the `sim/pipeline/sweep_runner.py` engine; the
batch job scripts in `sim/runs/` (master_run.py, followup_run.py,
followup2_run.py) drive it for the programme sweeps and are named per row.
Each az file holds float32 trace, dt, f0, az_deg, first-arrival energy E1 and
time t1_s, coda_db and tail.

| Directory | Size | Azimuths | Producer | Meaning |
|---|---|---|---|---|
| girdle_seed11_ppw8_dev/ | 1.7 MB | 60 at 6 deg | sweep_runner.py | THE development specimen: seed 11 girdle (kappa -8, axis [1,0,0]) at ppw 8. Central sweep behind the Sec. 4.2/S7 identification result, the f = 1 rung of the contrast ladder (S5), and the girdle half of the seed 11 matched fabric pair (S8). The specimen most exposed to selection, hence the replication programme. |
| girdle_seed11_ppw6_axis_perp/ | 1.4 MB | 60 at 6 deg | sweep_runner.py | Seed 11 girdle (axis [1,0,0] in the scan plane) at ppw 6: the ppw 6 rung of the development specimen; girdle-geometry family member for the facet predictor and axis-window work. |
| girdle_seed11_ppw6_axis_par/ | 1.4 MB | 60 at 6 deg | sweep_runner.py | Seed 11 girdle with axis [0,0,1] parallel to the probe axis, ppw 6. Axis-parallel member of the girdle geometry family, contrasted against girdle_seed11_ppw6_axis_perp for fabric-type and pooling analyses. |
| girdle_seed11_ppw6_axis_tilt20/ | 253 KB | 11 of 60 (PARTIAL) | sweep_runner.py | Seed 11 girdle with axis tilted 20 deg from z ([0.342,0,0.94]), ppw 6. Tilted-axis girdle case for the facet predictor family; incomplete. |
| singlemax_seed11_ppw8_twin/ | 1.7 MB | 60 at 6 deg | sweep_runner.py | Seed 11 single maximum (kappa 3.93, axis [0.866,0.5,0]) at ppw 8: the single-maximum twin of girdle_seed11_ppw8_dev on the bit-identical tessellation; the seed 11 matched fabric pair of tab:ensemble (S8). |
| mx_girdle_s<seed>_ppw8/ (11 dirs: seeds 7,13,17,23,29,41,47,53,67,71,89) | 8.2 MB | 30 at 12 deg each | master_run.py (7,17,23,41,53,71,89) and followup2_run.py (13,29,47,67) via sweep_runner | Ensemble members: independent girdle tessellations (kappa -8, axis [1,0,0]) at ppw 8, one directory per seed. With girdle_seed11_ppw8_dev (seed 11) the first seven form the eight-tessellation identification and replication ensemble (Sec. 4.2, S7, S8); the four followup seeds extend it to the twelve-tessellation set the beam-local and axis-window analyses use (S10). |
| mx_single_s<seed>_ppw8/ (7 dirs: seeds 7,17,23,41,53,71,89) | 5.2 MB | 30 at 12 deg each | master_run.py (17,23,41) and followup_run.py (7,53,71,89) via sweep_runner | Single-maximum twins (kappa 3.93, axis [0.866,0.5,0]) on bit-identical tessellations to the mx_girdle seeds. With singlemax_seed11_ppw8_twin they complete the eight matched fabric pairs behind the -2.86 dB fabric main effect (S8, Sec. 4.4). |
| girdle_seed11_ppw8_contrast_f{000,025,050,075}/ (4 dirs) | 2.9 MB | 30 at 12 deg each | master_run.py via sweep_runner (treatment = contrast) | Contrast-scaling ladder rungs on the seed 11 girdle at ppw 8: each grain's stiffness interpolated a fraction f = fff/100 from the orientation-averaged tensor toward its own, so scattered power must scale as f^2. girdle_seed11_ppw8_dev is the f = 1 rung. Physical-vs-numerical partition (S5 tab:ladder) and the identification switch-on test (Sec. 4.3). |
| girdle_seed11_ppw6_zerocontrast/ and girdle_seed11_ppw10_zerocontrast/ | 1.4 MB | 30 at 12 deg each | master_run.py via sweep_runner (treatment = zero) | Zero-contrast numerical-floor controls at the outer ladder rungs: seed 11 girdle geometry with zero acoustic contrast at ppw 6 and 10. Any recorded coda is staircase artefact, giving the numerical floor at each resolution (S5). |
| girdle_seed11_ppw8_uniform_axis/ | 1.4 MB | 60 at 6 deg | sweep_runner.py | Uniform-orientation control: seed 11 girdle geometry at ppw 8 with every grain given the SAME c-axis, so the tessellation is acoustically invisible; any coda is pure staircase artefact. The 26/48-rank control of Sec. 4.3. |
| girdle_seed11_ppw6_uniform_axis/ | 406 KB | 22 of 60 (PARTIAL) | sweep_runner.py | The same uniform-orientation zero-contrast control at ppw 6; numerical-floor rung at the coarse grid, incomplete. |
| girdle_seed23_ppw6_ladder/ and girdle_seed23_ppw10_ladder/ | 1.5 MB | 30 at 12 deg each | followup_run.py via sweep_runner | Second resolution ladder: seed 23 girdle at ppw 6 and 10. With girdle_seed23_ppw8_ensemble as the middle rung they test whether the seed 11 convergence behaviour (steps -3.72/-2.54 dB) belongs to the regime or to that one specimen (S5 ladder repeat). |
| singlemax_seed11_ppw5_ladder/ and singlemax_seed11_ppw10_ladder/ | 323 KB | 6 at 60 deg each | sweep_runner.py | Seed 11 single-maximum (axis [0.866,0.5,0]) resolution-ladder rungs at ppw 5 and 10; singlemax_seed11_ppw10_ladder supplies the ppw 10 level of tab:reconcile (S5). |
| singlemax_seed11_ppw10_ladder_halfdamp/ | 104 KB | 3 at 120 deg (by design) | sweep_runner.py | Damping variant of the ppw 10 rung: fluid_damp halved to 0.012. Absorbing-layer exclusion evidence (the +1.74 dB halved-damping figure, S5). |
| girdle_seed11_ppw10_licensing/ | 953 KB | 30 at 12 deg | master_run.py via sweep_runner | The ppw 10 LICENSING run: seed 11 girdle at ppw 10, run first to decide whether the tessellation matrix could be collected at ppw 8. Also the ppw 10 girdle level in tab:reconcile (S5). |
| singlemax_seed11_ppw8_axis_normal/ | 906 KB | 30 at 12 deg | followup2_run.py via sweep_runner | Blinding-geometry control: seed 11 single maximum with axis [0,0,1] NORMAL to the scan plane. Every c-axis makes the same angle to every ray, so the facet-visibility account predicts almost no coda; tests the mechanism no in-plane sweep could. |
| girdle_seed11_ppw8_weak_kappa3/ | 766 KB | 30 at 12 deg | followup2_run.py via sweep_runner | Weak-girdle variant: seed 11 girdle at kappa -3 instead of -8. Tests the claim that the fabric axis stays unresolvable across the girdle range at a concentration where the nominal two-fold term is an order of magnitude larger. |
| girdle_seed11_ppw8_clustered/ | 768 KB | 30 at 12 deg | followup2_run.py via sweep_runner | Spatially clustered orientation variant: seed 11 girdle with spatial_corr 0.6 (all other sweeps use 0). Pairs against girdle_seed11_ppw8_dev, differing only in the orientation correlation; real ice is clustered. |
| singlemax_seed11_ppw8_12az_check/ | 342 KB | 12 at 30 deg | sweep_runner.py | Legacy 12-azimuth check sweep: seed 11 single maximum at ppw 8; a quick ppw 6 to 8 consistency check for the facet predictor. Readers label it "12-azimuth check" / "LEGACY ppw8". |
| isotropic_seed41_ppw6_calibration/ | 7.9 MB | 360 at 1 deg | sweep_runner.py | Isotropic calibration sweep: seed 41, near-isotropic fabric (concentration 0.001), ppw 6. The no-fabric control of the axis-window analyses (S10); its cached prediction is faxwin_fabric_pred_isotropic_seed41_ppw6_calibration.npz. |
| singlemax_seed11_ppw6_rigid2/ | 7.9 MB | 360 at 1 deg | sweep_runner.py | Seed 11 single maximum (axis [0.866,0.5,0]) at ppw 6, run under the corrected rigid2 axes convention (the name records the convention). Dense-azimuth legacy-geometry sweep feeding the ToF axis recovery figure (S12) and facet-model statistics. |
| singlemax_seed17_ppw6_kappa8/ | 2.0 MB | 90 at 4 deg | sweep_runner.py | Strong single maximum: kappa 8, axis [0.5,0.866,0], seed 17, ppw 6. Strong-concentration case of the time-of-flight axis and concentration recovery (S12, tab:axisrecovery, fig:tofaz). |
| singlemax_seed23_ppw6_heldout_axis/ | 7.9 MB | 360 at 1 deg | sweep_runner.py | Out-of-sample axis-recovery sweep: seed 23 single maximum with a held-out axis [-0.342,0.94,0], ppw 6. Tests the ToF axis estimator on a geometry it was not developed on (S12). |
| singlemax_seed7_ppw6_fittest_legacy/ | 7.6 MB | 360 at 1 deg | sweep_runner.py; arc images inside by sim/pipeline/arc_backproject.py | Legacy fit-test sweep: seed 7 single maximum axis [1,0,0], 2 MHz, ppw 6, record_factor 2.2, pre-rigid2 axes convention (its fitted axis reads 58.9 deg in the legacy frame). The 2 MHz partner of singlemax_seed7_ppw6_5mhz_production in the S13 frequency comparison; also holds arc_image/arc_enhanced backprojection npz and STOP/GRID_STOP flag files. |
| singlemax_seed7_ppw6_5mhz_production/ | 2.5 MB | 40 of 360 (PARTIAL) | sweep_runner.py | 5 MHz production sweep: seed 7 single maximum axis [1,0,0], ppw 6. The high-frequency side of the 2 vs 5 MHz dependence (-6.87 dB, residual about f^-1/2, S13); incomplete. |
| singlemax_seed11_ppw6_inversion_cal/ | 6.2 MB | 337 of 360 (PARTIAL) | sweep_runner.py | Validation-calibration sweep for the sweep-directory fabric inversion: seed 11 single maximum, ppw 6, record_factor 2.2, legacy axes convention. Quoted for its 0.21 dB tail figure in fit_sweep's calibration notes; incomplete. |

## out/tesscache

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| tess_s<seed>_p<ppw>_k<kappa>.npz (57 files) | 37 MB total | CACHE | sim/analysis/tessellation_replication.py (cache writer; downstream modules require the cache or rebuild with the identical recipe) | Cached rasterised candidate tessellation: labels = int16 grain-label volume on the ppw grid (about 416x416x146 at p8), axes = per-grain c-axis unit vectors, seeds = generator points, h = grid spacing. The 52 p8 k-8 entries are the identification candidate pool: 12 real specimen seeds (7,11,13,17,23,29,41,47,53,67,71,89) plus 40 distractor seeds 100-139; the 4 k3.93 entries (s11,17,23,41) are single-max builds; tess_s11_p6_k-8 is the sole ppw 6 entry. |
| gen_s<seed>_p8.npz (48 files: real seeds + distractors 100-139) | 249 KB total | CACHE | sim/analysis/candidate_geometry.py (replay_generators) | Replayed Laguerre GENERATORS of one candidate: pts = seed points, w = power weights, recovered by replaying the deterministic DiskSpecimen build stream up to grid labelling. Cached because the weight-scale bisection costs about twelve seconds per seed. Feeds the S7 candidate-set geometry result (nearest wrong candidate 2.36 mm vs 0.38-0.50 mm tolerance). |
| replay_s<seed>_p8.npz (8 files: the true girdle seeds 7,11,17,23,41,53,71,89) | 44 KB total | CACHE | sim/analysis/microstructure_error.py | Full replayed build state of one real tessellation: pts, weights, lognormal radii, base and scale of the weight bisection, keep mask of surviving grains, and ok flags recording the bit-exact checks against the cached label volume and c-axis draw. Perturbed for the S7 microstructure-error displacement sweep (0.5 mm displacement kills first rank). |

## out/observables

| File | Size | Kind | Producer | Meaning |
|---|---|---|---|---|
| observable_matrix.npz | 2.0 MB | ARCHIVE | sim/analysis/observable_matrix.py | The observable side of the property-by-observable matrix over 41 sweeps (not all 48): A = per-azimuth A-scan observables (41 x 360 az x 72, NaN-padded), S = azimuth-aggregated scalars (41 x 72), B = B-scan observables needing the whole stack (41 x 1789), with obs_names/bobs_names and per-sweep metadata (sweep, seed, kappa, ppw, axis, n_az, az_step, dt_common, t_end). Estimators are local, tapered-window, band-limited 0.8-3.0 MHz, source-referenced, on each trace's own dt. Feeds the S8 screen. |
| observable_matrix_S.csv and observable_matrix_B.csv | small | ARCHIVE | sim/analysis/observable_matrix.py | The two comma-separated summaries written beside the npz: per-sweep values of the 72 azimuth-aggregated A-scan scalars (S) and the 1789 B-scan observables (B); for human inspection. |

## ORPHANS AND SPECIAL CASES

Four files with no paper linkage (fw_fabric25b.npz,
fw_fabric25b_axes.npy, ensemble_2mhz_fabric00.npz,
skeptic_sweep.log) were pruned to the sibling archive on
2026-08-05: nothing in either document quotes them and nothing
in the release reads them. Files with no producer in the
repository but with readers or a documented role remain, listed
below. Search method for each: repo-wide
grep for the filename stem, for savez/save/open calls building the name, and
for WRITES docstring lines; readers were searched the same way.

* `sim/results/coda_levels.npz`: no writer found anywhere in the repo, but
  readers exist (sim/analysis/azimuthal_autocorrelation.py; analysis README
  rows S6, S8, S14). Likely a one-off extraction from out/sweeps that
  predates the restructure. Treat as an archive: it cannot currently be
  regenerated by any script.

* `sim/fe_crosscheck/fe_p2_test.npz`: not an orphan on the read side (two
  probes consume it, both skipping gracefully if absent) but no current
  call site writes this filename; fe_mesh.py's savez signature
  (nodes/tets/grain) matches it exactly and mesh_polycrystal(order=2)
  produces its TET10 layout, so it is a deliberately committed fixture whose
  generating call was one-off. It is the only npz whitelisted in .gitignore.

Related notes: the sim/ref disconnection recorded by the first
edition of this manifest is FIXED (all five ref-path constants
now resolve to the shared sim/ref one level up); coda_shape_cal_2mhz exists only as
.HOLD and .legacy, meaning the shape calibration is deliberately OFF and
renaming the HOLD file back silently re-enables it; predict_field.py would
recreate the t2p_* caches in sim/analysis rather than sim/analysis/
facet_model; exact_rc_psib_sweep/3.py now read bicrystal_val.npz from
sim/analysis/facet_model in-repo; and cross_fabric_coda_rows.npy plus predcache/observable_panel_results.npy are pickled object
arrays that need allow_pickle=True and are fragile across numpy versions.

"""Fabric inversion on a SWEEP directory: the validated fit_fabric
machinery (Watson ODF, factorised forward, Huber loss) fed by whatever
azimuths a sweep has produced so far.

Reuses fit_fabric's functions verbatim - the observables per azimuth
(absolute coda RMS in the clean window, E1 ToF) are recomputed from the
saved traces with the exact same formulas, the geometric envelope is
built from the SWEEP'S OWN specimen and frequency (not the 2 MHz ladder
standard), and the same grid + Nelder-Mead runs. The slow one-time part
(the rung-3 envelope per azimuth, ~1-3 s each) is cached in the sweep
directory, so refits after more azimuths arrive are fast.

CLI:  python fit_sweep.py <sweep_name>   -> writes <dir>/fit_result.json
Works from 6 azimuths (the validated minimum) up to the full 360.
"""
import json
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys
import time

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

from scipy.signal import hilbert               # noqa: E402
from scipy.optimize import minimize            # noqa: E402

import fit_fabric as FF                        # noqa: E402
import ladder                                  # noqa: E402
import specimen as S                           # noqa: E402
import sweep_runner as SW                      # noqa: E402
from specimen import DiskSpecimen              # noqa: E402

C_REF = 3850.0
D = 0.100
W0, W1 = ladder.W_CLEAN

# ── coda-estimator upgrades (work on SAVED traces, no resimulation) ──
# FD_BANDS > 1: frequency diversity - split the coda's spectral support
# into sub-bands and combine their window energies: each sub-band sees
# a different speckle realization of the same facets, so glint variance
# drops while the fabric-driven mean is unchanged.
#
# HISTORY. The first FD implementation was BROKEN (found 2026-07-31):
# it applied a BRICK-WALL spectral mask to the WHOLE record and only
# then windowed, so each sub-band trace was convolved with a sinc whose
# 1/t sidelobes dragged E1 (about 45 dB louder than the coda) into the
# 24-36 us window. Measured on singlemax_seed11_ppw6_rigid2, 360 az:
#   corr(FD3 observable, plain window RMS)        = +0.004
#   corr(FD3 observable, WHOLE-RECORD band energy)= +0.872 (0.25 dB)
#   azimuthal std 4.07 -> 0.51 dB, k=2 1.53 -> 0.03 dB
# The observable had stopped measuring the coda window at all: its k=2
# was 0.03 dB on a kappa 3.93 specimen and 0.05 dB on the kappa~0
# isotropic_seed41_ppw6_calibration control. Every kappa produced through it is void. It was
# pinned to FD_BANDS=1 (plain window RMS) as a mitigation.
#
# REPAIRED 2026-07-31 (this block). Three things had to change, and
# only the third is what actually buys anything:
#  1. GATE FIRST. The record is zeroed outside W_CLEAN + FD_GUARD with
#     a raised-cosine ramp BEFORE any band split, so E1 is multiplied
#     by exactly zero and cannot leak through any mask sidelobe.
#     Verified: gate-then-full-band correlates 1.0000 with the plain
#     window RMS and moves k=2 by 0.008 dB - the gate is inert on its
#     own, which is the point.
#  2. TAPERED masks. A COLA Hann bank (centres on equal splits, full
#     width 2*spacing, so the masks sum to 1) instead of brick walls.
#     Measured difference vs gated-rectangular and vs a time-domain
#     4th-order Butterworth bank: all three agree to <0.06 dB in k=2
#     once the gate is in place. Taper choice is NOT what mattered.
#  3. SPAN + COMBINER, which is what mattered. See FD_SPAN and
#     FD_COMBINE below.
FD_BANDS = 3
# FD_SPAN: the bank must cover the coda's ACTUAL spectral support, not
# the pulse band. Measured on the 24-36 us window (2026-07-31): the
# window spectrum PEAKS at ~4.9 MHz on a 2 MHz probe and 64-75% of its
# energy sits above 3 MHz, where the two-way E1 pulse spectrum is 35-45
# dB down - i.e. the coda window is dominated by numerical grid noise
# piling up below the grid's spatial Nyquist c/(2h) = f0*ppw/2 = 6.0
# MHz (it collapses by 50 dB above 5.4 MHz, the classic signature).
# Restricting the bank to the pulse band (the old 0.5-1.5 f0) throws
# away 3/4 of the energy AND fails the isotropic control outright:
# gate + a SINGLE 1-3 MHz Hann mask, no split at all, gives k=2 =
# 1.68 dB at phi2 13.7 deg on isotropic_seed41_ppw6_calibration (p<1e-4) vs 1.20 dB (p=0.058)
# on singlemax_seed11_ppw6_rigid2 - a BIGGER 2-theta on NO fabric than on kappa 3.93,
# and every 1-3 MHz sub-band variant inherits it (iso 0.74-1.98 dB,
# p<0.01, at phi2 7-24 deg). Narrow-band glint has a long azimuthal
# correlation length and forges a 2-theta line out of nothing.
# Span is derived, not tuned: [0.5*f0, f0*ppw/2] scales to any
# frequency/resolution (5 MHz ppw 6 -> 2.5-15 MHz).
FD_SPAN = (0.5, None)       # (x f0, x f0) - None = f0*ppw/2, grid Nyq
FD_GUARD = 3e-6             # flat gate guard either side of W_CLEAN
FD_RAMP = 2e-6              # raised-cosine ramp outside the guard
# FD_COMBINE: 'geo' = geometric mean of the sub-band window energies
# (mean of their dB). NOT the arithmetic mean the original code used:
# over a bank that covers the whole support, sum(sub-band window
# energies) = total window energy by Parseval, so the ARITHMETIC mean
# is the plain FD1 observable divided by the band count - measured
# rho(full-band arithmetic FD3, FD1) = +0.995. Arithmetic FD is not
# frequency diversity at all; it is FD1 with extra steps, which is why
# the band-count evidence below never showed up as a real gain.
# The geometric mean is the standard log-domain multi-look speckle
# estimator, and it is also what stops the ~4.9 MHz grid-noise spike
# from owning 75% of the observable: each sub-band gets equal weight
# in dB, so the physical pulse band gets a fair share.
FD_COMBINE = "geo"
# Measured (2026-07-30, matched seed-7 azimuths): the coda's spectral
# correlation half-width is ~167 kHz at BOTH frequencies (set by the
# 12 us window, not f0), so the 1-6 MHz support carries ~30 mutually
# independent looks and any FD_BANDS from 2 to 8 is defensible. 3 is
# kept (the pre-existing value, not a tuned one); the acceptance scan
# below is flat over that whole range, so nothing hinges on it.
#
# ACCEPTANCE (2026-07-31; isotropic control, 360 az, no calibration,
# raw per-azimuth dB, k=2 tested against a phase-randomised surrogate
# whose smooth red background is fitted to harmonics k=1..20 excluding
# k=2,4 - a PERMUTATION null is invalid here, glint has a 3-7 deg
# azimuthal correlation length):
#            singlemax_seed11_ppw6_rigid2       singlemax_seed23_ppw6_heldout_axis       isotropic_seed41_ppw6_calibration (kappa 0.001)
#   FD1      1.53 dB p=0.22    2.05 dB p=0.002   0.37 dB p=0.94
#   FD3geo   1.61 dB p=0.012   2.13 dB p=0.0002  0.07 dB p=0.996
#   az std   4.07 -> 3.08      3.52 -> 3.22      3.54 -> 3.06
# The repaired estimator turns singlemax_seed11_ppw6_rigid2's k=2 from undetectable
# into detectable while the control stays dead. Do NOT read the
# fabric/control amplitude RATIO as 22x: iso's 0.07 dB is a noise draw
# (its own background is 1.16 dB), so the denominator is meaningless -
# the p-values are the honest statement. Robustness scan over 36
# (span x band-count) combinations: rigid p<=0.039 and iso p>=0.71 in
# EVERY one, so the win is not a selection artifact.
# NOT FIXED, and not fixable here: the k=2 PHASE of this observable
# still does not read the fabric axis (rigid phi2 141.0 vs axis 28.8;
# oos 82.8 vs 104.3 - inconsistent with each other by 43 deg under any
# common offset, and by 50 deg after the geometric envelope is divided
# out, so it is not a geo contamination either). Neither FD1 nor any
# repaired variant achieves it. Axis information lives in the C2T
# channel (per-time-bin coherent 2-theta), which does read it.
#
# ⚠ THE REPAIR DOES NOT BUY A TRUSTWORTHY KAPPA. Two-stage fits, cal
# off, sigma 1.82 (2026-07-31), with a 15-deg-block bootstrap (blocks
# >> the 3-7 deg glint correlation length), 120 resamples:
#   singlemax_seed11_ppw6_rigid2  kappa 1.66  boot 5-95% [1.04, 2.50]   truth 3.93
#   singlemax_seed23_ppw6_heldout_axis    kappa 3.48  boot 5-95% [2.45, 5.18]   truth 3.93
#   isotropic_seed41_ppw6_calibration      kappa 0.83  boot 5-95% [0.48, 3.02]   truth 0.001
# The point estimates improve a lot on the control (FD1 gave rigid
# 1.563 vs iso 1.546 - literally no separation; FD3geo gives 1.66 vs
# 0.83), but the CONTROL'S OWN bootstrap interval reaches kappa 3.02
# and swallows singlemax_seed11_ppw6_rigid2 whole. A fabric-free specimen can still
# return kappa 3 by chance. Freezing the axis and scanning it makes
# the same point: isotropic_seed41_ppw6_calibration's kappa wanders 0.30-1.81 over frozen axes.
# So: the k=2 OBSERVABLE now discriminates fabric from no-fabric, the
# fitted KAPPA still does not. Do not quote a coda-channel kappa
# without its bootstrap interval.
#
# ROOT CAUSE, and it is not in this file: at ppw 5-6 the 24-36 us
# window is mostly NUMERICAL GRID NOISE. Coda-window spectral peak vs
# resolution (2 MHz probe, all sweeps, 2026-07-31):
#   ppw  5 -> 4.13 MHz, 99.4% of window energy above 3 MHz, +29.5 dB
#   ppw  6 -> 4.82 MHz, 86.3% above 3 MHz, +20.1 dB     <- production
#   ppw  8 -> 3.24 MHz, 31.8% above 3 MHz,  +5.1 dB
#   ppw 10 -> 2.80 MHz, 28.1% above 3 MHz,  +6.5 dB
# ("+X dB" = peak level above the level at f0.) The artifact sits at
# 0.8x the grid Nyquist at ppw 5-6 and CONVERGES AWAY by ppw 8-10,
# where the window reverts to the physical 1-3 MHz pulse band. The
# geometric mean over a full-support bank is a mitigation - it caps
# the grid-noise band at 1/nb of the observable instead of 3/4 - not
# a cure. The real fix is resimulating the 360-az sweeps at ppw >= 8.
FD_ACCEPTANCE_NOTE = "see block above; kappa needs ppw>=8 sweeps"


def _fd_bands(f0_mhz):
    # One band count for every frequency now that FD_SPAN scales with
    # f0*ppw: the 5 MHz sweeps used 8 sub-bands of the (broken)
    # brick-wall path, and their "30 vs 12 independent looks" evidence
    # is about the SPAN, which is now handled by FD_SPAN directly.
    return FD_BANDS


def _fd_gate(n, dt):
    """Unity across W_CLEAN + FD_GUARD, raised-cosine down to exactly
    zero over FD_RAMP outside it. Applied BEFORE any band split so no
    mask sidelobe can pull E1 into the window."""
    t = np.arange(n) * dt
    g = np.zeros(n)
    lo_f, hi_f = W0 - FD_GUARD, W1 + FD_GUARD
    g[(t >= lo_f) & (t <= hi_f)] = 1.0
    m = (t > lo_f - FD_RAMP) & (t < lo_f)
    g[m] = 0.5 * (1 - np.cos(np.pi * (t[m] - (lo_f - FD_RAMP)) / FD_RAMP))
    m = (t > hi_f) & (t < hi_f + FD_RAMP)
    g[m] = 0.5 * (1 + np.cos(np.pi * (t[m] - hi_f) / FD_RAMP))
    return g


def _fd_coda(tr, dt, f0, ppw, nb):
    """Gated-then-split frequency-diversity coda amplitude.

    Gate -> COLA Hann sub-band bank over FD_SPAN -> per-band W_CLEAN
    envelope energy -> FD_COMBINE. Returns a linear amplitude on the
    same footing as the plain window RMS (a free gain absorbs the
    offset, so only its azimuthal shape is fitted)."""
    n = len(tr)
    fs = 1.0 / dt
    fr = np.fft.rfftfreq(n, dt)
    F = np.fft.rfft(tr * _fd_gate(n, dt))
    lo = FD_SPAN[0] * f0
    hi = (FD_SPAN[1] * f0) if FD_SPAN[1] else 0.5 * f0 * ppw
    edges = np.linspace(lo, hi, nb + 1)
    d = edges[1] - edges[0]
    i0, i1 = int(W0 * fs), int(W1 * fs)
    be = []
    for b in range(nb):
        c = 0.5 * (edges[b] + edges[b + 1])
        mask = np.zeros_like(fr)
        sel = np.abs(fr - c) <= d
        mask[sel] = 0.5 * (1 + np.cos(np.pi * (fr[sel] - c) / d))
        env = np.abs(hilbert(np.fft.irfft(F * mask, n)))
        be.append((env[i0:i1] ** 2).mean())
    be = np.maximum(np.asarray(be, float), 1e-300)
    if FD_COMBINE == "geo":
        return float(10 ** (np.mean(10 * np.log10(be)) / 20))
    if FD_COMBINE == "med":
        return float(np.sqrt(np.median(be)))
    return float(np.sqrt(be.mean()))
# AZ_SMOOTH_DEG > 0: average measured coda (in dB) over neighbouring
# azimuths within +/- this many degrees before fitting - glint
# decorrelates look-to-look, the fabric pattern is smooth on 10+ deg.
AZ_SMOOTH_DEG = 1.5


# ── tail channel (trace-mining study 2026-07-29) ─────────────────────
# Post-E1 tail = RMS envelope in TAIL_W re E1: 15.2 dB swing across
# fabric strengths vs 1.2 dB independent-draw spread in the saved
# traces - the strongest untapped channel. Fitted as a THIRD stage-2
# channel (same factorised doctrine as the coda, own free gain); this
# flag disables it in one place. Records shorter than TAIL_W (e.g.
# record_factor 2.2 ends ~57 us) use the available part of the window
# if at least 2 us of it exists, else the observable is NaN and the
# channel skips itself.
# DISABLED 2026-07-30 (azimuthal-decomposition study): the tail
# window's 2-theta azimuthal shape (~1.8-1.9 dB at lab-frame phase
# ~112-118 deg on BOTH seed11 and seed23) is almost entirely a
# specimen-INDEPENDENT lab-frame systematic g(t) (plausibly rotated-
# grid rasterisation) - after cross-specimen subtraction only 0.19 dB
# of fabric signal remains. Fitting the tail's azimuthal shape against
# a fabric model therefore fits an artifact. Re-enable once g(t) is
# calibrated out (isotropic-specimen run) or the rasterisation source
# is fixed. The tail LEVEL diagnostics in fit_result.json remain.
TAIL_CHANNEL = False
TAIL_W = (54e-6, 66e-6)
# Use the un-factorised shape grid (shape_grid.npz) as the coda model
# when a sweep has one. OFF: the grid is a documented negative result
# for (alpha, kappa) fitting (realization glint in candidate draws;
# jackknife 2026-07-29: kappa 2.48 grid vs 3.97 factorised on singlemax_seed7_ppw6_fittest_legacy,
# truth 3.93). Keep the grid machinery for realization studies.
USE_SHAPE_GRID = False
# Coda sigma for the SWEEP estimator (FD3geo + az-smoothing).
# RE-MEASURED 2026-07-31 on the repaired observable. The old 1.8 came
# from a 2026-07-29 jackknife run through the BROKEN brick-wall FD3
# path, so it was a property of a leaked-E1 observable and had to be
# redone. Method: clean-band residual rms of the two-stage fit with
# the axis FROZEN at each sweep's known sample axis and the shape
# calibration OFF (ref/coda_shape_cal_2mhz.npz parked as .HOLD.npz),
# iterated to self-consistency because sigma enters the Huber gain:
#   singlemax_seed11_ppw6_rigid2 (axis frozen 28.8):  1.66 dB   (kappa 1.67, 220 az)
#   singlemax_seed23_ppw6_heldout_axis   (axis frozen 104.3): 1.98 dB   (kappa 1.95, 220 az)
# The identical measurement on the FD1 incumbent gives 2.00 / 1.97 dB,
# so the repaired estimator is ~0.34 dB tighter on rigid and unchanged
# on oos, and 1.8 was accidentally close to right all along. chi2 moves
# a little with the new sigma and the new observable - expected, and
# chi2 values are not comparable across estimator versions.
# fit_fabric keeps its own 2.5 - that value is right for the plain FD1
# estimator it uses on the 6 ref azimuths.
SIG_CODA_SWEEP = 1.82
# SIG_TAIL (dB): measured on singlemax_seed7_ppw6_fittest_legacy (record_factor 2.2, so only
# 54-57.2 us of the window exists there): rot/rot+180 pair differences
# of the absolute tail dB give 1.9 dB rms per-azimuth incoherent
# scatter (2.71/sqrt(2)); after the mod-180 fold ~1.4 dB remains.
# 1.0 dB (jackknife 2026-07-29): the folded/smoothed tail residual is
# 0.74 dB (singlemax_seed7_ppw6_fittest_legacy) / 0.21 dB (valcal); at the earlier 2.0 the channel
# was inert (tail ON identical to OFF on both sweeps).
SIG_TAIL = 1.0

OBS_VER = (f"fd{FD_BANDS}{FD_COMBINE}-le25-t4"  # invalidates the cache
           if FD_BANDS > 1 else "fd1-le25-t4")  # on estimator change
# t2 (2026-07-30): band-limited centroid (all caches regenerate once)
# t3 (2026-07-31): FD_BANDS 3 -> 1, the brick-wall leakage bug above
# t4 (2026-07-31): FD repaired - gated/Hann/full-support/geometric,
#                  FD_BANDS back to 3 (all caches regenerate once)

# ── coherent 2-theta axis channel (azimuthal-decomposition study
# 2026-07-30) ────────────────────────────────────────────────────────
# The coherent 2-theta component of the FULL-BAND envelope dB over
# 30-44 us is fabric-locked: its phase reads the specimen fabric axis
# with NO calibration (validated: singlemax_seed11_ppw6_rigid2 phi2 27.2 vs axis 28.8;
# singlemax_seed23_ppw6_heldout_axis 97.2 vs 104.3 - vs ToF-template stage-1 errors 10.4/24.8).
# Estimator (from the study's azdecomp.py/followup.py - do not vary):
# per-0.5-us-bin least squares of m0 + A2 cos2 + B2 sin2 + A4 cos4 +
# B4 sin4 over azimuth, then a COHERENT complex average of the
# (a2 + i b2) vectors over the window bins; incoherent glint phases
# cancel, the fabric phase survives. It MUST be computed on the
# full-band envelope: FD sub-banding destroys the cross-band phase
# coherence that is the mechanism. The 30-44 us window is the
# validated near-artifact-free choice (the lab-frame systematic g(t)
# lives mainly at 12-32 us and in the tail).
C2T_CHANNEL = True
# 30-36 us (was 30-44): the window jackknife (2026-07-31, three-
# specimen tests) measured 30-36 closest to truth on BOTH rigid
# sweeps (+0.9/-4.3 deg vs -1.6/-7.1 at 30-44), with a shared lab-
# locked downward phi2 drift when the window extends into 36-44 =
# the artifact's fade-band phase leaking in. Window systematic to
# quote: +-2.5 deg (1 sigma over window choice).
C2T_W = (30e-6, 36e-6)
C2T_DTBIN = 0.5e-6
# Minimum azimuths, measured on singlemax_seed11_ppw6_rigid2 progressive-order
# prefixes (2026-07-31): n<=24 -> phi2 errors up to 16 deg with
# bootstrap se 24-47 (no information beyond ToF); n=48-96 -> errors
# 3-12 deg, se 9-37 (comparable to ToF, honest se); n>=128 -> errors
# 1.5-9 deg, se 6-17 (beats ToF). Enable at 48: below that the se
# estimate itself is unreliable, above it the (d/se)^2 weighting
# self-deweights an uncertain phase.
C2T_MIN_AZ = 48
C2T_MIN_BINS = 10           # >= ~80% of the 12-bin 30-36 us window
C2T_SE_FLOOR = 2.0          # deg; do not let a lucky bootstrap zealotise
C2T_N_BOOT = 1000
# ToF-information deflation when COMBINING with the 2-theta penalty:
# the ToF Huber sum treats azimuthally-CORRELATED glint residuals as
# independent 0.3-us noise, so its chi2 curvature implies ~1 deg axis
# precision while its validated axis errors are 10-25 deg - the sum
# overstates axis information by ~x200 (profiles on singlemax_seed11_ppw6_rigid2 /
# singlemax_seed23_ppw6_heldout_axis, 2026-07-31: chi2 rises 80-340 over the 8-15 deg the
# axis is actually wrong by). Dividing by a positive constant does
# NOT change the pure-ToF argmin, so the channel-off path is exactly
# the historical fit; with the channel on, the calibrated 2-theta
# phase dominates within its se and ToF breaks ties / rejects the
# wrong branch (its 1e9 guard survives deflation).
C2T_TOF_INFO = 200.0
C2T_VER = "c2t1-w3036"      # cache version (window + estimator)

# NOTE (2026-07-31): an orphaned lab-frame-2theta nuisance block sat
# here - written by a subagent that was killed mid-task before it ran
# any of the verification its own comment claimed ("kappa moved TOWARD
# truth on both rigid sweeps, beta landed at the known artifact
# phase"). It was also dead code: NUIS_2T was re-assigned False further
# down, so the later definition won. Removed. The live nuisance block
# (measured, and OFF) is with _lab_2theta below; the review
# subsequently showed the naive estimator is biased on the psi-frame
# keep band and that a free lab-frame 2-theta collapses onto alpha+90,
# i.e. it is a kappa degeneracy, not an artifact measurement.


def sweep_data(cfg):
    """fit_fabric-format rows from a sweep dir (its exact formulas).
    Per-azimuth observables are CACHED (obs_cache.npz): every fit and
    every grid inversion was re-extracting all traces (~90 s at 326
    azimuths) - with the cache a rerun costs seconds, which is what
    makes the live grid display actually feel live."""
    d = SW.sweep_dir(cfg["name"])
    cache_p = os.path.join(d, "obs_cache.npz")
    nb = _fd_bands(cfg["f0_mhz"])
    # 2 MHz keeps the historical version string (existing caches stay
    # valid); other band counts get their own tag
    obs_ver = OBS_VER if nb == FD_BANDS else OBS_VER + f"-b{nb}"
    cached = {}
    if os.path.exists(cache_p):
        z = np.load(cache_p, allow_pickle=False)
        if str(z["ver"]) == obs_ver:
            for az_, c_, e_, t_, td_, cm_, sc_ in zip(
                    z["az"], z["coda"], z["e1"], z["tof_us"],
                    z["tail_db"], z["cent_mhz"], z["speckle"]):
                cached[int(az_)] = (float(c_), float(e_), float(t_),
                                    float(td_), float(cm_), float(sc_))
    rows = []
    new_rows = False
    for az in SW.done_azimuths(cfg):
        if az in cached:
            c_, e_, t_, td_, cm_, sc_ = cached[az]
            rows.append(dict(rot=az, coda=c_, e1=e_, tof_us=t_,
                             tail_db=td_,
                             e1_db=20 * np.log10(max(e_, 1e-30)),
                             cent_mhz=cm_, speckle_contrast=sc_))
            continue
        new_rows = True
        z = np.load(os.path.join(d, f"az{az:03d}.npz"))
        tr, dt = np.asarray(z["trace"], float).ravel(), float(z["dt"])
        fs = 1.0 / dt
        e = np.abs(hilbert(tr))
        k0 = int(2 * D / C_REF * fs)
        a = max(k0 - int(2e-6 * fs), 0)
        seg = e[a:k0 + int(2e-6 * fs)]
        E1 = seg.max()
        # ToF from the LEADING EDGE (25% of peak, sub-sample), not the
        # envelope peak: forward scattering broadens/skews E1 most at
        # mid-psi azimuths, so the PEAK drifts late by up to ~0.8 us in
        # a scattering-correlated pattern that masqueraded as extra
        # velocity anisotropy (the model's ToF amplitude looked ~2/3 of
        # measured). The leading edge tracks the fastest coherent
        # energy, which is what the slowness-average model predicts.
        ipk = int(np.argmax(seg))
        thr = 0.25 * E1
        above = np.where(seg[:ipk + 1] >= thr)[0]
        j = int(above[0]) if len(above) else ipk
        if j > 0 and seg[j] > seg[j - 1]:
            frac = (thr - seg[j - 1]) / (seg[j] - seg[j - 1])
        else:
            frac = 0.0
        tof = (a + j - 1 + frac) / fs
        if nb > 1:
            # frequency-diversity coda, gated-then-split (see the
            # FD_BANDS block: gate first, Hann bank over the coda's
            # real spectral support, geometric-mean combine)
            f0v = float(z["f0"]) if "f0" in z.files else cfg["f0_mhz"]*1e6
            ppw = float(z["ppw"]) if "ppw" in z.files else cfg["ppw"]
            coda = _fd_coda(tr, dt, f0v, ppw, nb)
        else:
            coda = np.sqrt((e[int(W0 * fs):int(W1 * fs)] ** 2).mean())
        # ── trace-mining observables (2026-07-29 study) ──────────────
        # post-E1 tail RMS re E1 in TAIL_W (dB). NOTE the az*.npz
        # 'tail' key is a max-ratio stability FLAG, not this - compute
        # fresh. Truncated records (record_factor 2.2 ends ~57 us) use
        # whatever part of the window exists if >= 2 us, else NaN.
        lo_t = int(TAIL_W[0] * fs)
        hi_t = min(int(TAIL_W[1] * fs), len(tr))
        if hi_t - lo_t >= int(2e-6 * fs):
            tail_db = float(20 * np.log10(
                np.sqrt((e[lo_t:hi_t] ** 2).mean()) / max(E1, 1e-30)))
        else:
            tail_db = float("nan")
        # E1 spectral centroid (MHz) over +-2 us around the envelope
        # peak, rFFT magnitude weighting - gain-free, seed-stable.
        # BAND-LIMITED to 0.25-2 f0: the full-rfft version is biased
        # by the noise floor at 5 MHz (an azimuth read 8 MHz on it -
        # caught by the 2026-07-30 information study).
        kpk = a + ipk
        wn = int(2e-6 * fs)
        s2 = tr[max(kpk - wn, 0):min(kpk + wn, len(tr))]
        mag = np.abs(np.fft.rfft(s2))
        frq = np.fft.rfftfreq(len(s2), dt)
        f0c = float(z["f0"]) if "f0" in z.files else cfg["f0_mhz"] * 1e6
        bm = (frq >= 0.25 * f0c) & (frq <= 2.0 * f0c)
        cent = float((frq[bm] * mag[bm]).sum()
                     / max(mag[bm].sum(), 1e-30) / 1e6)
        # coda speckle contrast std(I)/mean(I), I = env^2, clean window
        # (monotonic with fabric but needs ensemble validation -
        # DIAGNOSTIC ONLY, never fitted)
        inten = e[int(W0 * fs):int(W1 * fs)] ** 2
        spk = float(inten.std() / max(inten.mean(), 1e-30))
        rows.append(dict(rot=az, coda=coda, e1=E1, tof_us=tof * 1e6,
                         tail_db=tail_db,
                         e1_db=20 * np.log10(max(E1, 1e-30)),
                         cent_mhz=cent, speckle_contrast=spk))
        cached[az] = (coda, E1, tof * 1e6, tail_db, cent, spk)
    if new_rows:
        ks = sorted(cached)
        tmp = cache_p[:-4] + f".tmp{os.getpid()}.npz"
        try:
            np.savez(tmp, ver=obs_ver, az=np.asarray(ks),
                     coda=np.asarray([cached[k][0] for k in ks]),
                     e1=np.asarray([cached[k][1] for k in ks]),
                     tof_us=np.asarray([cached[k][2] for k in ks]),
                     tail_db=np.asarray([cached[k][3] for k in ks]),
                     cent_mhz=np.asarray([cached[k][4] for k in ks]),
                     speckle=np.asarray([cached[k][5] for k in ks]))
            SW._atomic_replace(tmp, cache_p, critical=False)
        except OSError:
            pass
    rows.sort(key=lambda r_: r_["rot"])
    if AZ_SMOOTH_DEG > 0 and len(rows) > 4:
        # neighbour + MOD-180 averaging for BOTH observables. The
        # fabric patterns are exactly 180-deg periodic, but the
        # measurements are not: per-azimuth re-rasterisation jitters
        # the E1 leading edge by +-0.57 us rms between rot and rot+180
        # pairs (measured 2026-07-29) - enough to drag the stage-1
        # axis 10 deg once the second half arrived. Folding partners
        # and neighbours averages the jitter; the pattern is invariant.
        rr = np.array([r["rot"] for r in rows], float)
        cd = 20 * np.log10([r["coda"] for r in rows])
        tf = np.array([r["tof_us"] for r in rows])
        rr180 = rr % 180.0
        for i, r in enumerate(rows):
            dd = np.abs(rr180 - rr180[i])
            sel = np.minimum(dd, 180.0 - dd) <= AZ_SMOOTH_DEG
            r["coda"] = 10 ** (cd[sel].mean() / 20)
            r["tof_us"] = float(tf[sel].mean())
    return rows


GEO_VER = "g2-tail5466"    # regenerate pre-tail caches (no tail term)


def sweep_geo(cfg, rots):
    """Geometric (fabric-independent) coda envelopes for THIS sweep's
    specimen and frequency. Cached PER AZIMUTH and extended
    incrementally, so a continuously refitting sweep only pays for the
    azimuths that arrived since the last fit (the specimen build and
    all previously computed envelopes are reused from disk).

    Returns (geo, geo_tail): clean-window (W_CLEAN) and tail-window
    (TAIL_W) RMS of the SAME unit-contrast rung-3 envelope - one
    boundary_scatter call serves both channels. The born record runs
    to 67.5 us (ladder.standard_cfg), so the model tail window is
    never truncated even when the measured one is. Cache carries a
    version tag: pre-tail caches (no 'ver'/'geo_tail') regenerate in
    full, a one-time ~1-3 s/azimuth cost."""
    import born
    d = SW.sweep_dir(cfg["name"])
    cache = os.path.join(d, "fit_geo_cache.npz")
    have = {}
    if os.path.exists(cache):
        z = np.load(cache)
        if "ver" in z.files and str(z["ver"]) == GEO_VER:
            have = {int(r): (float(g), float(gt)) for r, g, gt in
                    zip(z["rots"], z["geo"], z["geo_tail"])}
    missing = [r for r in rots if int(r) not in have]
    if missing:
        print(f"[fit] computing {len(missing)} geometric envelopes "
              "(coda + tail windows)...", flush=True)
        f0 = cfg["f0_mhz"] * 1e6
        h = C_REF / f0 / cfg["ppw"]
        sp = DiskSpecimen(diameter_m=cfg["diameter_mm"] * 1e-3,
                          thickness_m=cfg["thickness_mm"] * 1e-3,
                          n_grains=cfg["n_grains"],
                          size_cv=cfg["size_cv"],
                          concentration=cfg["concentration"],
                          spatial_corr=cfg["spatial_corr"],
                          fabric_axis=tuple(cfg["fabric_axis"]),
                          seed=cfg["seed"])
        build = sp.build(h)
        mcfg = ladder.standard_cfg()
        mcfg.probe.f0 = f0
        dt_m = ladder.DT
        lo, hi = int(W0 / dt_m), int(W1 / dt_m)
        lo_t, hi_t = int(TAIL_W[0] / dt_m), int(TAIL_W[1] / dt_m)
        for rot in missing:
            _, env = born.boundary_scatter(dict(build), mcfg,
                                           azimuth_rad=np.radians(rot),
                                           unit_contrast=True)
            env = np.asarray(env, float)
            g = np.sqrt((env[lo:hi] ** 2).mean())
            gt = np.sqrt((env[lo_t:min(hi_t, len(env))] ** 2).mean())
            have[int(rot)] = (float(g), float(gt))
        ks = sorted(have)
        # per-process tmp name: two concurrent fits raced on a shared
        # tmp (one replaced it, the other crashed FileNotFound)
        tmp = cache[:-4] + f".tmp{os.getpid()}.npz"
        try:
            np.savez(tmp, ver=GEO_VER, rots=np.asarray(ks),
                     geo=np.asarray([have[k][0] for k in ks]),
                     geo_tail=np.asarray([have[k][1] for k in ks]))
            SW._atomic_replace(tmp, cache, critical=False)
        except OSError:
            pass                       # cache write is best-effort
    return (np.asarray([have[int(r)][0] for r in rots], float),
            np.asarray([have[int(r)][1] for r in rots], float))


def fit(cfg):
    # cross-process single-flight lock: the GUI's in-session guard is
    # lost on page reload, and two concurrent fits produced exactly the
    # Windows replace-while-open crash the runner already survived
    lock = os.path.join(SW.sweep_dir(cfg["name"]), "fit.lock")
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 600:
        print("[fit] another fit is already running (fit.lock fresh) - "
              "exiting; the running fit will deliver the result.",
              flush=True)
        return None
    with open(lock, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    # HEARTBEAT the lock while working: a 5 MHz geometric-envelope
    # build legitimately runs >> 600 s, and without touching the lock
    # the next auto-refit judged it stale and started a SECOND fit on
    # the same sweep (observed 2026-07-30: three piled-up fits). A
    # crashed fit stops beating and goes stale in 600 s as before.
    import threading
    _hb_stop = threading.Event()

    def _hb():
        while not _hb_stop.wait(60):
            try:
                os.utime(lock, None)
            except OSError:
                return
    threading.Thread(target=_hb, daemon=True).start()
    try:
        return _fit_locked(cfg)
    finally:
        _hb_stop.set()
        try:
            os.remove(lock)
        except OSError:
            pass


def cal_path(f0_mhz=2.0):
    """Per-FREQUENCY calibration file. The shape correction is learned
    against full-wave truth at one frequency and does not transfer
    (different ka, different facet-coherence ceiling): a 5 MHz fit must
    not silently load the 2 MHz curve (bug caught 2026-07-30 before
    the first 5 MHz auto-refit). No file for a frequency = no
    correction (zeros) until one is built from that frequency's data."""
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
                        "ref", f"coda_shape_cal_{f0_mhz:g}mhz.npz")


CAL_PATH = cal_path(2.0)          # the historical 2 MHz file


def shape_calibration(psi_deg, f0_mhz=2.0):
    """Twin-learned coda shape correction (dB, added to the MODEL) as a
    function of beam-to-fabric-axis angle psi.

    WHY: after the two-stage fit, the clean-band coda residuals retain
    a smooth ~2.5 dB S-shape vs psi (autocorr +0.60). Three mechanisms
    were tested and rejected on this data (2026-07-28): measured-E1
    illumination loss (corr +0.02), skew reception loss (right sign,
    only ~0.25 dB), ODF shear-contrast (wrong sign). The structure is
    the fast model's facet-coherence ceiling (rung 3 tracks ~58% of
    azimuth variance) - so it is CALIBRATED against full-wave truth
    (built from a completed sweep by --build-cal) rather than modeled.
    Learned per FREQUENCY (cal_path); a frequency with no cal file
    gets zeros. Validate on an independent seed/axis before trusting
    elsewhere."""
    p = cal_path(f0_mhz)
    if not os.path.exists(p):
        return np.zeros_like(np.asarray(psi_deg, float))
    z = np.load(p)
    return np.interp(np.asarray(psi_deg, float), z["psi"], z["corr_db"])


def build_calibration(cfg):
    """Fit a smooth curve to the clean-band residuals of a COMPLETED
    sweep's fit and save it as the model's shape correction."""
    p = os.path.join(SW.sweep_dir(cfg["name"]), "fit_result.json")
    with open(p, encoding="utf-8") as fh:
        fit = json.load(fh)
    rots = np.asarray(fit["rots"], float)
    res = np.asarray(fit["coda_res_db"])
    ex = set(fit["excluded_rots"])
    m = np.array([r not in ex for r in fit["rots"]])
    psi = _psi_from_axis(rots, fit["alpha_deg"])
    # smooth in psi: cubic poly is enough for an S-shape, and too stiff
    # to memorise glint (57 points, 4 dof)
    # INCREMENT mode: residuals from a cal-applied fit are the error
    # REMAINING on top of the existing curve - add, don't replace
    # (replacement discards the previous correction and made kappa
    # oscillate 4.1/4.9/2.5 across single-pass rebuilds)
    cp = cal_path(cfg["f0_mhz"])
    prev = None
    sources = []
    if os.path.exists(cp):
        zp = np.load(cp)
        prev = (zp["psi"], zp["corr_db"])
        # provenance: every sweep whose residuals ever contributed
        # (older files recorded only the LAST source - carry it over)
        if "sources" in zp:
            sources = [str(s) for s in zp["sources"]]
        elif "source" in zp:
            sources = [str(zp["source"])]
    if cfg["name"] not in sources:
        sources.append(cfg["name"])
    c = np.polyfit(psi[m], res[m], 3)
    # evaluate ONLY inside the data's psi range: a cubic extrapolated
    # into the excluded null band exploded to +16 dB on first build.
    # np.interp in shape_calibration then holds the endpoint values
    # outside this range.
    grid = np.linspace(psi[m].min(), psi[m].max(), 91)
    corr = np.polyval(c, grid)
    if prev is not None:
        corr = corr + np.interp(grid, prev[0], prev[1])
    # ORTHOGONALISE AGAINST KAPPA (2026-07-31). 10log10(E[R^2]) is a
    # smooth monotone function of psi, so a free cubic in psi absorbs
    # 93% of a kappa change (measured: a kappa 3.93 -> 1.48 signature
    # of 1.175 dB rms leaves 0.085 dB after removing a cubic - 20x
    # below SIG_CODA_SWEEP). That makes kappa UNIDENTIFIABLE whenever
    # this calibration is free, and the previously saved curve was 71%
    # kappa-shaped, worth a 2.84-kappa-unit shift - larger than the
    # error it was meant to be orthogonal to. Project the correction
    # onto the complement of d(shape)/d(kappa) so it can only remove
    # structure the fabric parameter cannot explain.
    dk = (_shape_db(grid, fit["kappa"] + 0.5)
          - _shape_db(grid, fit["kappa"] - 0.5))
    dk = dk - dk.mean()
    cz = corr - corr.mean()
    if float(dk @ dk) > 1e-12:
        corr = corr - (float(cz @ dk) / float(dk @ dk)) * dk
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    tmp = cp[:-4] + ".tmp.npz"
    np.savez(tmp, psi=grid, corr_db=corr, source=cfg["name"],
             sources=np.array(sources), f0_mhz=cfg["f0_mhz"],
             kappa_fit=fit["kappa"], alpha_fit=fit["alpha_deg"])
    SW._atomic_replace(tmp, cp, critical=True)
    print(f"[cal] saved {cp}: correction spans "
          f"[{corr.min():+.2f}, {corr.max():+.2f}] dB "
          f"(from {int(m.sum())} clean-band azimuths of "
          f"'{cfg['name']}')", flush=True)


def _psi_from_axis(rots, alpha_deg):
    """Beam-to-fabric-axis angle folded to [0, 90] degrees."""
    psi = np.abs((np.asarray(rots, float) - alpha_deg) % 180.0)
    return np.minimum(psi, 180.0 - psi)


# ── UN-FACTORISED forward model (the 2026-07-29 candidate study:
# geo x global-ODF tracks +0.13 of the measured azimuthal shape,
# full per-facet contrast +0.28 - the factorisation was the
# bottleneck). Too slow inside the optimiser (~0.25 s/azimuth per
# candidate fabric), so: precompute shapes on a coarse (alpha, kappa,
# azimuth) grid ONCE per sweep in the background, interpolate in the
# fit. Falls back to the factorised model when no grid exists. ──────
def _watson_axes_smooth(alpha_deg, kappa, ng, seed=1234):
    """Per-grain c-axes ~ Watson(axis(alpha), kappa), REPARAMETERISED:
    one fixed set of uniform draws (seed) mapped through the kappa-
    dependent inverse-CDF and rotated to alpha. Neighbouring grid
    candidates therefore share their randomness and the model varies
    SMOOTHLY across the grid - independent redraws would stack
    sampling roughness on top of the model's realization glint."""
    rng = np.random.default_rng(seed)
    u_colat, u_azim = rng.random(ng), rng.random(ng)
    ts = np.linspace(0.0, 1.0, 2001)      # |cos| of angle from axis
    cdf = np.cumsum(np.exp(kappa * ts ** 2))
    cdf /= cdf[-1]
    t = np.interp(u_colat, cdf, ts)
    st = np.sqrt(np.clip(1.0 - t ** 2, 0, 1))
    phi = 2 * np.pi * u_azim
    a_r = np.radians(alpha_deg)
    axis = np.array([np.cos(a_r), np.sin(a_r), 0.0])
    perp1 = np.array([-np.sin(a_r), np.cos(a_r), 0.0])
    perp2 = np.array([0.0, 0.0, 1.0])
    ax = (t[:, None] * axis + (st * np.cos(phi))[:, None] * perp1
          + (st * np.sin(phi))[:, None] * perp2)
    return ax / np.linalg.norm(ax, axis=1, keepdims=True)


GRID_ALPHAS = np.arange(0.0, 180.0, 7.5)          # probe-frame axis
GRID_KAPPAS = np.array([0.5, 1.0, 2.0, 3.5, 6.0, 12.0])
GRID_AZ_STEP = 5.0


def build_shape_grid(cfg):
    """Background builder: full-contrast rung-3 coda shape (window-RMS,
    dB) for every grid fabric at every grid azimuth. ~45 min per sweep
    at 2 MHz; saved to <sweep>/shape_grid.npz and reused by every
    subsequent refit."""
    import born

    d = SW.sweep_dir(cfg["name"])
    lock = os.path.join(d, "map.lock")
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    try:
        f0 = cfg["f0_mhz"] * 1e6
        h = C_REF / f0 / cfg["ppw"]
        sp = DiskSpecimen(diameter_m=cfg["diameter_mm"] * 1e-3,
                          thickness_m=cfg["thickness_mm"] * 1e-3,
                          n_grains=cfg["n_grains"],
                          size_cv=cfg["size_cv"],
                          concentration=cfg["concentration"],
                          spatial_corr=cfg["spatial_corr"],
                          fabric_axis=tuple(cfg["fabric_axis"]),
                          seed=cfg["seed"])
        build = sp.build(h)
        mcfg = ladder.standard_cfg()
        mcfg.probe.f0 = f0
        dt_m = ladder.DT
        lo, hi = int(W0 / dt_m), int(W1 / dt_m)
        az_grid = np.arange(cfg["az_start"], cfg["az_stop"],
                            GRID_AZ_STEP)
        # size candidate axes from the BUILD, not the config: the
        # specimen can carry more labels than n_grains (fragment
        # splitting) - config-sized axes crashed with IndexError 100/100
        ng = len(build["axes"])
        shape = np.zeros((len(GRID_ALPHAS), len(GRID_KAPPAS),
                          len(az_grid)), np.float32)
        t0 = time.time()
        for ia, al in enumerate(GRID_ALPHAS):
            for ik, kp in enumerate(GRID_KAPPAS):
                axes_c = _watson_axes_smooth(al, kp, ng)
                bc = dict(build, axes=axes_c)
                for iz, az in enumerate(az_grid):
                    _, env = born.boundary_scatter(
                        bc, mcfg, azimuth_rad=np.radians(float(az)))
                    v = np.sqrt((np.asarray(env, float)[lo:hi] ** 2)
                                .mean())
                    shape[ia, ik, iz] = 20 * np.log10(v + 1e-30)
            el = time.time() - t0
            print(f"[grid] alpha {al:.1f} done "
                  f"({ia+1}/{len(GRID_ALPHAS)}, {el/60:.0f} min, "
                  f"eta {el/(ia+1)*(len(GRID_ALPHAS)-ia-1)/60:.0f} "
                  f"min)", flush=True)
        tmp = os.path.join(d, "shape_grid.tmp.npz")
        np.savez(tmp, alphas=GRID_ALPHAS, kappas=GRID_KAPPAS,
                 az=az_grid, shape_db=shape)
        SW._atomic_replace(tmp, os.path.join(d, "shape_grid.npz"),
                           critical=True)
        print("[grid] saved shape_grid.npz", flush=True)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


_GRID_CACHE = {}


def _load_shape_grid(name):
    p = os.path.join(SW.sweep_dir(name), "shape_grid.npz")
    if not os.path.exists(p):
        return None
    key = (p, os.path.getmtime(p))
    if key not in _GRID_CACHE:
        _GRID_CACHE.clear()
        z = np.load(p)
        _GRID_CACHE[key] = dict(alphas=z["alphas"], kappas=z["kappas"],
                                az=z["az"], shape=z["shape_db"])
    return _GRID_CACHE[key]


def grid_shape_db(name, alpha_deg, kappa, rots):
    """Interpolated full-contrast model shape (dB) at the candidate
    fabric for the requested azimuths; None if no grid built yet.
    Trilinear: circular in alpha (mod 180) and azimuth, linear in
    log-kappa (clamped to the grid range)."""
    g = _load_shape_grid(name)
    if g is None:
        return None
    al = np.asarray(g["alphas"], float)
    kp = np.asarray(g["kappas"], float)
    azg = np.asarray(g["az"], float)
    sh = g["shape"]
    a = float(alpha_deg) % 180.0
    step_a = al[1] - al[0]
    ia0 = int(a // step_a) % len(al)
    ia1 = (ia0 + 1) % len(al)
    fa = (a - al[ia0]) / step_a % 1.0
    lk = np.log(np.clip(kappa, kp[0], kp[-1]))
    lkg = np.log(kp)
    ik1 = int(np.searchsorted(lkg, lk).clip(1, len(kp) - 1))
    ik0 = ik1 - 1
    fk = (lk - lkg[ik0]) / (lkg[ik1] - lkg[ik0])
    plane = ((1 - fa) * (1 - fk) * sh[ia0, ik0]
             + fa * (1 - fk) * sh[ia1, ik0]
             + (1 - fa) * fk * sh[ia0, ik1]
             + fa * fk * sh[ia1, ik1])
    # circular interpolation in azimuth
    rr = np.asarray(rots, float) % 360.0
    az_ext = np.r_[azg, azg[0] + 360.0]
    pl_ext = np.r_[plane, plane[:1]]
    return np.interp(rr, az_ext, pl_ext)


# ── lab-frame 2-theta nuisance (the g(t) artifact, 2026-07-31) ───────
# Every sweep carries a LAB-FRAME (azimuth-frame, not psi-frame) 2-theta
# artifact of ~1-2 dB from the rotated-grid rasterisation: beam-
# referenced staircase scattering + corridor resample phase screen.
# It is additive in the FIELD, so its dB amplitude/phase differ by
# specimen class (isotropic sweep fitted a NEGATIVE scale against the
# aligned-pair template) - no global template transfers. Within ONE
# sweep, though, it is a single stable complex 2-theta vector, so the
# honest fix is to fit it per sweep as a nuisance alongside the gain.
# It matters because it corrupts pattern AMPLITUDE, and kappa IS a
# pattern amplitude (measured corruption up to +-50%).
# MEASURED VERDICT 2026-07-31 - NEGATIVE, default OFF. Fitted on the
# production observable the artifact is only 0.19 dB (seed11, beta
# 82.4 deg - phase bang on the predicted 81) and 0.21 dB (seed23,
# beta 134). That is 10x SMALLER than the 1.5-2 dB measured in the
# raw per-time-bin full-band envelope, because the fit's coda
# observable is already FD-banded + azimuth-smoothed + mod-180 folded
# + window-RMS'd, all of which suppress a lab-frame 2-theta. And the
# cost is real: kappa FELL (seed11 1.48->1.43, seed23 1.91->1.62,
# truth 3.93) because a lab-frame 2-theta partly overlaps the fabric
# pattern whenever beta is near alpha (seed23: 35 deg apart, biggest
# loss). So the artifact does NOT live in the fitted coda scalar -
# it lives in the time-resolved envelope, where the C2T channel meets
# it and handles it by window choice (30-36 us). Kept, documented,
# and OFF; flip on only if a future observable is shown to carry it.
NUIS_2T = False
NUIS_B2_MAX_DB = 2.5        # cap: measured artifact is 1-2 dB


# ── pattern-amplitude efficiency (THE kappa fix, 2026-07-31) ─────────
# The measured clean-band coda 2-theta AMPLITUDE is a constant ~0.657
# of what the forward model predicts, on both rigid sweeps:
#     seed11  2.006 / 3.038 = 0.660
#     seed23  1.986 / 3.038 = 0.654       (agree to 1%)
# Crucially the ORACLE model - one given every grain's TRUE c-axis -
# misses by the same factor, so this is a forward-model amplitude
# CEILING (the documented rung-3 facet-coherence limit: rung 3 tracks
# ~58% of azimuth variance), not a fabric error. Model realization
# spread on that amplitude is sd 0.80 dB, yet the two measurements
# agree to 0.02 dB - it is specimen-independent, hence calibratable.
# The lab-frame artifact does not explain it either (isotropic_seed41_ppw6_calibration 2-theta
# floor is only 0.443 dB = 4.9% of the measured 2-theta power).
# Because kappa is read from modulation DEPTH, an over-predicted model
# amplitude forces kappa down: this single factor accounts for 71%
# (seed11) and 87% (seed23) of the standing kappa gap.
# OUT-OF-SPECIMEN CROSS-VALIDATION: eta from seed11 applied to seed23
# gives kappa 3.86; eta from seed23 applied to seed11 gives 4.01
# (truth 3.93; the fits without it read 1.91 and 1.48).
# ⚠ UNVALIDATED IN KAPPA. Both calibration specimens have the SAME
# truth kappa 3.93, so the transfer test proves specimen-independence
# but says NOTHING about whether eta varies with kappa. If it does,
# this is partly circular. THE decisive test is a sweep at a different
# truth kappa (e.g. 2 or 8) - until then treat recovered kappa near
# 3.9 as calibrated, not predicted.
# ⚠ DISABLED 2026-07-31 - FAILS THE ISOTROPIC CONTROL. With eta on and
# the FD bug fixed, isotropic_seed41_ppw6_calibration (truth kappa 0.001) fits kappa 7.95 and
# a1 0.861: the coda channel attributes pure glint to fabric, and eta
# (which scales the model modulation DOWN) makes that worse, not
# better. The 0.657 was also derived against the pre-fix FD3
# observable, whose 2-theta was 0.03 dB rather than the ~2 dB it was
# calibrated from. GATE FOR RE-ENABLING: any kappa estimator must
# first return LOW kappa on isotropic_seed41_ppw6_calibration. Until that gate passes, kappa
# from the coda channel is not a measurement. Keep the machinery and
# the finding; ship no correction.
SHAPE_ETA = {}              # keyed by f0 in MHz; empty = no correction


def shape_eta(f0_mhz):
    return SHAPE_ETA.get(round(float(f0_mhz), 3), 1.0)


def _apply_eta(model_db, eta, keep=None):
    """Scale the model's azimuthal MODULATION about its mean (the free
    gain absorbs the mean, so only the depth matters)."""
    if eta == 1.0:
        return model_db
    m = np.asarray(model_db, float)
    ref = m[keep].mean() if keep is not None and np.any(keep) else m.mean()
    return ref + eta * (m - ref)


def _shape_db(psi_deg, kappa):
    """Model coda shape (dB) vs beam-to-axis angle psi at a candidate
    kappa: 20*log10(sqrt(E[R^2])) = 10*log10(E[R^2])."""
    axis = np.array([1.0, 0.0, 0.0])
    out = []
    for p in np.atleast_1d(np.asarray(psi_deg, float)):
        beam = -np.array([np.cos(np.radians(p)),
                          np.sin(np.radians(p)), 0.0])
        er2, _ = FF.odf_moments(axis, kappa, beam)
        out.append(10.0 * np.log10(max(float(er2), 1e-30)))
    return np.asarray(out)
NUIS_MIN_AZ = 20            # need azimuth coverage to separate it
NUIS_MIN_SPAN_DEG = 120.0


def _lab_2theta(r0, rots_keep):
    """Closed-form lab-frame 2-theta component of coda residuals.

    Returns (nuisance_curve_db, B2_db, beta_deg). The curve is
    subtracted from the residuals before the robust gain profile, so
    no optimiser dimensions are added. Declines (zeros) when azimuth
    coverage cannot separate a 2-theta from the fabric pattern."""
    rk = np.asarray(rots_keep, float)
    if (not NUIS_2T or len(rk) < NUIS_MIN_AZ
            or (rk.max() - rk.min()) < NUIS_MIN_SPAN_DEG):
        return np.zeros_like(r0), 0.0, float("nan")
    ph = np.radians(2.0 * rk)
    d = r0 - np.median(r0)
    c = 2.0 * np.mean(d * np.exp(-1j * ph))
    b2 = float(min(abs(c), NUIS_B2_MAX_DB))
    beta = float(np.degrees(np.angle(c)) / 2.0 % 180.0)
    return b2 * np.cos(ph - np.radians(2.0 * beta)), b2, beta


def _coda_chi2(kappa, alpha_deg, data, geo, keep, name=None,
               f0_mhz=2.0, ret_nuis=False):
    """Huber coda misfit (with the free gain) on a fixed subset.
    Model = the factorised geo x ODF envelope (the ladder doctrine:
    mean-field smoothness is CORRECT for (alpha, kappa) fitting).
    The UN-FACTORISED grid shape is available behind USE_SHAPE_GRID
    but is OFF by default: candidate Watson draws carry their own
    realization glint (documented negative result), and the 2026-07-29
    processing jackknife measured the cost directly on singlemax_seed7_ppw6_fittest_legacy -
    grid model kappa 2.48 vs factorised 3.97 (truth 3.93). The twin
    calibration is added to the MODEL in dB space either way."""
    from scipy.optimize import minimize_scalar
    rots = [d["rot"] for d in data]
    model_db = grid_shape_db(name, alpha_deg, kappa, rots) \
        if (name and USE_SHAPE_GRID) else None
    if model_db is None:
        c_odf, _, _, _ = FF.predict_basis(alpha_deg, kappa, rots, geo)
        model_db = 20 * np.log10(c_odf + 1e-30)
        # eta scales ONLY the FABRIC modulation. c_odf = geo*sqrt(E[R^2]),
        # and the geometric envelope carries its own azimuthal structure
        # (specimen boundary geometry, nothing to do with fabric) - the
        # first attempt scaled the product and pushed kappa the WRONG way.
        geo_db = 20 * np.log10(np.asarray(geo, float) + 1e-30)
        model_db = geo_db + _apply_eta(model_db - geo_db,
                                       shape_eta(f0_mhz), keep)
    cal = shape_calibration(_psi_from_axis(rots, alpha_deg), f0_mhz)
    coda_m = np.array([d["coda"] for d in data])
    r0 = (20 * np.log10(coda_m) - model_db - cal)[keep]
    if len(r0) < 3:
        return (1e9, np.zeros(0), 0.0, float("nan")) if ret_nuis \
            else (1e9, np.zeros(0))
    nuis, b2, beta = _lab_2theta(r0, np.asarray(rots, float)[keep])
    r1 = r0 - nuis
    g = minimize_scalar(
        lambda x: np.sum(FF.huber((r1 - x) / SIG_CODA_SWEEP)),
        bounds=(r1.min() - 1, r1.max() + 1), method="bounded")
    chi2 = float(np.sum(FF.huber((r1 - g.x) / SIG_CODA_SWEEP)))
    return (chi2, r1 - g.x, b2, beta) if ret_nuis else (chi2, r1 - g.x)


def _jlist(v):
    """JSON-safe list: NaN -> null (json.dump would emit bare NaN,
    which the GUI's strict JSON parser rejects)."""
    return [float(x) if np.isfinite(x) else None
            for x in np.asarray(v, float)]


def _jstats(vals):
    """Azimuth-mean/median/std of one diagnostic observable."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(mean=None, median=None, std=None, n=0)
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                std=float(v.std()), n=int(v.size))


def _fold_smooth_db(rots, vals_db):
    """Neighbour + mod-180 averaging of a dB observable - the same
    conditioning sweep_data applies to coda/ToF (the pattern is
    180-periodic, the glint is not). NaN-aware: NaN entries STAY NaN
    (a short record is short at every azimuth; do not invent data)."""
    v = np.asarray(vals_db, float)
    if AZ_SMOOTH_DEG <= 0 or len(v) <= 4:
        return v.copy()
    rr180 = np.asarray(rots, float) % 180.0
    out = np.full_like(v, np.nan)
    fin = np.isfinite(v)
    for i in range(len(v)):
        if not fin[i]:
            continue
        dd = np.abs(rr180 - rr180[i])
        sel = (np.minimum(dd, 180.0 - dd) <= AZ_SMOOTH_DEG) & fin
        out[i] = v[sel].mean()
    return out


def _tail_chi2(kappa, alpha_deg, tail_abs_db, rots, geo_tail, keep):
    """Huber tail misfit with its OWN free gain, mirroring the coda
    channel. Model = the factorised doctrine only: unit-contrast
    geometric TAIL_W-window RMS x sqrt(E[R^2]) (FF.predict_basis fed
    the tail geo; no un-factorised grid exists for this window).

    Measured side = ABSOLUTE tail RMS (dB), i.e. tail_db + e1_db, NOT
    the re-E1 ratio: with a free gain only the azimuthal SHAPE scores,
    and dividing by per-azimuth E1 would inject E1's glint shape into
    a fitted channel - exactly what the trace study says must not be
    fitted (E1's per-azimuth shape is glint; only its MEAN informs).
    No shape calibration is applied: the twin-learned curve was
    trained on the 24-36 us coda window, not this one."""
    from scipy.optimize import minimize_scalar
    c_odf, _, _, _ = FF.predict_basis(alpha_deg, kappa, rots, geo_tail)
    model_db = 20 * np.log10(c_odf + 1e-30)
    ok = np.asarray(keep) & np.isfinite(tail_abs_db)
    r0 = (tail_abs_db - model_db)[ok]
    if len(r0) < 3:
        return 0.0, np.zeros(0)
    g = minimize_scalar(
        lambda x: np.sum(FF.huber((r0 - x) / SIG_TAIL)),
        bounds=(r0.min() - 1, r0.max() + 1), method="bounded")
    return float(np.sum(FF.huber((r0 - g.x) / SIG_TAIL))), r0 - g.x


def _c2t_design(az_deg):
    """The study's harmonic design matrix: [1, cos2, sin2, cos4, sin4]
    (the 4-theta columns stay in the regression so 4-theta structure
    cannot alias into the 2-theta coefficients on partial coverage)."""
    th = np.radians(np.asarray(az_deg, float))
    return np.column_stack([np.ones_like(th), np.cos(2 * th),
                            np.sin(2 * th), np.cos(4 * th),
                            np.sin(4 * th)])


def coherent_2theta(cfg):
    """Coherent 2-theta axis observable of a sweep (see the C2T block
    above for doctrine). Returns dict(phi2_deg, a2_db, se_deg, n_bins,
    n_az) or None when the channel must decline:
      - legacy (non-rigid2) sweeps: the estimator lives in the lab
        frame, and on the legacy 'singlemax_seed7_ppw6_fittest_legacy' it reads 58.9 deg vs the
        raw-fit a_probe 171.6 (67 deg apart, checked 2026-07-31) -
        the legacy axes-vs-grains mismatch corrupts the phase, so the
        channel is rigid2-only;
      - fewer than C2T_MIN_AZ azimuths (measured instability);
      - records too short for >= C2T_MIN_BINS full bins of C2T_W.
    Needs the RAW traces (not obs_cache), so the result is cached in
    <sweep>/c2t_cache.npz keyed by version + the azimuth list; traces
    are opened with with-blocks and copied out (a GPU sweep may be
    writing the directory concurrently - unreadable files are
    skipped, like the study's loader)."""
    import hashlib
    if cfg.get("axes_convention") != "rigid2":
        return None
    azs = SW.done_azimuths(cfg)
    if len(azs) < C2T_MIN_AZ:
        return None
    d = SW.sweep_dir(cfg["name"])
    key = hashlib.sha1(
        (C2T_VER + ":" + ",".join(str(a) for a in azs))
        .encode()).hexdigest()[:16]
    cache_p = os.path.join(d, "c2t_cache.npz")
    if os.path.exists(cache_p):
        try:
            with np.load(cache_p) as z:
                if str(z["key"]) == key:
                    return dict(phi2_deg=float(z["phi2_deg"]),
                                a2_db=float(z["a2_db"]),
                                se_deg=float(z["se_deg"]),
                                n_bins=int(z["n_bins"]),
                                n_az=int(z["n_az"]))
        except Exception:
            pass                        # stale/corrupt cache: recompute
    tb = np.arange(C2T_W[0], C2T_W[1] - 1e-12, C2T_DTBIN)
    az_ok, Lrows, nb_min = [], [], len(tb)
    for az in azs:
        try:
            with np.load(os.path.join(d, f"az{az:03d}.npz")) as z:
                tr = np.array(z["trace"], float).ravel()
                dtr = float(z["dt"])
        except Exception:
            continue                    # mid-write file: skip
        fs = 1.0 / dtr
        env = np.abs(hilbert(tr))
        lo = np.round(tb * fs).astype(int)
        hi = np.round((tb + C2T_DTBIN) * fs).astype(int)
        full = hi <= len(tr)            # only FULL bins count
        if full.sum() < C2T_MIN_BINS:
            continue
        row = 20 * np.log10(np.sqrt(
            [(env[a:b] ** 2).mean() for a, b in
             zip(lo[full], hi[full])]) + 1e-30)
        nb_min = min(nb_min, len(row))
        az_ok.append(float(az))
        Lrows.append(row)
    if len(az_ok) < C2T_MIN_AZ or nb_min < C2T_MIN_BINS:
        return None
    L = np.stack([r[:nb_min] for r in Lrows])
    azv = np.asarray(az_ok, float)
    X = _c2t_design(azv)
    c = np.linalg.lstsq(X, L, rcond=None)[0]
    v = (c[1] + 1j * c[2]).mean()
    phi2 = float(np.degrees(0.5 * np.angle(v)) % 180.0)
    a2 = float(np.abs(v))
    # phase uncertainty: bootstrap over AZIMUTHS (resamples both the
    # per-azimuth glint and the angular coverage)
    rng = np.random.default_rng(1234)
    ph = np.empty(C2T_N_BOOT)
    for k in range(C2T_N_BOOT):
        idx = rng.integers(0, len(azv), len(azv))
        cb = np.linalg.lstsq(X[idx], L[idx], rcond=None)[0]
        ph[k] = np.degrees(0.5 * np.angle(
            (cb[1] + 1j * cb[2]).mean())) % 180.0
    se = float(np.std((ph - phi2 + 90.0) % 180.0 - 90.0))
    out = dict(phi2_deg=phi2, a2_db=a2, se_deg=se,
               n_bins=int(nb_min), n_az=len(azv))
    tmp = cache_p[:-4] + f".tmp{os.getpid()}.npz"
    try:
        np.savez(tmp, key=key, ver=C2T_VER, **out)
        SW._atomic_replace(tmp, cache_p, critical=False)
    except OSError:
        pass                            # cache write is best-effort
    return out


def _fit_locked(cfg):
    """TWO-STAGE fit - the doctrine made structural.

    The single-stage Huber fit breaks on full angular coverage: with
    360-degree sweeps a large FRACTION of azimuths sit in the two
    known bad-physics bands (the psi~0 deep null where the incoherent
    model over-predicts by ~6-11 dB, and the psi~65-85 walk-off band),
    and a robust loss cannot outvote a third of its data (measured:
    6 curated azimuths -> axis error 7.4 deg; 35 full-coverage
    azimuths -> 54 deg!).

    Stage 1: fabric AXIS from ToF alone - arrival times are clean at
    every azimuth (no null, no walk-off amplitude physics).
    Stage 2: kappa + gain from coda ONLY in the well-modeled band
    (psi in [15, 65] or [85, 90] measured from the STAGE-1 axis).
    Because the axis is frozen before the band is chosen, the mask
    cannot chase the parameters - the exploitability trap that killed
    candidate-dependent masking (documented in fit_fabric.py) does
    not apply."""
    data = sweep_data(cfg)
    if len(data) < 6:
        raise RuntimeError(f"only {len(data)} azimuths saved; "
                           "the fit needs at least 6")
    rots = [r["rot"] for r in data]
    print(f"[fit] {len(rots)} azimuths; building geometric envelopes "
          "(cached after first run)...", flush=True)
    t0 = time.time()
    geo, geo_tail = sweep_geo(cfg, rots)
    print(f"[fit] envelopes ready ({time.time()-t0:.0f}s)", flush=True)

    tof_m = np.array([d["tof_us"] for d in data])

    # tail channel measured side: absolute tail dB (see _tail_chi2 for
    # why not re-E1), folded/smoothed like the coda. NaN where the
    # record ends before enough of TAIL_W exists.
    tail_abs = _fold_smooth_db(
        rots, np.array([d["tail_db"] + d["e1_db"] for d in data]))
    n_tail_ok = int(np.isfinite(tail_abs).sum())
    tail_on = bool(TAIL_CHANNEL) and n_tail_ok >= max(6, len(rots) // 2)
    if TAIL_CHANNEL and not tail_on:
        print(f"[fit] tail channel SKIPPED: only {n_tail_ok} of "
              f"{len(rots)} azimuths have >=2 us of the "
              f"{TAIL_W[0]*1e6:.0f}-{TAIL_W[1]*1e6:.0f} us window "
              "(record too short)", flush=True)

    # Stage-1 objective: TEMPLATE fit. The model's ToF(psi) SHAPE is
    # right (min along axis, max near psi~50 - the qP curve) but its
    # amplitude is ~2/3 of measured and it carries a calibration
    # offset; fitting measured = a + b*model with free (a, b>=0)
    # scores only the SHAPE - the same free-gain doctrine the coda
    # observable already uses. (A raw-residual ToF fit at 40 azimuths
    # preferred to FLATTEN the pattern: axis 111 deg, kappa 1.75 -
    # measured 2026-07-28, kept as the reason this is a template fit.
    # A 2-phi harmonic is also wrong: qP(psi) has strong 4-phi
    # content, harmonic axis came out 140 deg.)
    # us, post-shape scatter. 0.3 = measured template residual rms
    # 0.27-0.37 across leading-edge pick variants (jackknife
    # 2026-07-29; the old 0.2 over-weighted ToF and pushed points
    # into the Huber tail). Use ~0.45 if fitting with AZ_SMOOTH = 0.
    SIG_TOF_SHAPE = 0.3

    def tof_chi2(alpha_deg, kappa):
        _, _, tof_p, _ = FF.predict_basis(alpha_deg, kappa, rots, geo)
        t = tof_p - tof_p.mean()
        denom = float(t @ t)
        if denom < 1e-12:
            return 1e9
        b = float(t @ (tof_m - tof_m.mean())) / denom
        if b < 0.2:                           # wrong orientation branch
            return 1e9
        r = (tof_m - tof_m.mean()) - b * t
        return float(np.sum(FF.huber(r / SIG_TOF_SHAPE)))

    # coherent 2-theta axis constraint (see C2T block / coherent_2theta
    # for doctrine and the decline conditions). phi2 lives in the same
    # raw/lab frame as the template alpha (rigid2 sweeps only), so the
    # penalty applies to alpha BEFORE the legacy negation.
    c2t = None
    if C2T_CHANNEL:
        try:
            c2t = coherent_2theta(cfg)
        except Exception as e:                # channel is optional -
            print(f"[fit] 2-theta channel failed ({e}) - "
                  "pure-ToF stage 1", flush=True)
    if c2t is not None:
        print(f"[fit] 2-theta channel: phi2 {c2t['phi2_deg']:.1f} "
              f"+- {c2t['se_deg']:.1f} deg (A2 {c2t['a2_db']:.2f} dB, "
              f"{c2t['n_az']} az, {c2t['n_bins']} bins)", flush=True)
    elif C2T_CHANNEL:
        print("[fit] 2-theta channel: declined (legacy convention, "
              f"< {C2T_MIN_AZ} azimuths, or record too short) - "
              "pure-ToF stage 1", flush=True)

    def stage1_chi2(alpha_deg, kappa):
        """ToF template chi2, PLUS the von-Mises-style 2-theta phase
        penalty when the channel is live (ToF then deflated by its
        measured information overstatement, C2T_TOF_INFO - a pure
        rescale, so with c2t None this IS the historical objective)."""
        c = tof_chi2(alpha_deg, kappa)
        if c2t is None:
            return c
        dphi = (alpha_deg - c2t["phi2_deg"] + 90.0) % 180.0 - 90.0
        return (c / C2T_TOF_INFO
                + (dphi / max(c2t["se_deg"], C2T_SE_FLOOR)) ** 2)

    # stage 1: axis (+ starting kappa) from the ToF pattern shape
    # (2-theta-constrained when the channel is live)
    best = None
    for a in range(0, 180, 2):
        for k in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
            c = stage1_chi2(a, k)
            if best is None or c < best[0]:
                best = (c, a, k)
    res1 = minimize(lambda p: stage1_chi2(p[0], np.exp(p[1])),
                    x0=[best[1], np.log(best[2])], method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=1e-4, maxiter=2000))
    a_fit = float(res1.x[0] % 180.0)
    print(f"[fit] stage 1 (ToF shape"
          + (" + 2-theta phase" if c2t is not None else "")
          + f"): axis {a_fit:.1f} deg", flush=True)

    # stage 2: kappa + gain from coda in the predicted-clean band
    psi = _psi_from_axis(rots, a_fit)
    keep = (psi >= 15.0) & ~((psi > 65.0) & (psi < 85.0))
    print(f"[fit] stage 2 (coda): {keep.sum()} of {len(rots)} azimuths "
          f"in the clean band (excluded: null psi<15, walk-off "
          f"65<psi<85)", flush=True)
    # kappa is scored against BOTH pattern shapes at the frozen axis:
    # clean-band coda alone under-constrains it at 2 MHz (measured:
    # kappa slid to 0), while the ToF shape's curvature vs kappa adds
    # the missing leverage. Bounded log-kappa keeps NM out of the
    # degenerate flat-pattern corner.
    def stage2(logk):
        k = float(np.exp(np.clip(logk, np.log(0.3), np.log(30.0))))
        c = (_coda_chi2(k, a_fit, data, geo, keep,
                        name=cfg["name"], f0_mhz=cfg["f0_mhz"])[0]
             + tof_chi2(a_fit, k))
        if tail_on:
            c += _tail_chi2(k, a_fit, tail_abs, rots, geo_tail,
                            keep)[0]
        return c

    ks = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    k0 = min(ks, key=lambda k: stage2(np.log(k)))
    res2 = minimize(lambda p: stage2(p[0]), x0=[np.log(k0)],
                    method="Nelder-Mead",
                    options=dict(xatol=1e-4, fatol=1e-4, maxiter=500))
    k_fit = float(np.exp(np.clip(res2.x[0], np.log(0.3), np.log(30.0))))

    # final report with STAGE-CONSISTENT residuals (the raw FF.objective
    # residuals include the calibration offset/gain the template fit
    # removes, which made healthy fits look broken in the GUI):
    # coda gain from the clean band applied to ALL azimuths (excluded
    # ones then show their true null/walk-off excess), ToF residuals
    # from the template (free offset + gain).
    from scipy.optimize import minimize_scalar
    chi2_coda, _, nuis_b2, nuis_beta = _coda_chi2(
        k_fit, a_fit, data, geo, keep, name=cfg["name"],
        f0_mhz=cfg["f0_mhz"], ret_nuis=True)
    if nuis_b2 > 0:
        print(f"[fit] lab-frame 2-theta nuisance: {nuis_b2:.2f} dB at "
              f"beta {nuis_beta:.1f} deg (rasterisation artifact; "
              f"measured ~81 deg in the coda band)", flush=True)
    chi2_tail = (_tail_chi2(k_fit, a_fit, tail_abs, rots, geo_tail,
                            keep)[0] if tail_on else 0.0)
    chi2 = chi2_coda + tof_chi2(a_fit, k_fit) + chi2_tail
    c_odf, _, tof_p, _ = FF.predict_basis(a_fit, k_fit, rots, geo)
    model_db = (grid_shape_db(cfg["name"], a_fit, k_fit, rots)
                if USE_SHAPE_GRID else None)
    if model_db is None:
        model_db = 20 * np.log10(c_odf + 1e-30)
        geo_db = 20 * np.log10(np.asarray(geo, float) + 1e-30)
        model_db = geo_db + _apply_eta(model_db - geo_db,
                                       shape_eta(cfg["f0_mhz"]), keep)
    coda_m = np.array([d["coda"] for d in data])
    r0 = (20 * np.log10(coda_m) - model_db
          - shape_calibration(_psi_from_axis(rots, a_fit),
                              cfg["f0_mhz"]))
    # report curves carry the SAME nuisance the fit used (else the
    # residual chart shows an artifact the chi2 already accounted for)
    nuis_all = (nuis_b2 * np.cos(np.radians(2.0 * (np.asarray(rots, float)
                                                   - nuis_beta)))
                if nuis_b2 > 0 else np.zeros(len(rots)))
    r0 = r0 - nuis_all
    gain = minimize_scalar(
        lambda x: np.sum(FF.huber((r0[keep] - x) / SIG_CODA_SWEEP)),
        bounds=(r0[keep].min() - 1, r0[keep].max() + 1),
        method="bounded").x
    # tail channel report curves: clean-band gain applied to ALL
    # azimuths (mirrors the coda reporting; excluded/short azimuths
    # then show their true excess), NaN -> null in the JSON
    tail_res = np.full(len(rots), np.nan)
    model_tail_curve = np.full(len(rots), np.nan)
    if tail_on:
        c_t, _, _, _ = FF.predict_basis(a_fit, k_fit, rots, geo_tail)
        model_t_db = 20 * np.log10(c_t + 1e-30)
        rt0 = tail_abs - model_t_db
        okt = keep & np.isfinite(tail_abs)
        g_t = minimize_scalar(
            lambda x: np.sum(FF.huber((rt0[okt] - x) / SIG_TAIL)),
            bounds=(float(np.nanmin(rt0)) - 1,
                    float(np.nanmax(rt0)) + 1),
            method="bounded").x
        tail_res = rt0 - g_t
        model_tail_curve = model_t_db + g_t
    t = tof_p - tof_p.mean()
    b_t = max(float(t @ (tof_m - tof_m.mean())) / float(t @ t), 0.2)
    det = dict(coda_res=r0 - gain,
               tof_res=(tof_m - tof_m.mean()) - b_t * t,
               # absolute curves for measured-vs-inverted overlays:
               meas_coda_db=20 * np.log10(coda_m),
               model_coda_db=(model_db + gain
                              + shape_calibration(
                                  _psi_from_axis(rots, a_fit),
                                  cfg["f0_mhz"])),
               meas_tof_us=tof_m,
               model_tof_us=tof_m.mean() + b_t * t)
    ax = np.asarray(cfg["fabric_axis"], float)
    truth_alpha = float(np.degrees(np.arctan2(ax[1], ax[0])) % 180.0)
    # FRAME CONVENTION: sweeps generated by the FIXED rotated_grid
    # (2026-07-29, axes_convention='rigid2' in config.json) are rigid
    # rotations, and the template axis IS the specimen-frame axis.
    # LEGACY sweeps (no tag) were generated by the buggy rotated_grid
    # that spun the c-axes by +rot instead of -rot: for those the
    # fitted axis is the NEGATIVE of the specimen-frame axis (the
    # empirical negation discovered on seed11/axis30, kept here so old
    # data stays fittable). Per-cell/per-realisation use of legacy
    # sweeps remains compromised (axes vs grains mismatch): regenerate.
    a_probe = a_fit
    if cfg.get("axes_convention") != "rigid2":
        a_fit = float((-a_fit) % 180.0)
    # calibration circularity check: if this sweep's residuals ever
    # trained the shape calibration, its recovery metrics are
    # IN-SAMPLE (the cal has absorbed this specimen's glint)
    cal_in_sample = False
    if os.path.exists(cal_path(cfg["f0_mhz"])):
        zc = np.load(cal_path(cfg["f0_mhz"]))
        srcs = ([str(s) for s in zc["sources"]] if "sources" in zc
                else [str(zc["source"])] if "source" in zc else [])
        cal_in_sample = cfg["name"] in srcs
        if cal_in_sample:
            print("[fit] WARNING: the coda shape calibration was "
                  "trained on THIS sweep - recovery metrics below are "
                  "IN-SAMPLE (circular). Validate on an independent "
                  "sweep before quoting them.", flush=True)
    out = dict(
        cal_in_sample=cal_in_sample,
        nuis_b2_db=float(nuis_b2),
        nuis_beta_deg=(float(nuis_beta) if nuis_b2 > 0 else None),
        n_azimuths=len(rots),
        method="two-stage (axis from ToF; kappa from clean-band coda)",
        n_coda_used=int(keep.sum()),
        excluded_rots=[int(r) for r, k in zip(rots, keep) if not k],
        alpha_deg=a_fit, alpha_probe_deg=a_probe, kappa=k_fit,
        a1=float(S.watson_eigenvalue(k_fit)),
        chi2=float(chi2),
        truth_alpha_deg=truth_alpha,
        truth_kappa=float(cfg["concentration"]),
        truth_a1=float(S.watson_eigenvalue(cfg["concentration"])),
        alpha_err_deg=float(min(abs(a_fit - truth_alpha),
                                180 - abs(a_fit - truth_alpha))),
        coda_res_db=[float(x) for x in det["coda_res"]],
        tof_res_us=[float(x) for x in det["tof_res"]],
        meas_coda_db=[float(x) for x in det["meas_coda_db"]],
        model_coda_db=[float(x) for x in det["model_coda_db"]],
        meas_tof_us=[float(x) for x in det["meas_tof_us"]],
        model_tof_us=[float(x) for x in det["model_tof_us"]],
        # coherent 2-theta axis channel (stage-1 constraint; None
        # fields when the channel declined and stage 1 was pure ToF)
        c2t_used=bool(c2t is not None),
        c2t_phi2_deg=(float(c2t["phi2_deg"]) if c2t else None),
        c2t_a2_db=(float(c2t["a2_db"]) if c2t else None),
        c2t_se_deg=(float(c2t["se_deg"]) if c2t else None),
        c2t_n_bins=(int(c2t["n_bins"]) if c2t else None),
        c2t_n_az=(int(c2t["n_az"]) if c2t else None),
        # tail channel (third stage-2 channel, TAIL_CHANNEL gate)
        tail_channel=tail_on, sig_tail_db=SIG_TAIL,
        meas_tail_db=_jlist(tail_abs),
        model_tail_db=_jlist(model_tail_curve),
        tail_res_db=_jlist(tail_res),
        # REPORT, DO NOT FIT: azimuth-aggregate values of the mined
        # observables, for kappa cross-checks against future ensemble
        # calibrations. E1 attenuation has NO validated forward model
        # (coherent decoherence, open question) and speckle contrast
        # needs ensemble validation - neither belongs in the objective.
        # (tail_db here is re-E1 per the study's definition; the FITTED
        # tail uses the absolute level, see _tail_chi2.)
        diagnostics={k: _jstats([d_[k] for d_ in data])
                     for k in ("tail_db", "e1_db", "cent_mhz",
                               "speckle_contrast")},
        rots=rots, fitted_at=time.ctime(),
    )
    p = os.path.join(SW.sweep_dir(cfg["name"]), "fit_result.json")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    SW._atomic_replace(p + ".tmp", p, critical=True)
    print(f"[fit] RECOVERED: axis {a_fit:.1f} deg (truth {truth_alpha:.1f}"
          f", err {out['alpha_err_deg']:.1f}), kappa {k_fit:.2f} "
          f"(truth {cfg['concentration']}), a1 {out['a1']:.3f} "
          f"(truth {out['truth_a1']:.3f}), chi2 {chi2:.1f}", flush=True)
    return out


def misfit_map(cfg, n_tbins=16, cell_mm=5.0,
               t_map=(16e-6, 48e-6), data_map=False):
    """data_map=False (default): build ONLY the ground-truth fabric
    maps (rotation/tilt vs actual local c-axes) - fast, no trace loop.
    data_map=True additionally builds the data-driven coda misfit map;
    retired from the GUI 2026-07-28 per Jerome but KEPT because it is
    the only map possible on REAL ice (no truth grid in the lab) - it
    returns the day the scanner produces data.

    Spatial misfit map: the coda is TIME-resolved, and time maps to
    depth along each beam (r = v*t/2), so each azimuth's per-time-bin
    residual can be laid out along its diameter and accumulated into
    grid squares over the disk. Colour = local measured-vs-model coda
    misfit (dB): the first step of 'areas of disorder' mapping.

    Uses the CURRENT fit (axis, kappa, calibration) for the model side;
    zero-means the map over covered cells (the absolute gain is a fit
    nuisance, spatial CONTRAST is the information). Saves
    <sweep>/misfit_map.npz for the GUI."""
    from scipy.signal import hilbert as _hilb
    import ladder as L

    d = SW.sweep_dir(cfg["name"])
    lock = os.path.join(d, "map.lock")      # GUI polls while this exists
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    try:
        return _misfit_map_locked(cfg, d, n_tbins, cell_mm, t_map,
                                  _hilb, L, data_map)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _misfit_map_locked(cfg, d, n_tbins, cell_mm, t_map, _hilb, L,
                       data_map=False):
    with open(os.path.join(d, "fit_result.json"), encoding="utf-8") as fh:
        fit = json.load(fh)
    alpha, kappa = fit["alpha_deg"], fit["kappa"]
    rots = SW.done_azimuths(cfg)
    axis = np.array([np.cos(np.radians(alpha)),
                     np.sin(np.radians(alpha)), 0.0])

    tens_cache = os.path.join(d, "cof_cell_tensors.npz")
    if not data_map and os.path.exists(tens_cache):
        # truth map refresh against a new fit: tensors cached, no
        # specimen rebuild needed - seconds, cheap enough to auto-run
        cof_truth_map(cfg, None, fit, cell_mm)
        return
    f0 = cfg["f0_mhz"] * 1e6
    h = C_REF / f0 / cfg["ppw"]
    sp = DiskSpecimen(diameter_m=cfg["diameter_mm"] * 1e-3,
                      thickness_m=cfg["thickness_mm"] * 1e-3,
                      n_grains=cfg["n_grains"], size_cv=cfg["size_cv"],
                      concentration=cfg["concentration"],
                      spatial_corr=cfg["spatial_corr"],
                      fabric_axis=tuple(cfg["fabric_axis"]),
                      seed=cfg["seed"])
    build = sp.build(h)
    if not data_map:                 # truth maps only (GUI default)
        cof_truth_map(cfg, build, fit, cell_mm)
        return
    mcfg = ladder.standard_cfg()
    mcfg.probe.f0 = f0
    import born

    # the MAP uses the full pre-E1 record (16-48 us -> depths spanning
    # most of the disk), unlike the FIT which stays in the clean window
    # 24-36 us: for a qualitative disorder map the early coherent-facet
    # zone and late pre-E1 buildup are still informative, they just
    # carry the model's known systematic there. Coverage = union of the
    # depth band over available probe azimuths (a full 360-degree sweep
    # fills the disk; a 0-90 sweep lights a fan).
    tb = np.linspace(t_map[0], t_map[1], n_tbins + 1)
    R = cfg["diameter_mm"] * 1e-3 / 2.0
    ncell = int(np.ceil(2 * R / (cell_mm * 1e-3)))
    acc = np.zeros((ncell, ncell))
    cnt = np.zeros((ncell, ncell))

    for az in rots:
        z = np.load(os.path.join(d, f"az{az:03d}.npz"))
        tr, dt = np.asarray(z["trace"], float).ravel(), float(z["dt"])
        fs = 1.0 / dt
        e = np.abs(_hilb(tr))
        # unit_contrast: factorised convention (geometric x sqrt(er2));
        # the old full-contrast call double-counted contrast with the
        # sqrt(er2) factor below (fixed 2026-07-29, same as grid_inversion)
        _, env = born.boundary_scatter(dict(build), mcfg,
                                       azimuth_rad=np.radians(az),
                                       unit_contrast=True)
        env = np.asarray(env, float)
        fs_m = 1.0 / L.DT
        beam_ang = np.radians(az)
        psi = _psi_from_axis([az], alpha)[0]
        er2, _ = FF.odf_moments(axis, kappa,
                                -np.array([np.cos(beam_ang),
                                           np.sin(beam_ang), 0.0]))
        cal = float(shape_calibration(np.array([psi]),
                                      cfg["f0_mhz"])[0])
        probe = np.array([R * np.cos(beam_ang), R * np.sin(beam_ang)])
        bdir = -np.array([np.cos(beam_ang), np.sin(beam_ang)])
        for k in range(n_tbins):
            s0, s1 = int(tb[k] * fs), int(tb[k + 1] * fs)
            meas = np.sqrt((e[s0:s1] ** 2).mean())
            m0, m1 = int(tb[k] * fs_m), int(tb[k + 1] * fs_m)
            mod = (np.sqrt((env[m0:m1] ** 2).mean())
                   * np.sqrt(max(er2, 1e-30)))
            if mod <= 0 or meas <= 0:
                continue
            res_db = 20 * np.log10(meas / mod) - cal
            t_mid = 0.5 * (tb[k] + tb[k + 1])
            # subtract the Ricker envelope-peak source delay (1.2/f0)
            r_depth = C_REF * max(t_mid - 1.2 / f0, 0.0) / 2.0
            pt = probe + r_depth * bdir
            ix = int((pt[0] + R) / (2 * R) * ncell)
            iy = int((pt[1] + R) / (2 * R) * ncell)
            if 0 <= ix < ncell and 0 <= iy < ncell:
                acc[iy, ix] += res_db
                cnt[iy, ix] += 1
    mp = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    mp -= np.nanmean(mp)                     # spatial CONTRAST only
    edges = (np.arange(ncell + 1) / ncell * 2 * R - R) * 1e3   # mm
    tmp = os.path.join(d, "misfit_map.tmp.npz")
    np.savez(tmp, map_db=mp, counts=cnt, edges_mm=edges,
             n_tbins=n_tbins, alpha=alpha, kappa=kappa)
    SW._atomic_replace(tmp, os.path.join(d, "misfit_map.npz"),
                       critical=True)
    cov = int((cnt > 0).sum())
    print(f"[map] {cov} covered cells, misfit contrast rms "
          f"{np.nanstd(mp):.2f} dB -> saved misfit_map.npz", flush=True)
    cof_truth_map(cfg, build, fit, cell_mm)


def cof_truth_map(cfg, build, fit, cell_mm=5.0):
    """GROUND-TRUTH fabric error map (synthetic specimens only): for
    each disk square, the orientation tensor of the c-axes of the
    grains ACTUALLY inside it (volume-weighted from the label grid)
    gives the true local mean axis and a1; colour = angle between the
    INVERTED global axis and the TRUE local axis. This evaluates the
    'per-area orientation' goal directly: where a single-fabric
    description matches local reality and where local mapping would be
    needed. (In the lab there is no truth grid - the data-driven coda
    misfit map is the field instrument; this one is its calibrator.)"""
    d = SW.sweep_dir(cfg["name"])
    # the per-cell orientation tensors depend only on the SPECIMEN and
    # the grid, never on the fit - cache them once per sweep so map
    # rebuilds against each new fitted axis take seconds, which is what
    # makes auto-updating the map affordable
    tens_cache = os.path.join(d, "cof_cell_tensors.npz")
    if os.path.exists(tens_cache):
        zc = np.load(tens_cache)
        T, tot = zc["T"], zc["tot"]
        ncell, D_m = int(zc["ncell"]), float(zc["D_m"])
    else:
        if build is None:
            raise RuntimeError("tensor cache missing and no build "
                               "provided")
        labels = np.asarray(build["labels"])
        axes = np.asarray(build["axes"], float)
        h_spec = float(build["h"])
        nx_s = labels.shape[0]
        D_m = nx_s * h_spec
        ncell = int(np.ceil(D_m / (cell_mm * 1e-3)))
        # voxel -> in-plane cell index (labels axes are (x, y, z))
        xi = np.minimum((np.arange(nx_s) * h_spec / (cell_mm * 1e-3))
                        .astype(int), ncell - 1)
        ng = len(axes)
        w = np.zeros((ncell * ncell, ng))
        lab2 = labels.reshape(nx_s, labels.shape[1], -1)
        for iz in range(lab2.shape[2]):        # slice-wise, memory-lean
            sl = lab2[:, :, iz]
            ok = sl >= 0
            cells = (xi[:, None] * ncell
                     + xi[None, :labels.shape[1]])[ok]
            np.add.at(w, (cells, sl[ok]), 1.0)
        # per-cell orientation tensor of TRUE c-axes (sign-symmetric)
        outer = np.einsum("gi,gj->gij", axes, axes)  # (ng,3,3)
        T = np.einsum("cg,gij->cij", w, outer)
        tot = w.sum(axis=1)
        okm = tot > 0
        T[okm] /= tot[okm][:, None, None]
        tmpc = tens_cache[:-4] + ".tmp.npz"
        np.savez(tmpc, T=T, tot=tot, ncell=ncell, D_m=D_m)
        SW._atomic_replace(tmpc, tens_cache, critical=True)
    ok = tot > 0
    ang = np.full(ncell * ncell, np.nan)      # full 3D angle
    rot_err = np.full(ncell * ncell, np.nan)  # in-plane rotation error
    tilt = np.full(ncell * ncell, np.nan)     # out-of-plane tilt
    a1_loc = np.full(ncell * ncell, np.nan)
    alpha_f = fit["alpha_deg"]
    alpha = np.radians(alpha_f)
    fit_ax = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    for c in np.where(ok)[0]:
        vals, vecs = np.linalg.eigh(T[c])
        v = vecs[:, -1]
        a1_loc[c] = vals[-1]
        cosang = abs(float(v @ fit_ax))
        ang[c] = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
        # split into the two physical degrees of freedom:
        # tilt = elevation of the TRUE local axis out of the disk
        # plane (the inversion assumes 0 - in-plane fabric)
        tilt[c] = np.degrees(np.arcsin(np.clip(abs(v[2]), 0, 1)))
        # rotation = in-plane angle of the true axis vs the fitted
        # axis (mod 180; what the azimuthal method measures)
        rloc = np.degrees(np.arctan2(v[1], v[0])) % 180.0
        dr = abs(rloc - (alpha_f % 180.0))
        rot_err[c] = min(dr, 180.0 - dr)
    shp = (ncell, ncell)
    ang = ang.reshape(shp).T                   # (y, x) for plotting
    rot_err = rot_err.reshape(shp).T
    tilt = tilt.reshape(shp).T
    a1_loc = a1_loc.reshape(shp).T
    edges = (np.arange(ncell + 1) * cell_mm * 1e-3 - D_m / 2) * 1e3
    d = SW.sweep_dir(cfg["name"])
    tmp = os.path.join(d, "cof_truth_map.tmp.npz")
    np.savez(tmp, angle_err_deg=ang, rot_err_deg=rot_err,
             tilt_deg=tilt, a1_local=a1_loc,
             a1_fit=fit["a1"], edges_mm=edges,
             alpha_fit_deg=alpha_f)
    SW._atomic_replace(tmp, os.path.join(d, "cof_truth_map.npz"),
                       critical=True)
    print(f"[map] TRUTH map: median 3D axis err {np.nanmedian(ang):.1f}"
          f" deg (rotation {np.nanmedian(rot_err):.1f}, tilt "
          f"{np.nanmedian(tilt):.1f}), local a1 "
          f"{np.nanmedian(a1_loc):.3f} (fit {fit['a1']:.3f}) "
          f"-> saved cof_truth_map.npz", flush=True)


def grain_axis_errors(cfg):
    """PER-CRYSTAL absolute axis errors, exactly as specified: each
    grain's true c-axis is described by two angles - YAW (in-plane
    rotation, atan2(ay, ax) mod 180) and PITCH (elevation out of the
    disk plane, arcsin(|az|)) - and compared against the INVERTED
    description (yaw = fitted alpha, pitch = 0, in-plane assumed):
        d_yaw   = |yaw_true - alpha_fit|   folded to <= 90 deg
        d_pitch = |pitch_true - 0|         (= pitch_true)
    Saves <sweep>/grain_axis_errors.npz with one row per crystal."""
    d = SW.sweep_dir(cfg["name"])
    lock = os.path.join(d, "map.lock")      # GUI polls while present
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    try:
        with open(os.path.join(d, "fit_result.json"),
                  encoding="utf-8") as fh:
            fit = json.load(fh)
        alpha_f = float(fit["alpha_deg"]) % 180.0
        f0 = cfg["f0_mhz"] * 1e6
        h = C_REF / f0 / cfg["ppw"]
        sp = DiskSpecimen(diameter_m=cfg["diameter_mm"] * 1e-3,
                          thickness_m=cfg["thickness_mm"] * 1e-3,
                          n_grains=cfg["n_grains"],
                          size_cv=cfg["size_cv"],
                          concentration=cfg["concentration"],
                          spatial_corr=cfg["spatial_corr"],
                          fabric_axis=tuple(cfg["fabric_axis"]),
                          seed=cfg["seed"])
        axes = np.asarray(sp.build(h)["axes"], float)
        yaw = np.degrees(np.arctan2(axes[:, 1], axes[:, 0])) % 180.0
        pitch = np.degrees(np.arcsin(np.clip(np.abs(axes[:, 2]),
                                             0, 1)))
        dyaw = np.abs(yaw - alpha_f)
        dyaw = np.minimum(dyaw, 180.0 - dyaw)
        dpitch = pitch                      # inverted pitch is 0
        tmp = os.path.join(d, "grain_axis_errors.tmp.npz")
        np.savez(tmp, grain=np.arange(len(axes)), yaw_true=yaw,
                 pitch_true=pitch, d_yaw=dyaw, d_pitch=dpitch,
                 alpha_fit_deg=alpha_f)
        SW._atomic_replace(tmp,
                           os.path.join(d, "grain_axis_errors.npz"),
                           critical=True)
        print(f"[grains] {len(axes)} crystals: |d_yaw| median "
              f"{np.median(dyaw):.1f} deg (max {dyaw.max():.1f}), "
              f"|d_pitch| median {np.median(dpitch):.1f} deg "
              f"(max {dpitch.max():.1f})", flush=True)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


if __name__ == "__main__":
    if sys.argv[1] == "--build-cal":
        build_calibration(SW.load(sys.argv[2]))
    elif sys.argv[1] == "--map":
        misfit_map(SW.load(sys.argv[2]))
    elif sys.argv[1] == "--grain-errors":
        grain_axis_errors(SW.load(sys.argv[2]))
    elif sys.argv[1] == "--build-grid":
        build_shape_grid(SW.load(sys.argv[2]))
    else:
        fit(SW.load(sys.argv[1]))

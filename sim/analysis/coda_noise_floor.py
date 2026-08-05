"""Noise floor of the Sec. 5.2 coda claims.

Every number in Sec. 5.2 was measured on noiseless synthetic traces. That
is not a criticism of the simulation, it is a statement about what the
result currently licenses: it says the coda CARRIES the information, and
says nothing about the instrument needed to read it. This module answers
the second question by putting measurement noise on the stored traces and
rerunning the identical statistics until each claim breaks.

NOISE MODEL, and why this one.

  Additive, stationary, Gaussian, and band-limited to the 0.8 to 3.0 MHz
  analysis band of Sec. 3, added to the raw trace before any processing.
  The band choice is the conservative one and it matters. A receiver's
  thermal and digitiser noise is white over a much wider band, and the
  analysis filter throws most of it away; injecting white noise at the
  96 MHz simulation rate and quoting its full-band level would flatter
  the method by the filter's rejection ratio, which is about 16 dB here.
  Putting the noise IN the analysis band means every decibel of the
  quoted SNR is a decibel the processing has to fight. The wideband case
  is measured too, as a conversion factor, in section E of the output.

  The noise is drawn independently per azimuth, which is what an
  uncorrelated receiver noise floor does, and it is drawn at each
  azimuth's OWN sample rate: the CFL step differs by a few parts per
  thousand between azimuths, so a noise vector shared by sample index
  would be a different physical signal at each azimuth.

SNR DEFINITION, and why against the backwall.

  SNR = 20 log10 (E1 / sigma), where E1 is the peak of the analytic
  envelope of the backwall echo, per azimuth, averaged over the sweep,
  and sigma is the rms of the added noise. Both are things a practitioner
  reads off one A-scan: the height of the backwall echo, and the rms of a
  quiet stretch of the same trace. No quantity in this definition depends
  on knowing the answer.

  The number this has to be read against is the coda-to-backwall ratio
  the simulation itself produces, which section A of the output measures
  rather than assumes. It is about -60 dB in the envelope convention the
  paper uses and -64 dB in signal rms, so an instrument needs roughly
  64 dB of in-band SNR merely to put the coda at the noise, before any
  question of whether the pattern inside it survives.

COHERENT CONTAMINATION.

  White noise is the wrong worry for a contact measurement. The failure
  that actually happens is a transducer or couplant reverberation: a
  ringing that arrives at a FIXED time, with the same waveform at every
  azimuth, because it lives in the probe and not in the specimen. Every
  statistic here is azimuthal, and the centring convention removes the
  per-time mean across azimuths, so an exactly azimuth-invariant addition
  is largely cancelled by construction. What is not cancelled is its
  cross term with the coda, and what is not cancelled at all is the part
  that jitters. Both are measured: an exactly repeatable ring, and the
  same ring with 10 per cent rms amplitude jitter between azimuths, which
  is what varying couplant thickness gives.

  The ring is a 2 MHz tone burst with a 6 us decay constant starting at
  20 us, band-limited to the receiver band, and its level is quoted as
  its rms INSIDE the 24 to 36 us coda gate relative to the backwall peak.
  That is the same convention as the coda level itself, so contamination
  and noise can be compared decibel for decibel at equal in-gate power.

WHAT IS MEASURED, per realisation, on the eight ppw 8 girdle
tessellations and the 30 azimuths they share:

  r_field    the azimuth-by-time field correlation of Sec. 5.2 item 1
  rank_reg   the rank of the true azimuthal registration among the 30
             rigid rotations, and the count of tessellations ranking
             first, which is six of eight in the manuscript
  rank_ident the rank of the true tessellation among 48 candidates, and
             the four-of-eight count of Sec. 5.2 item 3

Nothing about the predictors depends on the noise, so the marched
candidate geometry and the facet field are built once and cached; only
the measurement is redrawn. The noiseless pass is run first and printed
against the stored tessellation_replication.npz, so a discrepancy in the
harness shows up before any noise result is read.

NO CUDA. The 48 candidate tessellations are read from out/tesscache. The
GPU labeller is replaced at import in the same way sample_matrix.py does
it, so that a cache miss falls back to NumPy instead of reaching CUDA.

READS
  out/sweeps/girdle_perp_ppw8/az*.npz               seed 11
  out/sweeps/mx_girdle_s{7,17,23,41,53,71,89}_ppw8/az*.npz
  out/tesscache/tess_s<seed>_p8_k-8.npz             48 candidates
  sim/analysis/tessellation_replication.npz         noiseless reference
WRITES
  sim/analysis/nfcache/march_s<seed>.npz            marched geometry
  sim/analysis/nfcache/field_s<seed>.npy            facet field
  sim/analysis/coda_noise_floor.npz                 sections A to E
  sim/analysis/coda_noise_floor_speckle.npz         section F

A fifth claim is carried as section F because it is the one most exposed
to a noise floor and it costs nothing to add: the speckle rejection of
Sec. 4 rests on the implied gamma shape lying BELOW UNITY at the shortest
gate, and additive noise is fully developed speckle, so it can only push
that number up. Its estimator is imported from speckle_looks.py rather
than reimplemented.

Usage
  python coda_noise_floor.py                 full sweep, 20 realisations
  python coda_noise_floor.py --reals 6       faster, wider error bars
  python coda_noise_floor.py --only speckle  section F alone, ~3 minutes
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import binom, t as student

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
    sys.path[:0] = [os.path.join(sys.path[0], _d)
                    for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import _t2_common as C2                                   # noqa: E402
import facet_predictors as FP                             # noqa: E402
import shift_null_2d as S2                                # noqa: E402
import speckle_looks as SL                                # noqa: E402
from specimen import DiskSpecimen                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
TESS = os.path.join(ROOT, "out", "tesscache")
NFC = os.path.join(HERE, "nfcache")

AZ_COMMON = np.arange(0, 360, 12)
PPW, KAPPA, AXIS = 8.0, -8.0, (1.0, 0.0, 0.0)
CONE_DEG = np.degrees(FP.HALF)
N_RAY = 600

GIRDLE = [("girdle_perp_ppw8", 11), ("mx_girdle_s7_ppw8", 7),
          ("mx_girdle_s17_ppw8", 17), ("mx_girdle_s23_ppw8", 23),
          ("mx_girdle_s41_ppw8", 41), ("mx_girdle_s53_ppw8", 53),
          ("mx_girdle_s71_ppw8", 71), ("mx_girdle_s89_ppw8", 89)]
DISTRACTORS = list(range(100, 140))
CANDS = sorted(set([s for _, s in GIRDLE] + DISTRACTORS))

# The centring the manuscript quotes: mean and azimuthal harmonics k = 1
# to 4 stripped at every time sample, then the per-azimuth level removed.
# Verified rather than assumed. The noiseless control below reprints r and
# the two counts for all four modes and flags the one that reproduces
# tessellation_replication.npz; only "harm" does, on all three statistics
# and on every tessellation, which also settles that the stored npz was
# written with the command-line override and not the module default.
MODE = "harm"

TGRID = np.arange(20e-6, 42e-6, C2.DT_C)
GATE = (TGRID >= C2.CODA_W[0]) & (TGRID < C2.CODA_W[1])
EDGES = np.concatenate([TGRID - C2.DT_C / 2, [TGRID[-1] + C2.DT_C / 2]])
KERN = C2.pulse_power_kernel(C2.DT_C)

# Backwall-peak SNR grid, in dB. The region worth resolving was found by
# a coarse probe first rather than guessed: the statistics are intact at
# 70 dB and gone by 40 dB, so the grid is dense between.
SNR_GRID = [90.0, 80.0, 75.0, 70.0, 65.0, 62.5, 60.0, 57.5, 55.0, 52.5,
            50.0, 47.5, 45.0, 42.5, 40.0, 35.0, 30.0]

# In-gate rms of the coherent ring, dB re the backwall peak, on the same
# convention as the coda level so the two axes can be compared directly.
RING_GRID = [-110.0, -100.0, -90.0, -85.0, -80.0, -75.0, -70.0, -65.0,
             -60.0, -55.0, -50.0, -45.0, -40.0, -35.0]
RING_T0, RING_TAU, RING_F = 20e-6, 6e-6, 2.0e6

# Three coherent variants. A reverberation that repeats EXACTLY at every
# azimuth is the benign limit and the one a lock-in argument would assume;
# what a contact measurement actually delivers is a reverberation whose
# amplitude and arrival time move as the couplant layer changes. The
# delay figure is set by the round trip through a 20 um change in
# couplant thickness at 1500 m/s, which is 27 ns; 25 ns is used, a
# twentieth of a period at 2 MHz and so a 18 degree phase wander.
RING_VAR = [("exact", 0.0, 0.0),
            ("amplitude jitter 10 pct", 0.10, 0.0),
            ("delay jitter 25 ns", 0.0, 25e-9)]

REF = dict(r_field=0.162, sd_field=0.079, n_reg=6, n_ident=4, n_cand=48)

# Noiseless backwall peak per sweep and the unit ring per azimuth, filled
# by prepare() before any perturbation closure runs. Module level so the
# closures do not have to carry them.
REFE1 = {}
RINGC = {}


# ─────────────────────────── no-CUDA guarantee ────────────────────────
def _label_plane_by_plane(x, z, R, seeds, weights):
    """Exact all-seeds Laguerre argmin in NumPy, one z-plane at a time.

    Only reached on a cache miss. specimen.py labels with CuPy; this
    module runs on the CPU while a GPU job holds the device, so the GPU
    entry point is replaced rather than guarded against.
    """
    nx, nz = len(x), len(z)
    X, Y = np.meshgrid(x, x, indexing="ij")
    inside = (X ** 2 + Y ** 2) <= R ** 2
    p_xy = np.stack([X[inside], Y[inside]], axis=1)
    d_xy = (((p_xy[:, None, :] - seeds[None, :, :2]) ** 2).sum(-1)
            - weights[None, :])
    raw = np.full((nx, nx, nz), -1, np.int32)
    for iz in range(nz):
        d2 = d_xy + (z[iz] - seeds[:, 2])[None, :] ** 2
        plane = np.full((nx, nx), -1, np.int32)
        plane[inside] = d2.argmin(axis=1).astype(np.int32)
        raw[:, :, iz] = plane
    return raw


DiskSpecimen._label_grid_gpu = staticmethod(_label_plane_by_plane)


# ───────────────────────────── measurement ────────────────────────────
def load_raw(sweep):
    """Unprocessed traces and their own time steps, on the common grid."""
    d = os.path.join(FP.OUT, sweep)
    az, tr, dt = [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if a not in AZ_COMMON:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr.append(np.asarray(z["trace"], float).ravel())
            dt.append(float(z["dt"]))
        az.append(a)
    o = np.argsort(az)
    return (np.array(az)[o], [tr[i] for i in o], [dt[i] for i in o])


_SOS = {}


def sos_for(dt, band=None):
    """Cached band-pass. Every azimuth has its own dt and so its own fs."""
    b = C2.BAND if band is None else band
    key = (dt, b)
    if key not in _SOS:
        fs = 1.0 / dt
        _SOS[key] = butter(4, [b[0] / (fs / 2),
                               min(b[1], 0.45 * fs) / (fs / 2)],
                           btype="band", output="sos")
    return _SOS[key]


def band_noise(n, dt, sigma, rng, band=None):
    """Gaussian noise band-limited to `band`, rms fixed to sigma exactly.

    Normalising each realisation to the requested rms removes the chi
    fluctuation of the noise power, so a level on the SNR axis means one
    SNR and not a distribution of them. The realisation-to-realisation
    scatter reported below is therefore scatter in the noise PATTERN, not
    in how much of it there is.
    """
    w = sosfiltfilt(sos_for(dt, band), rng.standard_normal(n))
    return w * (sigma / np.sqrt(np.mean(w ** 2)))


def ring(nt, dt, t0=RING_T0, tau=RING_TAU, f=RING_F):
    """Transducer reverberation and its in-gate rms, unit amplitude.

    Built in physical time and band-limited by the receiver, so the same
    reverberation is the same physical signal at every azimuth even
    though the azimuths do not share a sample rate. Returning the rms
    inside the coda gate lets the level be set in the same decibels the
    coda level is quoted in.
    """
    t = np.arange(nt) * dt
    y = np.where(t >= t0,
                 np.exp(-(t - t0) / tau) * np.sin(2 * np.pi * f * (t - t0)),
                 0.0)
    y = sosfiltfilt(sos_for(dt), y)
    fs = 1.0 / dt
    g = slice(int(C2.CODA_W[0] * fs), int(C2.CODA_W[1] * fs))
    return y, float(np.sqrt(np.mean(y[g] ** 2)))


def measure(trs, dts):
    """Everything the Sec. 5.2 statistics read, from noisy traces.

    Reproduces facet_predictors.measure and the coda_field of
    tessellation_replication.py exactly, on traces held in memory rather
    than reread from disk. E1 for the SNR reference is the UNFILTERED
    analytic peak, which is what an oscilloscope shows; the field
    normalisation uses the in-band peak, which is what the published
    pipeline uses. Both are carried.
    """
    e1u, e1f, t1, rows, cd = [], [], [], [], []
    for tr, dt in zip(trs, dts):
        fs = 1.0 / dt
        y = sosfiltfilt(sos_for(dt), tr)
        env = np.abs(hilbert(y))
        k0 = int(2 * C2.DIA / C2.C_REF * fs)
        w2, w3 = int(2e-6 * fs), int(3e-6 * fs)
        lo2, lo3 = max(k0 - w2, 0), max(k0 - w3, 0)
        e1u.append(np.abs(hilbert(tr))[lo2:k0 + w2].max())
        seg = env[lo3:k0 + w3]
        e1f.append(seg.max())
        t1.append((lo3 + int(np.argmax(seg))) * dt)
        g = slice(int(C2.CODA_W[0] * fs), int(C2.CODA_W[1] * fs))
        cd.append(np.sqrt((env[g] ** 2).mean()))
        rows.append(np.interp(TGRID, np.arange(len(tr)) * dt, env))
    e1u, e1f = np.array(e1u), np.array(e1f)
    E = np.array(rows) / float(np.mean(e1f))
    return dict(E2=E[:, GATE] ** 2,
                c_az=2.0 * C2.DIA / (np.array(t1) - C2.T0_SRC),
                e1u=float(np.mean(e1u)),
                coda_db=float(20 * np.log10(np.mean(cd) / np.mean(e1u))))


# ──────────────────────────── predictors ──────────────────────────────
def tess(seed):
    """Label volume, c-axes and seed points of a candidate, from cache."""
    p = os.path.join(TESS, f"tess_s{seed}_p{PPW:g}_k{KAPPA:g}.npz")
    if not os.path.exists(p):                       # cache miss: CPU build
        h = C2.C_REF / C2.F0 / PPW
        b = DiskSpecimen(diameter_m=C2.DIA, thickness_m=0.035,
                         n_grains=100, size_cv=0.35, concentration=KAPPA,
                         spatial_corr=0.0, fabric_axis=tuple(AXIS),
                         seed=seed).build(h)
        np.savez_compressed(p, labels=b["labels"], axes=b["axes"],
                            seeds=b["seeds"], h=h)
    with np.load(p) as z:
        return (z["labels"].astype(np.int32), z["axes"], z["seeds"],
                float(z["h"]))


def march(seed):
    """Beam-crossed boundary weights and ranges, geometry only, cached."""
    os.makedirs(NFC, exist_ok=True)
    p = os.path.join(NFC, f"march_s{seed}.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            w, s, off = z["w"], z["s"], z["off"]
        return [(w[off[i]:off[i + 1]].astype(float),
                 s[off[i]:off[i + 1]].astype(float))
                for i in range(len(AZ_COMMON))]
    lab, axes, sd, h = tess(seed)
    ws, ss = [], []
    for a in AZ_COMMON:
        _, w, s = C2.facet_events(lab, axes, sd, h, a, use_dv=False,
                                  use_direc=True, n_ray=N_RAY,
                                  th_max_deg=CONE_DEG, const_v=True)
        ws.append(w)
        ss.append(s)
    off = np.concatenate([[0], np.cumsum([len(x) for x in ws])])
    np.savez_compressed(p, w=np.concatenate(ws).astype(np.float32),
                        s=np.concatenate(ss).astype(np.float32), off=off)
    return list(zip(ws, ss))


def field_pred(seed):
    """Eq. (facetmodel) as an azimuth-by-time power field, cached."""
    os.makedirs(NFC, exist_ok=True)
    p = os.path.join(NFC, f"field_s{seed}.npy")
    if os.path.exists(p):
        return np.load(p)
    lab, axes, sd, h = tess(seed)
    rows = []
    for a in AZ_COMMON:
        t, w, _ = C2.facet_events(lab, axes, sd, h, a, use_dv=True,
                                  use_direc=True, n_ray=N_RAY,
                                  th_max_deg=CONE_DEG)
        rows.append(np.histogram(t, EDGES, weights=w)[0])
    P = np.apply_along_axis(lambda r: np.convolve(r, KERN, mode="same"),
                            1, np.array(rows))[:, GATE]
    np.save(p, P)
    return P


_BE = {}


def bin_events(seed, ev, c_az):
    """Marched ranges mapped to two-way time with the MEASURED speed.

    Memoised on (candidate, measured speed vector). The speed comes from
    the backwall arrival, which sits 45 dB or more above the noise at
    every level swept here, so it usually does not move between
    realisations and the same candidate field is rebuilt identically
    hundreds of times. The cache is keyed on the speeds themselves, so a
    realisation that DOES move the backwall gets its own field; nothing
    is approximated.
    """
    key = (seed, c_az.tobytes())
    if key in _BE:
        return _BE[key]
    rows = [np.histogram(2.0 * s / c_az[q] + C2.T0_SRC, EDGES,
                         weights=w)[0] for q, (w, s) in enumerate(ev)]
    P = np.apply_along_axis(
        lambda r: np.convolve(r, KERN, mode="same"), 1,
        np.array(rows))[:, GATE]
    if len(_BE) > 600:
        _BE.clear()
    _BE[key] = P
    return P


# ──────────────────────────── statistics ──────────────────────────────
def field_claim(E2, P):
    """r at true registration and its rank among the 30 rigid rotations."""
    n = E2.shape[0]
    r = np.array([S2.score(np.roll(E2, s, axis=0), P, MODE)
                  for s in range(n)])
    return float(r[0]), int(np.sum(r >= r[0]))


def ident_claim(E2, c_az, own, marched):
    """Rank of the true tessellation among the 48 candidates."""
    v = np.array([S2.score(E2, bin_events(c, marched[c], c_az), MODE,
                           log=False) for c in CANDS])
    i = CANDS.index(own)
    return int(np.sum(v >= v[i])), float(v[i])


def pooled(v):
    """Fisher-z t test with the tessellation as the unit of replication."""
    z = np.arctanh(np.clip(np.asarray(v, float), -0.999, 0.999))
    m = len(z)
    tstat = z.mean() / (z.std(ddof=1) / np.sqrt(m))
    return float(np.tanh(z.mean())), float(student.sf(tstat, m - 1))


def p_count(k, m, q):
    """One-sided binomial level for k of m first ranks at chance 1/q."""
    return float(binom.sf(k - 1, m, 1.0 / q))


def realisation(raw, marched, fields, perturb):
    """One draw: perturb every sweep, rerun all three statistics."""
    r_f, rk_f, rk_i, lev = [], [], [], []
    for j, (name, seed) in enumerate(GIRDLE):
        az, trs, dts = raw[name]
        noisy = perturb(j, trs, dts, name)
        m = measure(noisy, dts)
        r, rk = field_claim(m["E2"], fields[seed])
        ri, _ = ident_claim(m["E2"], m["c_az"], seed, marched)
        r_f.append(r)
        rk_f.append(rk)
        rk_i.append(ri)
        lev.append(m["coda_db"])
    r_f = np.array(r_f)
    return dict(r=r_f, rank_reg=np.array(rk_f), rank_id=np.array(rk_i),
                n_reg=int(np.sum(np.array(rk_f) == 1)),
                n_id=int(np.sum(np.array(rk_i) == 1)),
                r_mean=float(r_f.mean()), r_sd=float(r_f.std(ddof=1)),
                p_pool=pooled(r_f)[1], coda_db=np.array(lev))


# ───────────────────────────── reporting ──────────────────────────────
def summarise(rows):
    """Mean and sd over realisations of every recorded outcome."""
    out = {}
    for k in ("r_mean", "n_reg", "n_id", "p_pool"):
        v = np.array([r[k] for r in rows], float)
        out[k] = (float(v.mean()),
                  float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    for k in ("rank_reg", "rank_id"):
        v = np.array([r[k] for r in rows], float)
        out[k] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
                  if len(v) > 1 else 0.0)
    out["coda_db"] = float(np.mean([r["coda_db"].mean() for r in rows]))
    return out


def crossing(x, y, thr, rising=False):
    """Linear interpolation of where y crosses thr along x. INFERRED."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1):
        a, b = y[i], y[i + 1]
        if (a >= thr > b) or (rising and a <= thr < b):
            if b == a:
                continue
            return float(x[i] + (thr - a) * (x[i + 1] - x[i]) / (b - a))
    return np.nan


def table(head, levels, sums, unit):
    """One contamination axis. Error bars are sd over realisations."""
    print("=" * 79)
    print(head)
    print("=" * 79)
    print(f"{unit:>10}{'coda/E1':>9}{'r_field':>15}{'rk_reg':>8}"
          f"{'first/8':>13}{'rk_id':>7}{'first/8':>13}")
    for lv, s in zip(levels, sums):
        print(f"{lv:>10}{s['coda_db']:>9.1f}"
              f"{s['r_mean'][0]:>8.3f} +-{s['r_mean'][1]:<5.3f}"
              f"{s['rank_reg'][0]:>8.1f}"
              f"{s['n_reg'][0]:>8.2f} +-{s['n_reg'][1]:<4.2f}"
              f"{s['rank_id'][0]:>7.1f}"
              f"{s['n_id'][0]:>8.2f} +-{s['n_id'][1]:<4.2f}")


def prepare(raw):
    """Noiseless backwall peak per sweep and the unit ring per azimuth.

    The backwall peak is the SNR reference and must be the NOISELESS one,
    otherwise the noise would set its own denominator.
    """
    for name, _ in GIRDLE:
        az, trs, dts = raw[name]
        e = []
        for q, (tr, dt) in enumerate(zip(trs, dts)):
            fs = 1.0 / dt
            k0, w = int(2 * C2.DIA / C2.C_REF * fs), int(2e-6 * fs)
            e.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
            RINGC[(name, q)] = ring(len(tr), dt)
        REFE1[name] = float(np.mean(e))


def ring_perturb(lvl, a_jit, d_jit, rep):
    """Perturbation closure for one coherent-contamination setting.

    The level is the ring's rms inside the coda gate, and it is held at
    that level by the NOMINAL ring's own rms, so a delay-jittered ring is
    not silently rescaled: only its arrival moves.
    """
    def pert(j, trs, dts, name):
        e1 = REFE1[name]
        rng = np.random.default_rng([int(-lvl), rep, j,
                                     int(a_jit * 1000), int(d_jit * 1e10)])
        amp = rng.normal(1.0, a_jit, len(trs)) if a_jit > 0 \
            else np.ones(len(trs))
        dtau = rng.normal(0.0, d_jit, len(trs)) if d_jit > 0 \
            else np.zeros(len(trs))
        out = []
        for q, (tr, dt) in enumerate(zip(trs, dts)):
            y, rms = RINGC[(name, q)]
            if d_jit > 0:
                y = ring(len(tr), dt, t0=RING_T0 + dtau[q])[0]
            out.append(tr + amp[q] * e1 * 10 ** (lvl / 20.0) / rms * y)
        return out
    return pert


def gate_levels(trs, dts):
    """10 log10 of the audited gate power at each gate end, per azimuth.

    The estimator is speckle_looks.gate_power run on traces held in
    memory: 2 E[x^2] over the gate on the band-limited trace, referenced
    to the source amplitude of the same trace. Reproducing it here rather
    than reimplementing it keeps the speckle test on the same decibel the
    subsection uses.
    """
    out = {e: [] for e in SL.GATE_ENDS_US}
    for tr, dt in zip(trs, dts):
        rec = dict(trace=tr, dt=dt, band=SL.bandpass(tr, 1.0 / dt),
                   src=np.abs(hilbert(tr)).max() ** 2)
        for e in SL.GATE_ENDS_US:
            out[e].append(10 * np.log10(
                SL.gate_power(rec, SL.CODA_GATE[0], e * 1e-6)))
    return {e: np.array(v) for e, v in out.items()}


def speckle_shapes(raw, perturb):
    """Implied gamma shape against gate length, over the eight sweeps.

    The Sec. 4 speckle rejection rests on the level scatter being LARGER
    than any average of independent looks allows: the implied shape is
    below unity at the shortest gate and grows as T^0.18 instead of T^1.
    Additive noise is fully developed speckle with many looks in the
    gate, so it can only pull the scatter down and the implied shape up.
    That makes this the claim most exposed to a noise floor, and the one
    whose failure mode is a FALSE NEGATIVE: enough noise turns the coda
    into the speckle the subsection says it is not.
    """
    rows = {e: [] for e in SL.GATE_ENDS_US}
    for j, (name, seed) in enumerate(GIRDLE):
        az, trs, dts = raw[name]
        lv = gate_levels(perturb(j, trs, dts, name), dts)
        for e in SL.GATE_ENDS_US:
            rows[e].append(lv[e])
    spans, shapes, sds = [], [], []
    for e in SL.GATE_ENDS_US:
        sd, _ = SL.pooled_scatter(rows[e])
        spans.append(e - SL.CODA_GATE[0] * 1e6)
        sds.append(sd)
        shapes.append(SL.invert_looks(SL.level_sd, sd))
    return (np.array(spans), np.array(sds), np.array(shapes),
            SL.scaling_exponent(spans, shapes))


def speckle_report(raw, reals, noise_perturb):
    """Section F: the noise floor of the speckle rejection of Sec. 4."""
    def none(j, trs, dts, name):
        return trs

    sp, sd0, sh0, ex0 = speckle_shapes(raw, none)
    print("=" * 79)
    print("F. SPECKLE REJECTION. Implied gamma shape against gate length,")
    print("   pooled over the eight tessellations, gate start held at")
    print(f"   {SL.CODA_GATE[0]*1e6:.0f} us. The subsection rejects the "
          f"speckle family on two counts:")
    print("   the shape is BELOW UNITY at the shortest gate, where no")
    print("   average of independent looks can lie, and it grows as T^0.18")
    print("   instead of the T^1 that independent looks require.")
    print("=" * 79)
    print(f"{'SNR dB':>9}" + "".join(f"{s:>7.0f}" for s in sp)
          + f"{'  exponent':>11}{'shape<1?':>10}")
    print(f"{'noiseless':>9}" + "".join(f"{v:>7.2f}" for v in sh0)
          + f"{ex0:>11.2f}{'yes' if sh0[0] < 1 else 'no':>10}")
    out = []
    for snr in SNR_GRID:
        acc = [speckle_shapes(raw, noise_perturb(snr, rep))
               for rep in range(reals)]
        sh = np.mean([a[2] for a in acc], axis=0)
        sh_sd = np.std([a[2] for a in acc], axis=0, ddof=1)
        ex = float(np.mean([a[3] for a in acc]))
        out.append((snr, sh, sh_sd, ex))
        print(f"{snr:>9.1f}" + "".join(f"{v:>7.2f}" for v in sh)
              + f"{ex:>11.2f}{'yes' if sh[0] < 1 else 'no':>10}",
              flush=True)
    print(f"\n  shortest gate is {sp[0]:.0f} us; its shape is "
          f"{sh0[0]:.2f} noiseless, and the sd over {reals}")
    print("  realisations is in the npz beside the mean.")
    sh1 = np.array([o[1][0] for o in out])
    x = np.array(SNR_GRID)
    ok = x[sh1 < 1.0]
    c = crossing(x, 1.0 - sh1, 0.0)
    print(f"  shape stays below unity down to "
          f"{'none' if len(ok) == 0 else f'{ok.min():.1f}'} dB"
          f"{'' if np.isnan(c) else f', interpolated {c:.1f} dB'}")
    exs = np.array([o[3] for o in out])
    ok = x[exs <= 0.5]
    print(f"  exponent stays below 0.5, i.e. clearly under the T^1 of")
    print(f"  independent looks, down to "
          f"{'none' if len(ok) == 0 else f'{ok.min():.1f}'} dB")
    np.savez(os.path.join(HERE, "coda_noise_floor_speckle.npz"),
             snr=np.array(SNR_GRID), spans=sp, shape0=sh0, exp0=ex0,
             sd0=sd0, shape=np.array([o[1] for o in out]),
             shape_sd=np.array([o[2] for o in out]),
             exponent=exs, reals=reals)


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--reals", type=int, default=20)
    ap.add_argument("--skip-ring", action="store_true")
    ap.add_argument("--only", choices=("all", "speckle"), default="all")
    a = ap.parse_args()

    t00 = time.time()
    raw = {n: load_raw(n) for n, _ in GIRDLE}
    for n, _ in GIRDLE:
        assert len(raw[n][0]) == len(AZ_COMMON), n
    prepare(raw)
    print(f"loaded {len(GIRDLE)} sweeps x {len(AZ_COMMON)} azimuths "
          f"({time.time() - t00:.0f} s)", flush=True)

    def noise_perturb(snr, rep):
        def pert(j, trs, dts, name):
            sg = REFE1[name] * 10 ** (-snr / 20.0)
            return [tr + band_noise(len(tr), dt, sg,
                                    np.random.default_rng(
                                        [int(snr * 10), rep, j, q]))
                    for q, (tr, dt) in enumerate(zip(trs, dts))]
        return pert

    if a.only == "speckle":
        speckle_report(raw, a.reals, noise_perturb)
        return

    t0 = time.time()
    fields = {s: field_pred(s) for _, s in GIRDLE}
    marched = {c: march(c) for c in CANDS}
    print(f"predictors ready, {len(CANDS)} candidates "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ── A. noiseless control and the ratio the instrument must beat ────
    def none(j, trs, dts, name):
        return trs

    base = realisation(raw, marched, fields, none)
    print("\n" + "=" * 79)
    print("A. NOISELESS CONTROL. The harness must reproduce Sec. 5.2 before")
    print("   any noise result is read.")
    print("=" * 79)
    ref = np.load(os.path.join(HERE, "tessellation_replication.npz"))
    fr = ref["field"]
    print(f"{'quantity':<34}{'this harness':>14}{'stored npz':>13}"
          f"{'Sec. 5.2':>11}")
    print(f"{'field r, mean over 8':<34}{base['r_mean']:>14.4f}"
          f"{fr[:, 0].mean():>13.4f}{REF['r_field']:>11.3f}")
    print(f"{'field r, sd over 8':<34}{base['r_sd']:>14.4f}"
          f"{fr[:, 0].std(ddof=1):>13.4f}{REF['sd_field']:>11.3f}")
    print(f"{'registration first of 30, of 8':<34}{base['n_reg']:>14d}"
          f"{int(np.sum(fr[:, 1] == 1)):>13d}{REF['n_reg']:>11d}")
    io = ref["ident_own"][:8, 1].astype(int)
    print(f"{'identification first of 48, of 8':<34}{base['n_id']:>14d}"
          f"{int(np.sum(io == 1)):>13d}{REF['n_ident']:>11d}")
    print(f"\n  per tessellation, noiseless:")
    print(f"  {'sweep':<20}{'seed':>5}{'r':>8}{'rank_reg':>10}"
          f"{'rank_id':>9}{'coda dB':>9}")
    for i, (n, s) in enumerate(GIRDLE):
        print(f"  {n:<20}{s:>5}{base['r'][i]:>8.3f}"
              f"{base['rank_reg'][i]:>10d}{base['rank_id'][i]:>9d}"
              f"{base['coda_db'][i]:>9.1f}")
    print("\n  centring modes, for the record "
          "(r mean, reg first, id first):")
    keep = MODE
    for md in ("raw", "col", "double", "harm"):
        MODE = md
        b = realisation(raw, marched, fields, none)
        flag = "  <- reproduces Sec. 5.2" if (
            abs(b["r_mean"] - REF["r_field"]) < 0.005
            and b["n_reg"] == REF["n_reg"]
            and b["n_id"] == REF["n_ident"]) else ""
        print(f"    {md:<8}{b['r_mean']:>8.3f}{b['n_reg']:>6d}"
              f"{b['n_id']:>6d}{flag}")
    MODE = keep

    coda_env = base["coda_db"].mean()
    print(f"\n  MEASURED coda-to-backwall ratio, band-limited envelope rms")
    print(f"  over the 24-36 us gate against the unfiltered backwall peak:")
    print(f"    {coda_env:.1f} dB, mean of the eight tessellations, "
          f"range [{base['coda_db'].min():.1f}, "
          f"{base['coda_db'].max():.1f}]")

    # signal-rms convention, measured on the same traces
    sig = []
    for n, _ in GIRDLE:
        az, trs, dts = raw[n]
        s_, e_ = [], []
        for tr, dt in zip(trs, dts):
            fs = 1.0 / dt
            y = sosfiltfilt(sos_for(dt), tr)
            g = slice(int(C2.CODA_W[0] * fs), int(C2.CODA_W[1] * fs))
            s_.append(np.sqrt((y[g] ** 2).mean()))
            k0, w = int(2 * C2.DIA / C2.C_REF * fs), int(2e-6 * fs)
            e_.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
        sig.append(20 * np.log10(np.mean(s_) / np.mean(e_)))
    coda_sig = float(np.mean(sig))
    print(f"    {coda_sig:.1f} dB in signal rms, which is the convention "
          f"the SNR axis uses")
    print(f"  An instrument therefore reaches coda = noise at an in-band "
          f"backwall SNR of\n  {-coda_sig:.0f} dB, and that is the number "
          f"every threshold below should be read against.")

    # ── B. white-noise sweep ──────────────────────────────────────────
    print("\nsweeping SNR ...", flush=True)
    sums, allrows = [], []
    for snr in SNR_GRID:
        rows = [realisation(raw, marched, fields, noise_perturb(snr, rep))
                for rep in range(a.reals)]
        sums.append(summarise(rows))
        allrows.append(rows)
        s = sums[-1]
        print(f"  SNR {snr:5.1f} dB  r {s['r_mean'][0]:+.3f}  "
              f"reg {s['n_reg'][0]:.2f}/8  id {s['n_id'][0]:.2f}/8  "
              f"({time.time() - t00:.0f} s)", flush=True)

    print()
    table(f"B. WHITE IN-BAND NOISE, mean +- sd over {a.reals} "
          f"realisations. rk_reg is the rank of\n   the true registration "
          f"among 30 rigid rotations, rk_id the rank of the true\n   "
          f"tessellation among 48 candidates, first/8 the count of "
          f"tessellations at rank 1.",
          [f"{s:.1f}" for s in SNR_GRID] + ["noiseless"],
          sums + [summarise([base])], "SNR dB")
    print(f"   Sec. 5.2 is the noiseless row: r = {REF['r_field']:.3f}, "
          f"{REF['n_reg']} of 8 registrations first,\n   "
          f"{REF['n_ident']} of 8 identifications first. Bin(8, 1/30) needs "
          f"2 of 8 for p < 0.05\n   ({p_count(2, 8, 30):.3f}); "
          f"Bin(8, 1/48) needs 2 of 8 ({p_count(2, 8, 48):.3f}).")

    # ── C. coherent contamination ─────────────────────────────────────
    ringsum = {}
    if not a.skip_ring:
        for lab, ajit, djit in RING_VAR:
            # an exactly repeatable ring has no random part, so one
            # realisation is the whole distribution
            nrep = 1 if (ajit == 0 and djit == 0) else a.reals
            print(f"\nsweeping coherent ring, {lab} ...", flush=True)
            ss = []
            for lvl in RING_GRID:
                rows = [realisation(raw, marched, fields,
                                    ring_perturb(lvl, ajit, djit, rep))
                        for rep in range(nrep)]
                ss.append(summarise(rows))
                s = ss[-1]
                print(f"  ring {lvl:6.1f} dB  r {s['r_mean'][0]:+.3f}  "
                      f"reg {s['n_reg'][0]:.2f}/8  id {s['n_id'][0]:.2f}/8"
                      f"  ({time.time() - t00:.0f} s)", flush=True)
            ringsum[lab] = ss
            print()
            table(f"C. COHERENT RING, {lab}. Fixed arrival at "
                  f"{RING_T0*1e6:.0f} us, {RING_TAU*1e6:.0f} us decay,\n"
                  f"   2 MHz, same waveform at every azimuth. Level is the "
                  f"ring's rms INSIDE the\n   coda gate, dB re backwall "
                  f"peak, the same convention as the {coda_sig:.0f} dB "
                  f"coda,\n   so the two contamination axes are comparable "
                  f"decibel for decibel.",
                  [f"{v:.1f}" for v in RING_GRID], ss, "ring dB")

    # ── D. thresholds ─────────────────────────────────────────────────
    print("\n" + "=" * 79)
    print("D. WHERE EACH CLAIM BREAKS. Grid points are MEASURED; the dB")
    print("   figures in brackets are linear interpolation between them,")
    print("   so they are INFERRED and carry the grid spacing as error.")
    print("=" * 79)
    x = np.array(SNR_GRID)
    rm = np.array([s["r_mean"][0] for s in sums])
    nr = np.array([s["n_reg"][0] for s in sums])
    ni = np.array([s["n_id"][0] for s in sums])

    def report(label, y, thr, note):
        c = crossing(x, y, thr)
        ok = x[y >= thr]
        v = "none" if len(ok) == 0 else f"{ok.min():.1f}"
        i = "" if np.isnan(c) else f"[{c:.1f}]"
        print(f"  {label:<44}{v:>7}{i:>9}   {note}")

    print(f"  {'criterion':<44}{'last':>7}{'interp':>9}   note")
    report("field r at half its noiseless value", rm,
           0.5 * base["r_mean"], f"r >= {0.5 * base['r_mean']:.3f}")
    report("field r significant, tessellation as unit",
           -np.array([s["p_pool"][0] for s in sums]), -0.05,
           "one-sided t on Fisher z")
    report("registration recovers 6 of 8", nr, 5.5, "published count")
    report("registration significant, Bin(8,1/30)", nr, 1.5,
           f"2 of 8 gives p = {p_count(2, 8, 30):.3f}")
    report("identification recovers 4 of 8", ni, 3.5, "published count")
    report("identification significant, Bin(8,1/48)", ni, 1.5,
           f"2 of 8 gives p = {p_count(2, 8, 48):.3f}")
    # The gate-integrated LEVEL is a separate observable: it feeds the
    # column-crossing correlation of Sec. 5.4, and noise biases it upward
    # long before it disturbs any pattern statistic, because noise power
    # simply adds to coda power inside the gate.
    lv = np.array([s["coda_db"] for s in sums])
    report("coda level biased by less than 1 dB",
           -(lv - coda_env), -1.0, "level feeds the Sec. 5.4 r = 0.794")
    report("coda level biased by less than 3 dB",
           -(lv - coda_env), -3.0, "upward bias, noise adds power")
    print(f"\n  coda-to-backwall the simulation produces: "
          f"{coda_sig:.1f} dB signal rms, {coda_env:.1f} dB envelope rms")
    print(f"  so an SNR threshold of S dB corresponds to a coda-to-noise "
          f"ratio of S {coda_sig:+.1f} dB.")

    if ringsum:
        print("\n  Coherent ring: the HIGHEST in-gate level, dB re backwall")
        print("  peak, at which each criterion still holds. Compare against")
        print(f"  the coda's own {coda_sig:.1f} dB: a figure above that is a")
        print("  contamination stronger than the signal it sits on.")
        for lab, ss in ringsum.items():
            xr = np.array(RING_GRID)
            rmr = np.array([s["r_mean"][0] for s in ss])
            nrr = np.array([s["n_reg"][0] for s in ss])
            nir = np.array([s["n_id"][0] for s in ss])
            print(f"    {lab}")
            for nm, y, thr in (("field r at half", rmr,
                                0.5 * base["r_mean"]),
                               ("registration 6 of 8", nrr, 5.5),
                               ("registration 2 of 8", nrr, 1.5),
                               ("identification 4 of 8", nir, 3.5),
                               ("identification 2 of 8", nir, 1.5)):
                ok = xr[y >= thr]
                v = f"{ok.max():.1f} dB" if len(ok) else "fails everywhere"
                print(f"      {nm:<26}{v:>18}")

    # ── E. wideband conversion factor ─────────────────────────────────
    print("\n" + "=" * 79)
    print("E. IF THE NOISE IS WIDEBAND. The sweep above puts every decibel")
    print("   inside the analysis band. A receiver whose noise is white to")
    print("   10 MHz delivers the same in-band power at a higher quoted")
    print("   full-band SNR by this factor, measured not assumed:")
    print("=" * 79)
    az, trs, dts = raw[GIRDLE[0][0]]
    rng = np.random.default_rng(0)
    n_wb = band_noise(len(trs[0]), dts[0], 1.0, rng, band=(0.1e6, 10e6))
    n_ib = sosfiltfilt(sos_for(dts[0]), n_wb)
    fac = 20 * np.log10(1.0 / np.sqrt(np.mean(n_ib ** 2)))
    print(f"   {fac:.1f} dB. A wideband SNR of S dB is an in-band SNR of "
          f"S + {fac:.1f} dB,\n   so every threshold in section D is "
          f"{fac:.1f} dB more forgiving on such an instrument.")

    np.savez(os.path.join(HERE, "coda_noise_floor.npz"),
             snr=np.array(SNR_GRID),
             r_mean=rm, r_sd=np.array([s["r_mean"][1] for s in sums]),
             n_reg=nr, n_reg_sd=np.array([s["n_reg"][1] for s in sums]),
             n_id=ni, n_id_sd=np.array([s["n_id"][1] for s in sums]),
             rank_reg=np.array([s["rank_reg"][0] for s in sums]),
             rank_id=np.array([s["rank_id"][0] for s in sums]),
             coda_level=np.array([s["coda_db"] for s in sums]),
             base_r=base["r"], base_reg=base["rank_reg"],
             base_id=base["rank_id"], base_coda=base["coda_db"],
             coda_env=coda_env, coda_sig=coda_sig, wideband_gain=fac,
             ring_level=np.array(RING_GRID),
             ring_var=np.array(list(ringsum)),
             ring_r=np.array([[s["r_mean"][0] for s in ringsum[k]]
                              for k in ringsum]) if ringsum else 0,
             ring_reg=np.array([[s["n_reg"][0] for s in ringsum[k]]
                                for k in ringsum]) if ringsum else 0,
             ring_id=np.array([[s["n_id"][0] for s in ringsum[k]]
                               for k in ringsum]) if ringsum else 0,
             reals=a.reals)
    print(f"\ntotal {time.time() - t00:.0f} s")


if __name__ == "__main__":
    main()

"""Fabric-axis and concentration recovery from time of flight.

Supplies the three quantities Section 5.1 reports: the recovered axis
error per specimen, the recovered against realised Watson concentration,
and the axis error against azimuth count.

Figure tofaz does NOT rest on this module. sim/figures/fig_tofaz.py
draws rigid_seed11, kappa8_seed17 and oos_seed23 from the
fit_result.json each of those sweeps carries; all three are at ppw 6,
none of them is among the twelve sweeps analysed here, and that figure
folds azimuths modulo 180 and smooths before fitting, which this module
does not. The two are independent routes to the same claim, so their
numbers agree only in the aggregate and are not interchangeable.

READS
  out/sweeps/<name>/az*.npz         'trace' and 'dt'. The azimuth is
                                    taken from the file name; the
                                    'az_deg' the file also carries is
                                    never read.
  out/sweeps/<name>/config.json     seed, concentration, fabric_axis,
                                    az_step. The resolution is fixed by
                                    the sweep names below and 'ppw' is
                                    not read.
  the twelve sweeps with a known fabric axis:
      eight girdle tessellations, kappa = -8,  nominal axis 0 deg
      four single-maximum tessellations, kappa = +3.93, nominal 30 deg
  the seed-11 contrast ladder cs_f000..cs_f075 with girdle_perp_ppw8 as
  its f = 1 rung, and the three homogeneous-crystal sweeps zc_s11_ppw6,
  zerocontrast_ppw8, zc_s11_ppw10, which are used as controls on the
  time-of-flight observable itself.
  The realised microstructure is rebuilt on the CPU from
  sim/core/specimen.py at BUILD_H and cached alongside this file; the
  c-axis draw depends only on the realised grain count, which is stable
  in h (from 1.0 mm down to 0.35 mm) but not across seeds: the twelve
  tessellations realise 100 to 108 grains of the 100 requested, and the
  104 of Section 5.1 is what seed 11 realises.

WRITES
  stdout, and one tofaxis_build_s<seed>_k<kappa>_a<axis>.npz cache per
  tessellation beside this file, written on first use and read
  thereafter. Most of the printed numbers are Section 5.1's. The
  arrival-time budget is Section 3.3's, and the pick repeatability, the
  effective sample sizes and the correlation lengths belong with
  Appendix B.

The time-of-flight pick is the manuscript's: the leading edge of the
backwall arrival, the first crossing of 25 per cent of the envelope peak
with sub-sample interpolation, in a +-2 us window about the isotropic
arrival 2D/c. Nothing is band-limited and nothing is folded, so every
azimuth enters as it was recorded. The azimuth count in the subsampling
table is a count of looks and not of independent looks: the residual
correlates over 9.0 deg in the median, and the same 30-azimuth sweeps
return effective sample sizes of 10.6 to 30.0, printed beneath that
table.
"""
import json
import os
import sys

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) +
                   r"\sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import fit_fabric as FF                          # noqa: E402
import forward as FW                             # noqa: E402
import specimen as SP                            # noqa: E402

SWD = ((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) +
       r"\out\sweeps"))
HERE = os.path.dirname(os.path.abspath(__file__))

# Section 3.3: 100 mm diameter, and 3850 m/s is the isotropic reference
# used to set the grid, not the realised diameter-average speed.
C_REF, DIA, THK = 3850.0, 0.100, 0.035
# Section 3.3: leading edge at 25 per cent of the envelope peak. The
# search half-window is wide enough to hold the realised arrival, which
# Section 3.3 puts at 52.6 us against the nominal 51.9 us.
TOF_FRAC, TOF_HALF_W = 0.25, 2.0e-6
# Coarse rebuild of the label volume. Only the grain volumes are taken
# from it, and they are converged well before this (the volume-weighted
# sample axis moves by 0.04 deg between h = 1.0 and 0.35 mm).
BUILD_H = 0.5e-3

GIRDLE = ("girdle_perp_ppw8", "mx_girdle_s7_ppw8", "mx_girdle_s17_ppw8",
          "mx_girdle_s23_ppw8", "mx_girdle_s41_ppw8",
          "mx_girdle_s53_ppw8", "mx_girdle_s71_ppw8",
          "mx_girdle_s89_ppw8")
SINGLE = ("singlemax_ppw8", "mx_single_s17_ppw8", "mx_single_s23_ppw8",
          "mx_single_s41_ppw8")
LADDER = (("cs_f000_s11_ppw8", 0.00), ("cs_f025_s11_ppw8", 0.25),
          ("cs_f050_s11_ppw8", 0.50), ("cs_f075_s11_ppw8", 0.75),
          ("girdle_perp_ppw8", 1.00))
HOMOGENEOUS = ("zc_s11_ppw6", "zerocontrast_ppw8", "zc_s11_ppw10")

PSI = np.arange(0.0, 90.001, 0.5)
ALPHAS = np.arange(0.0, 180.0, 0.25)
# Two decades either side of zero. The sign of kappa is left free so the
# fit has to choose girdle against single maximum on the evidence.
KGRID = np.r_[-np.geomspace(0.3, 40.0, 60)[::-1],
              np.geomspace(0.3, 40.0, 60)]
# Every specimen carries the 12-degree grid; the two 60-azimuth sweeps
# additionally reach the 48 to 90 range Section 5.1 claims convergence
# over, so far as 60.
COUNTS = (30, 20, 15, 12, 10, 6)
COUNTS_LONG = (60, 48, 40)

_TEMPLATE = {}
_ASSIGN = SP.DiskSpecimen._assign


def _chunked_assign(pts, seeds, weights, k=12):
    """The stock assignment materialises an (n, 12, 3) temporary, which
    is several GB on a production grid. Same arithmetic, in slices."""
    out = np.empty(len(pts), np.int16)
    for i in range(0, len(pts), 400000):
        out[i:i + 400000] = _ASSIGN(pts[i:i + 400000], seeds, weights, k)
    return out


SP.DiskSpecimen._assign = staticmethod(_chunked_assign)
# The GPU labeller is exact where the CPU shortlist is not, but the two
# differ in 3 cells of 10.7M and nothing here is sensitive to that. The
# CPU path is forced so this module never touches CUDA.
SP.DiskSpecimen._label_grid_gpu = staticmethod(lambda *a, **k: None)


# ── loading ──────────────────────────────────────────────────────────
def load_config(name):
    p = os.path.join(SWD, name, "config.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_tof(name, level=TOF_FRAC):
    """Azimuth (deg) and leading-edge time of flight (us) per trace."""
    d = os.path.join(SWD, name)
    az, tof = [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        with np.load(os.path.join(d, f)) as z:
            trace = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        env = np.abs(hilbert(trace))
        k0 = int(2.0 * DIA / C_REF * fs)
        lo = max(k0 - int(TOF_HALF_W * fs), 0)
        seg = env[lo:k0 + int(TOF_HALF_W * fs)]
        ipk = int(np.argmax(seg))
        thr = level * seg[ipk]
        above = np.nonzero(seg[:ipk + 1] >= thr)[0]
        j = int(above[0]) if above.size else ipk
        if j > 0 and seg[j] > seg[j - 1]:
            sub = (thr - seg[j - 1]) / (seg[j] - seg[j - 1])
        else:
            sub = 0.0
        az.append(int(f[2:5]))
        tof.append((lo + j - 1 + sub) / fs * 1e6)
    order = np.argsort(az)
    return np.asarray(az, float)[order], np.asarray(tof)[order]


def load_specimen(seed, kappa, axis):
    """Realised c-axes and grain volumes of one tessellation."""
    tag = f"tofaxis_build_s{seed}_k{kappa:g}_a{axis[0]:.3f}.npz"
    p = os.path.join(HERE, tag)
    if os.path.exists(p):
        with np.load(p) as z:
            return np.asarray(z["axes"]), np.asarray(z["vol"])
    built = SP.DiskSpecimen(diameter_m=DIA, thickness_m=THK,
                            n_grains=100, size_cv=0.35,
                            concentration=kappa, spatial_corr=0.0,
                            fabric_axis=tuple(axis), seed=seed).build(
                                BUILD_H)
    axes = np.asarray(built["axes"], float)
    lab = built["labels"]
    vol = np.bincount(lab[lab >= 0].ravel(),
                      minlength=len(axes)).astype(float)
    np.savez_compressed(p, axes=axes, vol=vol)
    return axes, vol


# ── geometry and truth ───────────────────────────────────────────────
def fold_180(delta):
    """Axial angular difference, folded onto [0, 90] degrees."""
    d = abs(float(delta)) % 180.0
    return min(d, 180.0 - d)


def azimuth_of(vec):
    return float(np.degrees(np.arctan2(vec[1], vec[0])) % 180.0)


def sample_axis(axes, vol, kappa):
    """Volume-weighted orientation tensor and the Watson axis it implies.

    The Watson axis is the largest eigenvector for a single maximum and
    the smallest for a girdle, since the girdle axis is its normal.
    """
    w = vol / vol.sum()
    tensor = np.einsum("g,gi,gj->ij", w, axes, axes)
    ev, vecs = np.linalg.eigh(tensor)
    order = np.argsort(ev)[::-1]
    ev, vecs = ev[order], vecs[:, order]
    return (vecs[:, 0] if kappa > 0 else vecs[:, 2]), ev


def realised_kappa(ev, kappa_nominal):
    """The Watson kappa whose ensemble eigenvalue matches the realised
    one, so each tessellation is scored against its own fabric rather
    than against the number the generator was asked for."""
    target = ev[0] if kappa_nominal > 0 else ev[2]
    grid = np.r_[-np.linspace(40.0, 0.01, 800),
                 np.linspace(0.01, 40.0, 800)]
    pred = np.array([SP.watson_eigenvalue(k) for k in grid])
    same = grid > 0 if kappa_nominal > 0 else grid < 0
    g, p = grid[same], pred[same]
    return float(g[np.argmin(np.abs(p - target))])


def angle_between(u, v):
    c = abs(float(np.dot(u, v)) / np.linalg.norm(u) / np.linalg.norm(v))
    return float(np.degrees(np.arccos(min(c, 1.0))))


# ── forward model ────────────────────────────────────────────────────
def qp_velocity(cos_to_c):
    """Quasi-longitudinal speed at an angle from the c-axis."""
    psi = np.arccos(np.clip(np.abs(cos_to_c), 0.0, 1.0))
    return np.interp(psi, FW._PSI, FW._VQP)


def beam_direction(az_deg):
    """Section 3.3 geometry: the probe sits on the rim at azimuth az and
    fires along the diameter, so the beam axis is (cos az, sin az)."""
    a = np.radians(np.asarray(az_deg, float))
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)


def template(kappa):
    """Two-way time of flight (us) against beam-to-axis angle, for
    grains drawn from Watson(axis, kappa). Cached: the ensemble average
    is a quadrature over 1500 directions per point."""
    key = round(float(kappa), 4)
    if key not in _TEMPLATE:
        axis = np.array([1.0, 0.0, 0.0])
        inv_v = [FF.odf_moments(axis, key, -beam_direction(p))[1]
                 for p in PSI]
        _TEMPLATE[key] = 2.0 * DIA * np.asarray(inv_v) * 1e6
    return _TEMPLATE[key]


def psi_from_axis(az, alpha):
    d = np.abs((np.asarray(az, float) - alpha) % 180.0)
    return np.minimum(d, 180.0 - d)


def model_tof(az, alpha, kappa):
    return np.interp(psi_from_axis(az, alpha), PSI, template(kappa))


def oracle_tof(axes, vol, az):
    """Time of flight from the realised c-axes, with no fitted anything.

    This is the scattering-free limit of the specimen: the f = 0 rung of
    the contrast ladder is a homogeneous medium carrying exactly this
    orientation average, so it measures the same quantity.
    """
    w = vol / vol.sum()
    out = []
    for n in np.atleast_2d(beam_direction(az)):
        out.append(2.0 * DIA * float(np.sum(w / qp_velocity(axes @ n)))
                   * 1e6)
    return np.asarray(out)


# ── estimators ───────────────────────────────────────────────────────
def harmonic(az, y, k):
    """Amplitude and phase of y ~ A cos(k(theta - phi)), phi in degrees.

    The phase is MINUS the argument of the transform. The sign matters:
    it is the difference between an axis estimate and its reflection.
    """
    th = np.radians(np.asarray(az, float))
    x = np.asarray(y, float)
    x = x - x.mean()
    trans = (2.0 / len(th)) * np.sum(x * np.exp(-1j * k * th))
    return abs(trans), float((-np.degrees(np.angle(trans)) / k)
                             % (360.0 / k))


def template_phase(kappa, k=2):
    """Phase of the k-th harmonic of the template, measured from its own
    axis. Recovering the axis from a harmonic means subtracting this."""
    az = np.arange(0.0, 360.0, 1.0)
    return harmonic(az, model_tof(az, 0.0, kappa), k)


def two_fold_null_kappa(lo=-20.0, hi=-0.5):
    """The girdle concentration at which the template's 180-degree
    periodic component vanishes and reverses.

    Below it the two-fold minimum lies along the axis, above it along
    the girdle plane, so the axis a template fit returns swings by 90
    degrees across this point. It is the reason the girdle axis is not
    identified: kappa = -8 sits close to it and the surviving two-fold
    amplitude is a fiftieth of the four-fold.
    """
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g = template(mid)
        if (g[0] - g[-1]) * (template(lo)[0] - template(lo)[-1]) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_axis_template(az, tof, kappa, alphas=ALPHAS):
    """Section 5.1 stage one: template fit with a free offset and a free
    non-negative gain. The gain is held non-negative because a negative
    one is the same template rotated by 90 degrees, which the alpha
    search already covers."""
    yc = np.asarray(tof, float) - np.mean(tof)
    best = (np.inf, np.nan, np.nan)
    for alpha in alphas:
        m = model_tof(az, alpha, kappa)
        mc = m - m.mean()
        den = float(mc @ mc)
        gain = max(float(mc @ yc) / den, 0.0) if den > 0 else 0.0
        ssr = float(np.sum((yc - gain * mc) ** 2))
        if ssr < best[0]:
            best = (ssr, float(alpha), gain)
    return best[1], best[2], float(np.sqrt(best[0] / len(yc)))


def axis_from_two_fold(az, tof, kappa):
    """The 180-degree periodic component alone, with the template used
    only to fix which way round the phase reads."""
    amp, phase = harmonic(az, tof, 2)
    return (phase - template_phase(kappa, 2)[1]) % 180.0, amp


def fit_axis_kappa_joint(az, tof, kgrid=KGRID):
    """Axis and concentration together, gain still free. Concentration
    then enters only through the SHAPE of the pattern."""
    best = (np.inf, np.nan, np.nan)
    for kappa in kgrid:
        alpha, gain, rms = fit_axis_template(az, tof, kappa)
        ssr = rms * rms * len(tof)
        if ssr < best[0]:
            best = (ssr, alpha, float(kappa))
    return best[1], best[2]


def fit_kappa_amplitude(az, tof, alpha, kgrid=KGRID):
    """Concentration from the modulation DEPTH, with the axis held fixed
    and the gain held at the model's own prediction. This is the only
    route by which the absolute anisotropy of the specimen enters."""
    yc = np.asarray(tof, float) - np.mean(tof)
    best = (np.inf, np.nan)
    for kappa in kgrid:
        m = model_tof(az, alpha, kappa)
        ssr = float(np.sum((yc - (m - m.mean())) ** 2))
        if ssr < best[0]:
            best = (ssr, float(kappa))
    return best[1]


def pick_repeatability(az, tof):
    """rms scatter of a single pick, from partners half a turn apart.

    The specimen at az + 180 is the specimen at az, so the physics is
    common and the difference is twice the incoherent pick error.
    """
    have = {int(a): t for a, t in zip(az, tof)}
    diffs = [have[a] - have[(a + 180) % 360] for a in have
             if (a + 180) % 360 in have]
    if not diffs:
        return np.nan
    return float(np.sqrt(np.mean(np.square(diffs)) / 2.0))


def effective_sample_size(resid, step_deg):
    """Appendix B: N / (1 + 2 sum rho_k), lag sum truncated at the first
    zero crossing. Returns N_eff and the integral correlation length."""
    n = len(resid)
    y = np.asarray(resid, float) - np.mean(resid)
    power = np.abs(np.fft.fft(y)) ** 2
    rho = np.real(np.fft.ifft(power)) / n
    rho = rho / rho[0]
    m = 1
    while m < n // 2 and rho[m] > 0:
        m += 1
    den = 1.0 + 2.0 * sum((1 - j / n) * rho[j] for j in range(1, m))
    ell = step_deg * (0.5 + sum(rho[j] for j in range(1, m)))
    return float(n / den), float(ell)


def subsets(n_total, count):
    """Every distinct maximally spread subset of `count` azimuths."""
    out = []
    seen = set()
    for off in range(n_total):
        idx = (off + np.round(np.arange(count) * n_total / count)
               ).astype(int) % n_total
        key = frozenset(idx.tolist())
        if len(key) == count and key not in seen:
            seen.add(key)
            out.append(np.array(sorted(key)))
    return out


# ── computation ──────────────────────────────────────────────────────
def compute_specimens(names):
    out = []
    for name in names:
        cfg = load_config(name)
        az, tof = load_tof(name)
        kappa_nom = float(cfg["concentration"])
        nominal = np.asarray(cfg["fabric_axis"], float)
        axes, vol = load_specimen(cfg["seed"], kappa_nom,
                                  tuple(cfg["fabric_axis"]))
        svec, ev = sample_axis(axes, vol, kappa_nom)
        if float(np.dot(svec, nominal)) < 0.0:
            svec = -svec
        orc = oracle_tof(axes, vol, az)
        design = np.c_[np.ones(len(az)), orc]
        resid = tof - design @ np.linalg.lstsq(design, tof,
                                               rcond=None)[0]
        neff, ell = effective_sample_size(resid, float(cfg["az_step"]))
        out.append(dict(
            name=name, az=az, tof=tof, oracle=orc, resid=resid,
            step=int(cfg["az_step"]), kappa_nom=kappa_nom,
            kappa_real=realised_kappa(ev, kappa_nom),
            nominal=azimuth_of(nominal), sample=azimuth_of(svec),
            floor_3d=angle_between(svec, nominal), ev=ev,
            neff=neff, ell=ell,
            pick_sd=pick_repeatability(az, tof)))
    return out


def compute_axes(spec):
    """Every axis estimator, per specimen, scored against the sample
    axis."""
    for s in spec:
        k = s["kappa_nom"]
        a_t, _, rms = fit_axis_template(s["az"], s["tof"], k)
        a_j, k_j = fit_axis_kappa_joint(s["az"], s["tof"])
        a_2, amp2 = axis_from_two_fold(s["az"], s["tof"], k)
        a_o, _ = axis_from_two_fold(s["az"], s["oracle"], k)
        halves = [fit_axis_template(s["az"], s["tof"], kk)[0]
                  for kk in (k / 2.0, k, 2.0 * k)]
        s.update(a_tpl=a_t, rms=rms, a_joint=a_j,
                 kappa_shape=k_j, a_two=a_2, amp2=amp2, a_oracle=a_o,
                 e_tpl=fold_180(a_t - s["sample"]),
                 e_joint=fold_180(a_j - s["sample"]),
                 e_two=fold_180(a_2 - s["sample"]),
                 e_oracle=fold_180(a_o - s["sample"]),
                 e_tpl45=min(abs((a_t - s["sample"]) % 45.0),
                             45.0 - abs((a_t - s["sample"]) % 45.0)),
                 tpl_sensitivity=max(fold_180(halves[i] - halves[j])
                                     for i in range(3)
                                     for j in range(3)))
    return spec


def compute_kappa(spec):
    for s in spec:
        s["kappa_amp"] = fit_kappa_amplitude(s["az"], s["tof"],
                                             s["sample"])
        s["kappa_amp_oracle"] = fit_kappa_amplitude(
            s["az"], s["oracle"], s["sample"])
        m = model_tof(s["az"], s["sample"], s["kappa_real"])
        mc = m - m.mean()
        s["gain_at_truth"] = float(mc @ (s["tof"] - s["tof"].mean())
                                   / (mc @ mc))
        s["amp2_model"] = harmonic(s["az"], m, 2)[0]
        s["amp2_oracle"] = harmonic(s["az"], s["oracle"], 2)[0]
    return spec


def compute_counts(spec):
    """Axis error against azimuth count, over every maximally spread
    subset of that size."""
    table = {}
    for s in spec:
        n = len(s["az"])
        rows = {}
        for c in COUNTS + COUNTS_LONG:
            if c > n:
                continue
            errs_t, errs_2 = [], []
            for idx in subsets(n, c):
                a, _, _ = fit_axis_template(s["az"][idx], s["tof"][idx],
                                            s["kappa_nom"])
                errs_t.append(fold_180(a - s["sample"]))
                a2, _ = axis_from_two_fold(s["az"][idx], s["tof"][idx],
                                           s["kappa_nom"])
                errs_2.append(fold_180(a2 - s["sample"]))
            rows[c] = (np.array(errs_t), np.array(errs_2))
        table[s["name"]] = rows
    return table


def compute_controls():
    """The two controls on the observable itself: a homogeneous single
    crystal, where the pattern is known in closed form, and the contrast
    ladder, whose f = 0 rung removes grain-scale scattering while
    keeping the specimen's bulk anisotropy."""
    cfg = load_config("girdle_perp_ppw8")
    axes, vol = load_specimen(cfg["seed"], cfg["concentration"],
                              tuple(cfg["fabric_axis"]))
    homo = []
    for name in HOMOGENEOUS:
        az, tof = load_tof(name)
        # master_run's zero-contrast treatment copies the FIRST grain's
        # stiffness into every grain, so the specimen is one crystal and
        # the pattern is 2D / v_qL with no free parameter but an offset.
        pred = 2.0 * DIA / qp_velocity(beam_direction(az) @ axes[0]) * 1e6
        design = np.c_[np.ones(len(az)), pred - pred.mean()]
        coef = np.linalg.lstsq(design, tof, rcond=None)[0]
        homo.append(dict(name=name, n=len(az), gain=float(coef[1]),
                         offset=float(coef[0] - tof.mean()),
                         rms=float(np.std(tof - design @ coef)),
                         ptp_meas=float(np.ptp(tof)),
                         ptp_pred=float(np.ptp(pred)),
                         corr=float(np.corrcoef(tof, pred)[0, 1])))
    ladder = []
    for name, frac in LADDER:
        az, tof = load_tof(name)
        # the f = 1 rung has 60 azimuths and the rest have 30, so the
        # ladder is read on the 12-degree grid they share
        keep = np.isin(az.astype(int) % 12, [0])
        az, tof = az[keep], tof[keep]
        orc = oracle_tof(axes, vol, az)
        a2, p2 = harmonic(az, tof, 2)
        a4, p4 = harmonic(az, tof, 4)
        ladder.append(dict(name=name, frac=frac, sd=float(tof.std()),
                           mean=float(tof.mean()), a2=a2, p2=p2,
                           a4=a4, p4=p4, n=len(az),
                           corr=float(np.corrcoef(tof, orc)[0, 1])))
    az0 = load_tof("cs_f000_s11_ppw8")[0]
    return homo, ladder, harmonic(az0, oracle_tof(axes, vol, az0), 2)


def compute_arrival_budget(names=("girdle_perp_ppw8", "singlemax_ppw8")):
    """Where the backwall arrival actually sits.

    Section 3.3 reads a realised diameter-average speed off the envelope
    peak. The Ricker source puts its envelope peak 1.2/f0 = 0.6 us after
    t = 0, which Section 5.1 states separately, so that reading carries
    the wavelet delay as though it were slowness.
    """
    rows = []
    for name in names:
        cfg = load_config(name)
        axes, vol = load_specimen(cfg["seed"], cfg["concentration"],
                                  tuple(cfg["fabric_axis"]))
        az, lead = load_tof(name)
        peak = []
        d = os.path.join(SWD, name)
        for f in sorted(os.listdir(d)):
            if not (f.startswith("az") and f.endswith(".npz")):
                continue
            with np.load(os.path.join(d, f)) as z:
                trace = np.asarray(z["trace"], float).ravel()
                dt = float(z["dt"])
            fs = 1.0 / dt
            env = np.abs(hilbert(trace))
            k0 = int(2.0 * DIA / C_REF * fs)
            lo = max(k0 - int(TOF_HALF_W * fs), 0)
            seg = env[lo:k0 + int(TOF_HALF_W * fs)]
            peak.append((lo + int(np.argmax(seg))) / fs * 1e6)
        exact = float(np.mean(oracle_tof(axes, vol, az)))
        rows.append(dict(name=name, lead=float(lead.mean()),
                         peak=float(np.mean(peak)), exact=exact,
                         v_exact=2.0 * DIA / (exact * 1e-6),
                         v_peak=2.0 * DIA / (np.mean(peak) * 1e-6)))
    return rows


def draw_arrival_budget(rows):
    print("\nARRIVAL-TIME BUDGET (us) and the speed each pick implies")
    print(f"  {'sweep':22s} {'lead':>7s} {'peak':>7s} {'exact':>7s} "
          f"{'peak-ex':>8s} {'lead-ex':>8s} {'v_exact':>8s} "
          f"{'v_peak':>8s}")
    for r in rows:
        print(f"  {r['name']:22s} {r['lead']:7.3f} {r['peak']:7.3f} "
              f"{r['exact']:7.3f} {r['peak'] - r['exact']:+8.3f} "
              f"{r['lead'] - r['exact']:+8.3f} {r['v_exact']:8.1f} "
              f"{r['v_peak']:8.1f}")
    print(f"  Ricker envelope-peak delay 1.2/f0 = "
          f"{1.2 / 2.0:.2f} us, and the isotropic reference is "
          f"{C_REF:.0f} m/s")


# ── reporting ────────────────────────────────────────────────────────
def draw_controls(homo, ladder, oracle2):
    print("CONTROLS ON THE TIME-OF-FLIGHT OBSERVABLE")
    print("homogeneous single crystal: measured against 2D/v_qL, one "
          "free offset")
    print(f"  {'sweep':20s} {'n':>3s} {'ptp_meas':>9s} {'ptp_pred':>9s} "
          f"{'gain':>6s} {'resid':>7s} {'r':>7s}")
    for h in homo:
        print(f"  {h['name']:20s} {h['n']:3d} {h['ptp_meas']:9.3f} "
              f"{h['ptp_pred']:9.3f} {h['gain']:6.3f} {h['rms']:7.3f} "
              f"{h['corr']:+7.4f}")
    print("contrast ladder, seed 11 girdle, matched azimuths: f = 0 is "
          "scattering-free")
    print(f"  {'sweep':20s} {'f':>5s} {'n':>3s} {'mean':>8s} {'sd':>6s} "
          f"{'A2':>6s} {'phi2':>6s} {'A4':>6s} {'phi4':>6s} {'r_orc':>7s}")
    for r in ladder:
        print(f"  {r['name']:20s} {r['frac']:5.2f} {r['n']:3d} "
              f"{r['mean']:8.3f} {r['sd']:6.3f} {r['a2']:6.3f} "
              f"{r['p2']:6.1f} {r['a4']:6.3f} {r['p4']:6.1f} "
              f"{r['corr']:+7.4f}")
    print(f"  volume-weighted oracle for the same specimen: "
          f"A2 {oracle2[0]:.3f} us at phi2 {oracle2[1]:.1f} deg")


def draw_floor(spec):
    print("\nSAMPLING FLOOR: nominal against volume-weighted sample axis")
    print(f"{'sweep':22s} {'nom':>6s} {'sample':>7s} {'d_plane':>8s} "
          f"{'d_3D':>6s} {'a1':>6s} {'a2':>6s} {'a3':>6s} {'k_real':>7s}")
    for s in spec:
        print(f"{s['name']:22s} {s['nominal']:6.1f} {s['sample']:7.2f} "
              f"{fold_180(s['sample'] - s['nominal']):8.2f} "
              f"{s['floor_3d']:6.2f} {s['ev'][0]:6.3f} {s['ev'][1]:6.3f} "
              f"{s['ev'][2]:6.3f} {s['kappa_real']:7.2f}")
    for tag, sel in (("girdle", spec[:8]), ("single max", spec[8:])):
        d = np.array([fold_180(s["sample"] - s["nominal"]) for s in sel])
        d3 = np.array([s["floor_3d"] for s in sel])
        print(f"  {tag:11s} in plane {d.mean():5.2f} +- {d.std(ddof=1):.2f}"
              f" deg, 3-D {d3.mean():5.2f} +- {d3.std(ddof=1):.2f} deg")


def draw_axes(spec):
    print("\nAXIS RECOVERY, error against the sample axis (deg)")
    print(f"{'sweep':22s} {'n':>3s} {'a_tpl':>7s} {'err':>6s} "
          f"{'a_2f':>7s} {'err':>6s} {'orc':>6s} {'mod45':>6s} "
          f"{'dk_sens':>8s} {'rms_us':>7s} {'pick':>6s}")
    for s in spec:
        print(f"{s['name']:22s} {len(s['az']):3d} {s['a_tpl']:7.2f} "
              f"{s['e_tpl']:6.2f} {s['a_two']:7.2f} {s['e_two']:6.2f} "
              f"{s['e_oracle']:6.2f} {s['e_tpl45']:6.2f} "
              f"{s['tpl_sensitivity']:8.2f} {s['rms']:7.3f} "
              f"{s['pick_sd']:6.3f}")
    for tag, sel in (("girdle", spec[:8]), ("single max", spec[8:])):
        for key, lab in (("e_tpl", "template"), ("e_two", "two-fold"),
                         ("e_joint", "template, free k"),
                         ("e_oracle", "oracle 2f"),
                         ("e_tpl45", "template mod 45")):
            e = np.array([s[key] for s in sel])
            print(f"  {tag:11s} {lab:16s} median {np.median(e):6.2f} "
                  f"mean {e.mean():6.2f} worst {e.max():6.2f} "
                  f"within 10 deg {int((e < 10).sum())}/{len(e)}")
        e = np.array([fold_180(s["a_tpl"] - s["nominal"]) for s in sel])
        print(f"  {tag:11s} {'template vs nominal':16s} "
              f"median {np.median(e):6.2f} mean {e.mean():6.2f}")
    k0 = two_fold_null_kappa()
    a2 = harmonic(np.arange(0.0, 360.0, 1.0),
                  model_tof(np.arange(0.0, 360.0, 1.0), 0.0, -8.0), 2)[0]
    a4 = harmonic(np.arange(0.0, 360.0, 1.0),
                  model_tof(np.arange(0.0, 360.0, 1.0), 0.0, -8.0), 4)[0]
    print(f"  the girdle template's two-fold component vanishes and "
          f"reverses at kappa = {k0:.2f};")
    print(f"  at the simulated kappa = -8 it is {a2:.4f} us against a "
          f"four-fold {a4:.4f} us, a ratio of 1 to {a4 / a2:.0f}")


def draw_kappa(spec):
    print("\nCONCENTRATION, scored against each tessellation's own "
          "realised kappa")
    print(f"{'sweep':22s} {'k_nom':>6s} {'k_real':>7s} {'k_amp':>7s} "
          f"{'err':>7s} {'k_orc':>7s} {'err':>7s} {'k_shape':>8s} "
          f"{'gain':>6s} {'A2_m':>6s} {'A2_orc':>7s} {'A2_mod':>7s}")
    for s in spec:
        print(f"{s['name']:22s} {s['kappa_nom']:6.2f} "
              f"{s['kappa_real']:7.2f} {s['kappa_amp']:7.2f} "
              f"{s['kappa_amp'] - s['kappa_real']:+7.2f} "
              f"{s['kappa_amp_oracle']:7.2f} "
              f"{s['kappa_amp_oracle'] - s['kappa_real']:+7.2f} "
              f"{s['kappa_shape']:8.2f} {s['gain_at_truth']:6.3f} "
              f"{s['amp2']:6.3f} {s['amp2_oracle']:7.3f} "
              f"{s['amp2_model']:7.3f}")
    for tag, sel in (("girdle", spec[:8]), ("single max", spec[8:])):
        for key, lab in (("kappa_amp", "amplitude"),
                         ("kappa_amp_oracle", "amplitude, oracle"),
                         ("kappa_shape", "shape, free gain")):
            v = np.array([s[key] for s in sel])
            e = np.array([s[key] - s["kappa_real"] for s in sel])
            print(f"  {tag:11s} {lab:18s} {v.mean():7.2f} "
                  f"(nominal {sel[0]['kappa_nom']:+.2f}); error vs "
                  f"realised {e.mean():+6.2f} +- {e.std(ddof=1):5.2f}, "
                  f"worst {e[np.argmax(np.abs(e))]:+6.2f}")


def draw_leverage(spec):
    """How far the modulation depth moves per unit kappa, against how
    well the depth is measured. This is what decides whether the
    concentration deficit is leverage or calibration."""
    print("\nLEVERAGE OF THE MODULATION DEPTH ON KAPPA")
    for tag, sel in (("girdle", spec[:8]), ("single max", spec[8:])):
        k0 = sel[0]["kappa_nom"]
        az = np.arange(0.0, 360.0, 6.0)
        step = 0.25 * abs(k0)
        hi = harmonic(az, model_tof(az, 0.0, k0 + step), 2)[0]
        lo = harmonic(az, model_tof(az, 0.0, k0 - step), 2)[0]
        slope = (hi - lo) / (2.0 * step)
        meas = np.array([s["amp2"] for s in sel])
        orc = np.array([s["amp2_oracle"] for s in sel])
        model = harmonic(az, model_tof(az, 0.0, k0), 2)[0]
        print(f"  {tag:11s} dA2/dkappa {slope:7.4f} us   model A2 "
              f"{model:6.3f} us")
        print(f"  {'':11s} measured A2 {meas.mean():6.3f} +- "
              f"{meas.std(ddof=1):.3f}, oracle A2 {orc.mean():6.3f} +- "
              f"{orc.std(ddof=1):.3f} us")
        print(f"  {'':11s} implied sigma_kappa "
              f"{meas.std(ddof=1) / abs(slope):6.2f} from measurement, "
              f"{orc.std(ddof=1) / abs(slope):5.2f} from realisation "
              f"alone")


def draw_counts(spec, table):
    print("\nAXIS ERROR AGAINST AZIMUTH COUNT, template estimator (deg)")
    head = "".join(f"{c:>7d}" for c in COUNTS)
    print(f"{'sweep':22s}{head}")
    for s in spec:
        row = table[s["name"]]
        print(f"{s['name']:22s}"
              + "".join(f"{np.median(row[c][0]):7.1f}" for c in COUNTS))
    for tag, sel in (("girdle", spec[:8]), ("single max", spec[8:])):
        for j, lab in ((0, "template"), (1, "two-fold")):
            cells, tail = "", ""
            for c in COUNTS:
                pooled = np.concatenate([table[s["name"]][c][j]
                                         for s in sel])
                cells += f"{np.median(pooled):7.1f}"
                tail += f"{np.percentile(pooled, 90):7.1f}"
            print(f"{tag + ' ' + lab:22s}{cells}")
            print(f"{'  90th percentile':22s}{tail}")
    print("  (aggregate rows: over specimens and over every maximally "
          "spread subset of that size)")
    print("\nthe two 60-azimuth sweeps, which alone reach the range "
          "Section 5.1 claims convergence over")
    head = "".join(f"{c:>7d}" for c in COUNTS_LONG + COUNTS)
    print(f"{'sweep':22s}{head}")
    for s in spec:
        row = table[s["name"]]
        if 60 not in row:
            continue
        print(f"{s['name']:22s}"
              + "".join(f"{np.median(row[c][0]):7.1f}"
                        for c in COUNTS_LONG + COUNTS))
    print(f"\n{'sweep':22s} {'step':>5s} {'N':>4s} {'N_eff':>6s} "
          f"{'ell_deg':>8s}")
    for s in spec:
        print(f"{s['name']:22s} {s['step']:5d} {len(s['az']):4d} "
              f"{s['neff']:6.1f} {s['ell']:8.1f}")
    ell = np.array([s["ell"] for s in spec])
    print(f"  integral correlation length of the time-of-flight "
          f"residual: {np.median(ell):.1f} deg median, "
          f"{ell.min():.1f} to {ell.max():.1f}")


def compute_threshold_robustness(names, fracs=(0.15, 0.25, 0.35)):
    """Does the recovered axis depend on where on the leading edge the
    pick is taken? Section 3.3 fixes 25 per cent, and a referee will ask
    whether that choice carries the result."""
    rows = []
    for name in names:
        cfg = load_config(name)
        axes, vol = load_specimen(cfg["seed"], cfg["concentration"],
                                  tuple(cfg["fabric_axis"]))
        svec, _ = sample_axis(axes, vol, float(cfg["concentration"]))
        if float(np.dot(svec, np.asarray(cfg["fabric_axis"],
                                         float))) < 0.0:
            svec = -svec
        got = []
        for f in fracs:
            az, tof = load_tof(name, level=f)
            a, _, _ = fit_axis_template(az, tof,
                                        float(cfg["concentration"]))
            got.append(fold_180(a - azimuth_of(svec)))
        rows.append((name, got))
    return fracs, rows


def draw_threshold_robustness(fracs, rows):
    print("\nPICK-THRESHOLD ROBUSTNESS, axis error (deg)")
    print(f"{'sweep':22s}" + "".join(f"{f:>8.2f}" for f in fracs))
    for name, got in rows:
        print(f"{name:22s}" + "".join(f"{g:8.2f}" for g in got))


def main():
    homo, ladder, oracle2 = compute_controls()
    draw_controls(homo, ladder, oracle2)
    draw_arrival_budget(compute_arrival_budget())
    spec = compute_specimens(GIRDLE + SINGLE)
    draw_floor(spec)
    compute_axes(spec)
    draw_axes(spec)
    compute_kappa(spec)
    draw_kappa(spec)
    draw_leverage(spec)
    draw_counts(spec, compute_counts(spec))
    draw_threshold_robustness(*compute_threshold_robustness(SINGLE))


if __name__ == "__main__":
    main()

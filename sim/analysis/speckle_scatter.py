"""Per-azimuth scatter of the coda level against the Rayleigh floor.

Supports Section 4, "Grain population and speckle statistics", which
compares a measured per-azimuth scatter with the speckle floor implied by
the number of grain boundaries the beam crosses inside the coda gate. Two
jobs, in the order the section needs them:

  1. establish what the measured scatter actually is, over the eight
     independent girdle tessellations rather than one realisation, with
     an interval on it,
  2. derive the Rayleigh prediction from first principles instead of
     quoting it, and say how far the two are apart.

A third block exists only to close an audit. An earlier draft quoted
4.34 dB for the measured scatter with no surviving derivation. The
normalisation search below enumerates the reference, band, envelope
definition, gate, azimuth grid and pooling rule the figure could have
come from, and reports both what the paper's own conventions give and how
many of the alternatives land on 4.34 by chance. The second count is the
useful one: a two-decimal target is hit often enough in a space that size
that a match would not identify anything.

Reads, all under ../../out/sweeps, each directory az*.npz with keys
'trace' and 'dt' plus a config.json:
    girdle_perp_ppw8, mx_girdle_s{7,17,23,41,53,71,89}_ppw8
        the eight independent girdle tessellations at ppw 8
    singlemax_ppw8, mx_single_s{17,23,41}_ppw8
        the four single-maximum sweeps, for the normalisation search only
    girdle_perp, lic_girdle_s11_ppw10
        seed 11 at ppw 6 and ppw 10, to show the scatter has converged
    zerocontrast_ppw8
        the zero-contrast control, which bounds the numerical floor
Rebuilds the eight tessellations through sim/specimen.py to count grain
boundaries along the beam axis. That build is done on the CPU here: this
module is analysis, and the release machine may have no CUDA device.

Levels are referenced to the SOURCE amplitude. Section 4, "Choice of
reference", rules the backwall echo inadmissible because refinement
raises it while lowering the coda, and the same reference is used here so
that a scatter and a resolution step are the same kind of number.

WHY THE CROSSING COUNT IS THE ONE THAT MATTERS. The gate holds a handful
of resolvable arrivals, not a handful of scatterers. What sets the
fluctuation of a gate-averaged level is how many independent looks the
gate averages over, and along the beam axis that is the number of grain
boundaries crossed inside it. The count of grains touched by the whole
beam column is larger and is not the relevant number: those grains are
not resolved from one another in range.
"""
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "out" / "sweeps"

# Acquisition and analysis constants as given in Section 3, and the coda
# gate used throughout the paper.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
CODA_GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)

# Gates the audit is allowed to try. The first is the paper's; the others
# trim or extend it by a pulse length or more at either end, which spans
# every gate any script in the project has used.
AUDIT_GATES = {"24-36": (24e-6, 36e-6), "22-38": (22e-6, 38e-6),
               "25-35": (25e-6, 35e-6), "24-34": (24e-6, 34e-6),
               "26-40": (26e-6, 40e-6), "30-40": (30e-6, 40e-6)}

# The common azimuth grid of the matrix. The two seed-11 sweeps hold 60
# records at 6 degrees and are decimated onto it so that every
# realisation is measured over the same beam positions.
AZ_STEP_DEG = 12
AZ_COMMON = np.arange(0, 360, AZ_STEP_DEG)

GIRDLE = [("girdle_perp_ppw8", 11), ("mx_girdle_s7_ppw8", 7),
          ("mx_girdle_s17_ppw8", 17), ("mx_girdle_s23_ppw8", 23),
          ("mx_girdle_s41_ppw8", 41), ("mx_girdle_s53_ppw8", 53),
          ("mx_girdle_s71_ppw8", 71), ("mx_girdle_s89_ppw8", 89)]
SINGLE = [("singlemax_ppw8", 11), ("mx_single_s17_ppw8", 17),
          ("mx_single_s23_ppw8", 23), ("mx_single_s41_ppw8", 41)]

# Resolution series at fixed tessellation, to show that the scatter has
# stopped moving by ppw 8 and that quoting the ppw 8 ensemble is not a
# choice of the coarser answer.
RESOLUTION = [("girdle_perp", 6), ("girdle_perp_ppw8", 8),
              ("lic_girdle_s11_ppw10", 10)]

CONF = 0.95
AUDIT_TARGET = 4.34

# Harmonic orders removed when asking how much of the scatter is a
# deterministic azimuthal signature rather than speckle. The girdle axis
# lies in the rotation plane, so a fabric term appears at order 2 and its
# first overtone at order 4; nothing above that is expected.
HARMONICS = (2, 4)


# ------------------------------------------------------------------ load

def _envelopes(trace, dt):
    """Analytic envelopes of the raw and band-limited trace."""
    fs = 1.0 / dt
    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                 btype="band", output="sos")
    return np.abs(hilbert(trace)), np.abs(hilbert(sosfiltfilt(sos, trace)))


def sweep_records(name, step_deg=AZ_STEP_DEG):
    """Per-azimuth amplitudes for one sweep, before any normalisation.

    Returns (azimuths, dict of amplitude arrays). Every quantity the
    normalisation search can choose between is measured once here, so
    that the search costs nothing beyond arithmetic and no combination
    is measured on a different code path from any other.
    """
    folder = SWEEPS / name
    keys = ["src", "src_bl", "e1", "e1_bl"]
    for gate in AUDIT_GATES:
        keys += ["%s|%s|%s" % (gate, band, kind)
                 for band in ("raw", "bl") for kind in
                 ("rms", "max", "mean")]
    az, out = [], {k: [] for k in keys}
    for entry in sorted(os.listdir(folder)):
        if not (entry.startswith("az") and entry.endswith(".npz")):
            continue
        degrees = int(entry[2:5])
        if degrees % step_deg:
            continue
        with np.load(folder / entry) as z:
            trace = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        env, env_bl = _envelopes(trace, dt)
        k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
        az.append(degrees)
        out["src"].append(env.max())
        out["src_bl"].append(env_bl.max())
        out["e1"].append(env[max(k0 - w, 0):k0 + w].max())
        out["e1_bl"].append(env_bl[max(k0 - w, 0):k0 + w].max())
        for gate, (t0, t1) in AUDIT_GATES.items():
            a, b = int(t0 * fs), int(t1 * fs)
            for band, e in (("raw", env), ("bl", env_bl)):
                seg = e[a:b]
                out["%s|%s|rms" % (gate, band)].append(
                    np.sqrt((seg ** 2).mean()))
                out["%s|%s|max" % (gate, band)].append(seg.max())
                out["%s|%s|mean" % (gate, band)].append(seg.mean())
    return np.array(az), {k: np.array(v, float) for k, v in out.items()}


def coda_level(rec, gate="24-36", band="raw", kind="rms", ref="src",
               factor=20.0):
    """One per-azimuth level vector from a record dictionary."""
    num = rec["%s|%s|%s" % (gate, band, kind)]
    if ref == "absolute":
        den = np.ones_like(num)
    else:
        den = rec[ref + ("_bl" if band == "bl" else "")]
    return factor * np.log10(num / den)


def load_matrix(family, step_deg=AZ_STEP_DEG):
    """(seeds, per-azimuth level matrix, raw records) for one fabric."""
    seeds, rows, recs = [], [], []
    for name, seed in family:
        az, rec = sweep_records(name, step_deg)
        with open(SWEEPS / name / "config.json") as fh:
            cfg = json.load(fh)
        if cfg["seed"] != seed:
            raise ValueError("%s carries seed %s, expected %d"
                             % (name, cfg["seed"], seed))
        if step_deg == AZ_STEP_DEG and not np.array_equal(az, AZ_COMMON):
            raise ValueError("%s does not cover the common azimuth grid"
                             % name)
        seeds.append(seed)
        rows.append(coda_level(rec))
        recs.append((az, rec))
    return np.array(seeds), np.array(rows), recs


def axis_crossings(seeds, ppw=8.0, azimuths=AZ_COMMON):
    """Grain boundaries crossed on the beam axis inside the coda gate.

    One row of counts per tessellation, one entry per azimuth. Built at
    the production spacing rather than a coarse one: a boundary is only
    counted where the rasterised labels change, so a coarser grid loses
    the thin cells and would undercount the very quantity in question.

    The label volume is produced by an exact all-seeds power argmin taken
    one z plane at a time. That is the same rule the GPU path in
    specimen.py applies, written out here so the module runs without a
    CUDA device and without competing for one.
    """
    import sys
    sys.path.insert(0, str(ROOT / "sim"))
    from specimen import DiskSpecimen

    def cpu_label(x, z, radius, centres, weights):
        nx, nz = len(x), len(z)
        gx, gy = np.meshgrid(x, x, indexing="ij")
        inside = (gx ** 2 + gy ** 2) <= radius ** 2
        pxy = np.stack([gx[inside], gy[inside]], axis=1)
        # The in-plane part of the power distance does not depend on z,
        # so it is formed once and the plane loop only adds one column.
        base = (((pxy[:, None, :] - centres[None, :, :2]) ** 2).sum(-1)
                - weights[None, :])
        raw = np.full((nx, nx, nz), -1, np.int32)
        for iz in range(nz):
            d2 = base + (z[iz] - centres[:, 2])[None, :] ** 2
            plane = np.full((nx, nx), -1, np.int32)
            plane[inside] = d2.argmin(axis=1).astype(np.int32)
            raw[:, :, iz] = plane
        return raw

    DiskSpecimen._label_grid_gpu = staticmethod(cpu_label)
    h = C_REF / F0 / ppw
    d0, d1 = CODA_GATE[0] * C_REF / 2, CODA_GATE[1] * C_REF / 2
    rows = []
    for seed in seeds:
        built = DiskSpecimen(
            diameter_m=DIA, thickness_m=0.035, n_grains=100, size_cv=0.35,
            concentration=-8.0, spatial_corr=0.0, fabric_axis=(1.0, 0.0, 0.0),
            seed=int(seed)).build(h)
        labels = np.asarray(built["labels"])
        nx, ny, nz = labels.shape
        counts = []
        for degrees in azimuths:
            a = np.radians(float(degrees))
            n = np.array([np.cos(a), np.sin(a), 0.0])
            # Sampled at a quarter cell so that no boundary is stepped
            # over; the labels themselves still set the resolution.
            s = np.arange(d0, d1, h / 4)
            q = DIA / 2 * n[None, :] - s[:, None] * n[None, :]
            qi = np.clip(np.rint((q[:, 0] + nx * h / 2) / h - 0.5), 0,
                         nx - 1).astype(int)
            qj = np.clip(np.rint((q[:, 1] + ny * h / 2) / h - 0.5), 0,
                         ny - 1).astype(int)
            line = labels[qi, qj, nz // 2]
            counts.append(int((np.diff(line) != 0).sum()))
        rows.append(counts)
    return np.array(rows, float)


# --------------------------------------------------------------- compute

def chi2_interval(sd, df):
    """Chi-square interval on a standard deviation of known dof."""
    return (sd * np.sqrt(df / stats.chi2.ppf(0.5 + CONF / 2, df)),
            sd * np.sqrt(df / stats.chi2.ppf(0.5 - CONF / 2, df)))


def pooled_scatter(rows):
    """Per-azimuth scatter pooled within realisations, and its interval.

    Pooling within rather than across removes the level differences
    between tessellations, which are the subject of the ensemble section
    and not of this one. The interval is the chi-square interval on a
    pooled standard deviation, whose degrees of freedom are checked
    against the azimuthal autocorrelation before it is quoted.
    """
    var = np.array([r.var(ddof=1) for r in rows])
    df = int(sum(len(r) - 1 for r in rows))
    sd = float(np.sqrt(var.mean()))
    lo, hi = chi2_interval(sd, df)
    return sd, lo, hi, df


def realisation_interval(rows):
    """Interval on the scatter treating the tessellation as the unit.

    The chi-square interval counts every azimuth as a degree of freedom.
    This one counts only the eight realisations, so it is the
    conservative reading, and the two are reported together: if they
    disagree the azimuths are not behaving as independent draws.
    """
    sd = np.array([r.std(ddof=1) for r in rows])
    half = stats.t.ppf(0.5 + CONF / 2, len(sd) - 1) * sd.std(ddof=1) \
        / np.sqrt(len(sd))
    return float(sd.mean()), float(half)


def independent_azimuths(rows, step_deg=AZ_STEP_DEG):
    """Effective azimuth count from the circular autocorrelation.

    Summed to the first sign change, which is the usual truncation: the
    tail of a mean-removed circular series sums to zero exactly and
    carries no information about the correlation length.
    """
    ess = []
    for r in rows:
        y = r - r.mean()
        n = len(y)
        f = np.fft.rfft(y, n=2 * n)
        ac = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
        rho = ac / ac[0]
        m = 1
        while m < n // 2 and rho[m] > 0:
            m += 1
        den = 1 + 2 * sum((1 - j / n) * rho[j] for j in range(1, m))
        ess.append(n / max(den, 1e-9))
    return float(np.mean(ess))


def harmonic_residual(rows, azimuths):
    """Scatter left after removing a deterministic azimuthal signature.

    Fits a constant plus the HARMONICS orders in azimuth to each
    realisation and pools the residual. A fabric signature is coherent in
    the laboratory frame and appears at those orders; speckle does not.
    If the residual is close to the raw scatter then what is being
    compared with a speckle floor is in fact speckle.
    """
    t = np.radians(azimuths)
    cols = [np.ones_like(t)]
    for k in HARMONICS:
        cols += [np.cos(k * t), np.sin(k * t)]
    design = np.column_stack(cols)
    var, df = [], 0
    for r in rows:
        coef, *_ = np.linalg.lstsq(design, r, rcond=None)
        res = r - design @ coef
        dof = len(r) - design.shape[1]
        var.append(res @ res / dof)
        df += dof
    return float(np.sqrt(np.mean(var))), df


def rayleigh_level_sd():
    """Standard deviation of the level of a single-look Rayleigh envelope.

    The envelope of a circular Gaussian field is Rayleigh, so its square
    is exponential and the level in decibels is a scaled logarithm of an
    exponential variate. The variance of that logarithm is the trigamma
    function at one, giving (10/ln 10) pi/sqrt(6) exactly, with no
    dependence on the mean level.
    """
    return 10.0 / np.log(10.0) * np.pi / np.sqrt(6.0)


def nlook_level_sd(n):
    """Level standard deviation after averaging n independent looks.

    A gate that averages n independent speckle intensities returns a
    gamma variate of shape n, whose log has variance equal to the
    trigamma function at n. This reduces to rayleigh_level_sd at n = 1
    and to (10/ln 10)/sqrt(n) for large n, and it is the exact form of
    the reduction that the manuscript writes as a division by sqrt(n).
    """
    return 10.0 / np.log(10.0) * np.sqrt(polygamma(1, n))


def rayleigh_monte_carlo(n_looks, draws=2_000_000, seed=0):
    """The two closed forms above, sampled rather than trusted."""
    rng = np.random.default_rng(seed)
    single = (20 * np.log10(np.sqrt(rng.exponential(1.0, draws)))).std(ddof=1)
    gate = (10 * np.log10(rng.gamma(n_looks, 1.0 / n_looks,
                                    draws))).std(ddof=1)
    return float(single), float(gate)


def normalisation_search(records, target=AUDIT_TARGET, tol=0.005):
    """Every scatter the pipeline could have produced, and the near misses.

    records maps a group name to a list of (azimuths, amplitude dict).
    Enumerates reference, band, envelope definition, decibel convention
    and pooling rule, and returns the full table plus the count landing
    on target. The count is the point of the exercise: it measures how
    little a two-decimal match would prove.
    """
    rule = {"pooled": lambda v: np.sqrt(np.mean([r.var(ddof=1) for r in v])),
            "mean sd": lambda v: np.mean([r.std(ddof=1) for r in v]),
            "concatenated": lambda v: np.concatenate(v).std(ddof=1)}
    table, hits = [], 0
    for group, recs in records.items():
        for gate, band, kind, ref, factor, how in itertools.product(
                sorted(AUDIT_GATES), ("raw", "bl"), ("rms", "max", "mean"),
                ("src", "e1", "absolute"), (20.0, 10.0), sorted(rule)):
            rows = [coda_level(rec, gate, band, kind, ref, factor)
                    for _az, rec in recs]
            value = float(rule[how](rows))
            table.append((group, gate, band, kind, ref, factor, how, value))
            hits += abs(value - target) < tol
    return table, hits


# ---------------------------------------------------------------- report

def main():
    seeds, rows, recs = load_matrix(GIRDLE)
    sd, lo, hi, df = pooled_scatter(rows)
    mean_sd, half = realisation_interval(rows)

    print("per-azimuth scatter of the coda level, dB re source, "
          "ppw 8, gate %.0f-%.0f us" % (CODA_GATE[0] * 1e6,
                                        CODA_GATE[1] * 1e6))
    print("  %-6s %10s %10s" % ("seed", "level", "az sd"))
    for seed, row in zip(seeds, rows):
        print("  %-6d %10.2f %10.2f" % (seed, row.mean(), row.std(ddof=1)))
    print("  pooled over %d tessellations  %.2f dB, "
          "%.0f%% interval [%.2f, %.2f] on %d dof"
          % (len(seeds), sd, 100 * CONF, lo, hi, df))
    print("  same quantity with the tessellation as the unit  "
          "%.2f +- %.2f dB" % (mean_sd, half))
    print("  effective independent azimuths per revolution %.1f of %d"
          % (independent_azimuths(rows), len(AZ_COMMON)))

    res, res_df = harmonic_residual(rows, AZ_COMMON)
    res_lo, res_hi = chi2_interval(res, res_df)
    print("  after removing orders %s in azimuth  %.2f dB on %d dof, "
          "[%.2f, %.2f], %.0f%% of the raw scatter"
          % (", ".join(str(k) for k in HARMONICS), res, res_df, res_lo,
             res_hi, 100 * res / sd))

    print("\nresolution series at fixed tessellation, seed 11")
    for name, ppw in RESOLUTION:
        az, rec = sweep_records(name)
        lev = coda_level(rec)
        print("  ppw %-3d %-22s %.2f dB over %d azimuths"
              % (ppw, name, lev.std(ddof=1), len(lev)))
    az, rec = sweep_records("zerocontrast_ppw8")
    floor = coda_level(rec).std(ddof=1)
    print("  zero-contrast control at ppw 8   %.2f dB, so the numerical "
          "floor carries none of it" % floor)

    print("\nRayleigh speckle floor, closed form")
    single = rayleigh_level_sd()
    print("  single look, (10/ln 10) pi/sqrt(6)            %.3f dB" % single)
    cross = axis_crossings(seeds)
    n_bar = float(cross.mean())
    per_seed = cross.mean(axis=1)
    print("  boundary crossings on the beam axis in the gate")
    for seed, row in zip(seeds, cross):
        print("    seed %-4d %.2f +- %.2f" % (seed, row.mean(),
                                              row.std(ddof=1)))
    print("    pooled over the eight  %.2f, spread of the eight means %.2f"
          % (n_bar, per_seed.std(ddof=1)))
    mc_single, mc_gate = rayleigh_monte_carlo(n_bar)
    print("  Monte Carlo check  single look %.2f dB, %.2f-look gate %.2f dB"
          % (mc_single, n_bar, mc_gate))
    heur = single / np.sqrt(n_bar)
    exact = nlook_level_sd(n_bar)
    print("  prediction at N = %.2f   %.2f dB by division by sqrt(N), "
          "%.2f dB exact" % (n_bar, heur, exact))
    print("  seed 11 alone, N = %.2f  %.2f dB by division by sqrt(N), "
          "%.2f dB exact" % (per_seed[0], single / np.sqrt(per_seed[0]),
                             nlook_level_sd(per_seed[0])))
    print("  total scatter %.2f dB exceeds the two predictions by %.2f "
          "and %.2f dB" % (sd, sd - heur, sd - exact))
    print("  incoherent residual %.2f dB differs from the exact form by "
          "%+.2f dB" % (res, res - exact))

    print("\nnormalisation audit for the %.2f dB figure" % AUDIT_TARGET)
    groups = {"girdle 8": recs,
              "single maximum 4": load_matrix(SINGLE)[2],
              "seed 11 ppw 6": [sweep_records("girdle_perp")],
              "seed 11 ppw 10": [sweep_records("lic_girdle_s11_ppw10")]}
    table, hits = normalisation_search(groups)
    print("  the paper's own conventions, rms envelope, 20 log10, pooled")
    print("  %-18s %-6s %-5s %-9s %s"
          % ("group", "gate", "band", "ref", "scatter"))
    for row in table:
        if (row[0] != "girdle 8" or row[1] != "24-36" or row[3] != "rms"
                or row[5] != 20.0 or row[6] != "pooled"):
            continue
        print("  %-18s %-6s %-5s %-9s %.2f dB"
              % (row[0], row[1], row[2], row[4], row[7]))
    print("  %d combinations enumerated, %d land on %.2f dB"
          % (len(table), hits, AUDIT_TARGET))
    near = sorted(table, key=lambda r: abs(r[7] - AUDIT_TARGET))[:5]
    print("  closest five overall")
    for row in near:
        print("    %-18s %-6s %-5s %-5s %-9s %-4.0f %-13s %.2f dB" % row)


if __name__ == "__main__":
    main()

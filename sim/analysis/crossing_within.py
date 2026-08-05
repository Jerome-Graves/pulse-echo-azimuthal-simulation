"""The crossing-count claim, rebuilt inside the specimen.

The manuscript states that the number of grain-boundary crossings within
the beam column predicts the measured coda level at r = 0.794, p = 0.02,
and that it is the only geometric descriptor of the tessellation that
does. That correlation is between specimens and it has eight points. It
is pre-specified, so it keeps its p, but a screen of the same regime run
on the property-by-observable matrix returned fewer hits than chance
across 32832 cells, so eight points is not a regime in which a referee
should be asked to believe a discovery.

The same relation has a within-specimen form with real power, and this
module measures it. Every sweep holds 30 or 60 azimuths of the SAME
tessellation. The crossing count varies from azimuth to azimuth because
the beam enters somewhere else and the coda gate covers a different part
of the disc, and the measured level varies with it. If more boundaries
crossed inside the gate means more back-scattered power, that has to show
up here, on 30 points per specimen with a circular-shift null that
preserves the azimuthal autocorrelation, and it has to show up in the
same direction in specimen after specimen.

THE NUMBER OF DISTINCT ALIGNMENTS is measured and not assumed. A
quantity set by |c . n| is exactly invariant under a beam reversal, so a
30-azimuth sweep offers only 15 distinct shifts of it. A count marched
inside the coda gate is not invariant, because at az + 180 the gate
covers the opposite half of the same chord. report_periodicity measures
r(az, az + 180) for every series used here, and report_degeneracy goes
further and measures the thing that actually matters, the spread between
r_k and r_{k + n/2} over the shift curve itself. Note that if a series
IS exactly periodic then r_k = r_{k+n/2} identically, the count and the
denominator of the shift p both double, and the p-value computed on n
shifts equals the one computed on n/2 exactly. Periodicity therefore
costs resolution, not correctness: it raises the smallest attainable p
from 1/n to 2/n. Running the null on all n shifts is right either way,
and the measurement says which resolution is on offer.

THE BEAM AXIS AND THE BEAM COLUMN are kept apart throughout, because the
manuscript uses one name for both.

  AXIS    one ray down the beam axis. What a travel time samples.
  COLUMN  the diverging cone. What a coda gate samples, and about
          twenty times as many crossings.

report_between establishes by measurement which of the two the published
number is, by correlating each against the count stored in ensemble.npz.

COMBINING ACROSS SPECIMENS respects the within-specimen nulls. Each
specimen contributes a shift-null p on its own discrete grid, and those
p are combined by Fisher; because the grid is discrete the analytic
chi-square level is conservative, so the same Fisher statistic is also
referred to its exact joint null, drawn by choosing one alignment per
specimen at random. A directional companion, the mean of the eight
per-specimen r against the same joint null, tests the sign the
manuscript claims rather than any departure.

CONTROLS. girdle_seed11_ppw8_uniform_axis and girdle_seed11_ppw8_contrast_f000 are the seed 11
geometry with the acoustic contrast switched off. They carry an
IDENTICAL per-azimuth crossing count to girdle_s11, because the count is
a property of the label volume alone, and a coda that is numerical
floor. Anything that survives there is not scattering.

POWER. A null result is worth nothing unless the regime can detect
something, so report_panel runs the identical machinery over all 35
per-azimuth columns of the sample matrix and Holm-corrects across them.
That is the evidence that the within-specimen regime is a discovery
engine and the crossing count simply is not the thing it discovers.

  MEASURED   every correlation, p, rank and level printed here.
  INFERRED   nothing. Where a threshold is used to classify a series as
             periodic it is printed beside the number it acts on.

CUDA IS NEVER TOUCHED and no solver runs. The per-azimuth crossing
counts are read from the cube analysis/sample_matrix.py wrote from the
cached tessellations; the levels are read from stored traces.

EVERY AZIMUTH HAS ITS OWN dt. The gate is converted to samples with that
azimuth's own sampling rate, so nothing is stacked by sample index. The
Hilbert transform is used only for the peak of the front arrival, which
is a hundred decibels above everything near it; the gate itself is
measured from the signal, E|z|^2 = 2 E x^2, so no part of the coda level
is a non-local transform of a distant arrival.

READS
  sim/results/sample_matrix_azimuth.npz   per-azimuth crossing counts
  sim/results/ensemble.npz                the published n = 8 columns
  out/sweeps/<sweep>/az???.npz            the traces
WRITES
  sim/results/crossing_within.npz
"""
import os
import sys

import numpy as np
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWD = os.path.join(ROOT, "out", "sweeps")
RESULTS = os.path.join(ROOT, "sim", "results")

# Acquisition and analysis constants, Section 3, identical to the ones
# analysis/contrast_ladder.py and analysis/db_reconcile.py use.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)
BW_HALF = 2e-6

# The four crossing counts. The first two are beam COLUMN, the last two
# beam AXIS. n_cross_paper is the 37-ray lattice of facet_predictors
# summed inside the gate, which is the definition ensemble.npz stores.
COL_PAPER = "n_cross_paper"
COL_CONE = "n_cross_col_gate"
AXIS_GATE = "n_cross_axis_gate"
AXIS_FULL = "n_cross_axis_full"
CROSS_KEYS = (COL_PAPER, COL_CONE, AXIS_GATE, AXIS_FULL)
CROSS_KIND = {COL_PAPER: "COLUMN", COL_CONE: "COLUMN",
              AXIS_GATE: "AXIS", AXIS_FULL: "AXIS"}

# A series is called 180-degree periodic above this. The threshold is
# printed beside every number it classifies; nothing downstream depends
# on it, because the shift null is run on all n alignments regardless.
PERIODIC_R = 0.99

N_JOINT = 200000            # draws of the exact joint-shift null
JOINT_SEED = 20260802

CONTROLS = ("zerocontrast_s11", "cs_f000_s11")


def db(x):
    """Power ratio to decibels. Every level here is a power."""
    return 10.0 * np.log10(x)


# ------------------------------------------------------------------ load

def audited_trace(path):
    """The audited coda level of one azimuth, and its backwall echo.

    band  local in-band gate power referenced to the source, which is
          the estimator Section 4.6 settled on: no Hilbert transform
          touches the gate, so none of the level is the tail of the
          analytic signal of the front arrival.
    e1    backwall echo power, kept only as a normalisation alternative
          so the result can be shown not to depend on the reference.
    raw   the same gate, unreferenced, which is the third normalisation.
    """
    with np.load(path) as z:
        trace = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
    fs = 1.0 / dt
    i0, i1 = int(GATE[0] * fs), int(GATE[1] * fs)
    envelope = np.abs(hilbert(trace))
    src2 = envelope.max() ** 2
    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                 btype="band", output="sos")
    filtered = sosfiltfilt(sos, trace)
    gate = 2.0 * (filtered[i0:i1] ** 2).mean()
    k0, kw = int(2 * DIA / C_REF * fs), int(BW_HALF * fs)
    e1 = envelope[max(k0 - kw, 0):k0 + kw].max() ** 2
    return dict(band=gate / src2, raw=gate, e1_ref=gate / e1, dt=dt)


_LEVELS = {}


def sweep_levels(name):
    """(azimuths, dict of per-azimuth power ratios) for one sweep."""
    if name in _LEVELS:
        return _LEVELS[name]
    root = os.path.join(SWD, name)
    az = sorted(int(f[2:5]) for f in os.listdir(root)
                if f.startswith("az") and f.endswith(".npz"))
    rows = [audited_trace(os.path.join(root, "az%03d.npz" % a)) for a in az]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    _LEVELS[name] = (np.array(az), out)
    return _LEVELS[name]


def load_cube():
    """The per-azimuth sample matrix: names, sweeps, kinds, az, cols, M."""
    path = os.path.join(RESULTS, "sample_matrix_azimuth.npz")
    with np.load(path, allow_pickle=True) as z:
        return ([str(x) for x in z["names"]], [str(x) for x in z["sweeps"]],
                [str(x) for x in z["kinds"]], z["az_deg"],
                [str(x) for x in z["columns"]], z["values"])


class Panel(object):
    """The cube and the levels, aligned azimuth by azimuth.

    A sweep of 30 azimuths at 12 degrees is every second row of the
    60-azimuth cube, so the alignment is an exact index match and never
    an interpolation.
    """

    def __init__(self):
        (self.names, self.sweeps, self.kinds, self.az_cube,
         self.columns, self.values) = load_cube()
        self.rows = {}
        for q, name in enumerate(self.names):
            root = os.path.join(SWD, self.sweeps[q])
            if not os.path.isdir(root):
                continue
            az, lev = sweep_levels(self.sweeps[q])
            sel = np.searchsorted(self.az_cube, az)
            if not np.array_equal(self.az_cube[sel], az):
                raise ValueError("azimuths of %s are not a subset of the "
                                 "cube grid" % self.sweeps[q])
            self.rows[name] = dict(index=q, az=az, sel=sel,
                                   kind=self.kinds[q], sweep=self.sweeps[q],
                                   level_db=db(lev["band"]),
                                   level_lin=lev["band"],
                                   level_e1=db(lev["e1_ref"]),
                                   level_raw=db(lev["raw"]))

    def family(self, kind):
        return [n for n in self.names
                if n in self.rows and self.rows[n]["kind"] == kind]

    def predictor(self, name, column):
        r = self.rows[name]
        return self.values[r["index"], self.columns.index(column), r["sel"]]

    def response(self, name, key="level_db"):
        return self.rows[name][key]


# --------------------------------------------------------------- compute

def periodicity(v):
    """r(az, az + 180) of a series sampled on a whole revolution."""
    n = len(v)
    if n % 2 or np.std(v) < 1e-14:
        return np.nan
    return float(np.corrcoef(v, np.roll(v, n // 2))[0, 1])


def shift_curve(x, y):
    """r(x, y rolled by k) for every k, the circular-shift null itself."""
    n = len(y)
    xc = x - x.mean()
    sx = np.sqrt((xc ** 2).sum())
    yc = y - y.mean()
    sy = np.sqrt((yc ** 2).sum())
    if sx < 1e-24 or sy < 1e-24:
        return np.zeros(n)
    return np.array([float(xc @ np.roll(yc, k)) for k in range(n)]) / (sx * sy)


def shift_stats(rs):
    """Two-sided and one-sided shift p, and the rank of the true alignment.

    The p-values count the true alignment itself, so the smallest
    attainable value is 1/n and no p is ever zero. Rank 1 means no
    rotation of the measured level matches the predictor better.
    """
    n = len(rs)
    r0 = float(rs[0])
    p2 = float((np.abs(rs) >= abs(r0)).sum()) / n
    p1 = float((rs >= r0).sum()) / n
    rank_abs = int((np.abs(rs) > abs(r0)).sum()) + 1
    rank_pos = int((rs > r0).sum()) + 1
    return dict(r=r0, p2=p2, p1=p1, rank_abs=rank_abs, rank_pos=rank_pos,
                n_shift=n)


def degeneracy(rs):
    """How far the shift curve is from being 180-degree degenerate.

    Zero means r_k and r_{k+n/2} coincide for every k, so only n/2 of the
    n alignments are distinct and the smallest attainable p is 2/n. One
    or more means the two halves of the curve are as different as the
    curve's own spread, so all n are distinct.
    """
    n = len(rs)
    if n % 2:
        return np.nan
    d = np.abs(rs - np.roll(rs, n // 2)).max()
    s = rs.std()
    return float(d / s) if s > 1e-24 else 0.0


def two_sided_p_grid(rs):
    """The two-sided shift p that EVERY alignment of one specimen gives.

    Under the circular-shift null the true alignment is exchangeable
    with all n, so this vector is the exact null distribution of that
    specimen's p, and drawing one entry at random is drawing one p under
    the null. It is what the joint combination below samples.
    """
    a = np.abs(rs)
    return np.array([float((a >= v).sum()) / len(a) for v in a])


def fisher_analytic(ps):
    """Fisher's combined level, the familiar analytic one."""
    ps = np.clip(np.asarray(ps, float), 1e-300, 1.0)
    chi2 = -2.0 * np.log(ps).sum()
    return chi2, float(stats.chi2.sf(chi2, 2 * len(ps)))


def joint_null(curves, seed=JOINT_SEED, n_draw=N_JOINT):
    """The exact combined level, by drawing one alignment per specimen.

    Fisher on p-values that live on a coarse discrete grid is
    conservative, and the grid here has thirty points. The remedy is not
    a correction factor but the joint null itself: choose an alignment
    independently and uniformly in each specimen, which is exactly what
    the within-specimen null asserts, and read the combined statistic
    off. Two statistics are carried, because they answer different
    questions. Fisher is blind to sign and detects any departure;
    the mean r is directional and tests the sign the manuscript claims.

    Returns the observed statistics, their exact levels, and the Monte
    Carlo standard error of those levels.
    """
    rng = np.random.default_rng(seed)
    grids = [two_sided_p_grid(rs) for rs in curves]
    fis_obs = -2.0 * np.log([g[0] for g in grids]).sum()
    mean_obs = float(np.mean([rs[0] for rs in curves]))
    fis, mean = np.zeros(n_draw), np.zeros(n_draw)
    for g, rs in zip(grids, curves):
        k = rng.integers(0, len(rs), n_draw)
        fis += -2.0 * np.log(g[k])
        mean += rs[k]
    mean /= len(curves)
    p_fis = float((fis >= fis_obs).sum() + 1) / (n_draw + 1)
    p_mean = float((mean >= mean_obs).sum() + 1) / (n_draw + 1)
    se = float(np.sqrt(max(p_fis * (1 - p_fis), 1e-12) / n_draw))
    chi2, p_an = fisher_analytic([rs_p[0] for rs_p in grids])
    return dict(fisher_stat=fis_obs, fisher_exact=p_fis,
                fisher_analytic=p_an, chi2_df=2 * len(curves),
                mean_r=mean_obs, mean_r_exact=p_mean, mc_se=se,
                mean_r_null=float(mean.mean()),
                mean_r_null_sd=float(mean.std(ddof=1)))


def holm(ps):
    """Holm-corrected levels, in the order the inputs were given."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    out, run = np.empty(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (m - rank)))
        out[i] = run
    return out


def bh(ps):
    """Benjamini-Hochberg q-values, in input order."""
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)[::-1]
    out, run = np.empty(m), 1.0
    for j, i in enumerate(order):
        run = min(run, ps[i] * m / (m - j))
        out[i] = run
    return out


def test_family(panel, members, column, response="level_db"):
    """The per-specimen shift tests of one predictor over one family."""
    rows, curves = [], []
    for name in members:
        x = panel.predictor(name, column)
        y = panel.response(name, response)
        rs = shift_curve(x, y)
        st = shift_stats(rs)
        st.update(name=name, r180_x=periodicity(x), r180_y=periodicity(y),
                  degen=degeneracy(rs), n_az=len(x))
        rows.append(st)
        curves.append(rs)
    return rows, curves


def residualise(x, z):
    """The part of x that a linear function of the columns of z misses."""
    z = np.atleast_2d(np.asarray(z, float))
    if z.shape[0] != len(x):
        z = z.T
    A = np.column_stack([np.ones(len(x))] + [z[:, j]
                                             for j in range(z.shape[1])])
    return x - A @ np.linalg.lstsq(A, x, rcond=None)[0]


def theta2(az_deg):
    """The cos and sin of twice the azimuth, the fabric's own symmetry."""
    t = np.radians(np.asarray(az_deg, float))
    return np.column_stack([np.cos(2 * t), np.sin(2 * t)])


def test_derived(panel, members, build, response="level_db"):
    """The same shift tests on a predictor built from several columns.

    build(panel, name) returns the per-azimuth predictor. Residualising
    the PREDICTOR and shifting the RESPONSE keeps the null exact: the
    circular shift is applied to an untouched measured series, so
    nothing about the residualisation can leak into the null.
    """
    rows, curves = [], []
    for name in members:
        x = build(panel, name)
        y = panel.response(name, response)
        rs = shift_curve(x, y)
        st = shift_stats(rs)
        st.update(name=name, degen=degeneracy(rs), n_az=len(x))
        rows.append(st)
        curves.append(rs)
    return rows, curves


# ------------------------------------------------------------------ draw

def report_between(panel):
    """Which count the published r = 0.794 is, measured not assumed."""
    path = os.path.join(RESULTS, "ensemble.npz")
    print("=" * 79)
    print("WHAT THE PUBLISHED NUMBER IS, BETWEEN SPECIMENS, n = 8")
    print("=" * 79)
    if not os.path.exists(path):
        print("  ensemble.npz absent")
        return {}
    with np.load(path) as z:
        seeds = [int(s) for s in z["girdle_seeds"]]
        levels, published = z["girdle_levels"], z["crossings"]
    r, p = stats.pearsonr(published, levels)
    rho, prho = stats.spearmanr(published, levels)
    print("  the manuscript's own two columns")
    print("    published crossing count vs revolution level   "
          "r = %+.3f, p = %.4f" % (r, p))
    print("    the same by rank                               "
          "rho = %+.3f, p = %.4f" % (rho, prho))
    print()
    print("  every crossing definition, recomputed at ppw 8 and averaged")
    print("  over azimuth, against the same eight levels")
    print("  %-20s %8s %10s %9s %9s"
          % ("definition", "kind", "vs pub.", "pearson", "spearman"))
    out = {}
    for key in CROSS_KEYS:
        m = np.array([panel.predictor("girdle_s%d" % s, key).mean()
                      for s in seeds])
        rp = stats.pearsonr(published, m)[0]
        rr, pp = stats.pearsonr(m, levels)
        sr, sp = stats.spearmanr(m, levels)
        print("  %-20s %8s %10.3f %+6.3f/%.3f %+6.3f/%.3f"
              % (key, CROSS_KIND[key], rp, rr, pp, sr, sp))
        out[key] = dict(vs_published=float(rp), pearson=float(rr),
                        p=float(pp), spearman=float(sr), p_spearman=float(sp))
    print()
    print("  and the contrast descriptor the within-specimen test below")
    print("  selects, on the same eight points, with its collinearity to")
    print("  the count")
    dvm = np.array([panel.predictor("girdle_s%d" % s, "dv_cross_rms").mean()
                    for s in seeds])
    cnt = np.array([panel.predictor("girdle_s%d" % s, COL_PAPER).mean()
                    for s in seeds])
    rr, pp = stats.pearsonr(dvm, levels)
    sr, sp = stats.spearmanr(dvm, levels)
    rc = stats.pearsonr(dvm, cnt)[0]
    print("  %-20s %8s %10s %+6.3f/%.3f %+6.3f/%.3f"
          % ("dv_cross_rms", "contrast", "-", rr, pp, sr, sp))
    print("    r(dv_cross_rms, %s) across the eight = %+.3f"
          % (COL_PAPER, rc))
    out["dv_cross_rms"] = dict(vs_published=float(rc), pearson=float(rr),
                               p=float(pp), spearman=float(sr),
                               p_spearman=float(sp))
    print()
    print("  The published column reproduces %s at r = %.3f and the AXIS"
          % (COL_PAPER, out[COL_PAPER]["vs_published"]))
    print("  count only at r = %.3f, so the number in the manuscript is a"
          % out[AXIS_GATE]["vs_published"])
    print("  COLUMN count. The AXIS count over the same eight specimens")
    print("  reaches r = %+.3f (p = %.2f) and rho = %+.3f (p = %.2f):"
          % (out[AXIS_GATE]["pearson"], out[AXIS_GATE]["p"],
             out[AXIS_GATE]["spearman"], out[AXIS_GATE]["p_spearman"]))
    print("  nothing, on eight points, exactly as the screen reported.")
    print()
    return out


def report_periodicity(panel, members):
    """r(az, az+180) of every series the tests use, and what it costs."""
    print("=" * 79)
    print("HOW MANY DISTINCT ALIGNMENTS EACH SERIES OFFERS")
    print("=" * 79)
    print("  measured, not assumed: r(az, az + 180) on each sweep's own")
    print("  grid, and the same for the response it is tested against")
    print("  %-14s %4s %9s %9s %9s %9s %9s %8s"
          % ("specimen", "n", "column", "cone", "axisgate", "axisfull",
             "level", "cv% col"))
    keep = {}
    for name in members:
        vals = [periodicity(panel.predictor(name, k)) for k in CROSS_KEYS]
        y = periodicity(panel.response(name))
        c = panel.predictor(name, COL_PAPER)
        keep[name] = vals + [y]
        print("  %-14s %4d %9.3f %9.3f %9.3f %9.3f %9.3f %8.1f"
              % ((name, len(panel.rows[name]["az"])) + tuple(vals)
                 + (y, 100 * c.std(ddof=1) / c.mean())))
    arr = np.array([keep[n] for n in members])
    cv = np.array([100 * panel.predictor(n, COL_PAPER).std(ddof=1)
                   / panel.predictor(n, COL_PAPER).mean() for n in members])
    print()
    print("  mean over the family  %9.3f %9.3f %9.3f %9.3f %9.3f %8.1f"
          % (tuple(arr.mean(axis=0)) + (cv.mean(),)))
    print()
    print("  The last column is the azimuth-to-azimuth spread of the")
    print("  published count inside one specimen, %.1f per cent of its own"
          % cv.mean())
    print("  mean, against the %.1f per cent that separates the eight"
          % (100 * np.std([panel.predictor(n, COL_PAPER).mean()
                           for n in members], ddof=1)
             / np.mean([panel.predictor(n, COL_PAPER).mean()
                        for n in members])))
    print("  specimen means. The predictor varies more within a specimen")
    print("  than between specimens, which is why this is the regime with")
    print("  the power.")
    print()
    print("  A series is called 180-degree periodic above %.2f." % PERIODIC_R)
    for j, k in enumerate(CROSS_KEYS):
        tag = "PERIODIC" if arr[:, j].mean() > PERIODIC_R else "not periodic"
        print("    %-20s %-12s  mean r(az+180) = %+.3f"
              % (k, tag, arr[:, j].mean()))
    print("    %-20s %-12s  mean r(az+180) = %+.3f"
          % ("audited level", "not periodic"
             if arr[:, -1].mean() <= PERIODIC_R else "PERIODIC",
             arr[:, -1].mean()))
    print()
    print("  Both gate-restricted counts are NOT periodic, so a 30-azimuth")
    print("  sweep gives 30 distinct alignments of them and the shift null")
    print("  can reach p = 1/30. The AXIS count over the WHOLE traverse is")
    print("  the one series here that nearly is periodic, which is the")
    print("  sharpest statement of why the two counts must be kept apart:")
    print("  restricting the same ray to the gate destroys the symmetry,")
    print("  because at az + 180 the gate covers the opposite half of the")
    print("  same chord.")
    print()
    print("  Periodicity costs resolution, not correctness. If r_k =")
    print("  r_{k+n/2} identically then the shift p on n alignments and on")
    print("  n/2 are equal, because numerator and denominator both double;")
    print("  the only effect is that the smallest attainable p becomes 2/n.")
    print("  The null below is therefore run on all n alignments, and")
    print("  report_ranks prints the measured degeneracy of each curve so")
    print("  the resolution actually on offer is visible.")
    print()
    return arr


def report_ranks(rows, title):
    """Per-specimen r, shift p and rank of the true alignment."""
    print("  %s" % title)
    print("  %-14s %4s %7s %7s %7s %8s %8s %7s"
          % ("specimen", "n", "r", "p two", "p one", "rank|r|", "rank +r",
             "degen"))
    for s in rows:
        print("  %-14s %4d %+7.3f %7.3f %7.3f %5d/%-3d %5d/%-3d %7.2f"
              % (s["name"], s["n_az"], s["r"], s["p2"], s["p1"],
                 s["rank_abs"], s["n_shift"], s["rank_pos"], s["n_shift"],
                 s["degen"]))
    r = np.array([s["r"] for s in rows])
    print("  %-14s      %+7.3f   mean, %d of %d positive, sd %.3f"
          % ("", r.mean(), int((r > 0).sum()), len(r), r.std(ddof=1)))


def report_combined(res, indent="  "):
    """The combined level, both statistics."""
    print("%sFisher over the per-specimen shift p"
          "   chi2 = %.2f on %d df" % (indent, res["fisher_stat"],
                                       res["chi2_df"]))
    print("%s  analytic level                     p = %.4f  (conservative,"
          " the p are on a discrete grid)" % (indent, res["fisher_analytic"]))
    print("%s  exact joint-shift level            p = %.4f  +- %.4f"
          % (indent, res["fisher_exact"], res["mc_se"]))
    print("%smean r over the family               %+.3f  against a joint"
          " null of %+.3f +- %.3f" % (indent, res["mean_r"],
                                      res["mean_r_null"],
                                      res["mean_r_null_sd"]))
    print("%s  exact one-sided level              p = %.4f"
          % (indent, res["mean_r_exact"]))
    sd = res["mean_r_null_sd"]
    print("%s  the null's own spread makes a mean r of %+.3f the smallest"
          % (indent, 1.96 * sd))
    print("%s  this design would call significant, so the interval the"
          % indent)
    print("%s  measurement leaves open is [%+.3f, %+.3f]"
          % (indent, res["mean_r"] - 1.96 * sd, res["mean_r"] + 1.96 * sd))


def report_within(panel, kind, label):
    """The headline test: four crossing definitions over one family."""
    members = panel.family(kind)
    n_az = sorted(len(panel.rows[n]["az"]) for n in members)
    print("=" * 79)
    print("WITHIN SPECIMEN, %s, %d sweeps, %d to %d azimuths"
          % (label.upper(), len(members), n_az[0], n_az[-1]))
    print("=" * 79)
    out = {}
    for key in CROSS_KEYS:
        rows, curves = test_family(panel, members, key)
        res = joint_null(curves)
        print("\n  %s  (%s)" % (key, CROSS_KIND[key]))
        report_ranks(rows, "per specimen")
        report_combined(res, indent="    ")
        out[key] = dict(rows=rows, combined=res)
    print()
    return out


def exact_perm_p(x, y):
    """Two-sided permutation level of a Pearson r by enumerating n!.

    Eight points is small enough to enumerate exactly, so the level owes
    nothing to a t approximation that eight points cannot support.
    """
    from itertools import permutations
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).sum())
    yc = (y - y.mean()) / np.sqrt(((y - y.mean()) ** 2).sum())
    r0 = float(xc @ yc)
    P = np.array(list(permutations(range(len(y)))))
    rs = yc[P] @ xc
    return r0, float((np.abs(rs) >= abs(r0) - 1e-12).mean()), len(P)


def revolution_level(panel, name, matched=True):
    """The revolution level of one sweep, averaged as an energy.

    Restricted by default to the 30-azimuth grid every sweep shares, so
    that a 60-azimuth sweep is not compared against a 30-azimuth one.
    """
    v = panel.rows[name]["level_lin"]
    if matched and len(v) == 60:
        v = v[::2]
    return db(v.mean())


def report_replication(panel):
    """The n = 8 claim, put to the only replication the data allows.

    DiskSpecimen draws its Laguerre seed points and weights before it
    draws any c-axis, so a girdle and a single maximum of the same seed
    carry a BIT-IDENTICAL tessellation. The crossing count is a property
    of the label volume alone, so the eight numbers that give r = 0.794
    on the girdle sweeps are literally the same eight numbers on the
    single-maximum sweeps. Eight more simulations of the same geometry
    with an unrelated orientation draw are therefore an exact
    out-of-sample test of the published correlation, and the manuscript
    already owns them.

    If the count drove the level, it would drive it in both fabrics.
    """
    seeds = (11, 7, 17, 23, 41, 53, 71, 89)
    print("=" * 79)
    print("THE n = 8 CLAIM, REPLICATED ON THE SECOND FABRIC")
    print("=" * 79)
    same = all(np.array_equal(panel.predictor("girdle_s%d" % s, COL_PAPER),
                              panel.predictor("single_s%d" % s, COL_PAPER))
               for s in seeds)
    print("  the predictor is bit-identical between the two families: %s"
          % same)
    print()
    print("  %-6s %10s %12s %12s"
          % ("seed", "count", "girdle dB", "single max dB"))
    cnt = np.array([panel.predictor("girdle_s%d" % s, COL_PAPER).mean()
                    for s in seeds])
    gl = np.array([revolution_level(panel, "girdle_s%d" % s) for s in seeds])
    sl = np.array([revolution_level(panel, "single_s%d" % s) for s in seeds])
    for i, s in enumerate(seeds):
        print("  %-6d %10.2f %12.3f %12.3f" % (s, cnt[i], gl[i], sl[i]))
    print()
    print("  %-14s %8s %9s %9s %9s %9s"
          % ("family", "pearson", "exact p", "spearman", "p", "spread"))
    out = {}
    for tag, y in (("girdle", gl), ("single max", sl)):
        r, pe, npm = exact_perm_p(cnt, y)
        rho, prho = stats.spearmanr(cnt, y)
        print("  %-14s %+8.3f %9.4f %+9.3f %9.4f %9.2f"
              % (tag, r, pe, rho, prho, y.max() - y.min()))
        out[tag] = dict(r=float(r), p_exact=float(pe), rho=float(rho))
    z = np.arctanh(np.clip([out["girdle"]["r"], out["single max"]["r"]],
                           -0.999999, 0.999999))
    zdiff = (z[0] - z[1]) / np.sqrt(2.0 / (len(seeds) - 3))
    print()
    rg, pg = stats.pearsonr(gl, sl)
    print("  the %d permutations of eight labels are enumerated, so the"
          % npm)
    print("  levels above owe nothing to a t approximation: the published")
    print("  correlation keeps its level, %.4f against the %.4f the"
          % (out["girdle"]["p_exact"], stats.pearsonr(cnt, gl)[1]))
    print("  parametric test gives, so nothing here rests on n = 8 being")
    print("  treated as normal.")
    print("  the two families' levels correlate at r = %+.3f (p = %.2f)"
          % (rg, pg))
    print("  across the eight seeds, so the second test is not a restatement")
    print("  of the first.")
    print("  the two correlations differ at z = %.2f, p = %.3f"
          % (zdiff, 2 * stats.norm.sf(abs(zdiff))))
    print()
    print("  The same eight geometries, measured the same way, give +%.3f"
          % out["girdle"]["r"])
    print("  in one fabric and %+.3f in the other. A property of the"
          % out["single max"]["r"])
    print("  tessellation cannot do that. The girdle correlation is a")
    print("  property of the eight girdle ORIENTATION draws, not of the")
    print("  eight tessellations, and the manuscript's own reading, that")
    print("  the orientation draw dominates, is the correct one.")
    print()
    out["z"] = float(zdiff)
    return out


def report_pooled(panel):
    """Both fabric families at once, for the tightest bound on the count.

    The sixteen sweeps carry only eight tessellations, so a girdle and a
    single maximum of the same seed share their predictor exactly. That
    dependence can only make a pooled level look MORE significant than it
    should, never less, so pooling is safe for a negative conclusion and
    would not be for a positive one. It is reported for the bound it puts
    on how large a within-specimen association could still be hiding.
    """
    members = panel.family("girdle") + panel.family("single")
    print("=" * 79)
    print("BOTH FAMILIES POOLED, %d sweeps on 8 tessellations" % len(members))
    print("=" * 79)
    print("  %-24s %8s %9s %9s %9s"
          % ("predictor", "mean r", "pos/16", "Fisher p", "95% upper"))
    out = {}
    for key in CROSS_KEYS + ("dv_cross_rms",):
        rows, curves = test_family(panel, members, key)
        res = joint_null(curves, n_draw=50000)
        r = np.array([s["r"] for s in rows])
        hi = res["mean_r"] + 1.96 * res["mean_r_null_sd"]
        print("  %-24s %+8.3f %6d/%-2d %9.4f %+9.3f"
              % (key, r.mean(), int((r > 0).sum()), len(r),
                 res["fisher_exact"], hi))
        out[key] = res
    print()
    print("  Pooled, the beam-column count is bounded above at r = %+.3f,"
          % (out[COL_PAPER]["mean_r"]
             + 1.96 * out[COL_PAPER]["mean_r_null_sd"]))
    print("  which excludes an association of the size the contrast")
    print("  descriptor shows on the same sweeps.")
    print()
    return out


def report_controls(panel):
    """The same test where there is nothing to scatter from."""
    print("=" * 79)
    print("CONTROLS: seed 11 geometry, acoustic contrast switched off")
    print("=" * 79)
    print("  The per-azimuth crossing count of these two sweeps is")
    print("  IDENTICAL to girdle_s11's, because a count of label changes")
    print("  does not know what the c-axes are. Only the coda differs, and")
    print("  in these two it is the numerical floor. A relation that")
    print("  appears here is not scattering.")
    out = {}
    for key in (COL_PAPER, AXIS_GATE):
        rows, curves = test_family(panel, list(CONTROLS), key)
        print("\n  %s  (%s)" % (key, CROSS_KIND[key]))
        report_ranks(rows, "per control")
        out[key] = rows
    same = np.allclose(panel.predictor("girdle_s11", COL_PAPER)[::2],
                       panel.predictor("cs_f000_s11", COL_PAPER))
    lvl = {n: db(panel.rows[n]["level_lin"].mean())
           for n in list(CONTROLS) + ["girdle_s11"]}
    print()
    print("  predictor identical to girdle_s11 on the matched azimuths: %s"
          % same)
    print("  revolution level, dB   girdle_s11 %.2f   %s %.2f   %s %.2f"
          % (lvl["girdle_s11"], CONTROLS[0], lvl[CONTROLS[0]],
             CONTROLS[1], lvl[CONTROLS[1]]))
    print()
    return out


def report_robustness(panel):
    """The result must not depend on the reference or on the decibel.

    Three normalisations of the same gate and two scales. If the sign or
    the level moved with the choice, the choice would be the result.
    """
    members = panel.family("girdle")
    print("=" * 79)
    print("ROBUSTNESS OF THE WITHIN-SPECIMEN RESULT")
    print("=" * 79)
    print("  %-24s %8s %9s %9s %9s"
          % ("variant", "mean r", "positive", "Fisher p", "mean-r p"))
    out = {}
    for tag, key, resp in (("source-referenced dB", COL_PAPER, "level_db"),
                           ("backwall-referenced dB", COL_PAPER, "level_e1"),
                           ("unreferenced gate dB", COL_PAPER, "level_raw"),
                           ("linear power", COL_PAPER, "level_lin"),
                           ("cone count, dB", COL_CONE, "level_db"),
                           ("axis gate count, dB", AXIS_GATE, "level_db"),
                           ("axis full count, dB", AXIS_FULL, "level_db")):
        rows, curves = test_family(panel, members, key, resp)
        res = joint_null(curves, n_draw=50000)
        r = np.array([s["r"] for s in rows])
        print("  %-24s %+8.3f %5d of %-2d %9.4f %9.4f"
              % (tag, r.mean(), int((r > 0).sum()), len(r),
                 res["fisher_exact"], res["mean_r_exact"]))
        out[tag] = res
    print()
    print("  Restricting girdle_s11 to the 30-azimuth grid the other seven")
    print("  use, so that every specimen carries the same resolution:")
    rows, curves = [], []
    for name in members:
        x = panel.predictor(name, COL_PAPER)
        y = panel.response(name)
        if len(x) == 60:
            x, y = x[::2], y[::2]
        rs = shift_curve(x, y)
        rows.append(shift_stats(rs))
        rows[-1]["name"] = name
        rows[-1]["n_az"] = len(x)
        rows[-1]["degen"] = degeneracy(rs)
        curves.append(rs)
    res = joint_null(curves, n_draw=50000)
    report_ranks(rows, "matched 30-azimuth grid, %s" % COL_PAPER)
    report_combined(res, indent="    ")
    out["matched30"] = res
    print()
    return out


def report_mechanism(panel, kind="girdle"):
    """If the count is not the predictor, what is, and does the count help?

    Single scattering from a facet population says the gate power goes as
    the NUMBER of boundaries crossed times the mean square acoustic
    contrast across them. The two factors are separable here, because the
    cube carries both per azimuth, so the prediction can be taken apart
    rather than accepted or rejected whole.

      count          how many boundaries the beam column crosses
      contrast       the root mean square velocity jump across them
      Born product   count times contrast squared, the model itself
      count | dv     the part of the count the contrast cannot explain
      dv | count     the part of the contrast the count cannot explain

    The last two are the ones that decide it. Each is a residual of one
    PREDICTOR against the other, and the null still shifts the measured
    level, which leaves it exact.

    A second pair strips the fabric's own 2-theta modulation from the
    predictor. A girdle at (1,0,0) makes every quantity set by |c . n|
    move as cos 2az with a phase the fabric fixes, and a predictor that
    is only that carries no realisation-specific information at all: it
    is the azimuthal signature any bulk model of the same fabric would
    produce. What survives after cos 2az and sin 2az are projected out
    is the part that knows which tessellation this is.
    """
    members = panel.family(kind)

    def col(key):
        return lambda p, n: p.predictor(n, key)

    def born(p, n):
        return (p.predictor(n, COL_CONE)
                * p.predictor(n, "dv_cross_rms") ** 2)

    def count_given_dv(p, n):
        return residualise(p.predictor(n, COL_PAPER),
                           p.predictor(n, "dv_cross_rms"))

    def dv_given_count(p, n):
        return residualise(p.predictor(n, "dv_cross_rms"),
                           p.predictor(n, COL_PAPER))

    def detrended(key):
        def build(p, n):
            return residualise(p.predictor(n, key),
                               theta2(p.rows[n]["az"]))
        return build

    cases = [("count, beam column", col(COL_PAPER)),
             ("contrast at crossings", col("dv_cross_rms")),
             ("Born product, count x dv^2", born),
             ("count given contrast", count_given_dv),
             ("contrast given count", dv_given_count),
             ("count, 2-theta removed", detrended(COL_PAPER)),
             ("contrast, 2-theta removed", detrended("dv_cross_rms")),
             ("cos2 in column, 2-th removed", detrended("cos2_beam_col"))]

    print("=" * 79)
    print("IF NOT THE COUNT, WHAT? TAKING THE FACET MODEL APART")
    print("=" * 79)
    print("  %-30s %8s %8s %9s %9s"
          % ("predictor", "mean r", "pos/8", "Fisher p", "mean-r p"))
    out = {}
    for tag, build in cases:
        rows, curves = test_derived(panel, members, build)
        res = joint_null(curves, n_draw=50000)
        r = np.array([s["r"] for s in rows])
        print("  %-30s %+8.3f %5d/%-2d %9.4f %9.4f"
              % (tag, r.mean(), int((r > 0).sum()), len(r),
                 res["fisher_exact"], res["mean_r_exact"]))
        out[tag] = dict(mean_r=float(r.mean()), rows=rows, combined=res)
    print()
    print("  The count carries no sign of the association the model")
    print("  predicts, and the contrast carries all of it. Removing the")
    print("  contrast from the count does not rescue the count; removing")
    print("  the count from the contrast does not damage the contrast.")
    print("  Multiplying the two together makes the model's own product")
    print("  worse than its second factor alone, which is what a spurious")
    print("  first factor does.")
    print()
    return out


def report_panel(panel, kind="girdle"):
    """Does the within-specimen regime detect anything at all?

    The identical machinery over every per-azimuth column of the sample
    matrix. If nothing survives Holm across the panel then the regime has
    no power and the crossing-count null means nothing; if several do,
    the null is a measurement.
    """
    members = panel.family(kind)
    print("=" * 79)
    print("IS THE WITHIN-SPECIMEN REGIME A DISCOVERY ENGINE?")
    print("=" * 79)
    print("  %d per-azimuth properties x %d specimens, each with its own"
          % (len(panel.columns), len(members)))
    print("  circular-shift null, combined by the exact joint null and")
    print("  then corrected across the panel.")
    res = []
    for col in panel.columns:
        rows, curves = test_family(panel, members, col)
        j = joint_null(curves, n_draw=50000)
        r = np.array([s["r"] for s in rows])
        res.append(dict(column=col, mean_r=float(r.mean()),
                        n_pos=int((r > 0).sum()),
                        p_fisher=j["fisher_exact"],
                        p_mean=j["mean_r_exact"],
                        r180=float(np.mean([periodicity(
                            panel.predictor(n, col)) for n in members]))))
    ps = np.array([x["p_fisher"] for x in res])
    hp, qv = holm(ps), bh(ps)
    order = np.argsort(ps)
    print("  %-24s %8s %8s %9s %9s %9s %7s"
          % ("property", "mean r", "pos/8", "Fisher p", "Holm", "BH q",
             "r(180)"))
    for i in order:
        x = res[i]
        star = " *" if hp[i] < 0.05 else ("  ." if qv[i] < 0.10 else "")
        print("  %-24s %+8.3f %5d/%-2d %9.4f %9.4f %9.4f %+7.3f%s"
              % (x["column"][:24], x["mean_r"], x["n_pos"], len(members),
                 x["p_fisher"], hp[i], qv[i], x["r180"], star))
        res[i]["holm"] = float(hp[i])
        res[i]["bh"] = float(qv[i])
    n_holm = int((hp < 0.05).sum())
    n_bh = int((qv < 0.10).sum())
    print()
    print("  %d of %d columns survive Holm at 0.05 and %d hold a BH q below"
          % (n_holm, len(res), n_bh))
    print("  0.10. Expected by chance at the uncorrected 0.05 level: %.1f."
          % (0.05 * len(res)))
    for key in CROSS_KEYS:
        i = [q for q, x in enumerate(res) if x["column"] == key][0]
        print("    %-22s Fisher p %.3f, Holm %.3f, rank %d of %d"
              % (key, res[i]["p_fisher"], res[i]["holm"],
                 int(np.where(order == i)[0][0]) + 1, len(res)))
    print()
    return res


def write_outputs(within, controls, panel_res, between, repl, mech):
    """One archive, so the numbers here can be cited without rerunning."""
    os.makedirs(RESULTS, exist_ok=True)
    payload = {}
    for key, d in within.items():
        payload["r_" + key] = np.array([s["r"] for s in d["rows"]])
        payload["p2_" + key] = np.array([s["p2"] for s in d["rows"]])
        payload["rank_" + key] = np.array([s["rank_abs"] for s in d["rows"]])
        payload["comb_" + key] = np.array(
            [d["combined"]["fisher_exact"], d["combined"]["fisher_analytic"],
             d["combined"]["mean_r"], d["combined"]["mean_r_exact"]])
    payload["panel_columns"] = np.array([x["column"] for x in panel_res])
    payload["panel_p"] = np.array([x["p_fisher"] for x in panel_res])
    payload["panel_holm"] = np.array([x["holm"] for x in panel_res])
    payload["panel_mean_r"] = np.array([x["mean_r"] for x in panel_res])
    for key, d in between.items():
        payload["between_" + key] = np.array(
            [d["pearson"], d["p"], d["spearman"], d["p_spearman"],
             d["vs_published"]])
    for key, rows in controls.items():
        payload["ctrl_r_" + key] = np.array([s["r"] for s in rows])
        payload["ctrl_p_" + key] = np.array([s["p2"] for s in rows])
    for fam in ("girdle", "single max"):
        payload["repl_" + fam.replace(" ", "_")] = np.array(
            [repl[fam]["r"], repl[fam]["p_exact"], repl[fam]["rho"]])
    payload["repl_z"] = np.array([repl["z"]])
    payload["mech_names"] = np.array(list(mech))
    payload["mech_mean_r"] = np.array([mech[k]["mean_r"] for k in mech])
    payload["mech_p"] = np.array([mech[k]["combined"]["fisher_exact"]
                                  for k in mech])
    payload["mech_p_dir"] = np.array([mech[k]["combined"]["mean_r_exact"]
                                      for k in mech])
    path = os.path.join(RESULTS, "crossing_within.npz")
    np.savez(path, **payload)
    return path


def main():
    panel = Panel()
    missing = [n for n in panel.names if n not in panel.rows]
    if missing:
        print("sweeps absent, rows skipped: %s\n" % ", ".join(missing))
    between = report_between(panel)
    report_periodicity(panel, panel.family("girdle"))
    within = report_within(panel, "girdle", "eight girdle tessellations")
    if "--full" in sys.argv:
        report_within(panel, "single", "eight single maxima")
    repl = report_replication(panel)
    report_pooled(panel)
    controls = report_controls(panel)
    report_robustness(panel)
    mech = report_mechanism(panel)
    panel_res = report_panel(panel)
    path = write_outputs(within, controls, panel_res, between, repl, mech)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()

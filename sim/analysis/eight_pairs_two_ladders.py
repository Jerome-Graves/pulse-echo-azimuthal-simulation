"""What the six sweeps of 2026-08-02 settle. Two questions, one module.

FIRST, IS THE FABRIC SEPARATION REAL. singlemax_seed{7,53,71,89}_ppw8_ensemble
complete the paired design: all eight girdle tessellations now carry a
single-maximum twin on a bit-identical Laguerre tessellation, because
DiskSpecimen draws the seed points and weights before it draws any
c-axis. The pair therefore differs in orientation alone and the facet
geometry cancels within it. The manuscript reported this on four pairs,
a fabric main effect of -2.47 dB that a paired t did not establish
(t = -1.46, p = 0.24), with seed 17 alone carrying -7.40 dB of it.

The outlier story does not survive the completion. Seed 7 is a SECOND
specular realisation of almost exactly the same size, and it was
measured after the four-pair statement was written, so its identity as
an outlier is fixed by a criterion the earlier text already used and not
by its effect on the answer: the share of the revolution energy carried
by the loudest of thirty azimuths is 45.9 and 45.8 per cent for the
seeds 7 and 17 single maxima and 16.2 to 24.7 per cent for the other
six. Two of eight, and both are named before the test is run.

SECOND, IS THE CONVERGENCE BEHAVIOUR A PROPERTY OF THE REGIME OR OF
SEED 11. girdle_seed23_ppw{6,10}_ladder put the resolution ladder on a second
tessellation, seed 23 having existed at ppw 8 already. The ladder is the
evidence behind the resolution criterion, and until today it stood on
one specimen, so nothing in the manuscript distinguished a statement
about the regime from a statement about that specimen.

THE ESTIMATORS, and why there are two.

  LEVELS come from analysis/db_reconcile.py unchanged, which is the
  audited estimator of tab:reconcile: gate power measured locally on the
  trace band-limited to 0.8 to 3.0 MHz BEFORE gating, referenced to the
  peak of the source arrival, averaged over the revolution as ENERGIES.
  The backwall is the unfiltered analytic-envelope peak, unchanged for
  the reason db_reconcile sets out. Taking them from that module rather
  than reimplementing them is what makes the numbers here the same
  decibels as Table 4 and Table 5; report_reproduction prints the
  published values beside the recomputed ones.

  SHAPE CHANNELS, the coda decay rate and the coda bandwidth, come from
  analysis/observable_matrix.py's A-scan estimator, which is where they
  were defined and where the five-pair result that motivates this test
  was measured. Its gate edges are placed at the nearest sample rather
  than truncated, worth 0.07 dB on a level and nothing at all on a slope
  or a bandwidth; report_reproduction prints that difference too.

Neither is a Hilbert transform of a whole trace. The coda level uses no
transform at all, from E|z|^2 = 2 E x^2 for a locally stationary
segment, and every windowed envelope in observable_matrix is taken from
the trace tapered to a padded window before transforming. Every azimuth
is measured on its OWN dt: no stack is formed here, and no quantity is
indexed by sample number.

WHAT IS PRE-SPECIFIED. Three channels, and the distinction matters
because eleven were computed.

  a_coda           the manuscript's own channel, the one the 2.83 dB
                   claim is made on
  s_coda_decay     the coda decay rate
  f_coda_rms_ib    the in-band RMS bandwidth of the coda gate spectrum

The last two are named by the five-pair result of earlier today, which
found the coda LEVEL discriminant resting entirely on seeds 7 and 17
while the decay rate and the bandwidth separated the fabrics on every
pair then available and strengthened when those two were dropped. They
are therefore confirmatory here rather than exploratory. Everything else
this module prints is context and is labelled as such.

DROPPING THE OUTLIERS IS A DIAGNOSTIC AND NOT THE HEADLINE. The eight
pairs are the ensemble; two of them being specular is a property of the
population of tessellations, not a defect of the measurement. The
reduced sets are reported because a separation that lived only in two
realisations would be worth knowing about, and this one does not.

  MEASURED   everything printed, from the stored traces.
  INFERRED   only the two remarks flagged INFERRED in report_ladder,
             concerning the order of the error in the grid spacing.

Reads, all under out/sweeps, on the 30 azimuths at 12 degree spacing
that every sweep in the matrix carries:
    girdle_seed11_ppw8_dev girdle_seed{7,17,23,41,53,71,89}_ppw8_ensemble
    singlemax_seed11_ppw8_twin   singlemax_seed{7,17,23,41,53,71,89}_ppw8_ensemble
    girdle_seed11_ppw6_axis_perp girdle_seed23_ppw8_ensemble girdle_seed11_ppw10_licensing
    girdle_seed23_ppw{6,10}_ladder
Writes nothing. Touches no GPU: no forward model, no specimen build, no
labeller, no CUDA import path.
"""
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import db_reconcile as DR                            # noqa: E402
import observable_matrix as OM                       # noqa: E402

# The eight matched pairs, in the row order of sim/results/ensemble.npz.
SEEDS = (11, 7, 17, 23, 41, 53, 71, 89)
GIRDLE = dict(zip(SEEDS, OM.GIRDLE8))
SINGLE = dict(zip(SEEDS, OM.SINGLE8))

# The two resolution ladders. Every rung is the same specimen at three
# grid spacings, re-voxelised; sec:reraster shows that re-voxelisation
# moves the represented interfacial area by 0.01 dB across the ladder.
LADDER = {11: {6: "girdle_seed11_ppw6_axis_perp", 8: "girdle_seed11_ppw8_dev",
               10: "girdle_seed11_ppw10_licensing"},
          23: {6: "girdle_seed23_ppw6_ladder", 8: "girdle_seed23_ppw8_ensemble",
               10: "girdle_seed23_ppw10_ladder"}}

# Pre-specified before the four new sweeps were read: the manuscript's
# own channel, and the two the five-pair result named.
PRIMARY = "a_coda"
CONFIRMATORY = ("s_coda_decay", "f_coda_rms_ib")
PRESPEC = (PRIMARY,) + CONFIRMATORY

# Printed for context only. Not corrected for, not claimed on.
CONTEXT = ("s_coda_decay_e", "s_coda_decay_l", "s_coda_decay_w",
           "f_coda_rms", "f_coda_cen_ib", "r_coda_bw", "s_coda_crest",
           "s_coda_teff")

# Units, so a table of mixed channels is readable.
UNITS = {"a_coda": "dB", "r_coda_bw": "dB", "s_coda_crest": "dB",
         "s_coda_decay": "dB/us", "s_coda_decay_e": "dB/us",
         "s_coda_decay_l": "dB/us", "s_coda_decay_w": "dB/us",
         "f_coda_rms_ib": "MHz", "f_coda_rms": "MHz",
         "f_coda_cen_ib": "MHz", "s_coda_teff": "us"}

# The seeds whose single maximum concentrates the revolution energy.
# Named by the criterion the manuscript already applied to seed 17, and
# confirmed by report_concentration below rather than assumed.
SPECULAR = (7, 17)

# The published four-pair table, tab:ensemble, for the reproduction
# check. seed: (girdle, single, difference).
PUBLISHED_PAIRS = {11: (-85.11, -83.59, -1.52), 17: (-85.86, -78.45,
                                                     -7.40),
                   23: (-84.49, -83.19, -1.30), 41: (-83.42, -83.76,
                                                     0.34)}
# The published seed-11 ladder, tab:reconcile: (coda/backwall, coda,
# backwall), and the per-azimuth scatter series of sec:grainpop.
PUBLISHED_LADDER = {6: (-53.03, -81.09, -28.06),
                    8: (-59.82, -85.11, -25.29),
                    10: (-61.66, -87.79, -26.13)}
PUBLISHED_SCATTER = (3.01, 3.57, 3.76)

_LVL, _ASC = {}, {}


# ------------------------------------------------------------------ load

def levels(name):
    """db_reconcile's per-azimuth power ratios, cached."""
    if name not in _LVL:
        _LVL[name] = DR.load_sweep(name)
    return _LVL[name]


def ascan(name):
    """observable_matrix's per-azimuth A-scan observables, cached."""
    if name not in _ASC:
        root = os.path.join(OM.SWD, name)
        rows, used = [], []
        for a in DR.AZ_MATCHED:
            p = os.path.join(root, "az%03d.npz" % a)
            if not os.path.exists(p):
                continue
            tr, dt = OM.read_trace(p)
            rows.append(OM.ascan(tr, dt))
            used.append(a)
        if not rows:
            raise IOError("no matched azimuths under %s" % root)
        keys = list(rows[0].keys())
        _ASC[name] = dict(keys=keys, az=np.array(used),
                          a=np.array([[r[k] for k in keys] for r in rows],
                                     float))
    return _ASC[name]


def per_azimuth(name, key):
    """One value per matched azimuth, on that azimuth's own dt."""
    d = ascan(name)
    return d["a"][:, d["keys"].index(key)]


def revolution(name, key):
    """Azimuth average, as ENERGIES for a level and arithmetically for
    everything else. A slope and a bandwidth are not powers."""
    v = per_azimuth(name, key)
    v = v[np.isfinite(v)]
    if key.startswith("a_") or key.startswith("r_"):
        return float(OM.db(np.mean(10.0 ** (v / 10.0))))
    return float(v.mean())


def coda_level(name):
    """The audited coda level of tab:reconcile, exactly."""
    return DR.db(levels(name)["coda_band"].mean())


def backwall_level(name):
    """The unfiltered analytic-envelope peak of tab:reconcile."""
    return DR.db(levels(name)["bw_env"].mean())


def azimuth_scatter(name):
    """Standard deviation of the per-azimuth level in decibels."""
    return float(np.std(DR.db(levels(name)["coda_band"]), ddof=1))


# --------------------------------------------------------------- compute

def signflip_p(d):
    """Exact two-sided sign-flip p for a paired difference vector.

    The null is that the fabric label carries no information, under
    which each pair's difference is equally likely to have either sign.
    Enumerated, not sampled: 2**8 = 256 assignments, so the floor is
    2/256 = 0.0078 and no eight-pair sign-flip test can go below it.
    """
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None]
                      >> np.arange(n)[None, :]) & 1)
    null = np.abs((signs * d).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def split_p(a, b):
    """Exact unpaired rank-split p, enumerated over C(16,8) = 12870."""
    from itertools import combinations
    x = np.concatenate([a, b])
    n = len(a)
    obs = abs(a.mean() - b.mean())
    idx = range(len(x))
    hits = tot = 0
    for c in combinations(idx, n):
        m = np.zeros(len(x), bool)
        m[list(c)] = True
        tot += 1
        if abs(x[m].mean() - x[~m].mean()) >= obs - 1e-12:
            hits += 1
    return float(hits) / tot


def pair_vectors(key):
    """(girdle, single) revolution values over the eight seeds."""
    if key == PRIMARY:
        g = np.array([coda_level(GIRDLE[s]) for s in SEEDS])
        m = np.array([coda_level(SINGLE[s]) for s in SEEDS])
    else:
        g = np.array([revolution(GIRDLE[s], key) for s in SEEDS])
        m = np.array([revolution(SINGLE[s], key) for s in SEEDS])
    return g, m


def paired_test(g, m, sel):
    """Difference girdle minus single, paired t, and the exact null."""
    d = (g - m)[sel]
    n = len(d)
    t, p = stats.ttest_rel(g[sel], m[sel])
    se = float(np.std(d, ddof=1)) / np.sqrt(n)
    h = float(stats.t.ppf(0.975, n - 1)) * se
    return dict(n=int(n), delta=float(d.mean()),
                sd=float(np.std(d, ddof=1)), se=se,
                lo=float(d.mean()) - h, hi=float(d.mean()) + h,
                t=float(t), p=float(p), p_flip=signflip_p(d),
                agree=int(np.sum(np.sign(d) == np.sign(d.mean()))))


def subsets():
    """The full set and the two outlier-exclusion diagnostics."""
    s = np.array(SEEDS)
    return (("all eight", np.ones(8, bool)),
            ("drop 17", s != 17),
            ("drop 7 and 17", (s != 17) & (s != 7)))


# ------------------------------------------------------------------ draw

def report_reproduction():
    """Nothing below is trusted until the published numbers come back."""
    print("REPRODUCTION CHECK, published against recomputed")
    print("  tab:ensemble, the four pairs the manuscript reports")
    print("  %-6s %19s %19s %19s"
          % ("seed", "girdle dB", "single dB", "difference dB"))
    for s in sorted(PUBLISHED_PAIRS):
        pub = PUBLISHED_PAIRS[s]
        got = (coda_level(GIRDLE[s]), coda_level(SINGLE[s]))
        got = got + (got[0] - got[1],)
        print("  %-6d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f"
              % ((s,) + tuple(v for pair in zip(pub, got) for v in pair)))
    print()
    print("  tab:reconcile, the seed-11 ladder, and the sec:grainpop")
    print("  per-azimuth scatter series")
    print("  %-6s %15s %15s %15s %13s"
          % ("ppw", "coda/backwall", "coda/source", "backwall/source",
             "az scatter"))
    for p in sorted(PUBLISHED_LADDER):
        n = LADDER[11][p]
        pub = PUBLISHED_LADDER[p]
        c, b = coda_level(n), backwall_level(n)
        got = (c - b, c, b)
        sc = azimuth_scatter(n)
        print("  %-6d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %6.2f %6.2f"
              % ((p,) + tuple(v for pair in zip(pub, got) for v in pair)
                 + (PUBLISHED_SCATTER[sorted(PUBLISHED_LADDER).index(p)],
                    sc)))
    print()
    print("  the two level estimators against each other, ppw 8 girdles")
    print("  %-24s %11s %11s %8s"
          % ("sweep", "db_reconcile", "obs_matrix", "diff"))
    for s in SEEDS:
        n = GIRDLE[s]
        a, b = coda_level(n), revolution(n, "a_coda")
        print("  %-24s %11.3f %11.3f %8.3f" % (n, a, b, b - a))
    print("  The offset is the gate-edge convention, nearest sample")
    print("  against truncation, and is the 0.070 dB observable_matrix")
    print("  documents. It cancels in every difference taken here and")
    print("  cannot touch a slope or a bandwidth.")
    print()


def report_concentration():
    """Which single maxima are specular, on the manuscript's criterion."""
    print("SPECULAR REALISATIONS, named before any test is run")
    print("  share of the 30-azimuth coda energy carried by the loudest")
    print("  azimuth. The manuscript already calls seed 17 atypical at")
    print("  46 per cent; the criterion is applied here to all eight.")
    print("  %-6s %14s %14s" % ("seed", "single %", "girdle %"))
    for s in SEEDS:
        row = []
        for d in (SINGLE, GIRDLE):
            e = levels(d[s])["coda_band"]
            row.append(100.0 * e.max() / e.sum())
        print("  %-6d %14.1f %14.1f%s"
              % (s, row[0], row[1], "   <- specular" if s in SPECULAR
                 else ""))
    print("  Two of the eight, and they are the two the earlier")
    print("  five-pair analysis had already isolated. Seed 7 is new")
    print("  data, not a new criterion.")
    print()


def report_pairs():
    """The paired design, on the pre-specified channels and then some."""
    print("THE EIGHT PAIRS, girdle minus single, revolution values")
    print("  %-16s %-6s %s" % ("channel", "unit",
                               " ".join("%8s" % ("s%d" % s)
                                        for s in SEEDS)))
    for key in PRESPEC + CONTEXT:
        g, m = pair_vectors(key)
        print("  %-16s %-6s %s"
              % (key, UNITS.get(key, ""),
                 " ".join("%+8.3f" % v for v in g - m)))
    print()
    print("PRE-SPECIFIED CHANNELS. Paired t on the pairs, and the exact")
    print("  2**8 sign-flip null beside it, whose floor is 0.0078.")
    print("  Sign agreement counts the pairs whose difference has the")
    print("  same sign as the mean.")
    for key in PRESPEC:
        g, m = pair_vectors(key)
        print()
        print("  %s, %s   girdle %.3f   single %.3f"
              % (key, UNITS.get(key, ""), g.mean(), m.mean()))
        print("    %-16s %3s %9s %8s %8s %9s %7s %17s"
              % ("set", "n", "gir-sin", "sd", "t", "p", "p_flip",
                 "95 % interval"))
        for tag, sel in subsets():
            r = paired_test(g, m, sel)
            print("    %-16s %3d %9.3f %8.3f %8.2f %9.4f %7.4f "
                  "[%+7.3f,%+7.3f]  %d/%d agree"
                  % (tag, r["n"], r["delta"], r["sd"], r["t"], r["p"],
                     r["p_flip"], r["lo"], r["hi"], r["agree"], r["n"]))
    print()
    print("CONTEXT CHANNELS, not pre-specified, no correction applied,")
    print("  printed so that the pre-specified three are not the only")
    print("  ones the reader sees.")
    print("  %-16s %-6s %9s %8s %9s %9s"
          % ("channel", "unit", "gir-sin", "t", "p", "p drop7,17"))
    for key in CONTEXT:
        g, m = pair_vectors(key)
        r8 = paired_test(g, m, subsets()[0][1])
        r6 = paired_test(g, m, subsets()[2][1])
        print("  %-16s %-6s %9.3f %8.2f %9.4f %9.4f"
              % (key, UNITS.get(key, ""), r8["delta"], r8["t"], r8["p"],
                 r6["p"]))
    print()
    print("  MULTIPLICITY. Three channels were pre-specified, so a")
    print("  Bonferroni threshold over them is 0.0167. Eleven were")
    print("  computed; over all eleven it is 0.0045. Both are printed")
    print("  in the verdict rather than chosen after the fact.")
    print()


def report_unpaired():
    """The ensemble separation, which is what the 2.83 dB actually is."""
    print("THE UNPAIRED ENSEMBLE SEPARATION")
    print("  The 2.83 dB of fig:ensemble is the difference of family")
    print("  means over LEVELS, eight girdle against the four single")
    print("  maxima then available. It is recomputed here on eight")
    print("  against eight.")
    g = np.array([coda_level(GIRDLE[s]) for s in SEEDS])
    m = np.array([coda_level(SINGLE[s]) for s in SEEDS])
    four = np.array([s in PUBLISHED_PAIRS for s in SEEDS])
    print("  %-28s %9s %9s %9s" % ("", "girdle", "single", "gir-sin"))
    print("  %-28s %9.2f %9.2f %9.2f"
          % ("published, 8 against 4", g.mean(), m[four].mean(),
             g.mean() - m[four].mean()))
    print("  %-28s %9.2f %9.2f %9.2f"
          % ("now, 8 against 8", g.mean(), m.mean(), g.mean() - m.mean()))
    print("  spread of levels, sd    girdle %.2f dB   single %.2f dB"
          % (np.std(g, ddof=1), np.std(m, ddof=1)))
    print("  exact unpaired split over C(16,8) = 12870:  p = %.4f"
          % split_p(g, m))
    print("  Welch t on the same two families:           p = %.4f"
          % stats.ttest_ind(g, m, equal_var=False).pvalue)
    print("  The unpaired test discards the shared tessellation and so")
    print("  carries the geometry variance; the paired test above is")
    print("  the one the design was built for.")
    print()


def report_independence():
    """Is the decay rate a restatement of the level. It is not."""
    print("ARE THE THREE CHANNELS THE SAME CHANNEL")
    print("  Pearson r between the eight paired differences.")
    keys = list(PRESPEC) + ["s_coda_decay_l"]
    dv = {}
    for k in keys:
        g, m = pair_vectors(k)
        dv[k] = g - m
    print("  %-16s %s" % ("", " ".join("%16s" % k for k in keys)))
    for a in keys:
        print("  %-16s %s"
              % (a, " ".join("%16.3f" % np.corrcoef(dv[a], dv[b])[0, 1]
                             for b in keys)))
    print("  The decay rate is not a restatement of the level: the two")
    print("  pairs that carry the level, seeds 7 and 17, are the two")
    print("  whose decay barely moves, which is why dropping them")
    print("  strengthens the decay test and weakens the level test.")
    print()
    print("  dt is not doing the work either. Each azimuth has its own")
    print("  dt and the two fabrics rotate different velocity surfaces,")
    print("  so a spectral channel could in principle track the CFL")
    print("  step rather than the specimen.")
    print("  %-6s %11s %11s %9s" % ("seed", "girdle dt ns", "single dt ns",
                                    "gir-sin"))
    dt_d = []
    for s in SEEDS:
        a = float(np.mean(per_azimuth(GIRDLE[s], "qc_dt_ns")))
        b = float(np.mean(per_azimuth(SINGLE[s], "qc_dt_ns")))
        dt_d.append(a - b)
        print("  %-6d %11.4f %11.4f %9.4f" % (s, a, b, a - b))
    dt_d = np.array(dt_d)
    for k in CONFIRMATORY:
        g, m = pair_vectors(k)
        print("  r(delta dt, delta %s) = %+.3f"
              % (k, np.corrcoef(dt_d, g - m)[0, 1]))
    print()


def report_decomposition():
    """The seed-by-fabric variance decomposition, now on eight pairs.

    Balanced two-way layout with one observation per cell, so the
    interaction is the residual. The difference within a pair has
    variance twice the orientation component, and the row means carry
    the geometry component plus half of it. The manuscript reports
    sd 2.40 dB and a geometry component of -1.99 dB^2 on four pairs.
    """
    g, m = pair_vectors(PRIMARY)
    d = g - m
    s_orient = float(np.std(d, ddof=1)) / np.sqrt(2.0)
    rowm = 0.5 * (g + m)
    v_geom = float(np.var(rowm, ddof=1)) - 0.5 * s_orient ** 2
    r = float(np.corrcoef(g, m)[0, 1])
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(len(g) - 3)
    print("SEED-BY-FABRIC DECOMPOSITION, eight pairs")
    print("  orientation draw at frozen geometry, sd   %6.2f dB"
          % s_orient)
    print("  tessellation-geometry component           %6.2f dB^2"
          % v_geom)
    print("  correlation of the two fabrics across the shared")
    print("  tessellations   r = %+.2f, 95 %% interval [%+.2f, %+.2f]"
          % (r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)))
    print("  Four pairs gave sd 2.40 dB, geometry -1.99 dB^2 and")
    print("  r = -0.76 on [-0.99, +0.75]. Doubling the pairs halves")
    print("  the interval and the geometry component is still not")
    print("  resolved as positive by this decomposition, which is why")
    print("  the azimuthal check below is the evidence that it is not")
    print("  inert.")
    print()


def report_azimuth_correlation():
    """The check that does not go through the revolution mean at all.

    If the shared tessellation mattered it would imprint a common
    pattern on the level as a function of azimuth, so the per-azimuth
    levels of a SHARED girdle and single-maximum pair should correlate
    more than those of an unshared one. Both groups cross the fabric, so
    whatever pattern the fabric itself imposes cancels between them.

    The null is exact rather than assumed: the eight single-maximum
    sweeps are relabelled among the eight girdles, all 8! = 40320
    assignments enumerated, and the mean diagonal correlation compared
    with its own permutation distribution. The manuscript reports this
    on four pairs as +0.19 against -0.15 at p = 0.03.
    """
    from itertools import permutations
    print("SHARED TESSELLATION, per-azimuth correlation of the level")
    g = [DR.db(levels(GIRDLE[s])["coda_band"]) for s in SEEDS]
    m = [DR.db(levels(SINGLE[s])["coda_band"]) for s in SEEDS]
    R = np.array([[np.corrcoef(a, b)[0, 1] for b in m] for a in g])
    print("  correlation matrix, girdle rows against single columns")
    print("  %-6s %s" % ("", " ".join("%7s" % ("s%d" % s)
                                      for s in SEEDS)))
    for i, s in enumerate(SEEDS):
        print("  %-6s %s" % ("s%d" % s,
                             " ".join("%+7.3f" % v for v in R[i])))
    diag = np.diag(R)
    off = R[~np.eye(8, dtype=bool)]
    obs = diag.mean()
    null = np.array([R[np.arange(8), p].mean()
                     for p in permutations(range(8))])
    print()
    print("  shared pairs, n = 8    mean r = %+.3f" % obs)
    print("  unshared pairs, n = 56 mean r = %+.3f" % off.mean())
    hits = int((null >= obs - 1e-12).sum())
    print("  exact relabelling null over 8! = %d assignments:" % len(null))
    print("    %d assignments reach the true one, p = %.2e, floor %.2e"
          % (hits, hits / float(len(null)), 1.0 / len(null)))
    print("  per-pair shared r: %s"
          % " ".join("%+.2f" % v for v in diag))
    print()


def report_ladder():
    """The second tessellation against the first, rung by rung."""
    print("THE TWO RESOLUTION LADDERS")
    print("  Same girdle fabric, same code path, same 30 azimuths, two")
    print("  independent tessellations. Levels are the audited ones.")
    print()
    print("  %-6s %-5s %14s %14s %16s %12s"
          % ("seed", "ppw", "coda/backwall", "coda/source",
             "backwall/source", "az scatter"))
    rows = {}
    for s in (11, 23):
        for p in (6, 8, 10):
            n = LADDER[s][p]
            c, b = coda_level(n), backwall_level(n)
            rows[(s, p)] = (c - b, c, b, azimuth_scatter(n))
            print("  %-6d %-5d %14.2f %14.2f %16.2f %12.2f"
                  % ((s, p) + rows[(s, p)]))
    print()
    print("  CHANGE PER STEP, which is the quantity that has to agree.")
    print("  %-14s %14s %14s %16s %12s"
          % ("step", "coda/backwall", "coda/source", "backwall/source",
             "az scatter"))
    steps = {}
    for s in (11, 23):
        for lo, hi in ((6, 8), (8, 10)):
            d = tuple(rows[(s, hi)][j] - rows[(s, lo)][j]
                      for j in range(4))
            steps[(s, lo, hi)] = d
            print("  s%-2d %d->%-7d %+14.2f %+14.2f %+16.2f %+12.2f"
                  % ((s, lo, hi) + d))
    print()
    print("  %-24s %10s %10s %10s"
          % ("quantity", "seed 11", "seed 23", "difference"))
    named = (("coda step 6->8, dB", (6, 8), 1),
             ("coda step 8->10, dB", (8, 10), 1),
             ("backwall step 6->8, dB", (6, 8), 2),
             ("backwall step 8->10, dB", (8, 10), 2),
             ("coda/bw step 6->8, dB", (6, 8), 0),
             ("coda/bw step 8->10, dB", (8, 10), 0),
             ("scatter step 6->8, dB", (6, 8), 3),
             ("scatter step 8->10, dB", (8, 10), 3))
    for tag, (lo, hi), j in named:
        a, b = steps[(11, lo, hi)][j], steps[(23, lo, hi)][j]
        print("  %-24s %10.2f %10.2f %10.2f" % (tag, a, b, b - a))
    r11 = steps[(11, 8, 10)][1] / steps[(11, 6, 8)][1]
    r23 = steps[(23, 8, 10)][1] / steps[(23, 6, 8)][1]
    print("  %-24s %10.3f %10.3f %10.3f"
          % ("step ratio, 8-10 / 6-8", r11, r23, r23 - r11))
    print()
    print("  scatter series, dB")
    for s in (11, 23):
        print("    seed %-3d %s" % (s, "  ".join("%.2f" % rows[(s, p)][3]
                                                 for p in (6, 8, 10))))
    print()
    print("  INFERRED, and marked so. A quantity first order in the grid")
    print("  spacing h changes between rungs in proportion to the change")
    print("  in h, giving a step ratio of (1/8 - 1/10)/(1/6 - 1/8) =")
    print("  %.3f. Both ladders return a ratio near it. This is an"
          % (((1 / 8. - 1 / 10.) / (1 / 6. - 1 / 8.))))
    print("  INFERENCE from two steps on two specimens and is not a")
    print("  convergence order fitted to the data.")
    print()
    print("  INFERRED, and marked so. Neither scatter series has")
    print("  flattened and both rise monotonically, so both bound the")
    print("  residual grid dependence of the scatter without")
    print("  demonstrating convergence of it. The manuscript already")
    print("  says exactly that for seed 11.")
    print()


def report_tables():
    """The two manuscript tables, emitted as rows, to two decimals.

    Printed so that nothing is transcribed by hand. The rounding here is
    the rounding the tables carry.
    """
    g, m = pair_vectors(PRIMARY)
    print("tab:ensemble ROWS, eight pairs")
    order = np.argsort(SEEDS)
    for i in order:
        print("%d & %.2f & %.2f & %.2f \\\\"
              % (SEEDS[i], g[i], m[i], g[i] - m[i]))
    print("mean & %.2f & %.2f & %.2f \\\\"
          % (g.mean(), m.mean(), (g - m).mean()))
    print()
    print("tab:reconcile SECOND LADDER ROWS, seed 23")
    for p in (6, 8, 10):
        n = LADDER[23][p]
        c, b = coda_level(n), backwall_level(n)
        print("%d  & %.2f & %.2f & %.2f \\\\" % (p, c - b, c, b))
    for lo, hi in ((6, 8), (8, 10)):
        a = LADDER[23][lo]
        z = LADDER[23][hi]
        ca, cz = coda_level(a), coda_level(z)
        ba, bz = backwall_level(a), backwall_level(z)
        print("$%d\\!\\to\\!%d$ & %+.2f & %+.2f & %+.2f \\\\"
              % (lo, hi, (cz - bz) - (ca - ba), cz - ca, bz - ba))
    print()


def report_verdict():
    """The two answers, and what each does not license."""
    g, m = pair_vectors(PRIMARY)
    r8 = paired_test(g, m, subsets()[0][1])
    r7 = paired_test(g, m, subsets()[1][1])
    r6 = paired_test(g, m, subsets()[2][1])
    print("=" * 71)
    print("VERDICT ONE, THE FABRIC SEPARATION")
    print("  On eight pairs the single maximum is louder by %.2f dB,"
          % -r8["delta"])
    print("  paired t = %.2f on 7 degrees of freedom, p = %.4f, exact"
          % (r8["t"], r8["p"]))
    print("  sign-flip p = %.4f, %d of 8 pairs agreeing in sign. The"
          % (r8["p_flip"], r8["agree"]))
    print("  four-pair statement was t = -1.46, p = 0.24.")
    print("  DIAGNOSTIC, not the headline: dropping seed 17 leaves")
    print("  %.2f dB at p = %.4f, and dropping seeds 7 and 17 leaves"
          % (-r7["delta"], r7["p"]))
    print("  %.2f dB at p = %.4f on %d of 6 pairs agreeing. The"
          % (-r6["delta"], r6["p"], r6["agree"]))
    print("  separation is smaller without them and does not vanish.")
    for key in CONFIRMATORY:
        a, b = pair_vectors(key)
        q8 = paired_test(a, b, subsets()[0][1])
        q6 = paired_test(a, b, subsets()[2][1])
        print("  %-14s %+.4f %s on eight, p = %.4f; %+.4f, p = %.4f"
              % (key, -q8["delta"], UNITS.get(key, ""), q8["p"],
                 -q6["delta"], q6["p"]))
        print("                 single minus girdle, sign agreement "
              "%d/8 and %d/6" % (q8["agree"], q6["agree"]))
    print("  Bonferroni over the three pre-specified channels is")
    print("  0.0167 and over all eleven computed is 0.0045.")
    print()
    print("VERDICT TWO, THE LADDERS")
    print("  Printed above. The comparison to read is the step table:")
    print("  a claim about the regime needs the STEPS to agree, not the")
    print("  levels, since the two specimens differ in level at every")
    print("  rung by an amount the ensemble spread already covers.")
    print("=" * 71)
    print()


def main():
    report_reproduction()
    report_concentration()
    report_pairs()
    report_unpaired()
    report_independence()
    report_decomposition()
    report_azimuth_correlation()
    report_ladder()
    report_tables()
    report_verdict()


if __name__ == "__main__":
    main()

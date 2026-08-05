"""The property-observable matrix, screened: which cells hold a relation.

The paper has tested a handful of cells chosen by intuition.  This module
takes the two sides that now exist as arrays, sample properties from
analysis/sample_matrix.py and waveform observables from
analysis/observable_matrix.py, and enumerates the cells properly.

THE CLAIM this file exists to support: the screen is a HYPOTHESIS
GENERATOR, not a confirmatory test.  Nothing it prints is a result.  A
cell that survives here has earned an independent test on data that did
not generate it, and the module says which test and whether that test
needs new simulation.

TWO REGIMES, kept apart everywhere, because their power differs by two
orders of magnitude.

  BETWEEN specimens.  n = 8 girdle tessellations, n = 7 complete single
      maxima.  A Spearman of 0.7 on eight points is a coin toss dressed
      up.  n is printed on every line and the null is the exact
      enumeration of all 8! label permutations, not a t approximation.

  WITHIN a specimen, across azimuth.  n = 30 or 60 per specimen and the
      project's own circular-shift null applies.  The predictor's period
      decides how many DISTINCT alignments exist: if p(az+180) = p(az)
      then rho(k) = rho(k + n/2) exactly, so a 30-azimuth sweep offers 15
      alignments, not 30.  Using 30 halves the p-value floor and
      manufactures significance.  The per-column count is read from
      sample_matrix_azimuth.npz:shift_alignments_30, not assumed.
      The single-specimen floor of 1/15 is too coarse to screen with, so
      cells are COMBINED across the 15 specimens under INDEPENDENT
      random rotations.  That is what buys the power.

DEDUPLICATION FIRST, and from the regime that can afford it.  Both sides
are full of restatements; screening 145 properties against 68
observables turns one effect into twenty hits and then pays a
multiplicity penalty for the privilege.  The observable side is
clustered on the WITHIN-specimen data, which pools 15 specimens by 30 to
60 azimuths after removing each specimen's own mean, and that clustering
is then reused for the between-specimen screen, where eight points could
never have estimated it.

CONTROLS ARE THE DECIDING TEST, not the statistic.  The five seed-11
control rows share a BIT-IDENTICAL tessellation with girdle_s11, which
this module re-verifies rather than assumes, so every purely geometric
per-azimuth predictor is literally the same series in the control as in
the specimen.  A cell that fires with the contrast switched off is
reading geometry or the numerics.  The contrast ladder f = 0, 0.25,
0.50, 0.75 adds what no single control can: a cell that is about
scattering should STRENGTHEN with f, and the module reports that slope.

CENSORING GUARD.  Correlation-length columns on a 30-azimuth sweep
return exactly the azimuth step.  That is an upper bound, not a
measurement, and such a column is dropped from the between-specimen
screen rather than silently regressed.

Reads, and writes nothing:
    sim/results/sample_matrix.npz
    sim/results/sample_matrix_azimuth.npz
    out/observables/observable_matrix.npz

CPU only.  No specimen build, no solver, no CUDA, no trace is opened.
"""
import itertools
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

ROOT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SAMPLE = os.path.join(ROOT, "sim", "results", "sample_matrix.npz")
SAMPLE_AZ = os.path.join(ROOT, "sim", "results", "sample_matrix_azimuth.npz")
OBS = os.path.join(ROOT, "out", "observables", "observable_matrix.npz")

CLUSTER_THR = 0.90
FDR_Q = 0.10
N_DRAW = 20000
N_PERM = 200000
MIN_AZ = 20
SEED = 20260802

CONTROL_SWEEPS = ("girdle_seed11_ppw8_uniform_axis", "girdle_seed11_ppw8_contrast_f000")
LADDER = ("girdle_seed11_ppw8_contrast_f000", "girdle_seed11_ppw8_contrast_f025", "girdle_seed11_ppw8_contrast_f050",
          "girdle_seed11_ppw8_contrast_f075")
LADDER_F = np.array([0.00, 0.25, 0.50, 0.75])

DESIGN_COLS = ("seed", "kappa", "contrast_f", "is_single_max", "is_girdle",
               "ppw", "h_mm", "vol_disc_cm3")
QC_OBS = ("qc_pedestal", "qc_oob", "qc_dt_ns", "qc_nsamp")

BSTATS = ("sd", "A1", "A2", "A3", "A4", "fr1", "fr2", "fr3", "fr4",
          "ph2", "ph4", "r180", "rmsd180", "sd_res", "acf_tau_res",
          "acf_e_res", "neff_res", "neff")
CENSORED_STATS = ("acf_e", "acf_tau", "acf_e_exp", "acf_e_res",
                  "acf_tau_res", "acf_e_exp_res", "dec_wave_e", "dec_env_e",
                  "dec_coh_e", "dec_wave_tau", "dec_env_tau", "dec_coh_tau",
                  "dec_wave_exp", "dec_env_exp", "dec_coh_exp")

SPEED_PRED = ("c_axis_ms", "c_vol_ms", "c_slow_ms", "t_axis_us",
              "path_axis_mm", "chord_axis_mm", "c_axis_ms_m", "c_vol_ms_m",
              "c_slow_ms_m", "chord_axis_mm_m")
TIME_OBS = ("t_bw_peak", "t_bw_on6", "t_bw_cen", "t_slow", "v_app_slow",
            "t_gate_cen", "t_coda_on70", "t_coda_on80", "t_coda_knee",
            "t_rec_cen", "t_rec_med", "t_front", "t_front_on20")
CONTRAST_PRED = ("dv_all_rms", "dv_all_mean", "dv_all_p90", "dv_cross_rms",
                 "dv_cross_mean", "v_sd_ms", "dv_ray_quad", "dv_cross_rms_m",
                 "dv_all_rms_m", "v_sd_ms_m")
CODA_OBS = ("a_coda", "a_coda_env", "a_coda_near", "a_coda_mid",
            "a_coda_far", "r_coda_bw", "r_coda_bwE")
COL_CROSS = ("n_cross_col_full", "n_cross_col_gate", "n_cross_col_gate_wt",
             "n_cross_paper", "n_cross_col_full_m", "n_cross_col_gate_m",
             "n_cross_paper_m")
SIZE_PRED = ("n_grain", "d_mean_mm", "d_median_mm", "d_max_mm",
             "area_total_cm2", "sv_per_mm", "n_grain_axis",
             "n_grain_col_gate", "n_grain_axis_m", "n_grain_col_gate_m")
A1_PRED = ("a1_count", "a1_vol", "a1_col", "a3_col", "a1_vol_minus_count",
           "a1_col_m", "a2_count", "a2_vol", "a3_count", "a3_vol")
FABRIC_PRED = ("fabric_axis_phi_deg", "fabric_axis_tilt_deg", "e1_phi_deg",
               "e1_tilt_deg", "e3_phi_deg", "e3_tilt_deg",
               "fabaxis_meas_phi_deg", "fabaxis_meas_tilt_deg")
FRONT_OBS = ("t_front", "t_front_on20", "a_front_abs", "f_front_cen",
             "f_front_rms")


# ----------------------------------------------------------------- load

def load():
    """Load both sides; nothing is recomputed from traces or specimens."""
    zs = np.load(SAMPLE, allow_pickle=True)
    za = np.load(SAMPLE_AZ, allow_pickle=True)
    zo = np.load(OBS, allow_pickle=True)
    d = {}
    d["p_names"] = [str(s) for s in zs["names"]]
    d["p_kinds"] = [str(s) for s in zs["kinds"]]
    d["p_sweeps"] = [str(s) for s in zs["sweeps"]]
    d["p_cols"] = [str(s) for s in zs["columns"]]
    d["P"] = np.asarray(zs["values"], float)
    d["near_const"] = [str(s) for s in zs["near_constant"]]
    d["az_cols"] = [str(s) for s in za["columns"]]
    d["az_deg"] = np.asarray(za["az_deg"], int)
    d["AZP"] = np.asarray(za["values"], float)
    d["k30"] = np.asarray(za["shift_alignments_30"], int)
    d["o_sweeps"] = [str(s) for s in zo["sweep"]]
    d["o_cols"] = [str(s) for s in zo["obs_names"]]
    d["b_cols"] = [str(s) for s in zo["bobs_names"]]
    d["A"] = np.asarray(zo["A"], float)
    d["AZO"] = np.asarray(zo["AZ"], float)
    d["S"] = np.asarray(zo["S"], float)
    d["B"] = np.asarray(zo["B"], float)
    d["n_az"] = np.asarray(zo["n_az"], int)
    d["az_step"] = np.asarray(zo["az_step"], float)
    return d


def join(d):
    """Rows present on both sides with enough azimuths to shift."""
    rows, dropped = [], []
    for i, sw in enumerate(d["p_sweeps"]):
        if sw not in d["o_sweeps"]:
            dropped.append((sw, "no observable row"))
            continue
        j = d["o_sweeps"].index(sw)
        n = int(d["n_az"][j])
        if n < MIN_AZ:
            dropped.append((sw, "only %d azimuths, cannot shift" % n))
            continue
        step = int(round(d["az_step"][j]))
        sub = np.arange(0, 60, step // 6)
        if not np.allclose(d["az_deg"][sub], d["AZO"][j, :n]):
            dropped.append((sw, "azimuth grids do not match"))
            continue
        rows.append(dict(sweep=sw, ip=i, io=j, n=n, step=step, sub=sub,
                         kind=d["p_kinds"][i], name=d["p_names"][i]))
    return rows, dropped


def verify_shared_geometry(d):
    """Re-verify that the seed-11 controls carry girdle_s11's geometry."""
    g = d["p_names"].index("girdle_s11")
    geo = ("n_cross_axis_full", "n_cross_axis_gate", "n_cross_col_full",
           "n_cross_col_gate", "n_cross_paper", "n_grain_axis",
           "n_grain_col_gate", "chord_axis_mm", "path_axis_mm",
           "normal_align", "facing_frac_10", "facing_frac_20")
    out = []
    for nm in ("zerocontrast_s11", "cs_f000_s11", "cs_f025_s11",
               "cs_f050_s11", "cs_f075_s11", "single_s11"):
        i = d["p_names"].index(nm)
        w = 0.0
        for c in geo:
            j = d["az_cols"].index(c)
            w = max(w, float(np.abs(d["AZP"][i, j] - d["AZP"][g, j]).max()))
        out.append((nm, w))
    return geo, out


# ------------------------------------------------------------ machinery

def zrank(x):
    """Rank-transform to zero mean and unit mean square, or None."""
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return None
    r = rankdata(x)
    r = r - r.mean()
    s = np.sqrt((r ** 2).mean())
    return r / s if s > 0 else None


def zrank_block(X):
    """Column-wise zrank; columns that fail come back as zeros."""
    Z = np.zeros_like(X, dtype=float)
    ok = np.zeros(X.shape[1], bool)
    for j in range(X.shape[1]):
        z = zrank(X[:, j])
        if z is not None:
            Z[:, j] = z
            ok[j] = True
    return Z, ok


def n_align(n, k30):
    """Distinct circular alignments for this predictor on n azimuths."""
    return int(k30) * n // 30


def lag_cube(Pz, Oz):
    """rho at every circular lag for every predictor-observable pair.

    Pz is (npred, n) and Oz is (nobs, n), both already rank-z-scored
    along the azimuth axis.  Returns (npred, nobs, n) where entry k is
    the Spearman rho of the predictor against the observable rolled by k.
    """
    n = Pz.shape[1]
    FP = np.fft.rfft(Pz, axis=1)
    FO = np.conj(np.fft.rfft(Oz, axis=1))
    return np.fft.irfft(FP[:, None, :] * FO[None, :, :], n, axis=2) / n


def bh(p, q):
    """Benjamini-Hochberg: return q-values and the reject mask."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    ranked = p[o] * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(qv, 1.0)
    return out, out <= q


def cluster(Z, thr=CLUSTER_THR):
    """Cluster the columns of a rank-z block; return medoid -> members."""
    m = Z.shape[1]
    if m < 2:
        return {j: [j] for j in range(m)}
    R = (Z.T @ Z) / Z.shape[0]
    D = np.clip(1.0 - np.abs(R), 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    lab = fcluster(linkage(squareform(D, checks=False), "average"),
                   1.0 - thr, criterion="distance")
    groups = {}
    for j, g in enumerate(lab):
        groups.setdefault(int(g), []).append(j)
    reps = {}
    for mem in groups.values():
        sub = np.abs(R[np.ix_(mem, mem)])
        reps[mem[int(np.argmax(sub.mean(axis=1)))]] = mem
    return reps


def null_rho(n, rng):
    """Null distribution of Spearman rho for n tie-free points, sorted.

    Exact enumeration of all n! label permutations where that is cheap,
    Monte Carlo otherwise.  It depends only on n, so one table serves
    every cell with that n.
    """
    z = zrank(np.arange(n, dtype=float))
    if n <= 8:
        perms = np.array(list(itertools.permutations(range(n))))
        return np.sort(np.abs((z[perms] * z[None, :]).mean(axis=1))), "exact"
    idx = np.array([rng.permutation(n) for _ in range(N_PERM)])
    return np.sort(np.abs((z[idx] * z[None, :]).mean(axis=1))), "mc"


def lookup_p(r, tab):
    """Two-sided p from a sorted table of |rho| under the null."""
    k = len(tab) - np.searchsorted(tab, np.abs(r) - 1e-12, side="left")
    return k / float(len(tab))


# ------------------------------------------------- within-specimen screen

def build_lags(d, rows, okeep):
    """Full lag cube per joined row, plus the rank blocks for clustering."""
    lags, pblocks, oblocks = {}, [], []
    for r in rows:
        P = d["AZP"][r["ip"]][:, r["sub"]].T
        O = d["A"][r["io"], :r["n"], :][:, okeep]
        Pz, pok = zrank_block(P)
        Oz, ook = zrank_block(O)
        lags[r["sweep"]] = (lag_cube(Pz.T, Oz.T), pok, ook)
        if r["kind"] in ("girdle", "single"):
            pblocks.append(Pz)
            oblocks.append(Oz)
    return lags, np.vstack(pblocks), np.vstack(oblocks)


def joint_shift(tabs, shifts, n_draw=N_DRAW):
    """Signed mean rho across specimens against independent rotations."""
    use = [(t, s) for t, s in zip(tabs, shifts) if t is not None]
    if len(use) < 2:
        return np.nan, np.nan, np.nan, 0
    obs = float(np.mean([t[0] for t, _ in use]))
    acc = np.zeros(n_draw)
    for t, s in use:
        acc += t[s]
    null = acc / len(use)
    p = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (n_draw + 1.0)
    sg = np.sign([t[0] for t, _ in use])
    frac = float(np.mean(sg == np.sign(obs)))
    return obs, p, frac, len(use)


def single_p(tab, k):
    """Exact two-sided shift p for one specimen over k alignments."""
    if tab is None:
        return np.nan
    return float(np.sum(np.abs(tab[:k]) >= abs(tab[0]))) / k


def within_screen(d, rows, lags, pj, oj, okeep, rng):
    """Screen every representative cell in the within-specimen regime."""
    fab = [r for r in rows if r["kind"] in ("girdle", "single")]
    gir = [r for r in rows if r["kind"] == "girdle"]
    sin = [r for r in rows if r["kind"] == "single"]
    ctl = [r for r in rows if r["sweep"] in CONTROL_SWEEPS]
    lad = [r for r in rows if r["sweep"] in LADDER]
    shift = {}
    for r in rows:
        for k30 in (15, 30):
            shift[(r["sweep"], k30)] = rng.integers(
                0, n_align(r["n"], k30), N_DRAW)

    def tab(r, p, o):
        cube, pok, ook = lags[r["sweep"]]
        return cube[p, o] if (pok[p] and ook[o]) else None

    res = []
    for p in pj:
        k30 = int(d["k30"][p])
        sh = {r["sweep"]: shift[(r["sweep"], k30)] for r in rows}
        for o in oj:
            t_f = [tab(r, p, o) for r in fab]
            rho, pv, sf, ns = joint_shift(t_f, [sh[r["sweep"]] for r in fab])
            rg, pg, _, _ = joint_shift([tab(r, p, o) for r in gir],
                                       [sh[r["sweep"]] for r in gir])
            rs, ps, _, _ = joint_shift([tab(r, p, o) for r in sin],
                                       [sh[r["sweep"]] for r in sin])
            cp = []
            for r in ctl:
                t = tab(r, p, o)
                cp.append((r["sweep"], np.nan if t is None else float(t[0]),
                           single_p(t, n_align(r["n"], k30))))
            lr = []
            for r in lad:
                t = tab(r, p, o)
                lr.append(np.nan if t is None else float(t[0]))
            res.append(dict(pj=p, oj=okeep[o], oloc=o, rho=rho, p=pv,
                            sign_frac=sf, n_spec=ns, rho_g=rg, p_g=pg,
                            rho_s=rs, p_s=ps, ctrl=cp, ladder=np.array(lr),
                            k30=k30))
    return res


def ladder_slope(lr, rho):
    """Signed growth of the cell along the contrast ladder, per unit f."""
    ok = np.isfinite(lr)
    if ok.sum() < 3:
        return np.nan
    y = lr[ok] * np.sign(rho)
    return float(np.polyfit(LADDER_F[ok], y, 1)[0])


def verdict(r):
    """Control verdict for one cell.

    KILLED     the cell is already there with the contrast switched off,
               at the same sign and comparable strength.
    CTRL-OPP   the control carries a strong relation of the OPPOSITE
               sign, which is not a kill but is not clean either.
    SURVIVES   dead at f = 0 and growing along the contrast ladder.
    WEAK       neither, usually because the ladder is one specimen of
               30 azimuths per rung and cannot resolve the trend.
    """
    sl = ladder_slope(r["ladder"], r["rho"])
    mag = abs(r["rho"])
    f0 = r["ladder"][0] if len(r["ladder"]) else np.nan
    top = r["ladder"][-1] if len(r["ladder"]) else np.nan
    live = [(v, q) for _, v, q in r["ctrl"] if np.isfinite(v)]
    for v, q in live:
        if np.sign(v) == np.sign(r["rho"]) and abs(v) >= 0.8 * mag \
                and q < 0.25:
            return "KILLED", sl
    if np.isfinite(f0) and np.isfinite(top):
        if np.sign(f0) == np.sign(r["rho"]) and abs(f0) >= 0.8 * abs(top):
            return "KILLED", sl
    for v, q in live:
        if np.sign(v) != np.sign(r["rho"]) and abs(v) >= 0.8 * mag \
                and q < 0.25:
            return "CTRL-OPP", sl
    if np.isfinite(sl) and sl > 0.20 and (not np.isfinite(f0)
                                          or abs(f0) < 0.6 * abs(top)):
        return "SURVIVES", sl
    return "WEAK", sl


# ------------------------------------------------ between-specimen screen

def censored(name, vals, steps):
    """True if a correlation-length column is pinned at the azimuth step."""
    tail = name.split("|")[-1]
    if tail not in CENSORED_STATS:
        return False
    ok = np.isfinite(vals)
    if not ok.any():
        return True
    return float(np.mean(vals[ok] <= 1.01 * steps[ok])) > 0.20


def between_screen(Pz, pok, Oz, ook, tabn):
    """All property-observable cells at one n, with the exact null."""
    n = Pz.shape[0]
    R = (Pz.T @ Oz) / n
    P = np.empty_like(R)
    for a in range(R.shape[0]):
        P[a] = [lookup_p(v, tabn) for v in R[a]]
    bad = ~np.outer(pok, ook)
    R[bad] = np.nan
    P[bad] = np.nan
    return R, P


# ------------------------------------------------------- dt confound check

def dt_confound(d, rows, okeep, rng):
    """Does the azimuthal dt mediate the within-specimen cells?

    Every azimuth has its own dt, set by the CFL limit and so by the
    fastest speed in the ROTATED medium.  dt is therefore a deterministic
    function of the azimuthal anisotropy, which is the same thing several
    predictors measure.  If an observable also tracks dt, a cell can be a
    numerical artefact rather than a physical relation.  This measures
    both legs directly instead of arguing about them.
    """
    fab = [r for r in rows if r["kind"] in ("girdle", "single")]
    jdt = d["o_cols"].index("qc_dt_ns")
    prho, orho = {}, {}
    for r in fab:
        dtz = zrank(d["A"][r["io"], :r["n"], jdt])
        if dtz is None:
            continue
        P = d["AZP"][r["ip"]][:, r["sub"]].T
        O = d["A"][r["io"], :r["n"], :][:, okeep]
        for j in range(P.shape[1]):
            z = zrank(P[:, j])
            if z is not None:
                prho.setdefault(j, []).append(float((z * dtz).mean()))
        for j in range(O.shape[1]):
            z = zrank(O[:, j])
            if z is not None:
                orho.setdefault(j, []).append(float((z * dtz).mean()))
    pm = {j: float(np.mean(v)) for j, v in prho.items()}
    om = {j: float(np.mean(v)) for j, v in orho.items()}
    return pm, om


# --------------------------------------------------------- fabric type

def paired_signflip(x, y):
    """Exact 2**n sign-flip null on the paired differences."""
    dd = np.asarray(y, float) - np.asarray(x, float)
    n = len(dd)
    if not np.all(np.isfinite(dd)) or np.allclose(dd, 0):
        return np.nan, np.nan
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    null = (signs * dd[None, :]).mean(axis=1)
    obs = dd.mean()
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-15))


def twosample_exact(x, y):
    """Exact rank-sum on all C(n, k) label splits of the pooled ranks."""
    v = np.concatenate([x, y])
    if not np.all(np.isfinite(v)) or np.ptp(v) == 0:
        return np.nan, np.nan
    z = zrank(v)
    n, k = len(v), len(x)
    combs = np.array(list(itertools.combinations(range(n), k)))
    means = z[combs].mean(axis=1)
    obs = z[:k].mean()
    p = float(np.mean(np.abs(means - z.mean()) >= abs(obs - z.mean())
                      - 1e-15))
    return float(z[k:].mean() - obs), p


# --------------------------------------------------------- classification

def classify(pname, oname):
    """Already in the paper, known physics, or neither."""
    o = oname.split("|")[0]
    p = pname
    if o in FRONT_OBS:
        return "NULL-CH", "front arrival is the source reference"
    if o.startswith("t_bw") and p in FABRIC_PRED:
        return "PAPER", "backwall arrival against fabric axis and type"
    if o in CODA_OBS and p in COL_CROSS:
        return "PAPER", "coda level against beam-column crossings"
    if o in CODA_OBS and p in SIZE_PRED:
        return "PAPER", "coda level against grain size, reported null"
    if o in CODA_OBS and p in A1_PRED:
        return "PAPER", "fabric strength a1 against coda"
    if p in SPEED_PRED and o in TIME_OBS:
        return "PHYSICS", "arrival time is path over speed, a solver check"
    if p in CONTRAST_PRED and o in CODA_OBS:
        return "PHYSICS", "contrast drives scattering, this is the ladder"
    return "NEW", ""


# ---------------------------------------------------------------- report

def fmt_p(p):
    if not np.isfinite(p):
        return "   n/a"
    if p <= 1.0 / (N_DRAW + 1.0):
        return "<5e-05"
    return "%6.4f" % p


def main():
    rng = np.random.default_rng(SEED)
    d = load()
    rows, dropped = join(d)
    print("=" * 79)
    print("PROPERTY-OBSERVABLE SCREEN.  THIS IS A SCREEN, NOT A TEST.")
    print("=" * 79)
    print("joined rows: %d of %d sample rows" % (len(rows),
                                                 len(d["p_sweeps"])))
    for sw, why in dropped:
        print("  dropped %-22s %s" % (sw, why))
    for kind in ("girdle", "single", "control"):
        s = [r["sweep"] for r in rows if r["kind"] == kind]
        print("  %-8s n=%d" % (kind, len(s)))
        print("      " + ", ".join(s))

    geo, chk = verify_shared_geometry(d)
    print()
    print("CONTROL PREMISE, re-verified: %d purely geometric per-azimuth"
          % len(geo))
    print("predictors, max |difference| against girdle_s11:")
    for nm, w in chk:
        print("  %-18s %.3e %s" % (nm, w, "IDENTICAL" if w == 0 else "-"))

    okeep = [j for j, c in enumerate(d["o_cols"]) if c not in QC_OBS]
    lags, Pblk, Oblk = build_lags(d, rows, okeep)
    preps = cluster(Pblk)
    oreps = cluster(Oblk)
    print()
    print("-" * 79)
    print("WITHIN-SPECIMEN REGIME.  15 specimens, 30 or 60 azimuths each,")
    print("circular shift on the DISTINCT alignments, combined across")
    print("specimens under independent rotations.")
    print("-" * 79)
    print("predictors  %d -> %d clusters at |rho| >= %.2f, within-specimen"
          % (len(d["az_cols"]), len(preps), CLUSTER_THR))
    for a, v in sorted(preps.items()):
        if len(v) > 1:
            print("   %-20s = %s" % (d["az_cols"][a],
                                     ", ".join(d["az_cols"][b] for b in v
                                               if b != a)))
    print("observables %d -> %d clusters at |rho| >= %.2f, within-specimen"
          % (len(okeep), len(oreps), CLUSTER_THR))
    for a, v in sorted(oreps.items()):
        if len(v) > 1:
            print("   %-20s = %s"
                  % (d["o_cols"][okeep[a]],
                     ", ".join(d["o_cols"][okeep[b]] for b in v if b != a)))

    pj, oj = sorted(preps), sorted(oreps)
    res = within_screen(d, rows, lags, pj, oj, okeep, rng)
    pv = np.array([r["p"] for r in res])
    qv, rej = bh(pv, FDR_Q)
    for r, q, k in zip(res, qv, rej):
        r["q"], r["rej"] = q, bool(k)
    print()
    print("cells tested          %d  (%d predictors x %d observables)"
          % (len(res), len(pj), len(oj)))
    print("expected false at 5%%  %.1f" % (0.05 * len(res)))
    print("hits at raw p<0.05    %d" % int(np.sum(pv < 0.05)))
    print("hits at BH q<=%.2f     %d" % (FDR_Q, int(rej.sum())))
    keep = sorted([r for r in res if r["rej"]], key=lambda r: (r["q"], -abs(
        r["rho"])))
    print()
    print("%-20s %-15s %6s %7s %4s %6s %6s %-8s %5s %-8s"
          % ("PREDICTOR", "OBSERVABLE", "rho", "q", "sgn", "p_gir",
             "p_sin", "CONTROL", "dr/df", "CLASS"))
    print("-" * 79)
    for r in keep:
        pn = d["az_cols"][r["pj"]]
        on = d["o_cols"][r["oj"]]
        v, sl = verdict(r)
        r["verdict"], r["slope"] = v, sl
        cls, _ = classify(pn, on)
        print("%-20s %-15s %+6.3f %7s %3.0f%% %6s %6s %-8s %+5.2f %-8s"
              % (pn, on, r["rho"], fmt_p(r["q"]), 100 * r["sign_frac"],
                 fmt_p(r["p_g"]), fmt_p(r["p_s"]), v,
                 sl if np.isfinite(sl) else 0.0, cls))

    print()
    print("CONTROL DETAIL for the surviving cells.  zero = zerocontrast,")
    print("cs = cs_f000 (both seed 11, contrast off).  nan means the")
    print("predictor is identically zero in that control, so only the")
    print("ladder can test the cell.  ladder rho at f = 0.00 to 0.75.")
    print()
    print("%-20s %-15s %-24s %-24s" % ("PREDICTOR", "OBSERVABLE",
                                       "control rho (shift p)", "ladder"))
    print("-" * 79)
    for r in keep:
        if r["verdict"] == "WEAK":
            continue
        pn, on = d["az_cols"][r["pj"]], d["o_cols"][r["oj"]]
        c = " ".join("%s%+.2f(%.2f)" % (s[:4], v, q) for s, v, q in r["ctrl"])
        lad = " ".join("%+.2f" % v for v in r["ladder"])
        print("%-20s %-15s %-24s %-24s" % (pn, on, c, lad))

    # ------------------------------------------------------ dt confound
    pm, om = dt_confound(d, rows, okeep, rng)
    print()
    print("dt CONFOUND CHECK.  Every azimuth has its own dt, set by the")
    print("CFL limit and so by the fastest speed in the rotated medium.")
    print("A cell whose predictor AND observable both track dt could be")
    print("numerical.  Mean within-specimen rho against qc_dt_ns:")
    print("  predictors, worst 6:")
    for j in sorted(pm, key=lambda k: -abs(pm[k]))[:6]:
        print("    %-22s %+.3f" % (d["az_cols"][j], pm[j]))
    print("  observables, worst 6:")
    for j in sorted(om, key=lambda k: -abs(om[k]))[:6]:
        print("    %-22s %+.3f" % (d["o_cols"][okeep[j]], om[j]))
    risky = [r for r in keep
             if abs(pm.get(r["pj"], 0)) > 0.3 and abs(om.get(r["oloc"], 0))
             > 0.3]
    print("  surviving cells with both legs above 0.3: %d%s"
          % (len(risky), "" if not risky else "  "
             + ", ".join("%s/%s" % (d["az_cols"][r["pj"]],
                                    d["o_cols"][r["oj"]]) for r in risky)))

    # ---------------------------------------------------- between regime
    print()
    print("-" * 79)
    print("BETWEEN-SPECIMEN REGIME.  n = 8 girdle, n = 7 single maximum.")
    print("Exact enumeration of all label permutations as the null.")
    print("-" * 79)
    bsel = [j for j, c in enumerate(d["b_cols"])
            if c.startswith("FIELD|") or c.split("|")[-1] in BSTATS]
    onm = [d["o_cols"][j] for j in okeep] + [d["b_cols"][j] for j in bsel]
    pk = [j for j, c in enumerate(d["p_cols"])
          if c not in DESIGN_COLS and c not in d["near_const"]]
    pnm = [d["p_cols"][j] for j in pk]

    for kind, nn in (("girdle", 8), ("single", 7)):
        gi = [i for i, k in enumerate(d["p_kinds"]) if k == kind]
        gsw = [d["p_sweeps"][i] for i in gi]
        gi = [i for i, s in zip(gi, gsw)
              if s in [r["sweep"] for r in rows]]
        gsw = [d["p_sweeps"][i] for i in gi]
        go = [d["o_sweeps"].index(s) for s in gsw]
        steps = d["az_step"][go]
        Ob = np.column_stack([d["S"][np.ix_(go, okeep)],
                              d["B"][np.ix_(go, bsel)]])
        cens = np.array([censored(n, Ob[:, i], steps)
                         for i, n in enumerate(onm)])
        Pg = d["P"][np.ix_(gi, pk)]
        Pz, pok = zrank_block(Pg)
        Oz, ook = zrank_block(Ob)
        ook &= ~cens
        # reuse the within-specimen observable clustering for the S block
        srep = {okeep[a]: [okeep[b] for b in v] for a, v in oreps.items()}
        keep_s = [i for i in range(len(okeep)) if okeep[i] in srep]
        brep = cluster(Oz[:, len(okeep):][:, ook[len(okeep):]])
        bidx = np.arange(len(okeep), len(onm))[ook[len(okeep):]]
        keep_b = [int(bidx[a]) for a in sorted(brep)]
        prep = cluster(Pz[:, pok])
        pidx = np.arange(len(pnm))[pok]
        keep_p = [int(pidx[a]) for a in sorted(prep)]
        cols = [i for i in keep_s + keep_b if ook[i]]
        tabn, how = null_rho(len(gi), rng)
        R, PP = between_screen(Pz[:, keep_p], pok[keep_p],
                               Oz[:, cols], ook[cols], tabn)
        flat = PP.ravel()
        good = np.isfinite(flat)
        q, rj = bh(flat[good], FDR_Q)
        print()
        print("%s: n = %d, null = %s (%d permutations)"
              % (kind.upper(), len(gi), how, len(tabn)))
        print("  properties  %d usable -> %d clusters" % (len(pnm),
                                                          len(keep_p)))
        print("  observables %d usable (%d censored) -> %d screened"
              % (int(ook.sum()), int(cens.sum()), len(cols)))
        print("  cells tested %d, expected false at 5%% = %.0f, "
              "raw p<0.05 %d, BH q<=%.2f %d"
              % (good.sum(), 0.05 * good.sum(),
                 int(np.sum(flat[good] < 0.05)), FDR_Q, int(rj.sum())))
        qq = np.full(flat.shape, np.nan)
        qq[good] = q
        order = np.argsort(np.where(np.isfinite(flat), flat, 2.0))
        print("  strongest 12 cells by p (NONE of which survives BH "
              "unless flagged):")
        print("  %-24s %-26s %6s %8s %7s %s"
              % ("PROPERTY", "OBSERVABLE", "rho", "p", "q", "CLASS"))
        shown = 0
        for i in order:
            if not good[i]:
                continue
            a, b = divmod(i, len(cols))
            pn, on = pnm[keep_p[a]], onm[cols[b]]
            cls, _ = classify(pn, on)
            print("  %-24s %-26s %+6.3f %8s %7.3f %s%s"
                  % (pn[:24], on[:26], R[a, b], fmt_p(flat[i]), qq[i], cls,
                     "  *BH*" if qq[i] <= FDR_Q else ""))
            shown += 1
            if shown >= 12:
                break
    # -------------------------------------------------------- fabric type
    print()
    print("-" * 79)
    print("FABRIC TYPE, on the PAIRED design.  A girdle and a single")
    print("maximum at the same seed share a bit-identical tessellation,")
    print("so the pair differs in orientation alone and the geometry")
    print("variance cancels.  Seven such pairs exist.  This is the same")
    print("question the paper answered as 2.83 dB, p = 0.24 on four")
    print("pairs, now on seven, on a pre-specified panel rather than on")
    print("the whole matrix, and with the exact 2**7 sign-flip null.")
    print("-" * 79)
    seeds = []
    for s in sorted({d["P"][r["ip"], 0] for r in rows
                     if r["kind"] == "girdle"}):
        g = [r for r in rows if r["kind"] == "girdle"
             and d["P"][r["ip"], 0] == s]
        m = [r for r in rows if r["kind"] == "single"
             and d["P"][r["ip"], 0] == s]
        if g and m:
            seeds.append((int(s), g[0], m[0]))
    print("pairs: %s" % ", ".join("s%d" % s for s, _, _ in seeds))
    panel = [("S", j) for j in sorted(oreps)]
    for j, c in enumerate(d["b_cols"]):
        if c.startswith("FIELD|") and not c.startswith("FIELD|n_az"):
            panel.append(("B", j))
        elif (c.split("|")[0] in ("a_bw_peak", "t_bw_peak", "a_coda",
                                  "s_coda_decay", "a_pre_bw", "a_post_bw",
                                  "a_shallow")
              and c.split("|")[-1] in ("sd", "A2", "A4", "fr2", "fr4",
                                       "r180")):
            panel.append(("B", j))
    nm, pp, tt, dd = [], [], [], []
    for src, j in panel:
        if src == "S":
            col = (lambda r: d["S"][r["io"], okeep[j]])
            label = d["o_cols"][okeep[j]]
        else:
            col = (lambda r: d["B"][r["io"], j])
            label = d["b_cols"][j]
        gx = np.array([col(r) for _, r, _ in seeds])
        sx = np.array([col(r) for _, _, r in seeds])
        ga = np.array([col(r) for r in rows if r["kind"] == "girdle"])
        sa = np.array([col(r) for r in rows if r["kind"] == "single"])
        if not (np.all(np.isfinite(gx)) and np.all(np.isfinite(sx))
                and np.all(np.isfinite(ga)) and np.all(np.isfinite(sa))):
            continue
        delta, p1 = paired_signflip(gx, sx)
        _, p2 = twosample_exact(ga, sa)
        nm.append(label)
        dd.append(delta)
        pp.append(p1)
        tt.append(p2)
    pp = np.array(pp)
    tt = np.array(tt)
    q1, r1 = bh(pp, FDR_Q)
    q2, r2 = bh(tt, FDR_Q)
    print("panel columns tested %d (pre-specified, not the whole matrix)"
          % len(nm))
    print("p_pair is the exact 2**7 sign flip on the 7 matched pairs; its")
    print("floor is 2/128 = 0.0156, and 0.0156 x %d exceeds 1, so BH on"
          % len(nm))
    print("the paired p ALONE can never reject on a panel this wide.")
    print("p_split is the exact unpaired rank split of all 8 girdle")
    print("against all 7 single maxima, C(15,8) = 6435, floor 3.1e-04.")
    print("BH is run on p_split; p_pair is shown as the sign check that")
    print("the geometry is not doing the work.")
    print("  paired p<0.05 %d, split p<0.05 %d, expected by chance %.1f"
          % (int(np.sum(pp < 0.05)), int(np.sum(tt < 0.05)),
             0.05 * len(nm)))
    print()
    print("  %-26s %10s %8s %9s %7s" % ("OBSERVABLE", "single-gir",
                                        "p_pair", "p_split", "q_split"))
    for i in np.argsort(tt)[:14]:
        print("  %-26s %+10.4g %8.4f %9.5f %7.3f%s"
              % (nm[i][:26], dd[i], pp[i], tt[i], q2[i],
                 "  *BH*" if r2[i] else ""))
    print()
    print("  per-pair single minus girdle for the leading level and")
    print("  spectral channels, seeds %s:"
          % " ".join("s%d" % s for s, _, _ in seeds))
    for want in ("a_coda", "r_coda_bw", "f_coda_rms_ib", "s_coda_decay_l",
                 "a_pre_bw", "a_post_bw", "a_shallow"):
        if want not in d["o_cols"]:
            continue
        j = d["o_cols"].index(want)
        v = [d["S"][m["io"], j] - d["S"][g["io"], j] for _, g, m in seeds]
        print("    %-16s %s" % (want, " ".join("%+6.2f" % x for x in v)))
    print()
    print("  THE KNOWN FAILURE MODE.  analysis/fabric_discriminant.py")
    print("  already reports the coda level as NOT a usable fabric")
    print("  discriminant: on five pairs the single maximum was louder")
    print("  by 3.53 dB at p = 0.104, and the contrast was carried by")
    print("  seeds 7 and 17, each of which has one azimuth of thirty")
    print("  holding 46 per cent of the revolution energy.  The test")
    print("  that matters for any new channel is therefore whether it")
    print("  survives DROPPING those two pairs.")
    print("    %-16s %8s %8s %8s %8s"
          % ("channel", "all 7", "p_pair", "drop 7,17", "p_pair"))
    drop = [k for k, (s, _, _) in enumerate(seeds) if s not in (7, 17)]
    for want in ("a_coda", "r_coda_bw", "a_coda_mid", "f_coda_rms_ib",
                 "s_coda_decay_l", "f_coda_rms"):
        j = d["o_cols"].index(want)
        gx = np.array([d["S"][g["io"], j] for _, g, _ in seeds])
        sx = np.array([d["S"][m["io"], j] for _, _, m in seeds])
        d7, p7 = paired_signflip(gx, sx)
        d5, p5 = paired_signflip(gx[drop], sx[drop])
        print("    %-16s %+8.3f %8.4f %+8.3f %8.4f"
              % (want, d7, p7, d5, p5))
    print("    the five-pair sign-flip floor is 2/32 = 0.0625, so 0.0625")
    print("    there means every remaining pair agreed in sign.")

    # -------------------------------------------------- window comparison
    print()
    print("-" * 79)
    print("WHICH TIME WINDOW CARRIES THE AZIMUTHAL RELATION BEST.")
    print("The published gate is 24 to 36 us.  The same predictors are")
    print("tested here against four windows on the same traces, and the")
    print("per-specimen rho are compared PAIRWISE across the 15")
    print("specimens with an exact 2**15 sign-flip null, so the")
    print("comparison does not depend on the shift null at all.")
    print("-" * 79)
    wins = [("a_shallow", "12-22 us"), ("a_coda", "24-36 us, published"),
            ("a_pre_bw", "40-48 us"), ("a_post_bw", "56-66 us")]
    fabr = [r for r in rows if r["kind"] in ("girdle", "single")]
    wj = [okeep.index(d["o_cols"].index(w)) for w, _ in wins]
    print("  %-16s %s" % ("PREDICTOR",
                          "  ".join("%-12s" % w for w, _ in wins)))
    for pn in ("skew_deg", "dv_all_rms", "dv_cross_rms", "dark_all_0.001"):
        p = d["az_cols"].index(pn)
        per = []
        for o in wj:
            v = []
            for r in fabr:
                cube, pok, ook = lags[r["sweep"]]
                v.append(cube[p, o, 0] if (pok[p] and ook[o]) else np.nan)
            per.append(np.array(v))
        base = per[1]
        cells = []
        for k, arr in enumerate(per):
            if k == 1:
                cells.append("%+0.3f       " % arr.mean())
            else:
                _, pw = paired_signflip(base, arr)
                cells.append("%+0.3f (%.3f)" % (arr.mean(), pw))
        print("  %-16s %s" % (pn, "  ".join(cells)))
    print("  bracketed p is the paired sign-flip against the published")
    print("  gate on the same 15 specimens; small means the window")
    print("  genuinely differs from 24-36 us, not that it is better.")
    print("  windows: %s" % ", ".join("%s = %s" % (w, t) for w, t in wins))
    print()
    print("  BUT FIRST, IS THERE ANYTHING IN THOSE WINDOWS.  A window")
    print("  whose level in zerocontrast sits close to its level in the")
    print("  specimens is not carrying grain scattering, and a")
    print("  correlation inside it means nothing.  Levels are")
    print("  source-referenced dB, azimuth-averaged as energies:")
    print("    %-14s %8s %8s %8s %8s %8s"
          % ("window", "gir mean", "sin mean", "zerocon", "cs_f000",
             "isotropic_seed41_ppw6_calibration"))
    for w, _t in wins:
        j = d["o_cols"].index(w)
        gv = np.mean([d["S"][r["io"], j] for r in rows
                      if r["kind"] == "girdle"])
        sv = np.mean([d["S"][r["io"], j] for r in rows
                      if r["kind"] == "single"])
        zc = d["S"][d["o_sweeps"].index("girdle_seed11_ppw8_uniform_axis"), j]
        cf = d["S"][d["o_sweeps"].index("girdle_seed11_ppw8_contrast_f000"), j]
        ig = d["S"][d["o_sweeps"].index("isotropic_seed41_ppw6_calibration"), j]
        print("    %-14s %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (w, gv, sv, zc, cf, ig))
    print("    margin over zerocontrast is the number that matters; a")
    print("    window with only a few dB of margin is not a scattering")
    print("    window whatever its correlations say.  isotropic_seed41_ppw6_calibration is a")
    print("    different grid (ppw 6) and is shown for scale only.")

    # ------------------------------------------------------- named cells
    print()
    print("-" * 79)
    print("CELLS THE BRIEF NAMED, reported whether or not they survive.")
    print("-" * 79)
    named = [("n_cross_axis_gate", "a_coda",
              "the axis-gate crossing count, never separated before"),
             ("n_cross_axis_full", "a_coda", "its full-traverse twin"),
             ("n_cross_col_gate", "a_coda",
              "the published column quantity, within specimen this time"),
             ("skew_p90_deg", "a_bw_peak",
              "walk-off against backwall amplitude"),
             ("skew_deg", "t_bw_peak", "walk-off against backwall time")]
    byname = {(d["az_cols"][r["pj"]], d["o_cols"][r["oj"]]): r for r in res}
    for pn, on, why in named:
        r = byname.get((pn, on))
        if r is None:
            print("  %-20s %-14s not a representative cell (clustered "
                  "into another)" % (pn, on))
            continue
        v = verdict(r)[0] if r["rej"] else "not a hit"
        print("  %-20s %-14s rho %+0.3f  p %s  q %.3f  %-9s %s"
              % (pn, on, r["rho"], fmt_p(r["p"]), r["q"], v, why))
    print()
    print("  correlation length against grain size, the cell the")
    print("  observable side called the one to look at first:")
    res60 = [r["sweep"] for r in rows if r["step"] <= 6
             and r["kind"] in ("girdle", "single")]
    print("    resolvable on %d of the 15 production specimens (%s)."
          % (len(res60), ", ".join(res60)))
    print("    Every other production sweep is censored at its 12 degree")
    print("    step.  The cell is NOT TESTABLE as the data stand.")

    print()
    print("-" * 79)
    print("READ THIS BEFORE QUOTING ANYTHING ABOVE.  This is a screen.")
    print("Surviving cells are hypotheses that have earned an independent")
    print("test on data that did not generate them.")
    print("-" * 79)


if __name__ == "__main__":
    sys.exit(main())

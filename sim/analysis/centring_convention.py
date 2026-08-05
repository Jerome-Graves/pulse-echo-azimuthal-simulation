"""Which centring convention the coda correlation of Sec. 5.2 is quoted at.

Three correlations for the same specimen, gate and predictor are in
circulation: 0.156 in the ablation table of Sec. 5.4, 0.289 in the
ceiling statement that closes it, and 0.366 in result 1 of Sec. 5.2 and
in the caption of Fig. 'codafield'. The manuscript never states which
structure was projected out of the two fields before they were
correlated, and that choice is what separates the numbers. This script
evaluates all four centrings defined in shift_null_2d.centre on the one
specimen so each circulating number can be attributed to a convention,
and scores every one of them against the circular-shift null it belongs
to, so that the smaller correlations are not penalised for having had
the easy structure removed first.

The exhaustive 2-D null is legitimate for all four modes. Each is a
projection whose subspace is invariant under both cyclic rolls: the
grand mean, the column mean, the row mean and the azimuthal harmonics
k = 0..4 at fixed time all map to themselves under a rotation in
azimuth and under a rotation in time. Centring therefore commutes with
the shift, and the whole null follows from one FFT cross-correlation
instead of a re-centring at each of the n_az * n_t offsets.

The contrast-scaling ladder is the arbiter between conventions. Every
rung carries the same tessellation, so one prediction serves all of
them, while the stiffness contrast is scaled by a fraction f and the
scattered power must therefore scale as f^2 (Sec. 4). A centring that
scores the f = 0 rung, which cannot scatter at all, is scoring
structure that no scattering put there.

Reads:
    <cache>/t2p_girdle_perp_ppw8__gp8.npz    measured and predicted
                                             fields, 60 azimuths,
                                             cached by predict_field.py
    <sweeps>/cs_f{000,025,050,075}_s11_ppw8/az*.npz   ladder rungs
    <sweeps>/girdle_perp_ppw8/az*.npz        the f = 1 rung

Writes nothing. Prints the correlation, the two nulls and the retained
variance for each convention, which are the numbers Sec. 5.2 and
Sec. 5.4 quote.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _t2_common as C                                   # noqa: E402
import shift_null_2d as S                                # noqa: E402

# The cached azimuth-by-time fields are analysis products rather than
# repository content, so they sit beside the repository. Every location
# that has held them is tried.
CACHE_DIRS = ((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim", "analysis", "facet_model"))),
              os.path.join(C.HERE, "facet_model"), C.HERE)

SWEEP = "girdle_perp_ppw8"          # girdle kappa = -8, seed 11, ppw 8
TAG = "gp8"                         # the tessellation that was simulated
MODES = ("raw", "col", "double", "harm")

# Eq. (eq:facetmodel) as the manuscript states it carries the reflection
# weighting and the facet directivity together, which is 'full'; 'geom'
# replaces the reflection coefficient by a constant and is the variant
# the identification tests use.
VARIANTS = ("full", "geom")

# Contrast-scaling ladder, Sec. 4: each grain's stiffness is
# interpolated a fraction f from the orientation-averaged tensor toward
# its own, so scattered power scales as f^2 and f = 0 cannot scatter.
LADDER = ((0.00, "cs_f000_s11_ppw8"), (0.25, "cs_f025_s11_ppw8"),
          (0.50, "cs_f050_s11_ppw8"), (0.75, "cs_f075_s11_ppw8"),
          (1.00, SWEEP))


# ------------------------------------------------------------------ load

def cache_path(name):
    for d in CACHE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("%s in none of %s" % (name, list(CACHE_DIRS)))


def load_field():
    """Measured and predicted power inside the coda gate.

    The measurement is a band-passed envelope and the model predicts
    power, so the envelope is squared; dividing by the backwall peak
    first removes the source amplitude.
    """
    with np.load(cache_path("t2p_%s__%s.npz" % (SWEEP, TAG))) as z:
        t_s = z["tgrid"]
        gate = (t_s >= C.CODA_W[0]) & (t_s < C.CODA_W[1])
        az_deg = z["az"].astype(int)
        meas = (z["meas"][:, gate] / float(z["e1"])) ** 2
        pred = {v: z["p_%s" % v][:, gate] for v in VARIANTS}
    return az_deg, t_s[gate], meas, pred


def load_ladder(az_full):
    """Every rung on the azimuths the coarser rungs actually hold.

    The ladder rungs are sampled every 12 degrees and the production
    sweep every 6, so all of them are cut down to the common grid; the
    prediction is cut down with them, by index rather than by
    interpolation, because interpolating in azimuth would invent the
    structure the test is about.
    """
    t_grid = np.arange(20e-6, 42e-6, C.DT_C)
    gate = (t_grid >= C.CODA_W[0]) & (t_grid < C.CODA_W[1])
    rungs, az_common = [], None
    for frac, name in LADDER:
        az, env, e1, _ = C.load_sweep(name, t_grid)
        if az_common is None:
            az_common = az
        take = np.array([int(np.where(az == a)[0][0]) for a in az_common])
        rungs.append((frac, (env[take][:, gate] / e1) ** 2))
    take_pred = np.array([int(np.where(az_full == a)[0][0])
                          for a in az_common])
    return az_common, rungs, take_pred


# --------------------------------------------------------------- compute

def all_shift_correlations(meas_c, pred_c):
    """r at every cyclic (azimuth, time) offset, exactly, by FFT.

    Valid because each centring projects onto a subspace both rolls
    preserve, so the centred measurement may be rolled directly instead
    of the field being re-centred after every roll.
    """
    cross = np.fft.rfft2(meas_c) * np.conj(np.fft.rfft2(pred_c))
    r = np.fft.irfft2(cross, s=meas_c.shape)
    return r / np.sqrt((meas_c ** 2).sum() * (pred_c ** 2).sum())


def null_stats(r_all, r0):
    """Excess, spread and exceedance of a set of shifted correlations."""
    null = np.delete(np.asarray(r_all).ravel(), 0)
    return dict(sd=float(null.std()),
                z=float((r0 - null.mean()) / null.std()),
                p=float((np.abs(r_all) >= abs(r0)).sum() / np.size(r_all)))


def score_mode(meas, pred, mode):
    """Correlation, both nulls and retained variance for one convention."""
    m_db = 10 * np.log10(meas + S.EPS)
    p_db = 10 * np.log10(pred + S.EPS)
    m_c, p_c = S.centre(m_db, mode), S.centre(p_db, mode)
    r_all = all_shift_correlations(m_c, p_c)
    r0 = float(r_all[0, 0])
    n_eff = S.n_eff(m_c, p_c)
    return dict(
        mode=mode, r=r0, n_shift=int(r_all.size), n_az=int(r_all.shape[0]),
        kept=float((m_c ** 2).sum() / ((m_db - m_db.mean()) ** 2).sum()),
        joint=null_stats(r_all, r0), azim=null_stats(r_all[:, 0], r0),
        n_eff=n_eff, p_par=S.t_p(r0, n_eff))


def structure_terms(meas, pred):
    """The two profiles that 'col' and 'double' remove, and the harmonics.

    The depth profile is the mean over azimuth at each time: the decay
    of backscatter with range, which any model with a beam and a gate
    reproduces without knowing any geometry. The azimuth level is the
    mean over time at each azimuth: the scalar backscatter level, which
    Sec. 5.2 already reports as a separate result. The girdle axis lies
    in the scan plane, so the fabric alone imposes a 180 degree periodic
    modulation on that level, at wavenumber k = 2.
    """
    m_db = 10 * np.log10(meas + S.EPS)
    p_db = 10 * np.log10(pred + S.EPS)
    level = m_db.mean(1)
    spec = np.abs(np.fft.rfft(level - level.mean())) ** 2
    return dict(depth=S.corr(m_db.mean(0), p_db.mean(0)),
                level=S.corr(level, p_db.mean(1)),
                harmonics=spec / spec.sum())


def compute(meas, pred, rungs, take_pred):
    table = {v: [score_mode(meas, pred[v], m) for m in MODES]
             for v in VARIANTS}
    terms = structure_terms(meas, pred["full"])
    ladder = [(frac, [S.score(m_f, pred["full"][take_pred], mode)
                      for mode in MODES]) for frac, m_f in rungs]
    # The field spans many decades, so the manuscript compares it in dB.
    # The linear comparison is kept because it is the one convention
    # that changes the answer more than any centring does, and it is
    # worth showing why: in the linear domain the range decay is a
    # multiplicative factor that no additive centring can remove.
    linear = [(mode, S.score(meas, pred["full"], mode, log=False))
              for mode in MODES]
    return table, terms, ladder, linear


# ---------------------------------------------------------------- report

def report(az_deg, t_s, table, terms, ladder, linear, az_ladder):
    print("%s <- %s : %d azimuths, gate %.0f-%.0f us, %d time samples"
          % (SWEEP, TAG, len(az_deg), C.CODA_W[0] * 1e6, C.CODA_W[1] * 1e6,
             len(t_s)))
    n_shift = table[VARIANTS[0]][0]["n_shift"]
    n_az = table[VARIANTS[0]][0]["n_az"]
    print("nulls: %d joint (az, time) shifts, floor %.2e;  %d azimuth-only"
          " shifts, floor %.4f" % (n_shift, 1.0 / n_shift, n_az, 1.0 / n_az))
    print()
    print("A. CORRELATION AND CIRCULAR-SHIFT NULLS")
    print("   kept = fraction of the measured dB variance the centring")
    print("   leaves;  z = (r - null mean) / null sd")
    print("   %-7s %7s %6s  %7s %6s %9s  %7s %6s %7s"
          % ("mode", "r", "kept", "sd 2D", "z 2D", "p 2D", "sd az", "z az",
             "p az"))
    for v in VARIANTS:
        print("   predictor variant '%s'" % v)
        for row in table[v]:
            print("   %-7s %+7.4f %6.3f  %7.4f %6.1f %9.5f  %7.4f %6.1f"
                  " %7.4f"
                  % (row["mode"], row["r"], row["kept"], row["joint"]["sd"],
                     row["joint"]["z"], row["joint"]["p"], row["azim"]["sd"],
                     row["azim"]["z"], row["azim"]["p"]))
    print()

    print("B. EFFECTIVE SAMPLE SIZE, 'full' predictor (App. stats)")
    print("   %-7s %9s %11s" % ("mode", "N_eff", "p"))
    for row in table["full"]:
        print("   %-7s %9.1f %11.2e"
              % (row["mode"], row["n_eff"], row["p_par"]))
    print()

    print("C. WHAT EACH CENTRING TAKES AWAY")
    print("   depth profile, measured vs predicted : r = %+.4f"
          % terms["depth"])
    print("   azimuth level, measured vs predicted : r = %+.4f"
          % terms["level"])
    print("   measured azimuth level, power fraction by wavenumber k")
    print("   " + "".join("%8s" % ("k=%d" % k) for k in range(1, 7)))
    print("   " + "".join("%8.3f" % terms["harmonics"][k]
                          for k in range(1, 7)))
    print()

    print("D. CONTRAST-SCALING LADDER, %d matched azimuths, 'full' predictor"
          % len(az_ladder))
    print("   scattered power scales as f^2, so a defensible centring")
    print("   returns nothing at f = 0 and rises monotonically")
    print("   %-6s" % "f" + "".join("%9s" % m for m in MODES))
    for frac, rs in ladder:
        print("   %-6.2f" % frac + "".join("%+9.4f" % r for r in rs))
    print()

    print("E. THE SAME FIELDS COMPARED IN LINEAR POWER RATHER THAN dB")
    print("   %-7s %9s" % ("mode", "r"))
    for mode, r in linear:
        print("   %-7s %+9.4f" % (mode, r))


def main():
    az_deg, t_s, meas, pred = load_field()
    az_ladder, rungs, take_pred = load_ladder(az_deg)
    table, terms, ladder, linear = compute(meas, pred, rungs, take_pred)
    report(az_deg, t_s, table, terms, ladder, linear, az_ladder)


if __name__ == "__main__":
    main()

"""Does the fig:identify caption's uncorrected-delay claim reproduce?

The identification of Sec. 5.2 maps range to time with the two-way
speed measured from the specimen's own backwall, corrected for the
source-wavelet delay T0_SRC = 1.2/f0 (coda_field and bin_events in
tessellation_replication.py). The manuscript's fig:identify caption
quotes what happens WITHOUT the correction: the count of correct
identifications falls from four of eight to three, only two of the
four successes survive, and seed 41 falls to rank 47 of 48. Those
numbers came from an investigation that predates the archive and no
stored array returns them, so they are remeasured here on the
identical published estimator (harm centring, gate 24 to 36 us, 48
candidates, the 30 shared azimuths) and archived.

Five things are computed, in this order, and nothing is reported
until the first has passed:

  1  GATE. The stored ident and ident_own rows of
     tessellation_replication.npz are recomputed for the eight
     girdle sweeps through the same code path and must return to
     zero difference. If any row does not return, the script stops.
  2  CORRECTED COUNT. Which of the eight sweeps rank first under
     the published corrected estimator.
  3  UNCORRECTED VARIANT, exactly as the coda_field docstring
     defines it: the per-azimuth speed is DIA divided by the RAW
     backwall arrival, without subtracting T0_SRC; bin_events is
     unchanged and still adds T0_SRC, so every predicted arrival is
     stretched late by the factor t1/(t1 - T0_SRC) of its own
     travel time. Per sweep: r, rank of 48, and that factor.
  4  SCALE CHECK, the no-free-parameter defence: starting from the
     uncorrected speed, the own score is swept in a speed scale and
     its per-sweep optimum is compared against the predicted
     t1/(t1 - T0_SRC), unity being the uncorrected estimator. The
     docstring records measured 1.0121 +- 0.0040 against predicted
     1.0115, unity optimal for none.
  5  PERCENTAGE. The eight-sweep mean of t1/(t1 - T0_SRC) decides
     between the 1.15 per cent of the caption and the 1.16 per
     cent one sentence of the coda_field docstring carries.

The measurement side is not reimplemented: coda_field returns the
corrected speed c_az = 2 DIA / (t1 - T0_SRC), so the raw arrival is
recovered exactly as t1 = 2 DIA / c_az + T0_SRC and the uncorrected
speed as 2 DIA / t1. Candidate geometry is marched once per
candidate and scored under both speeds, so the corrected numbers of
the gate and the uncorrected numbers of the caption come from the
same events, the same measured fields and the same scorer.

READS
  out/sweeps/girdle_seed11_ppw8_dev/az*.npz
  out/sweeps/girdle_seed{7,17,23,41,53,71,89}_ppw8_ensemble/az*.npz
  out/tesscache/tess_s<seed>_p8_k-8.npz          cached label volumes
  sim/analysis/tessellation_replication.npz      the gate reference
WRITES
  sim/analysis/identify_delay_audit.npz          every number printed
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# tessellation_replication reads sys.argv at import; pin the argument
# list so this script's own invocation cannot change its centring, then
# set the published one explicitly.
sys.argv = sys.argv[:1]
import tessellation_replication as TR                  # noqa: E402
import _t2_common as C2                                # noqa: E402
import shift_null_2d as S2                             # noqa: E402

TR.CENTRING = "harm"          # the centring the manuscript quotes

# Speed-scale grid of the no-free-parameter check. Step 5e-4 in the
# scale, so the predicted 1.0115 sits 23 steps from unity; the grid
# contains 1.0000 exactly, so "unity optimal" is a well-posed event.
SCALES = np.linspace(0.98, 1.05, 141)


def rank_block(v, i, cands):
    """The per-sweep summary of compute_identification, verbatim.

    (r_own, rank of n_cand, p_wrong, z against the wrong-candidate
    spread, runner-up seed, runner-up score), with the same tie rule
    np.sum(v >= v[i]) the stored ident_own rows were built with.
    """
    other = np.delete(v, i)
    j = int(np.argmax(other))
    runner = [c for c in cands if c != cands[i]][j]
    return (v[i], int(np.sum(v >= v[i])),
            (1 + int(np.sum(other >= v[i]))) / len(cands),
            (v[i] - other.mean()) / other.std(ddof=1),
            runner, other[j])


def parabolic_peak(x, y):
    """Vertex of the parabola through the argmax and its neighbours."""
    j = int(np.argmax(y))
    if j == 0 or j == len(y) - 1:
        return float(x[j])
    d = y[j - 1] - 2.0 * y[j] + y[j + 1]
    if d >= 0:
        return float(x[j])
    return float(x[j] + 0.5 * (y[j - 1] - y[j + 1]) / d * (x[1] - x[0]))


def main():
    tgrid = np.arange(20e-6, 42e-6, C2.DT_C)
    gate = (tgrid >= C2.CODA_W[0]) & (tgrid < C2.CODA_W[1])
    seeds = [s for _, s in TR.GIRDLE]
    names = [n for n, _ in TR.GIRDLE]
    cands = sorted(set(seeds + TR.DISTRACTORS))

    ref = np.load(os.path.join(HERE, "tessellation_replication.npz"))
    if not (np.array_equal(ref["cands"], cands)
            and list(ref["names"][:8]) == names):
        print("STOP: candidate list or sweep order differs from the "
              "stored tessellation_replication.npz")
        sys.exit(1)

    # Measured field, corrected speed, and the raw backwall arrival
    # recovered by inverting coda_field's own range-to-time map.
    meas = {}
    for name in names:
        E, c_az = TR.coda_field(name, tgrid)
        t1 = 2.0 * C2.DIA / c_az + C2.T0_SRC
        meas[name] = (E[:, gate] ** 2, c_az, t1)

    sc_cor = {n: {} for n in names}
    sc_raw = {n: {} for n in names}
    own_ev = {}
    t0 = time.time()
    for c in cands:
        ev = TR.march_geometry(TR.tessellation(c), TR.AZ_COMMON)
        for name, own in TR.GIRDLE:
            M, c_az, t1 = meas[name]
            P = TR.bin_events(ev, c_az, tgrid)[:, gate]
            sc_cor[name][c] = S2.score(M, P, TR.CENTRING, log=False)
            Pr = TR.bin_events(ev, 2.0 * C2.DIA / t1, tgrid)[:, gate]
            sc_raw[name][c] = S2.score(M, Pr, TR.CENTRING, log=False)
        if c in seeds:
            own_ev[c] = ev
    print(f"scored {len(cands)} candidates x {len(names)} sweeps x "
          f"two speed estimators in {time.time() - t0:.0f} s")

    ident_cor = np.array([[sc_cor[n][c] for c in cands] for n in names])
    ident_raw = np.array([[sc_raw[n][c] for c in cands] for n in names])
    hits_cor = np.array([rank_block(ident_cor[k], cands.index(own), cands)
                         for k, (_, own) in enumerate(TR.GIRDLE)])
    hits_raw = np.array([rank_block(ident_raw[k], cands.index(own), cands)
                         for k, (_, own) in enumerate(TR.GIRDLE)])

    # 1 ─────────────────────────── gate ──────────────────────────────
    d_ident = float(np.max(np.abs(ident_cor - ref["ident"][:8])))
    d_own = float(np.max(np.abs(hits_cor - ref["ident_own"][:8])))
    print("=" * 71)
    print("1. GATE. Stored ident / ident_own rows, eight girdle sweeps.")
    print(f"   max |diff| ident     = {d_ident:.3g}")
    print(f"   max |diff| ident_own = {d_own:.3g}")
    print("=" * 71)
    if d_ident != 0.0 or d_own != 0.0:
        print("STOP: the stored rows did not return; nothing after this")
        print("line is trustworthy and nothing is saved.")
        sys.exit(1)
    print("   both zero: the archive returns, the estimator is the")
    print("   published one.")

    # 2 ─────────────────────── corrected count ───────────────────────
    first_cor = [s for (_, s), h in zip(TR.GIRDLE, hits_cor)
                 if int(h[1]) == 1]
    print("\n" + "=" * 71)
    print("2. CORRECTED. The published estimator, speed from")
    print("   t1 - T0_SRC.")
    print("=" * 71)
    print(f"   {len(first_cor)} of {len(names)} rank first: seeds "
          + ", ".join(str(s) for s in first_cor))

    # 3 ────────────────────── uncorrected variant ─────────────────────
    stretch = np.array([np.mean(meas[n][2] / (meas[n][2] - C2.T0_SRC))
                        for n in names])
    print("\n" + "=" * 71)
    print("3. UNCORRECTED. Speed from the raw arrival, binning")
    print("   unchanged; every predicted arrival late by the factor")
    print("   t1/(t1 - T0_SRC) of its own travel time.")
    print("=" * 71)
    print(f"{'sweep':<20}{'seed':>5}{'r_cor':>9}{'rk':>4}"
          f"{'r_raw':>9}{'rk':>4}{'stretch':>10}")
    for k, (name, own) in enumerate(TR.GIRDLE):
        print(f"{name:<20}{own:>5}{hits_cor[k, 0]:>9.3f}"
              f"{int(hits_cor[k, 1]):>4}{hits_raw[k, 0]:>9.3f}"
              f"{int(hits_raw[k, 1]):>4}{stretch[k]:>10.5f}")
    first_raw = [s for (_, s), h in zip(TR.GIRDLE, hits_raw)
                 if int(h[1]) == 1]
    survive = [s for s in first_cor if s in first_raw]
    k41 = int(hits_raw[names.index("girdle_seed41_ppw8_ensemble"), 1])
    n_raw, n_sur = len(first_raw), len(survive)
    print(f"\n   uncorrected firsts: {n_raw} of {len(names)}"
          + (": seeds " + ", ".join(str(s) for s in first_raw)
             if first_raw else ""))
    print(f"   corrected successes surviving: {n_sur} of "
          f"{len(first_cor)}"
          + (": seeds " + ", ".join(str(s) for s in survive)
             if survive else ""))
    print(f"   seed 41 falls to rank {k41} of {len(cands)}")
    cap = [("count three of eight", n_raw == 3),
           ("two of the four survive", (n_sur, len(first_cor)) == (2, 4)),
           ("seed 41 rank 47 of 48", k41 == 47)]
    for txt, ok in cap:
        print(f"   caption says {txt:<24}"
              f"{'CONFIRMED' if ok else 'WRONG'}")

    # 4 ──────────────────────── scale check ──────────────────────────
    curves = np.zeros((len(names), len(SCALES)))
    for k, (name, own) in enumerate(TR.GIRDLE):
        M, c_az, t1 = meas[name]
        c_raw = 2.0 * C2.DIA / t1
        for j, g in enumerate(SCALES):
            P = TR.bin_events(own_ev[own], g * c_raw, tgrid)[:, gate]
            curves[k, j] = S2.score(M, P, TR.CENTRING, log=False)
    opt_grid = SCALES[np.argmax(curves, axis=1)]
    opt = np.array([parabolic_peak(SCALES, c) for c in curves])
    pred = stretch                    # t1/(t1 - T0_SRC), sweep mean
    i_unity = int(np.argmin(np.abs(SCALES - 1.0)))
    n_unity = int(np.sum(np.argmax(curves, axis=1) == i_unity))
    print("\n" + "=" * 71)
    print("4. SCALE CHECK. Own score against a speed scale applied to")
    print("   the UNCORRECTED speed; the wavelet alone predicts the")
    print("   optimum at t1/(t1 - T0_SRC), and unity is the")
    print("   uncorrected estimator itself.")
    print("=" * 71)
    print(f"{'sweep':<20}{'seed':>5}{'opt':>9}{'pred':>9}"
          f"{'opt-pred':>10}")
    for k, (name, own) in enumerate(TR.GIRDLE):
        print(f"{name:<20}{own:>5}{opt[k]:>9.4f}{pred[k]:>9.4f}"
              f"{opt[k] - pred[k]:>+10.4f}")
    print(f"\n   measured optimum  {opt.mean():.4f} +- "
          f"{opt.std(ddof=1):.4f} over the eight sweeps")
    print(f"   predicted         {pred.mean():.4f} from the wavelet "
          f"delay alone")
    print(f"   unity optimal for {n_unity} of {len(names)} sweeps")
    print("   docstring records measured 1.0121 +- 0.0040, predicted")
    print("   1.0115, unity optimal for none.")

    # 5 ──────────────────────── percentage ───────────────────────────
    pct = 100.0 * (stretch - 1.0)
    print("\n" + "=" * 71)
    print("5. PERCENTAGE. Late-stretch of the predicted arrivals,")
    print("   per sweep and pooled.")
    print("=" * 71)
    for name, p in zip(names, pct):
        print(f"   {name:<22}{p:>7.4f} %")
    print(f"\n   eight-sweep mean {pct.mean():.4f} +- "
          f"{pct.std(ddof=1):.4f} %")
    two_dp = f"{pct.mean():.2f}"
    print(f"   to two figures that is {two_dp} per cent; the caption's"
          f" 1.15 is "
          f"{'SUPPORTED' if two_dp == '1.15' else 'NOT supported'}, "
          f"the docstring's 1.16 is "
          f"{'SUPPORTED' if two_dp == '1.16' else 'NOT supported'}.")

    np.savez(os.path.join(HERE, "identify_delay_audit.npz"),
             az=TR.AZ_COMMON, cands=np.array(cands),
             names=np.array(names), seeds=np.array(seeds),
             gate_diff=np.array([d_ident, d_own]),
             ident_cor=ident_cor, ident_raw=ident_raw,
             hits_cor=hits_cor, hits_raw=hits_raw,
             first_cor=np.array(first_cor),
             first_raw=np.array(first_raw),
             survive=np.array(survive), rank41_raw=np.array(k41),
             t1=np.array([meas[n][2] for n in names]),
             c_az=np.array([meas[n][1] for n in names]),
             stretch=stretch, pct=pct,
             pct_mean=np.array(pct.mean()),
             pct_sd=np.array(pct.std(ddof=1)),
             scales=SCALES, scale_curves=curves,
             scale_opt=opt, scale_opt_grid=opt_grid,
             scale_pred=pred,
             scale_opt_mean=np.array(opt.mean()),
             scale_opt_sd=np.array(opt.std(ddof=1)),
             n_unity=np.array(n_unity))


if __name__ == "__main__":
    main()

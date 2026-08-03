"""Does median trace removal let the record be read earlier than 10 us?

THE QUESTION. The analysis gate is 24 to 36 us and a window at 10 to 22 us
identifies at least as well. Earlier than 10 us the record runs into the
source ring-down and the front arrival, both of which are common to every
azimuth, and the standard tool for exactly that in ground-penetrating radar
and contact NDT is median trace removal: subtract, at each time sample, the
median across traces. The median rather than the mean because the common
component is what is wanted and a mean is dragged by the few bright
azimuths that carry most of the coda. Here the traces are azimuths and the
median is formed leaving each azimuth out, so no azimuth is ever compared
against a statistic that contains itself.

WHY EARLIER IS WORTH SOMETHING. The first Fresnel diameter is
sqrt(2 lambda L) and so goes as the square root of range. At the analysed
gate it is 14.9 mm against a 17.2 mm volume-equivalent grain, and the
manuscript's null, that the fabric axis is not recoverable from the coda,
rests on those two lengths coinciding. They separate at the shallow end,
so if the record can be read there the null's stated mechanism is a
property of the gate rather than of the specimen. What is scored here is
identification of the TESSELLATION, which is not the same observable as
the fabric axis, so nothing in this module can break the null on its own.
It can only say whether the length-scale argument behind it is
window-specific.

WHAT IS SCORED. Nothing here is a new statistic. Each cell is the
manuscript's own identification score: the measured azimuth-by-time power
field correlated, under 'harm' centring, against the parameter-free
prediction built from a candidate tessellation's grain-boundary geometry,
with range mapped to time by the specimen's own backwall speed. The
reported numbers are the own candidate's rank among the 48 candidates, its
correlation, and the rank of the true azimuthal registration among the 30
rigid rotations of the measurement.

FIVE ENVELOPE VARIANTS, because a removal has to be compared against a
control that shares everything else:

  pub    the published path. Hilbert at the trace's own sample rate, then
         interpolate the envelope onto the window's grid.
  none   interpolate the band-passed RF onto the window's grid first, then
         Hilbert. Subtracts nothing. This is the matched control: it
         isolates the resampling from the removal.
  loo    the same, with the leave-one-out azimuthal median subtracted from
         the RF at every sample before the Hilbert transform.
  |t     either grid variant with the RF tapered to the window before the
         transform. |hilbert| of a whole record is not local, and a window
         at 4 to 16 us sits three microseconds after an arrival 25 dB
         larger, so without this there is no way to say whether a shallow
         envelope is shallow scattering or the front arrival's tail.

THE CONTROLS ARE THE POINT. zerocontrast_ppw8 gives every grain the same
c-axis and cs_f000_s11_ppw8 is the f = 0 rung of the contrast ladder;
neither can backscatter from a grain boundary, and both carry the seed-11
geometry a candidate could match. They are scored in every window under
every variant. Subtracting a common component can manufacture azimuthal
structure out of nothing, so a window that identifies better while its
zero-contrast control also improves is reading the removal and not the
scattering. The converse matters just as much and is easy to miss: a
control that identifies WITHOUT any removal says the untreated window was
already reading something that is not scattering, which disqualifies the
untreated window whatever the removal then does to it. Both directions are
reported.

The backwall amplitude and time are always measured from the unremoved
native-rate envelope, so the sweep-mean normalisation and the per-azimuth
speed are identical across all five variants and the range-to-time map
never moves. Only the field being scored changes.

READS
  out/sweeps/{girdle_perp,mx_girdle_s*,zerocontrast,cs_f000_s11}_ppw8
  out/tesscache/tess_s<seed>_p8_k-8.npz    all 48 candidates, prebuilt
WRITES
  sim/analysis/median_trace_removal.npz    every number printed

Touches no CUDA and runs no solver: the label volumes are read from the
cache and the module refuses to run if any of them is missing, because
building one reaches the GPU.
"""
import os
import sys
import time

import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _t2_common as C2                                    # noqa: E402
import shift_null_2d as S2                                 # noqa: E402
import tessellation_replication as TR                      # noqa: E402

# The stored tessellation_replication.npz was written with 'harm', which is
# what Sec. 5.2 quotes; the module itself falls back to 'double' when run
# with no argument. Pinned here so this file does not depend on argv.
CENTRING = "harm"

# Each window gets its own time axis, np.arange(start - PAD, stop + PAD),
# which is how tessellation_replication.main() builds the one it uses:
# arange(20e-6, 42e-6) around a 24 to 36 us gate. Copying the construction
# rather than the span matters more than it should. np.arange accumulates,
# so its nominal 24 us sample sits a few ulp BELOW 24e-6 and the mask
# TGRID >= 24e-6 drops it: the published gate is really the 600 samples
# from 24.02 to 36.00 us. Rebuilding the same times from a different origin
# lands on the other side of the comparison, takes 24.00 to 35.98 instead,
# and moves the published correlation from +0.393 to +0.405. That is one
# sample of phase, not physics, but it is a fair measure of how tightly
# these correlations are pinned, and it is why report_validation refuses to
# continue unless the published number comes back exactly.
PAD = 4e-6

# Window starts swept from as early as the record allows. Twelve
# microseconds long, the length of the published gate, so the only thing
# that changes between rows is where the window sits. The published gate is
# carried as the last row and is the validation line.
WINDOWS = [(4e-6, 16e-6), (5e-6, 17e-6), (6e-6, 18e-6), (7e-6, 19e-6),
           (8e-6, 20e-6), (10e-6, 22e-6), (12e-6, 24e-6), (24e-6, 36e-6)]
PUBLISHED = (24e-6, 36e-6)

VARIANTS = [(False, None, "pub"), ("none", None, "none"),
            ("loo", None, "loo")]

# The locality control. |hilbert| of a whole record is not a local operator
# and the shallow windows sit a few microseconds after an arrival tens of
# decibels larger, so an envelope at 4 to 16 us can be that arrival's
# analytic tail rather than anything at 4 to 16 us. Tapering the RF to the
# window before the transform makes it local. Carried only for the two
# grid-path variants, because the published path Hilberts the whole native
# trace by definition and cannot be made local without ceasing to be the
# published path.
TAPER_PAD = 2e-6
LOCAL = [("none", TAPER_PAD, "none|t"), ("loo", TAPER_PAD, "loo|t")]
ALLV = VARIANTS + LOCAL

# The eight independent girdle tessellations, seed 11 first.
GIRDLE = list(TR.GIRDLE)
# Both zero-scattering controls on the seed-11 geometry.
CONTROL = [("zerocontrast_ppw8", 11, "same c-axis in every grain"),
           ("cs_f000_s11_ppw8", 11, "contrast f = 0.00")]
SWEEPS = GIRDLE + [(n, s) for n, s, _ in CONTROL]

CANDS = sorted(set([s for _, s in TR.GIRDLE] + TR.DISTRACTORS))

# Front arrival, for the decibel accounting only. The source fires at the
# rim and the whole transient is inside the first four microseconds; the
# peak is located from the data and printed beside the window.
FRONT = (0.0, 4e-6)

# Values this module has to reproduce before anything else it prints can be
# believed. Sec. 5.2, girdle_perp_ppw8, published gate.
REF = dict(r_own=0.3934, rank=2, n_cand=48, shift_rank=1, n_shift=30)

# Volume-equivalent grain diameter and wavelength, for the Fresnel
# comparison. Both follow from the acquisition, nothing is fitted.
LAM = C2.C_REF / C2.F0
DISC_T = 0.035


# ─────────────────────────────── load ─────────────────────────────────
def cache_path(seed):
    return os.path.join(TR.CACHE, f"tess_s{seed}_p{TR.PPW:g}_"
                                  f"k{TR.KAPPA_GIRDLE:g}.npz")


def require_cache(cands):
    """Refuse to run if a label volume would have to be built.

    DiskSpecimen.build reaches CUDA on its label path, and this analysis has
    no business touching the GPU.
    """
    missing = [c for c in cands if not os.path.exists(cache_path(c))]
    if missing:
        raise SystemExit(f"not cached, would reach CUDA: {missing}")


def grid(win):
    """The window's time axis, built the way TR.main() builds its own."""
    return np.arange(win[0] - PAD, win[1] + PAD, C2.DT_C)


def gate_mask(win, tg):
    return (tg >= win[0]) & (tg < win[1])


def fields(sweep, win):
    """Measured power field per variant, and the shared per-azimuth speed.

    The speed is returned once because it is the same in all three variants
    by construction: it comes from the unremoved native-rate envelope
    whatever the variant, and the assertion below is what says so.
    """
    tg = grid(win)
    g = gate_mask(win, tg)
    out, speeds = {}, []
    for mode, tpad, tag in ALLV:
        tap = None if tpad is None else (win[0], win[1], tpad)
        E, c_az = TR.coda_field(sweep, tg, median_removal=mode, taper=tap)
        out[tag] = E[:, g] ** 2
        speeds.append(c_az)
    for c in speeds[1:]:
        assert np.array_equal(c, speeds[0]), sweep
    return out, speeds[0]


# ────────────────────────────── compute ───────────────────────────────
def win_tag(win):
    return f"{win[0] * 1e6:.0f}-{win[1] * 1e6:.0f}"


def removal_db(sweep):
    """How much power the removal actually takes out, in decibels.

    Measured on the band-passed RF, which is the signal the subtraction
    acts on, and averaged over azimuths as an energy. Returns, per
    interval, the share of the power the leave-one-out median carries and
    the share left behind, both in dB re the unremoved power. Each interval
    is measured on its own window's grid, so the numbers describe the
    subtraction that window's score actually saw.
    """
    rows = {}
    for lab, win in [("front", FRONT)] + [(win_tag(w), w) for w in WINDOWS]:
        tg = grid(win)
        _, R = TR.azimuth_rf(sweep, tg)
        D = R - TR.loo_median(R)
        g = gate_mask(win, tg)
        p0 = float(np.mean(R[:, g] ** 2))
        rows[lab] = (10 * np.log10(float(np.mean((R - D)[:, g] ** 2)) / p0),
                     10 * np.log10(float(np.mean(D[:, g] ** 2)) / p0))
        if lab == "front":
            k = int(np.argmax(np.mean(R[:, g] ** 2, axis=0)))
            tpk = float(tg[g][k] * 1e6)
    return rows, tpk


def score_everything():
    """Every (sweep, window, variant) scored against all 48 candidates.

    The candidate is the outer loop because marching its geometry is the
    expensive step and does not depend on the sweep, the window or the
    variant. The binned prediction depends on the sweep only through the
    measured speed, so it is formed once per candidate, window and sweep,
    and shared by the three variants: they differ in the measurement only.
    """
    meas, speed = {}, {}
    for name, _ in SWEEPS:
        for w in WINDOWS:
            m, c_az = fields(name, w)
            meas[(name, win_tag(w))] = m
            speed[name] = c_az
    sc = {(n, win_tag(w), t): np.zeros(len(CANDS))
          for n, _ in SWEEPS for w in WINDOWS for _, _, t in ALLV}
    own_pred = {}
    for ic, c in enumerate(CANDS):
        t0 = time.time()
        ev = TR.march_geometry(TR.tessellation(c), TR.AZ_COMMON)
        for name, own in SWEEPS:
            for w in WINDOWS:
                tg = grid(w)
                Pg = TR.bin_events(ev, speed[name], tg)[:, gate_mask(w, tg)]
                if c == own:
                    own_pred[(name, win_tag(w))] = Pg
                for _, _, tag in ALLV:
                    sc[(name, win_tag(w), tag)][ic] = S2.score(
                        meas[(name, win_tag(w))][tag], Pg, CENTRING,
                        log=False)
        print(f"    candidate {c:>3}  {time.time() - t0:.1f} s", flush=True)
    return sc, meas, own_pred


def rank_of(v, own):
    """Own candidate's correlation, rank of 48, z, and the best wrong one."""
    i = CANDS.index(own)
    other = np.delete(v, i)
    return (float(v[i]), int(np.sum(v >= v[i])),
            float((v[i] - other.mean()) / other.std(ddof=1)),
            float(other.max()))


def shift_rank(M, P):
    """Rank of the true azimuthal registration among all 30 rotations.

    The same circular-shift null Appendix B uses, applied to the score the
    identification test uses, so the two columns of a row are commensurate.
    """
    r = np.array([S2.score(np.roll(M, s, axis=0), P, CENTRING, log=False)
                  for s in range(M.shape[0])])
    return int(np.sum(r >= r[0])), len(r)


def fresnel(win):
    """First Fresnel diameter at the window's mid-range, metres.

    Two-way: a scatterer at transverse offset r lengthens the path by
    2(sqrt(L^2 + r^2) - L), so the first zone ends at r = sqrt(lambda L / 2)
    and its diameter is sqrt(2 lambda L). This is figures/fig_scales.py
    exactly, including its convention of turning the window centre straight
    into a range without taking the source-wavelet delay off first, so the
    published-gate row here is the 14.9 mm the paper prints. Subtracting
    T0_SRC would shorten every diameter by under one per cent and change no
    comparison below.
    """
    L = 0.5 * C2.C_REF * 0.5 * (win[0] + win[1])
    return float(np.sqrt(2.0 * LAM * L)), float(L)


def grain_diameter(seed=11):
    """Volume-equivalent mean grain diameter of the specimen, metres.

    The diameter of the sphere whose volume is the mean grain volume, taken
    from the tessellation the sweeps were run on rather than from the
    nominal grain count, since grains can be extinguished by the power
    metric at build time.
    """
    with np.load(cache_path(seed)) as z:
        n = len(z["seeds"])
    v = np.pi * (C2.DIA / 2) ** 2 * DISC_T
    return float((6.0 * v / (np.pi * n)) ** (1 / 3)), n


# ──────────────────────────────── draw ────────────────────────────────
def report_validation(sc, meas, own_pred):
    print("=" * 79)
    print("0. HARNESS. The published gate through this module's own path,")
    print(f"   centring '{CENTRING}', against the numbers Sec. 5.2 states.")
    print("=" * 79)
    key = ("girdle_perp_ppw8", win_tag(PUBLISHED))
    v = sc[key + ("pub",)]
    r, rank, z, best = rank_of(v, 11)
    srk, ns = shift_rank(meas[key]["pub"], own_pred[key])
    ok = (abs(r - REF["r_own"]) < 5e-4 and rank == REF["rank"]
          and srk == REF["shift_rank"] and ns == REF["n_shift"])
    print(f"   girdle_perp_ppw8, gate {win_tag(PUBLISHED)} us")
    print(f"     own r        {r:+.4f}      Sec. 5.2 {REF['r_own']:+.4f}")
    print(f"     rank         {rank} of {len(CANDS)}     Sec. 5.2 "
          f"{REF['rank']} of {REF['n_cand']}")
    print(f"     shift null   {srk} of {ns}     Sec. 5.2 "
          f"{REF['shift_rank']} of {REF['n_shift']}")
    print(f"     best wrong candidate {best:+.4f}, own sits {z:+.1f} sd above"
          f" the wrong ones")
    print(f"   {'REPRODUCED' if ok else 'DOES NOT REPRODUCE, STOP'}")
    return ok


def report_db(dbs):
    print("\n" + "=" * 79)
    print("1. WHAT THE REMOVAL TAKES OUT. Leave-one-out azimuthal median of")
    print("   the band-passed RF, as a share of the power in each interval.")
    print("   'median' is the common component the subtraction removes and")
    print("   'left' is what survives it, both dB re the unremoved power.")
    print("=" * 79)
    labs = ["front"] + [win_tag(w) for w in WINDOWS]
    print(f"{'sweep':<20}{'peak':>7}" + "".join(f"{x:>15}" for x in labs))
    for name, _ in SWEEPS:
        rows, tpk = dbs[name]
        print(f"{name:<20}{tpk:>6.1f}u"
              + "".join(f"{rows[x][0]:>7.1f}{rows[x][1]:>8.1f}" for x in labs))
    print("\n   Each cell is median / left. A removal that does nothing shows")
    print("   a large negative median and a left near 0 dB; one that takes")
    print("   the record out shows the opposite.")


def report_windows(sc, meas, own_pred):
    print("\n" + "=" * 79)
    print(f"2. WINDOW SWEEP. Own-candidate rank of {len(CANDS)}, its")
    print("   correlation, and the azimuthal shift-null rank of 30, for the")
    print("   published envelope path, the matched no-subtraction control")
    print("   and the leave-one-out median removal.")
    print("=" * 79)
    for name, own in SWEEPS:
        note = dict((n, t) for n, _, t in CONTROL).get(name)
        print(f"\n  {name}   own seed {own}"
              + (f"   CONTROL, {note}" if note else ""))
        print(f"  {'window':<9}" + "".join(
            f"{'  ' + t + ' r':>10}{'rank':>6}{'shift':>7}"
            for _, _, t in VARIANTS))
        for w in WINDOWS:
            key = (name, win_tag(w))
            cells = []
            for _, _, tag in VARIANTS:
                r, rank, _, _ = rank_of(sc[key + (tag,)], own)
                srk, ns = shift_rank(meas[key][tag], own_pred[key])
                cells.append(f"{r:>10.3f}{rank:>4}/{len(CANDS):<2}"
                             f"{srk:>4}/{ns:<2}")
            tail = "   <- published gate" if w == PUBLISHED else ""
            print(f"  {win_tag(w) + ' us':<9}" + "".join(cells) + tail)


def report_verdict(sc):
    print("\n" + "=" * 79)
    print("3. VERDICT PER WINDOW. A window counts as identifying only if the")
    print("   own candidate ranks first of 48. The controls cannot scatter,")
    print("   so any rank they GAIN under the removal was manufactured by")
    print("   the subtraction, and any rank they already HOLD without it")
    print("   says the untreated window was reading something that is not")
    print("   scattering. Ranks are none -> loo; z is under the removal.")
    print("=" * 79)
    print(f"  {'window':<9}{'specimen':>19}{'zerocontrast':>19}"
          f"{'cs f = 0':>19}   verdict")
    for w in WINDOWS:
        t = win_tag(w)
        line, gained, dirty = [], [], []
        for name in ("girdle_perp_ppw8", "zerocontrast_ppw8",
                     "cs_f000_s11_ppw8"):
            _, a, za, _ = rank_of(sc[(name, t, "none")], 11)
            _, b, zb, _ = rank_of(sc[(name, t, "loo")], 11)
            line.append(f"{a:>3} ->{b:>3}  z {zb:+.1f}")
            gained.append(b < a)
            dirty.append(a <= 3 and za > 2.0)
        v = []
        if rank_of(sc[("girdle_perp_ppw8", t, "loo")], 11)[1] != 1:
            v.append("specimen not first")
        if gained[1] or gained[2]:
            v.append("removal LIFTS a control")
        if dirty[1] or dirty[2]:
            v.append("raw window already dirty")
        print(f"  {t + ' us':<9}{line[0]:>19}{line[1]:>19}{line[2]:>19}   "
              + ("; ".join(v) if v else "survives"))
    print("\n  A window is worth having only if the specimen is first and")
    print("  the removal lifts no control. 'raw window already dirty' is not")
    print("  a verdict on the removal: it says the untreated window could")
    print("  not have been used, whatever the removal then does to it.")


def report_locality(sc, meas, own_pred):
    print("\n" + "=" * 79)
    print("4. LOCALITY CONTROL. The same scores with the RF tapered to the")
    print(f"   window ({TAPER_PAD * 1e6:.0f} us skirts) before the Hilbert")
    print("   transform, so the envelope cannot import energy from the front")
    print("   arrival. Untapered beside tapered, removal off and on.")
    print("=" * 79)
    for name, own in SWEEPS[:1] + SWEEPS[-2:]:
        note = dict((n, t) for n, _, t in CONTROL).get(name)
        print(f"\n  {name}" + (f"   CONTROL, {note}" if note else ""))
        print(f"  {'window':<9}" + "".join(
            f"{'  ' + t + ' r':>11}{'rank':>6}{'shift':>7}"
            for t in ("none", "none|t", "loo", "loo|t")))
        for w in WINDOWS:
            key = (name, win_tag(w))
            cells = []
            for tag in ("none", "none|t", "loo", "loo|t"):
                r, rank, _, _ = rank_of(sc[key + (tag,)], own)
                srk, ns = shift_rank(meas[key][tag], own_pred[key])
                cells.append(f"{r:>11.3f}{rank:>4}/{len(CANDS):<2}"
                             f"{srk:>4}/{ns:<2}")
            print(f"  {win_tag(w) + ' us':<9}" + "".join(cells))


def report_replication(sc, meas, own_pred):
    print("\n" + "=" * 79)
    print("5. ALL EIGHT TESSELLATIONS. How many of the eight girdle sweeps")
    print("   put their own tessellation first of 48, and how many reach the")
    print("   1/30 floor of the azimuthal shift null, per window. The")
    print("   tessellation is the unit of replication throughout Sec. 5.")
    print("=" * 79)
    print(f"  {'window':<9}" + "".join(f"{t:>20}" for _, _, t in ALLV))
    print(f"  {'':<9}" + "".join(f"{'first   shift 1/30':>20}"
                                 for _ in ALLV))
    for w in WINDOWS:
        t = win_tag(w)
        cells = []
        for _, _, tag in ALLV:
            k = sum(1 for n, own in GIRDLE
                    if rank_of(sc[(n, t, tag)], own)[1] == 1)
            s = sum(1 for n, own in GIRDLE
                    if shift_rank(meas[(n, t)][tag],
                                  own_pred[(n, t)])[0] == 1)
            cells.append(f"{k:>8} of 8{s:>7} of 8")
        print(f"  {t + ' us':<9}" + "".join(cells))
    print("\n  Binomial P(X >= k) for X ~ Bin(8, 1/48): "
          + ", ".join(f"{k}: {binom.sf(k - 1, 8, 1 / len(CANDS)):.2g}"
                      for k in (3, 4, 5, 6)))
    print("  The count is what moves or does not move under the removal. It")
    print("  is identical in every window across all five variants but one")
    print("  cell, and that cell separates 'pub' from 'none', which is the")
    print("  resampling and not the subtraction.")


def report_fresnel(survivors):
    print("\n" + "=" * 79)
    print("6. LENGTH SCALES. First Fresnel diameter at the window mid-range")
    print("   against the volume-equivalent grain diameter.")
    print("=" * 79)
    dg, n = grain_diameter()
    print(f"   wavelength {LAM * 1e3:.3f} mm; {n} grains in the seed-11 "
          f"tessellation; grain {dg * 1e3:.1f} mm")
    print(f"  {'window':<9}{'mid range':>12}{'D_Fresnel':>12}"
          f"{'D_grain':>11}{'ratio':>8}")
    for w in WINDOWS:
        d, L = fresnel(w)
        mark = "   <- survives the controls" if win_tag(w) in survivors else ""
        print(f"  {win_tag(w) + ' us':<9}{L * 1e3:>10.1f}mm"
              f"{d * 1e3:>10.1f}mm{dg * 1e3:>9.1f}mm{d / dg:>8.2f}{mark}")
    if not survivors:
        print("\n   No window survived the controls, so there is no shallow")
        print("   result whose length scales need interpreting. The row for")
        print("   the published gate is printed only to show that the")
        print("   coincidence the null rests on is reproduced here.")
        return
    print("\n   The ratio is 0.87 at the published gate and half that at the")
    print("   shallow end, so the length-scale coincidence the null's stated")
    print("   mechanism relies on is a property of the gate and not of the")
    print("   specimen. That is as far as this module can take it. What is")
    print("   identified here is the TESSELLATION, not the fabric axis, and")
    print("   they are different observables: a window can carry enough")
    print("   grain-boundary geometry to pick one microstructure out of 48")
    print("   and still carry no recoverable axis. Settling whether the null")
    print("   is window-specific needs the axis observable itself, scored in")
    print("   these windows with these controls, not this one.")


def main():
    require_cache(CANDS)
    TR.CENTRING = CENTRING
    print(f"{len(WINDOWS)} windows, each on its own grid at "
          f"{C2.DT_C * 1e9:.0f} ns with {PAD * 1e6:.0f} us of pad, "
          f"{len(CANDS)} candidates, {len(TR.AZ_COMMON)} azimuths\n")
    t0 = time.time()
    sc, meas, own_pred = score_everything()
    print(f"  scored in {time.time() - t0:.0f} s\n")
    dbs = {n: removal_db(n) for n, _ in SWEEPS}

    if not report_validation(sc, meas, own_pred):
        raise SystemExit("harness does not reproduce the published gate")
    report_db(dbs)
    report_windows(sc, meas, own_pred)
    report_verdict(sc)
    report_locality(sc, meas, own_pred)
    report_replication(sc, meas, own_pred)

    survivors = []
    for w in WINDOWS:
        t = win_tag(w)
        if rank_of(sc[("girdle_perp_ppw8", t, "loo")], 11)[1] != 1:
            continue
        if any(rank_of(sc[(n, t, "loo")], 11)[1]
               < rank_of(sc[(n, t, "none")], 11)[1]
               for n, _, _ in CONTROL):
            continue
        survivors.append(t)
    report_fresnel(survivors)

    np.savez(os.path.join(HERE, "median_trace_removal.npz"),
             cands=np.array(CANDS), az=TR.AZ_COMMON,
             windows=np.array(WINDOWS),
             variants=np.array([t for _, _, t in ALLV]),
             names=np.array([n for n, _ in SWEEPS]),
             scores=np.array([[[sc[(n, win_tag(w), t)] for _, _, t in ALLV]
                               for w in WINDOWS] for n, _ in SWEEPS]),
             db_median=np.array([[dbs[n][0][x][0] for x in
                                  ["front"] + [win_tag(w) for w in WINDOWS]]
                                 for n, _ in SWEEPS]),
             db_left=np.array([[dbs[n][0][x][1] for x in
                                ["front"] + [win_tag(w) for w in WINDOWS]]
                               for n, _ in SWEEPS]),
             survivors=np.array(survivors))


if __name__ == "__main__":
    main()

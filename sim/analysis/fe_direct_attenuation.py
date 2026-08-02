"""Direct-arrival attenuation, finite-difference against finite-element.

Supports the Section 4 and Appendix A claim that the two solvers agree on
the attenuation the grain structure imposes on the direct arrival. The
appendix quoted two numbers, -1.37 against -1.34 dB and -1.81 against
-1.74 dB, but no script recomputed either: they were read off a console
that is gone. This one recovers them from the archived traces, so the
claim has a file behind it.

Reads sim/fe_p2plus_ab*_traces.npz, the five archived arbiter rounds.
Runs nothing. Each archive holds the A trace (real random c-axes) and
the B trace (uniform crystal) for both solvers on the same 12-grain
specimen, so 20 log10(|A|/|B|) at the direct arrival is the attenuation
the grain boundaries impose, with the source function divided out.

The amplitude measure is the one fe_p2plus_ab.analyse() already uses for
its normaliser: the peak of the Hilbert envelope in a 2 us window centred
on the geometric arrival. It is quoted here because it is the harness's
own definition, not because it flatters the result; the window half-width
can be moved between 0.6 and 1.5 us without changing any value in the
third decimal, the direct arrival being isolated by 2 us from anything
else.

Only rounds 3 to 5 are valid comparisons. Rounds 1 and 2 ran the
finite-element side with a point force, which radiates shear where the
finite-difference monopole radiates none; they are printed to show the
size of that error, not as agreement.

Rounds 4 and 5 share a source-receiver placement and differ only in how
the finite-element monopole is discretised, so they are two measurements
of one path, not two paths. There are two placements in the archive and
there were never more: 6 to 15 mm and 12 to 21 mm, both along x through
the specimen centre, overlapping over 3 mm of their 9 mm length.
"""
import os

import numpy as np
from scipy.signal import hilbert

SIM = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim"
F0 = 1.0e6                                  # arbiter centre frequency
C_REF = 4046.0                              # c-axis speed used for timing
PATH_LEN = 0.009                            # both placements are 9 mm
HALF = 1.0e-6                               # arrival window half-width

# Round: (archive tag, placement label, whether both solvers realise the
# same multipole). The placement label is what makes rounds 4 and 5 one
# path rather than two.
ROUNDS = (
    ("ab", "6-15 mm", False, "point force, unpadded box"),
    ("ab2", "6-15 mm", False, "point force, quartic pad"),
    ("ab3", "6-15 mm", True, "monopole and dilatation"),
    ("ab4", "12-21 mm", True, "monopole, deep placement"),
    ("ab5", "12-21 mm", True, "Gaussian-ball monopole"),
)


def load(tag):
    """Return the archived A/B traces for one round, or None."""
    path = os.path.join(SIM, "fe_p2plus_%s_traces.npz" % tag)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return dict(
            fa=np.asarray(z["fa"], float).ravel(),
            fb=np.asarray(z["fb"], float).ravel(),
            dtf=float(z["dtf"]),
            ea=np.asarray(z["ea"], float).ravel(),
            eb=np.asarray(z["eb"], float).ravel(),
            dte=float(z["dte"]))


def direct_peak(trace, dt, half=HALF):
    """Peak envelope of the direct arrival, and the time it occurs."""
    fs = 1.0 / dt
    t_dir = PATH_LEN / C_REF + 1.2 / F0      # ricker centre delay included
    env = np.abs(hilbert(trace))
    k = int(t_dir * fs)
    lo, hi = max(k - int(half * fs), 0), k + int(half * fs)
    seg = env[lo:hi]
    return seg.max(), (lo + int(seg.argmax())) / fs


def attenuation(a, b, dt, half=HALF):
    """Grain attenuation of the direct arrival, dB, A relative to B."""
    pa, _ = direct_peak(a, dt, half)
    pb, _ = direct_peak(b, dt, half)
    return 20.0 * np.log10(pa / pb)


def compute():
    """One record per archived round; missing archives are skipped."""
    rows = []
    for tag, place, matched, note in ROUNDS:
        d = load(tag)
        if d is None:
            continue
        fd = attenuation(d["fa"], d["fb"], d["dtf"])
        fe = attenuation(d["ea"], d["eb"], d["dte"])
        rows.append(dict(tag=tag, place=place, matched=matched, note=note,
                         fd=fd, fe=fe, gap=abs(fd - fe),
                         b_fd=direct_peak(d["fb"], d["dtf"])[0]))
    return rows


def window_sensitivity(tag, halves=(0.6e-6, 0.8e-6, 1.0e-6, 1.2e-6,
                                    1.5e-6)):
    """Spread of one round's numbers over plausible arrival windows."""
    d = load(tag)
    if d is None:
        return None
    fd = [attenuation(d["fa"], d["fb"], d["dtf"], h) for h in halves]
    fe = [attenuation(d["ea"], d["eb"], d["dte"], h) for h in halves]
    return max(fd) - min(fd), max(fe) - min(fe)


def report(rows):
    print("direct-arrival attenuation, grain structure against uniform "
          "crystal")
    print("%-5s %-9s %-26s %8s %8s %8s"
          % ("round", "path", "source realisation", "FDTD", "FE", "gap"))
    for r in rows:
        print("%-5s %-9s %-26s %+8.3f %+8.3f %8.3f"
              % (r["tag"], r["place"], r["note"], r["fd"], r["fe"],
                 r["gap"]))

    ok = [r for r in rows if r["matched"]]
    bad = [r for r in rows if not r["matched"]]
    if bad:
        print("\nrounds 1 and 2 are the multipole mismatch, not a "
              "disagreement between discretisations:")
        for r in bad:
            print("  %-4s FE %+.2f dB against FDTD %+.2f dB, %.2f dB out"
                  % (r["tag"], r["fe"], r["fd"], r["gap"]))

    print("\nvalid cross-solver comparisons: %d, over %d placements"
          % (len(ok), len(set(r["place"] for r in ok))))
    if ok:
        gaps = [r["gap"] for r in ok]
        print("  agreement spans %.3f to %.3f dB (%s)"
              % (min(gaps), max(gaps),
                 ", ".join("%s %.3f" % (r["tag"], r["gap"]) for r in ok)))
        louder = all(r["fd"] < r["fe"] for r in ok)
        print("  finite-difference the more attenuated in all %d: %s"
              % (len(ok), "yes" if louder else "NO"))

    # The uniform-crystal reference is placement-invariant, so any change
    # in the A-side number is grain structure and not geometry. This is
    # what makes the second placement worth quoting at all.
    by_place = {}
    for r in rows:
        by_place.setdefault(r["place"], []).append(r)
    places = [p for p in dict.fromkeys(r["place"] for r in rows)]
    if len(places) == 2:
        p, q = places                        # first placement, then second
        bp, bq = by_place[p][0]["b_fd"], by_place[q][0]["b_fd"]
        print("\nplacement check, finite-difference side")
        print("  %s to %s: uniform-crystal direct moves %+.4f dB, "
              "grain attenuation %+.3f dB"
              % (p, q, 20.0 * np.log10(bq / bp),
                 by_place[q][0]["fd"] - by_place[p][-1]["fd"]))
        print("  the reference is placement-invariant and the specimen "
              "number is not, so the placements differ in structure")

    for tag in ("ab3", "ab4", "ab5"):
        s = window_sensitivity(tag)
        if s is not None:
            print("  %s window sensitivity 0.6-1.5 us: FDTD %.4f dB, "
                  "FE %.4f dB" % (tag, s[0], s[1]))


def main():
    report(compute())


if __name__ == "__main__":
    main()

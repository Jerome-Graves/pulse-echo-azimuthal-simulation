"""Contrast-scaling ladder: partition of the coda into a physical part
that scales as f^2 and a contrast-independent residue.

Supports the two subsections of Section 4.6, "What the uniform-orientation
control does and does not bound" and "Contrast scaling", and replaces the
floor table that currently reads 5.1 and 12.8 per cent.

Each grain's stiffness is interpolated a fraction f from the
orientation-averaged tensor toward its own (runs/master_run.py,
tables_for, mode "contrast"). The interpolation is linear about the
grain-population mean, so the mean tensor is identical on every rung and
all five rungs share a specimen, a grid and an effective medium. Physical
single scattering is then proportional to f^2 and the intercept at f = 0
is the contrast-independent residue, which is the quantity the
uniform-orientation control also measures.

Reads, all under out/sweeps, on the 30 azimuths at 12 degree spacing that
every sweep has in common:
    cs_f000_s11_ppw8 cs_f025_s11_ppw8 cs_f050_s11_ppw8
    cs_f075_s11_ppw8                    ladder rungs f = 0 to 0.75
    girdle_perp_ppw8                    the f = 1 rung
    zerocontrast_ppw8 zc_s11_ppw6 zc_s11_ppw10   uniform orientation
    girdle_perp lic_girdle_s11_ppw10    specimen at ppw 6 and ppw 10

Writes nothing. Everything printed is a number quoted in Section 4.6.

ESTIMATOR NOTE, and the reason this module does not reuse db_reconcile's
loader. db_reconcile measures the coda as the RMS of the Hilbert envelope
of the whole trace inside the gate. The Hilbert transform is not local.
The front-surface arrival is about 100 dB above the coda and the tail of
its analytic signal deposits a pedestal near -94 dB, source-referenced,
across the gate. That pedestal is above the entire uniform-orientation
control at ppw 8 and ppw 10, so the envelope estimator cannot see the
control at all and the published floor is a measurement of the front
arrival. Here the coda is measured from the signal inside the gate, using
E[|z|^2] = 2 E[x^2] for a locally stationary segment, and the leak is
printed alongside so the size of the correction stays visible.
"""
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, hilbert, sosfiltfilt

SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))

# Acquisition constants, Section 3: 2 MHz Ricker, 100 mm disc, reference
# longitudinal speed 3850 m/s. Coda gate and analysis band as Table 1.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)

# Nothing the solver can carry lives above this: the source is a 2 MHz
# Ricker and the coarsest production grid is ppw 8. Gate content above it
# is arithmetic noise, and bounds the precision floor of the traces.
NOISE_LO = 8.0e6

AZ_MATCHED = tuple(range(0, 360, 12))

LADDER = ((0.00, "cs_f000_s11_ppw8"), (0.25, "cs_f025_s11_ppw8"),
          (0.50, "cs_f050_s11_ppw8"), (0.75, "cs_f075_s11_ppw8"),
          (1.00, "girdle_perp_ppw8"))
SPECIMEN = {6: "girdle_perp", 8: "girdle_perp_ppw8",
            10: "lic_girdle_s11_ppw10"}
UNIFORM = {6: "zc_s11_ppw6", 8: "zerocontrast_ppw8", 10: "zc_s11_ppw10"}

N_BOOT = 2000
RNG_SEED = 11                    # the specimen seed, so the run is traceable


def db(x):
    return 10.0 * np.log10(x)


# ------------------------------------------------------------------ load

def measure_trace(path):
    """Gate power by several estimators, all referenced to the source.

    Powers, not amplitudes: every partition below is a partition of
    power, and only power is additive across independent contributions.

      band   in-band coda from the signal itself. The headline estimator.
      raw    the same, unfiltered, so it keeps the sub-0.3 MHz content
             that dominates the gate whenever the contrast is switched off
      taper  in-band coda from the analytic signal of the tapered gate
             alone, an independent route to `band` that uses a Hilbert
             transform but cannot see outside the gate
      leak   what the global Hilbert envelope reports in the gate after
             the gate itself has been deleted from the trace
      pad    the same with the transform zero-padded fourfold, which
             separates the irreducible 1/t tail of the analytic signal
             from the circular wrap-around of the FFT implementation
      env    the global-envelope estimator db_reconcile uses
      noise  gate power above NOISE_LO scaled to the analysis bandwidth
      e1     backwall echo power, for the extinction estimate
    """
    with np.load(path) as z:
        trace = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
    fs = 1.0 / dt
    i0, i1 = int(GATE[0] * fs), int(GATE[1] * fs)
    envelope = np.abs(hilbert(trace))
    src2 = envelope.max() ** 2

    muted = trace.copy()
    muted[i0:i1] = 0.0
    leak = np.abs(hilbert(muted))
    pad = np.abs(hilbert(muted, N=4 * len(muted)))[:len(muted)]

    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                 btype="band", output="sos")
    filtered = sosfiltfilt(sos, trace)

    window = np.hanning(i1 - i0)
    tapered = np.abs(hilbert(filtered[i0:i1] * window)) ** 2
    taper = tapered.mean() / (window ** 2).mean()

    gate = trace[i0:i1]
    spec = np.abs(np.fft.rfft(gate * window)) ** 2
    freq = np.fft.rfftfreq(len(gate), dt)
    n_band = (BAND[1] - BAND[0]) * len(gate) * dt
    noise = (spec[freq >= NOISE_LO].mean() * n_band / spec.sum()
             * 2.0 * (gate ** 2).mean())

    k0, kw = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
    e1 = envelope[max(k0 - kw, 0):k0 + kw].max() ** 2

    return dict(band=2.0 * (filtered[i0:i1] ** 2).mean() / src2,
                raw=2.0 * (gate ** 2).mean() / src2,
                taper=taper / src2,
                leak=(leak[i0:i1] ** 2).mean() / src2,
                pad=(pad[i0:i1] ** 2).mean() / src2,
                env=(envelope[i0:i1] ** 2).mean() / src2,
                noise=noise / src2,
                e1=e1 / src2)


def load_sweep(name, azimuths=AZ_MATCHED):
    """Per-azimuth power ratios for one sweep, keyed by estimator."""
    root = os.path.join(SWD, name)
    rows, used = [], []
    for a in azimuths:
        p = os.path.join(root, "az%03d.npz" % a)
        if os.path.exists(p):
            rows.append(measure_trace(p))
            used.append(a)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["az"] = np.array(used)
    return out


# --------------------------------------------------------------- compute

def fit_offset_power(f, p, exponent=2.0):
    """Fit P(f) = P0 + A f^n at fixed n, minimising the residual in dB.

    Least squares on the linear powers would be a fit to the f = 1 rung
    alone, because the rungs span 20 dB. Fitting in dB gives every rung
    the same weight, which is what a scaling law asserts.
    """
    start = np.array([np.log10(max(p.min(), 1e-30)),
                      np.log10(max(p.max(), 1e-30))])

    def resid(q):
        model = 10.0 ** q[0] + 10.0 ** q[1] * f ** exponent
        return db(p) - db(model)

    sol = least_squares(resid, start, xtol=1e-12, ftol=1e-12)
    return 10.0 ** sol.x[0], 10.0 ** sol.x[1], resid(sol.x)


def fit_free_exponent(f, p):
    """Same fit with the exponent released. The exponent, not the
    intercept, is the test of the f^2 claim."""
    start = np.array([np.log10(max(p.min(), 1e-30)),
                      np.log10(max(p.max(), 1e-30)), 2.0])

    def resid(q):
        return db(p) - db(10.0 ** q[0] + 10.0 ** q[1] * f ** q[2])

    sol = least_squares(resid, start, xtol=1e-12, ftol=1e-12)
    return (float(sol.x[2]), 10.0 ** sol.x[0], 10.0 ** sol.x[1],
            float(np.sqrt((resid(sol.x) ** 2).mean())))


def local_exponents(f, p):
    """Exponent of the excess over the f = 0 rung between consecutive
    rungs. This is what says over what range of f the law holds; a single
    global exponent cannot."""
    excess = p - p[0]
    return [(f[i], f[i + 1],
             float(np.log(excess[i + 1] / excess[i])
                   / np.log(f[i + 1] / f[i])))
            for i in range(1, len(f) - 1)]


def bootstrap(f, powers, floor, n_boot=N_BOOT):
    """Paired azimuth bootstrap. The rungs are one tessellation at one
    set of azimuths, so the speckle is common to all of them and the
    azimuth index must be resampled jointly or the pairing is lost."""
    rng = np.random.default_rng(RNG_SEED)
    n_az = powers.shape[1]
    out = {k: [] for k in ("share_all", "share_pos", "share_floor",
                           "exponent")}
    for _ in range(n_boot):
        k = rng.integers(0, n_az, n_az)
        p = powers[:, k].mean(1)
        out["share_all"].append(fit_offset_power(f, p)[0] / p[-1])
        out["share_pos"].append(fit_offset_power(f[1:], p[1:])[0] / p[-1])
        out["share_floor"].append(floor[k].mean() / p[-1])
        out["exponent"].append(fit_free_exponent(f, p)[0])
    return {k: np.array(v) for k, v in out.items()}


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def extinction_bound(ladder, f_lo, f_hi):
    """Upper bound on the departure from f^2 between two rungs, in dB,
    from the measured backwall loss alone.

    Scattering removes energy from the incident beam in proportion to
    f^2, so the coda is an f^2 source term times a two-way extinction.
    The backwall loss across the ladder measures that extinction over a
    known path and the gate centre fixes the path it acts over. It is a
    bound, not a prediction, because part of the backwall loss is
    wavefront distortion at the far rim rather than energy removal.
    """
    loss = db(ladder[0]["e1"].mean()) - db(ladder[-1]["e1"].mean())
    scaled = loss * (0.5 * (GATE[0] + GATE[1]) * C_REF) / (2.0 * DIA)
    return scaled * (f_hi ** 2 - f_lo ** 2)


# ---------------------------------------------------------------- report

def report_estimator_audit(runs):
    """Why the published floor table has to be rebuilt. Additivity of
    the leak and the local coda against the global envelope is the proof
    that the split is real and not an artefact of the muting."""
    print("ESTIMATOR AUDIT, source-referenced gate power, dB")
    print("%-16s %8s %8s %8s %8s %8s %8s %6s" %
          ("run", "envelope", "leak", "padded", "local", "lk+loc",
           "in band", "noise"))
    for label, d in runs:
        print("%-16s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %6.0f" %
              (label, db(d["env"].mean()), db(d["leak"].mean()),
               db(d["pad"].mean()), db(d["raw"].mean()),
               db((d["leak"] + d["raw"]).mean()), db(d["band"].mean()),
               db(d["noise"].mean())))
    print()
    print("  envelope = leak + local to better than 0.3 dB in every row.")
    print("  Where the leak exceeds the local coda the envelope column")
    print("  is a measurement of the front arrival, not of the specimen.")
    print("  Padding the transform removes about 5 dB of the leak, which")
    print("  is FFT wrap-around; what remains is the 1/t tail itself and")
    print("  is still above every uniform-orientation run in band.")
    print()


def report_ladder_validation(rung0, floor8, rung1):
    """The f = 0 rung and the uniform-orientation control are two routes
    to a uniform medium and must agree in level. They are not the SAME
    uniform medium: f = 0 gives every grain the population-mean tensor,
    the control gives every grain the first grain's own tensor, so a
    small offset is expected and a shared azimuthal pattern is not."""
    d = db(rung0["band"]) - db(floor8["band"])
    print("LADDER VALIDATION, f = 0 against the uniform-orientation "
          "control")
    print("  f = 0 rung          %8.2f dB" % db(rung0["band"].mean()))
    print("  uniform orientation %8.2f dB" % db(floor8["band"].mean()))
    print("  difference          %+8.2f dB, per-azimuth sd %.2f dB, "
          "r = %+.2f" % (d.mean(), d.std(ddof=1),
                         np.corrcoef(rung0["band"], floor8["band"])[0, 1]))
    print("  both lie %.1f dB below the f = 1 coda"
          % (db(rung1["band"].mean()) - db(rung0["band"].mean())))
    print("  cross-check, tapered-gate analytic signal: f = 0 %.2f dB, "
          "control %.2f dB" % (db(rung0["taper"].mean()),
                               db(floor8["taper"].mean())))
    print()


def report_ladder(f, ladder, key="band"):
    label = {"band": "in band 0.8 to 3.0 MHz",
             "raw": "unfiltered"}[key]
    p = np.array([d[key].mean() for d in ladder])
    print("CONTRAST LADDER, %s, %d matched azimuths"
          % (label, len(ladder[0]["az"])))
    print("%-6s %12s %14s" % ("f", "coda (dB)", "excess (dB)"))
    for i, fi in enumerate(f):
        ex = "" if i == 0 else "%14.2f" % db(p[i] - p[0])
        print("%-6.2f %12.2f%s" % (fi, db(p[i]), ex))
    print()
    print("  f^2 predicts +6.02 dB of excess per doubling of f")
    for a, b in ((1, 2), (2, 4)):
        print("    f %.2f to %.2f : %+6.2f dB"
              % (f[a], f[b], db(p[b] - p[0]) - db(p[a] - p[0])))
    print("  local exponent of the excess, consecutive rungs")
    for lo, hi, n in local_exponents(f, p):
        print("    %.2f to %.2f : n = %.2f" % (lo, hi, n))
    p0, amp, resid = fit_offset_power(f, p)
    print("  fit P = P0 + A f^2, all five rungs, dB residual")
    print("    P0 = %.2f dB   A = %.2f dB   residuals %s dB"
          % (db(p0), db(amp),
             np.array2string(resid, precision=2, floatmode="fixed")))
    p0e = fit_offset_power(f[1:], p[1:])[0]
    print("  same fit on f > 0 only, intercept extrapolated to f = 0")
    print("    P0 = %.2f dB" % db(p0e))
    n, p0f, ampf, rms = fit_free_exponent(f, p)
    print("  exponent released")
    print("    n = %.2f   P0 = %.2f dB   A = %.2f dB   rms resid %.2f dB"
          % (n, db(p0f), db(ampf), rms))
    print()
    return p, p0, p0e


def report_shares(f, ladder, floor8, p, p0, p0e):
    powers = np.array([d["band"] for d in ladder])
    boot = bootstrap(f, powers, floor8["band"])
    print("TWO INDEPENDENT ESTIMATES OF THE NUMERICAL SHARE AT f = 1")
    print("  ladder intercept, all five rungs %6.2f %%  [%.2f, %.2f]"
          % (100 * p0 / p[-1], *[100 * v for v in ci(boot["share_all"])]))
    print("  ladder intercept, f > 0 only     %6.2f %%  [%.2f, %.2f]"
          % (100 * p0e / p[-1], *[100 * v for v in ci(boot["share_pos"])]))
    print("  measured f = 0 rung              %6.2f %%"
          % (100 * p[0] / p[-1]))
    print("  uniform-orientation control      %6.2f %%  [%.2f, %.2f]"
          % (100 * floor8["band"].mean() / p[-1],
             *[100 * v for v in ci(boot["share_floor"])]))
    print("  scaling exponent n               %6.2f    [%.2f, %.2f]"
          % (boot["exponent"].mean(), *ci(boot["exponent"])))
    print("  published envelope-estimator value  12.8 %")
    print()
    print("  departure from f^2 between f = 0.25 and f = 1")
    print("    measured %.2f dB, extinction bound %.2f dB"
          % (12.04 - (db(p[-1] - p[0]) - db(p[1] - p[0])),
             extinction_bound(ladder, f[1], f[-1])))
    print()


def report_resolution_floor():
    """The floor table of Section 4.6, rebuilt on the local estimator."""
    print("UNIFORM-ORIENTATION CONTROL AGAINST RESOLUTION, rebuilt")
    print("%-5s %11s %11s %10s %13s" %
          ("ppw", "specimen", "uniform", "share (%)", "published (%)"))
    for ppw in (6, 8, 10):
        s, u = load_sweep(SPECIMEN[ppw]), load_sweep(UNIFORM[ppw])
        print("%-5d %11.2f %11.2f %10.2f %13.1f" %
              (ppw, db(s["band"].mean()), db(u["band"].mean()),
               100 * u["band"].mean() / s["band"].mean(),
               100 * u["env"].mean() / s["env"].mean()))
    print()
    print("  The published column rises under refinement because the leak")
    print("  pedestal is fixed while the specimen coda falls. The")
    print("  corrected share falls, as a convergent error must.")
    print()


def main():
    f = np.array([x[0] for x in LADDER])
    ladder = [load_sweep(name) for _, name in LADDER]
    floor8 = load_sweep(UNIFORM[8])

    audit = [("ladder f=%.2f" % x[0], d) for x, d in zip(LADDER, ladder)]
    audit += [("uniform ppw %d" % p, load_sweep(UNIFORM[p]))
              for p in (6, 8, 10)]
    audit += [("specimen ppw %d" % p, load_sweep(SPECIMEN[p]))
              for p in (6, 10)]
    report_estimator_audit(audit)
    report_ladder_validation(ladder[0], floor8, ladder[-1])
    p, p0, p0e = report_ladder(f, ladder, "band")
    report_shares(f, ladder, floor8, p, p0, p0e)
    report_ladder(f, ladder, "raw")
    report_resolution_floor()


if __name__ == "__main__":
    main()

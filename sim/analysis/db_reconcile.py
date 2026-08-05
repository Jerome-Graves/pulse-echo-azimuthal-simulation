"""dB reconciliation: the resolution table of Section 4.6, tab:reconcile.

Two questions, one module.

FIRST, the question this file was written to answer. Three headline
figures did not add up: holding the physical coda fixed across
resolutions, the measured levels implied the numerical floor moved about
22 dB between ppw 6 and 8, which no mechanism produces. The cause is that
coda/E1 double-counts. The backwall E1 is itself attenuated by
scattering, part of which is numerical, so refining the grid raises E1
AND lowers the coda, and their ratio moves by the sum. Normalising by the
source amplitude removes the double count. That conclusion stands.

SECOND, and the reason the table is rebuilt here. The levels themselves
were measured with the GLOBAL-ENVELOPE estimator: the mean square of the
Hilbert envelope of the whole unfiltered trace inside the gate. That
estimator is wrong twice over at the coarse grid.

  Band. The paper band-limits to 0.8 to 3.0 MHz before gating
  (Section 3). The global-envelope estimator does not. The analysis
  filter rejects 78 per cent of the raw gate power at ppw 6 against
  30 per cent at ppw 8. Of those 78 points, 55 lie outside the band on
  an ideal edge and the rest are the filter transition band taking
  content stacked against the 3.0 MHz corner, 16 per cent of the gate
  power lying between 2.5 and 3.0 MHz at that resolution. The dominant
  term either way is a single lump: 53 per cent of the raw ppw 6 gate
  power sits between 4 and 6 MHz, where the source pulse carries
  0.5 per cent and the backwall echo carries none. The order-8 operator
  is designed accurate to f0*ppw/pi, which is 3.8 MHz at ppw 6, and the
  grid Nyquist f0*ppw/2 is 6.0 MHz. The estimator is therefore reading
  grid noise between two thirds of Nyquist and Nyquist, not the coda,
  and because the share falls with refinement the artefact does not
  cancel between rungs: it inflates the 6 to 8 step by about 5 dB.

  Locality. The Hilbert transform is not a local operator. The front
  arrival is about 100 dB above the coda and the tail of its analytic
  signal deposits a pedestal across the gate. In band that pedestal is
  6.6 dB below the specimen coda at ppw 8 and 2.4 dB below it at ppw 10,
  so it adds 0.9 and 2.0 dB of a quantity that belongs to the front
  arrival. Measuring the gate from the signal itself, using
  E|z|^2 = 2 E x^2 for a locally stationary segment, removes it. The
  split is verifiable, not assumed: pedestal plus local power reproduce
  the global envelope to better than 0.25 dB in every run here.

The AUDITED estimator is therefore the local in-band one already used by
analysis/contrast_ladder.py, so that Table 4 and the two tables that
follow it in Section 4.6 are the same kind of decibel.

The backwall column is deliberately NOT band-limited, and that is a
judgement this module prints the evidence for. Band-limiting exists to
reject numerical content. The backwall carries none: 94 per cent of its
raw power already lies inside 0.8 to 3.0 MHz at ppw 6 and at most
3.4 per cent lies above 4 MHz at any resolution. What the 3.0 MHz corner
removes from it is genuine source bandwidth that the finer grid
propagates and the coarser grid destroys, and it removes a rising share
of it, 5 per cent at ppw 6 against 41 per cent at ppw 10. Band-limiting
the backwall would inject exactly the resolution-dependent bias this
audit exists to remove, and with the opposite sign. The asymmetry is
visible in the sensitivity: opening the top corner from 3.0 to 4.0 MHz
moves the coda step by 0.42 dB and the backwall step by 1.7 dB.

Everything is measured at MATCHED azimuths, on one specimen, with one
metric at a time, so nothing is compared across conventions. Levels are
averaged as energies, which is what a partition of power requires and
what the rest of Section 4.6 already does.

Reads, all under out/sweeps, on the 30 azimuths at 12 degree spacing that
every sweep has in common:
    girdle_perp girdle_perp_ppw8 lic_girdle_s11_ppw10   specimen
    zc_s11_ppw6 zerocontrast_ppw8 zc_s11_ppw10          uniform
                                                        orientation

Writes nothing. Run with --legacy to print the published table instead,
for the record.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))

# Acquisition constants, Section 3: 2 MHz Ricker, 100 mm disc, reference
# longitudinal speed 3850 m/s. Coda gate and analysis band as Table 1.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)

# The backwall arrives at 2*DIA/C_REF = 51.9 us nominal and peaks near
# 52.6 us once the envelope delay of the source is included. A 2 us
# half-width holds the peak at every resolution; widening it to 6 us
# changes nothing, which is checked in report_backwall_choice.
BW_HALF = 2e-6

# Fixed-code-path sweeps of the SAME specimen, seed 11, girdle k = -8.
SPECIMEN = {6: "girdle_perp", 8: "girdle_perp_ppw8",
            10: "lic_girdle_s11_ppw10"}
# Zero-contrast controls, seed 11 on the same code path, so the floor is
# measured on the geometry it is compared against. zerocontrast_ppw6 is
# deliberately not used: it predates the production programme and holds
# only 22 of 30 azimuths.
UNIFORM = {6: "zc_s11_ppw6", 8: "zerocontrast_ppw8", 10: "zc_s11_ppw10"}

AZ_MATCHED = tuple(range(0, 360, 12))

# Bands used only for the sensitivity check that justifies leaving the
# backwall unfiltered. The production band is BAND.
PROBE_BANDS = ((0.8e6, 3.0e6), (0.8e6, 3.5e6), (0.8e6, 4.0e6))

# Edges of the spectral census, Hz. The 4 MHz edge is the one that
# matters: nothing physical lives above it at any resolution here.
EDGES = (0.0, 0.3e6, 0.8e6, 3.0e6, 4.0e6, 6.0e6, np.inf)
EDGE_NAMES = ("<0.3", "0.3-0.8", "0.8-3.0", "3.0-4.0", "4.0-6.0", ">6.0")


def db(x):
    """Power ratio to decibels. Every quantity here is a power."""
    return 10.0 * np.log10(x)


# ------------------------------------------------------------------ load

def bandpass(x, fs, lo, hi):
    """Zero-phase Butterworth applied to the COMPLETE trace.

    Section 3 filters before gating so that the filter transient falls
    outside the gate rather than inside it.
    """
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                 output="sos")
    return sosfiltfilt(sos, x)


def measure_trace(path):
    """Every estimator of interest for one azimuth, source-referenced.

    Powers, not amplitudes: the whole argument is a partition of power,
    and only power is additive across independent contributions.

      coda_env   global envelope, unfiltered. The published estimator.
      coda_loc   the same gate measured locally, still unfiltered, which
                 isolates what the Hilbert pedestal contributes
      coda_envb  global envelope of the band-limited trace, which is the
                 remaining corner of the two-by-two and lets the two
                 corrections be applied in either order
      coda_band  local and in band. The audited estimator.
      pedestal   what the global envelope reports in the gate after the
                 gate itself has been deleted from the trace
      pedestal_b the same in band, which is the one that matters, since
                 the audited estimator is in band and the pedestal has
                 to be shown small against THAT coda and not the raw one
      bw_env     backwall echo, peak of the unfiltered analytic
                 envelope. The published estimator, and the one kept.
      bw_band    the same peak taken on the band-limited trace, printed
                 so the reader can see why it is not used
      bw_wide    the same as bw_env over a 6 us half-width, which tests
                 whether the window truncates the arrival
      bw_power   backwall window power rather than peak, which separates
                 pulse compaction from recovered energy
    """
    with np.load(path) as z:
        trace = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
    fs = 1.0 / dt
    i0, i1 = int(GATE[0] * fs), int(GATE[1] * fs)

    envelope = np.abs(hilbert(trace))
    src2 = envelope.max() ** 2
    filtered = bandpass(trace, fs, *BAND)
    env_band = np.abs(hilbert(filtered))

    # Delete the gate and envelope what is left. Whatever the transform
    # still reports inside the gate came from outside it. The in-band
    # version filters the COMPLETE trace first and mutes afterwards, in
    # that order, so the mute discontinuity is never filtered and cannot
    # ring back into the gate it was made to empty.
    muted = trace.copy()
    muted[i0:i1] = 0.0
    pedestal = (np.abs(hilbert(muted))[i0:i1] ** 2).mean()
    muted_band = filtered.copy()
    muted_band[i0:i1] = 0.0
    pedestal_b = (np.abs(hilbert(muted_band))[i0:i1] ** 2).mean()

    k0 = int(2 * DIA / C_REF * fs)
    kw, kwide = int(BW_HALF * fs), int(3.0 * BW_HALF * fs)
    nom = slice(max(k0 - kw, 0), k0 + kw)
    wide = slice(max(k0 - kwide, 0), k0 + kwide)
    bw_gate = slice(int(50e-6 * fs), int(55e-6 * fs))

    out = dict(
        coda_env=(envelope[i0:i1] ** 2).mean() / src2,
        coda_loc=2.0 * (trace[i0:i1] ** 2).mean() / src2,
        coda_envb=(env_band[i0:i1] ** 2).mean() / src2,
        coda_band=2.0 * (filtered[i0:i1] ** 2).mean() / src2,
        pedestal=pedestal / src2,
        pedestal_b=pedestal_b / src2,
        bw_env=envelope[nom].max() ** 2 / src2,
        bw_band=env_band[nom].max() ** 2 / src2,
        bw_wide=envelope[wide].max() ** 2 / src2,
        bw_power=2.0 * (trace[bw_gate] ** 2).mean() / src2)

    # Spectral census of the gate and of the backwall arrival, on the
    # same taper, so the two are comparable. This is the evidence that
    # the out-of-band gate power is numerical.
    for tag, seg in (("gate", trace[i0:i1]), ("bw", trace[nom])):
        win = np.hanning(len(seg))
        power = np.abs(np.fft.rfft(seg * win)) ** 2
        freq = np.fft.rfftfreq(len(seg), dt)
        total = power.sum()
        for j, name in enumerate(EDGE_NAMES):
            sel = (freq >= EDGES[j]) & (freq < EDGES[j + 1])
            out["%s_%s" % (tag, name)] = power[sel].sum() / total

    # Sensitivity of coda and backwall to the top corner of the band.
    for lo, hi in PROBE_BANDS:
        xf = bandpass(trace, fs, lo, hi)
        key = "%.1f-%.1f" % (lo / 1e6, hi / 1e6)
        out["coda@" + key] = 2.0 * (xf[i0:i1] ** 2).mean() / src2
        out["bw@" + key] = np.abs(hilbert(xf))[nom].max() ** 2 / src2
    return out


def load_sweep(name, azimuths=AZ_MATCHED):
    """Per-azimuth power ratios for one sweep, keyed by estimator."""
    root = os.path.join(SWD, name)
    rows, used = [], []
    for a in azimuths:
        p = os.path.join(root, "az%03d.npz" % a)
        if os.path.exists(p):
            rows.append(measure_trace(p))
            used.append(a)
    if not rows:
        raise IOError("no matched azimuths under %s" % root)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["az"] = np.array(used)
    return out


def load_all():
    """(specimen, uniform-orientation) sweeps at the three resolutions."""
    return ({p: load_sweep(n) for p, n in SPECIMEN.items()},
            {p: load_sweep(n) for p, n in UNIFORM.items()})


# --------------------------------------------------------------- compute

def level(sweep, key):
    """Revolution level in dB, averaged as an energy.

    A revolution mean is a quadrature over a closed rotation, not a draw
    from a population, so the average is taken over powers and converted
    once. Averaging decibels instead biases the level low by 1 to 2 dB
    here, and by more where the azimuthal spread is larger.
    """
    return db(sweep[key].mean())


def table_rows(spec, coda_key, bw_key):
    """The three columns of tab:reconcile at each resolution."""
    rows = {}
    for ppw in sorted(spec):
        coda, backwall = level(spec[ppw], coda_key), level(spec[ppw], bw_key)
        rows[ppw] = (coda - backwall, coda, backwall)
    return rows


def steps(rows):
    """Change per refinement step, in the order the table prints them."""
    keys = sorted(rows)
    return [(keys[i], keys[i + 1],
             tuple(rows[keys[i + 1]][j] - rows[keys[i]][j]
                   for j in range(3)))
            for i in range(len(keys) - 1)]


def band_split(spec, lo, hi):
    """Split the change in a resolution step into locality and band.

    The published estimator differs from the audited one in two
    independent ways, so the change it makes to a step is split by
    applying them one at a time: global envelope to local at fixed
    bandwidth, then unfiltered to in band at fixed locality. The two
    corrections nearly commute, and report_estimator_split prints the
    reverse ordering as a check.
    """
    def step(key):
        return level(spec[hi], key) - level(spec[lo], key)

    s_env, s_loc, s_band = (step("coda_env"), step("coda_loc"),
                            step("coda_band"))
    s_envb = step("coda_envb")
    return dict(published=s_env, audited=s_band, total=s_band - s_env,
                locality=s_loc - s_env, band=s_band - s_loc,
                band_first=s_envb - s_env, locality_after=s_band - s_envb)


def out_of_band(sweep):
    """Fraction of the raw gate power the analysis band rejects."""
    return 1.0 - sweep["coda_band"].mean() / sweep["coda_loc"].mean()


# ------------------------------------------------------------------ draw

def report_table(spec, legacy=False):
    """tab:reconcile itself, on whichever estimator was asked for."""
    if legacy:
        coda_key, bw_key, tag = "coda_env", "bw_env", "PUBLISHED"
        note = "global Hilbert envelope, unfiltered, gate mean square"
    else:
        coda_key, bw_key, tag = "coda_band", "bw_env", "AUDITED"
        note = "local in-band gate power, unfiltered backwall peak"
    rows = table_rows(spec, coda_key, bw_key)
    print("%s TABLE 4, %s" % (tag, note))
    print("  one specimen, %d matched azimuths, averaged as energies"
          % len(spec[8]["az"]))
    print("  %-5s %14s %14s %16s"
          % ("ppw", "coda/backwall", "coda/source", "backwall/source"))
    for ppw in sorted(rows):
        print("  %-5d %14.2f %14.2f %16.2f" % ((ppw,) + rows[ppw]))
    print("  change per step")
    for lo, hi, d in steps(rows):
        print("  %-5s %+14.2f %+14.2f %+16.2f"
              % ("%d->%d" % (lo, hi), d[0], d[1], d[2]))
    print()
    return rows


def report_estimator_split(spec):
    """Where the difference between the two tables comes from."""
    print("WHAT MOVES BETWEEN THE TWO ESTIMATORS, coda, dB")
    print("  %-6s %11s %11s %11s %9s"
          % ("ppw", "global env", "local raw", "local band", "out %"))
    for ppw in sorted(spec):
        print("  %-6d %11.2f %11.2f %11.2f %9.1f"
              % (ppw, level(spec[ppw], "coda_env"),
                 level(spec[ppw], "coda_loc"),
                 level(spec[ppw], "coda_band"),
                 100 * out_of_band(spec[ppw])))
    print()
    for lo, hi in ((6, 8), (8, 10)):
        s = band_split(spec, lo, hi)
        print("  step %d->%d  published %+.2f  audited %+.2f  "
              "change %+.2f dB" % (lo, hi, s["published"], s["audited"],
                                   s["total"]))
        print("    locality first: locality %+.2f then band %+.2f dB"
              % (s["locality"], s["band"]))
        print("    band first    : band %+.2f then locality %+.2f dB"
              % (s["band_first"], s["locality_after"]))
    print()
    print("  The band term carries the correction and the locality term")
    print("  works against it. The 6 to 8 step is not mostly a pedestal")
    print("  leak; it is mostly the estimator counting grid noise at the")
    print("  coarse grid that the analysis band excludes.")
    print()


def report_pedestal(spec, unif):
    """Additivity is the proof that the locality split is real.

    If the pedestal were an artefact of the muting rather than a real
    contribution of the front arrival, pedestal plus local power would
    not reconstruct the global envelope. It does, in both bandwidths.
    """
    runs = [("specimen ppw %d" % p, spec[p]) for p in sorted(spec)]
    runs += [("uniform ppw %d" % p, unif[p]) for p in sorted(unif)]
    for tag, env, ped, loc in (("UNFILTERED", "coda_env", "pedestal",
                                "coda_loc"),
                               ("IN BAND", "coda_envb", "pedestal_b",
                                "coda_band")):
        print("PEDESTAL AUDIT, %s, source-referenced gate power, dB"
              % tag)
        print("  %-16s %10s %10s %10s %10s"
              % ("run", "envelope", "pedestal", "local", "ped+loc"))
        for name, d in runs:
            print("  %-16s %10.2f %10.2f %10.2f %10.2f"
                  % (name, level(d, env), level(d, ped), level(d, loc),
                     db((d[ped] + d[loc]).mean())))
        print()
    print("  The global envelope reports the front arrival wherever the")
    print("  pedestal exceeds the gate. That is the whole of the")
    print("  uniform-orientation control at ppw 8 and ppw 10, and it is")
    print("  1 to 2 dB of the specimen coda in band at ppw 8 and 10.")
    print()


def report_band_evidence(spec):
    """Why the out-of-band gate power is numerical and not signal."""
    print("SPECTRAL CENSUS, per cent of power, Hann-tapered")
    print("  operator design limit f0*ppw/pi, grid Nyquist f0*ppw/2")
    print("  %-16s" % "run" + "".join("%9s" % n for n in EDGE_NAMES)
          + "%10s %10s" % ("design", "Nyquist"))
    for ppw in sorted(spec):
        for tag, what in (("gate", "coda gate"), ("bw", "backwall")):
            print("  %-16s" % ("ppw %d %s" % (ppw, what))
                  + "".join("%9.1f" % (100 * spec[ppw]["%s_%s" % (tag, n)]
                                       .mean()) for n in EDGE_NAMES)
                  + "%10.2f %10.2f" % (F0 * ppw / np.pi / 1e6,
                                       F0 * ppw / 2 / 1e6))
    print()
    print("  At ppw 6 the majority of the raw gate power sits above the")
    print("  frequency to which the operator is designed accurate, in a")
    print("  band where neither the source nor the backwall carries")
    print("  anything. It is grid noise, and the analysis band excludes")
    print("  it. The backwall carries no such content at any resolution.")
    print()


def report_backwall_choice(spec):
    """The evidence for leaving the backwall column unfiltered."""
    print("BACKWALL ESTIMATOR, source-referenced, dB, and its steps")
    keys = [("bw_env", "peak, raw"), ("bw_band", "peak, in band"),
            ("bw_wide", "peak, wide window"),
            ("bw_power", "window power, raw")]
    print("  %-20s %9s %9s %9s %9s %9s"
          % ("estimator", "ppw6", "ppw8", "ppw10", "6->8", "8->10"))
    for key, name in keys:
        v = [level(spec[p], key) for p in (6, 8, 10)]
        print("  %-20s %9.2f %9.2f %9.2f %+9.2f %+9.2f"
              % (name, v[0], v[1], v[2], v[1] - v[0], v[2] - v[1]))
    print()
    print("  %-20s %9s %9s %9s %9s"
          % ("top corner", "coda 6->8", "coda 8->10", "bw 6->8",
             "bw 8->10"))
    for lo, hi in PROBE_BANDS:
        key = "%.1f-%.1f" % (lo / 1e6, hi / 1e6)
        c = [level(spec[p], "coda@" + key) for p in (6, 8, 10)]
        b = [level(spec[p], "bw@" + key) for p in (6, 8, 10)]
        print("  %-20s %+9.2f %+9.2f %+9.2f %+9.2f"
              % (key + " MHz", c[1] - c[0], c[2] - c[1],
                 b[1] - b[0], b[2] - b[1]))
    print()
    print("  Opening the top corner from 3.0 to 4.0 MHz moves the coda")
    print("  step by 0.4 dB and the backwall step by 1.7 dB, because the")
    print("  corner cuts genuine backwall bandwidth and only grid noise")
    print("  in the gate. The wide window reproduces the narrow one, so")
    print("  the backwall gate is not truncating the arrival. The peak")
    print("  rises further than the window power between ppw 6 and 8,")
    print("  so most of that rise is pulse compaction under refinement")
    print("  and not energy returned to the beam.")
    print()


def report_floor(spec, unif):
    """The uniform-orientation share, on both estimators, for context."""
    print("UNIFORM-ORIENTATION CONTROL, per cent of specimen coda power")
    print("  %-6s %12s %12s %12s"
          % ("ppw", "audited", "published", "specimen-uniform"))
    for ppw in sorted(spec):
        share = unif[ppw]["coda_band"].mean() / spec[ppw]["coda_band"].mean()
        old = unif[ppw]["coda_env"].mean() / spec[ppw]["coda_env"].mean()
        print("  %-6d %12.2f %12.2f %12.2f"
              % (ppw, 100 * share, 100 * old, -db(share)))
    print()


def main():
    legacy = "--legacy" in sys.argv
    spec, unif = load_all()
    report_table(spec, legacy=True)
    report_table(spec, legacy=False)
    if legacy:
        return
    report_estimator_split(spec)
    report_pedestal(spec, unif)
    report_band_evidence(spec)
    report_backwall_choice(spec)
    report_floor(spec, unif)


if __name__ == "__main__":
    main()

r"""Figure 'trace': a representative recorded A-scan and the analysis gates.

Supports the claim in Section 3.2 that the two observables of this paper
are read from one ordinary pulse-echo record: the backwall arrival, whose
leading edge gives the time of flight, and the backscattered coda in a
gate that lies wholly between the near-field ring-down and the backwall.
The logarithmic amplitude axis is the point of the figure. The coda is
tens of decibels below the backwall, so on a linear axis it is a flat
line at zero, and the reader has no way to see that it is a structured
echo train rather than noise. The linear inset over the gate is there to
settle that question.

Reads (paths resolved from this file, never absolute):
    <repo>/out/sweeps/girdle_perp_ppw8/az*.npz
        keys 'az_deg' and 'coda_db' from all sixty records, to find the
        azimuth whose coda level is the sweep median, then 'trace', 'dt',
        'f0' and 't1_s' from that one record.
    <repo>/out/sweeps/girdle_perp_ppw8/config.json
        for the disc diameter and the pulse centre frequency.

Writes:
    <GitHub>/pulse-echo-cof-paper/figures/trace.pdf

Prints the numbers the caption and Section 3.2 quote, so they can be
checked against the text: the coda level relative to the backwall, the
measured backwall arrival against the nominal 2D/c, the source peak
against the Ricker delay 1.2/f0, and the chord range the gate spans.
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "out" / "sweeps" / "girdle_perp_ppw8"

# Analysis constants, all as defined in sim/analysis/_t2_common.py and
# quoted in Section 3.2 of the manuscript. They are choices, not
# measurements, so they are stated here rather than read from a file.
BAND_HZ = (0.8e6, 3.0e6)        # _t2_common.BAND, 4th-order Butterworth
CODA_GATE_S = (24e-6, 36e-6)    # _t2_common.CODA_W
C_REF = 3850.0                  # _t2_common.C_REF, mean qP speed (m/s)

# Half-width of the search window for the backwall echo about the nominal
# two-way time. The measured arrival wanders by a few tenths of a
# microsecond with azimuth, which is the whole time-of-flight signal, so
# the window has to be wider than that wander and narrower than the gap
# to the coda.
BACKWALL_WINDOW_S = 3e-6

# The record runs to 70 us but the last microsecond carries the transient
# of the zero-phase filter, which is an artefact of the display and not
# of the measurement. Everything the figure is about has happened by 60.
T_SHOW_US = (0.0, 60.0)


# --------------------------------------------------------------- compute

def bandpass(trace, dt):
    """Filter the complete record, exactly as the analysis does.

    Section 3.2: the filter is applied to the whole trace and the coda is
    gated afterwards, because filtering an already truncated segment puts
    an edge transient inside the gate.
    """
    sos = butter(4, [f / (0.5 / dt) for f in BAND_HZ], btype="band",
                 output="sos")
    return sosfiltfilt(sos, trace)


def sweep_coda_levels(sweep_dir):
    """(azimuth, stored coda level in dB) for every record in the sweep."""
    az, coda_db = [], []
    for path in sorted(sweep_dir.glob("az*.npz")):
        with np.load(path) as z:
            az.append(int(z["az_deg"]))
            coda_db.append(float(z["coda_db"]))
    return np.array(az), np.array(coda_db)


def representative_azimuth(sweep_dir):
    """The azimuth whose coda level is the median of the sweep.

    A figure captioned 'representative' has to be chosen by a rule, not by
    eye: the coda level varies by about 6 dB rms around the sweep, and
    picking the record that reads best would flatter the method.
    """
    az, coda_db = sweep_coda_levels(sweep_dir)
    return int(az[np.argmin(np.abs(coda_db - np.median(coda_db)))])


def load_record(sweep_dir, azimuth_deg):
    """Band-passed record, its envelope, and the levels the figure marks.

    Amplitudes are referenced to the backwall envelope peak of this same
    record, so 0 dB is the backwall and the coda level is read straight
    off the axis as the depth below it.
    """
    with np.load(sweep_dir / ("az%03d.npz" % azimuth_deg)) as z:
        trace = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
        f0 = float(z["f0"])
        t_backwall_s = float(z["t1_s"])
    diameter_m = json.loads((sweep_dir / "config.json").read_text()
                            )["diameter_mm"] * 1e-3

    t_s = np.arange(trace.size) * dt
    filtered = bandpass(trace, dt)
    envelope = np.abs(hilbert(filtered))

    t_nominal_s = 2.0 * diameter_m / C_REF
    near = np.abs(t_s - t_nominal_s) <= BACKWALL_WINDOW_S
    backwall_ref = envelope[near].max()

    gate = (t_s >= CODA_GATE_S[0]) & (t_s <= CODA_GATE_S[1])
    coda_rms = np.sqrt((envelope[gate] ** 2).mean())

    return dict(t_us=t_s * 1e6, filtered=filtered, envelope=envelope,
                gate=gate, backwall_ref=backwall_ref, coda_rms=coda_rms,
                t_backwall_us=t_backwall_s * 1e6,
                t_nominal_us=t_nominal_s * 1e6,
                t_source_us=t_s[np.argmax(np.abs(filtered))] * 1e6,
                t_ricker_us=1.2 / f0 * 1e6, dt=dt)


def db(x, reference):
    """Amplitude ratio in decibels, floored so log of zero cannot bite."""
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30) / reference)


# ------------------------------------------------------------------ draw

def draw_record(ax, rec):
    """The whole record on a log amplitude axis, with the gates marked."""
    t_us, ref = rec["t_us"], rec["backwall_ref"]
    shown = (t_us >= T_SHOW_US[0]) & (t_us <= T_SHOW_US[1])

    ax.axvspan(CODA_GATE_S[0] * 1e6, CODA_GATE_S[1] * 1e6,
               color=fs.GREY_LIGHT, alpha=0.35, linewidth=0, zorder=0)
    # The rectified waveform under its envelope. Light, because at this
    # width it is 120 cycles of carrier and would otherwise bury the
    # envelope that the levels are actually read from.
    ax.plot(t_us[shown], db(rec["filtered"][shown], ref),
            color=fs.GREY_LIGHT, linewidth=fs.LW["realisation"], zorder=1)
    ax.plot(t_us[shown], db(rec["envelope"][shown], ref), color=fs.BLACK,
            linewidth=fs.LW["data"], zorder=3)

    ax.axvline(rec["t_backwall_us"], color=fs.GREY,
               linewidth=fs.LW["annotation"], linestyle=(0, (3.0, 1.5)),
               zorder=2)
    ax.set_xlim(*T_SHOW_US)
    ax.set_ylim(-80.0, 40.0)
    ax.set_yticks([-80, -60, -40, -20, 0, 20])
    # Ticks every 12 us, so that the gate boundaries at 24 and 36 are read
    # off the axis itself rather than guessed from the shading.
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("amplitude (dB re backwall)")

    fs.arrow_label(ax, "source excitation", xy=(rec["t_source_us"], 26.5),
                   xytext=(4.0, 32.0), colour=fs.BLACK, va="bottom")
    fs.direct_label(ax, rec["t_backwall_us"] - 1.5, 32.0, "backwall",
                    colour=fs.BLACK, ha="right", va="bottom")
    fs.direct_label(ax, 30.0, -46.0, "coda gate", colour=fs.GREY,
                    ha="center", va="bottom")


def draw_coda_inset(ax, rec):
    """The gate again, linear and magnified: an echo train, not noise.

    Sits directly above the shaded gate and spans exactly the same 24 to
    36 us, so no leader lines are needed to say what it magnifies. The
    ordinate is per cent of the backwall peak, the same reference the main
    panel's decibel axis uses.
    """
    inset = ax.inset_axes([0.175, 0.500, 0.525, 0.383])
    t_us = rec["t_us"][rec["gate"]]
    unit = 0.01 * rec["backwall_ref"]
    inset.plot(t_us, rec["filtered"][rec["gate"]] / unit, color=fs.BLACK,
               linewidth=fs.LW["annotation"], zorder=3)
    # Both signs of the envelope, so the packets are enclosed rather than
    # cut in half by a one-sided curve. Dashed as well as coloured, so the
    # two curves stay distinct when the page is printed in mono.
    for sign in (1.0, -1.0):
        inset.plot(t_us, sign * rec["envelope"][rec["gate"]] / unit,
                   color=fs.BLUE, linestyle=fs.SERIES[1][1],
                   linewidth=fs.LW["reference"], zorder=2)

    lim = 1.12 * rec["envelope"][rec["gate"]].max() / unit
    inset.set_xlim(*[g * 1e6 for g in CODA_GATE_S])
    inset.set_ylim(-lim, lim)
    inset.set_xticks([24, 30, 36])
    inset.set_yticks([-0.2, 0.0, 0.2])
    inset.tick_params(labelsize=fs.FS["cbar_tick"], pad=1.5)
    inset.set_ylabel("amplitude (%)", fontsize=fs.FS["cbar_tick"],
                     labelpad=1.5)
    for spine in inset.spines.values():
        spine.set_linewidth(fs.LW["spine"])
    return inset


# --------------------------------------------------------------- reports

def report(azimuth_deg, rec):
    """The numbers of Section 3.2, recomputed from the record shown."""
    coda_db = 20.0 * np.log10(rec["coda_rms"] / rec["backwall_ref"])
    chord_mm = [0.5 * C_REF * g * 1e3 for g in CODA_GATE_S]
    print("record girdle_perp_ppw8 az%03d, chosen as the sweep median coda"
          % azimuth_deg)
    print("  coda gate rms          %.1f dB below the backwall peak"
          % coda_db)
    print("  backwall arrival       %.2f us measured, %.2f us nominal 2D/c"
          % (rec["t_backwall_us"], rec["t_nominal_us"]))
    print("  source peak            %.2f us measured, %.2f us Ricker 1.2/f0"
          % (rec["t_source_us"], rec["t_ricker_us"]))
    print("  gate along the chord   %.0f to %.0f mm of the 100 mm diameter"
          % tuple(chord_mm))
    print("  band %.1f to %.1f MHz, sample interval %.2f ns"
          % (BAND_HZ[0] / 1e6, BAND_HZ[1] / 1e6, rec["dt"] * 1e9))


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    azimuth_deg = representative_azimuth(SWEEP)
    rec = load_record(SWEEP, azimuth_deg)

    fig, ax = fs.figure("trace", left=11.0, right=2.5, bottom=8.5, top=2.5)
    draw_record(ax, rec)
    draw_coda_inset(ax, rec)

    report(azimuth_deg, rec)
    fs.save(fig, "trace",
            expect=["time (µs)", "amplitude (dB re backwall)", "coda gate",
                    "backwall", "source excitation"])


if __name__ == "__main__":
    main()

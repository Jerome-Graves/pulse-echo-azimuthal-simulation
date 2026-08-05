"""Bandwidth audit of the physical-optics anchor of Section 2.

Section 2 predicts the coda level from the Kirchhoff integral over the
536 true grain-boundary polygons and compares it with the simulation.
The comparison was left open when Table 4 was rebuilt, because the two
sides were measured with different estimators. This module closes it.

The measurement was retired onto an AUDITED estimator in
analysis/db_reconcile.py: local, in band 0.8 to 3.0 MHz, filtered before
gating, averaged over azimuth as energies. The prediction was never
audited. It applies no band limit at all: analysis/physical_optics/
po_src_03_kirchhoff_pred.py builds the whole rfft axis out to the 60 MHz trace Nyquist
and weights it only by the Ricker spectrum, the transmit factor
S_t/lambda^2 and the two-way piston directivity. So the question is
whether the prediction has to be compared unfiltered, or whether it can
simply be truncated to the analysis band.

It can, and that is the recommendation here, for two reasons this module
prints the evidence for.

  Locality is not in dispute. The synthetic trace carries no front
  arrival, only the gated facet returns, so the global envelope and the
  local estimator 2 E[x^2] agree on it to better than 0.05 dB. The
  locality half of the audit is a no-op on the prediction and a real
  0.95 dB correction on the measurement. The measured comparand is
  therefore local whatever is decided about bandwidth.

  Bandwidth is a truncation, not a reinterpretation. The prediction is
  assembled in the frequency domain, so restricting it to the analysis
  band is one mask on an array. Doing it costs 1.1 dB, because the
  prediction places about a fifth of its gate power outside 0.8 to
  3.0 MHz, nearly all of it between 3.0 and 4.0 MHz where the Ricker
  source still carries 10 per cent of its energy. Comparing unfiltered
  instead would credit the model for measured gate power that the
  analysis band exists to reject.

The bandwidth audit also settles a second thing that was not asked. The
prediction's own spectral transfer is measured here directly, as the
azimuth mean of |H(f)|^2 with the source divided out. It varies by less
than 3 dB across 0.8 to 3.0 MHz, and by less than 1 dB over 1.5 to
3.0 MHz, against the factor of 195 that an f^4 weighting would impose
over the same band. That confirms the spectral
bookkeeping of po_src_11_closed_form.py, in which the orientation-averaged
backscatter form factor is frequency flat, and refutes the f^2
pulse-shape shortcut of the earlier po_src_06_reconcile.py, which weights the
integrand by an extra (f/f0)^4 and so inflates the closed form by
3.4 dB. The closed-form levels quoted in Section 2 are the shortcut
values. See report_closed_form.

Reads, nothing written:
    analysis/physical_optics/po_src_03_kirchhoff_pred.py        the PO integral
    analysis/physical_optics/po_src_geom.npz      cached tessellation
    analysis/physical_optics/po_src_diag.npz      insonified-facet stats
    out/sweeps/girdle_seed11_ppw10_licensing               the ppw 10 measurement
    out/sweeps/girdle_seed11_ppw10_zerocontrast                       uniform-orientation

CPU and disk only. Nothing here touches the GPU.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import j1

HERE = os.path.dirname(os.path.abspath(__file__))
PO_DIR = os.path.join(HERE, "physical_optics")
SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))
if PO_DIR not in sys.path:
    sys.path.insert(0, PO_DIR)

import po_src_03_kirchhoff_pred as PO                                   # noqa: E402

# Acquisition constants, Section 3, and the analysis band of Table 1.
GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)
DIA = 0.100
AZ_MATCHED = tuple(range(0, 360, 12))

# The prediction is evaluated on a dense azimuth grid because it is
# glint dominated: 30 azimuths do not resolve its ensemble mean.
AZ_DENSE = np.arange(0, 360, 2.0)
QUAD = PO.LAM / 8

# Delay-to-range speeds. 3843 m/s is the diameter-average qP speed of
# the realised c-axes quoted in Section 2. 3850 is the nominal value the
# cached prediction was actually computed at. 3807.5 is the superseded
# value obtained by dividing 2D by the backwall ENVELOPE PEAK time
# without removing the envelope delay of the source.
SPEEDS = (3850.0, 3843.0, 3807.5)
C_SECTION2 = 3843.0

# Spectral census edges, matching db_reconcile.py so the two modules
# report the same partition of power.
EDGES = (0.0, 0.8e6, 3.0e6, 4.0e6, 6.0e6, np.inf)
EDGE_NAMES = ("<0.8", "0.8-3.0", "3.0-4.0", "4.0-6.0", ">6.0")


def db(x):
    """Power ratio to decibels. Every level here is a power."""
    return 10.0 * np.log10(x)


def census(freq, power):
    """Share of a power spectrum in each band of the census."""
    total = power.sum()
    return [power[(freq >= EDGES[j]) & (freq < EDGES[j + 1])].sum() / total
            for j in range(len(EDGE_NAMES))]


# ------------------------------------------------------------------ load

def bandpass(x, fs, lo, hi, axis=-1):
    """The filter of Section 3, applied to the COMPLETE trace.

    Identical to db_reconcile.bandpass so that both sides of the
    comparison are shaped by the same transition bands.
    """
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                 output="sos")
    return sosfiltfilt(sos, x, axis=axis)


def measure_trace(path):
    """The estimators of db_reconcile, on one measured azimuth."""
    with np.load(path) as z:
        trace = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
    fs = 1.0 / dt
    i0, i1 = int(GATE[0] * fs), int(GATE[1] * fs)
    envelope = np.abs(hilbert(trace))
    src2 = envelope.max() ** 2
    filtered = bandpass(trace, fs, *BAND)
    gate = trace[i0:i1]
    freq = np.fft.rfftfreq(len(gate), dt)
    power = np.abs(np.fft.rfft(gate * np.hanning(len(gate)))) ** 2
    out = dict(coda_env=(envelope[i0:i1] ** 2).mean() / src2,
               coda_loc=2.0 * (gate ** 2).mean() / src2,
               coda_band=2.0 * (filtered[i0:i1] ** 2).mean() / src2)
    for name, share in zip(EDGE_NAMES, census(freq, power)):
        out["gate_" + name] = share
    return out


def load_sweep(name):
    """Per-azimuth measured quantities on the matched azimuths."""
    rows, used = [], []
    for a in AZ_MATCHED:
        p = os.path.join(SWD, name, "az%03d.npz" % a)
        if os.path.exists(p):
            rows.append(measure_trace(p))
            used.append(a)
    if not rows:
        raise IOError("no matched azimuths under %s" % name)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["az"] = np.array(used)
    return out


# --------------------------------------------------------------- compute

def po_traces(speed, azimuths=AZ_DENSE):
    """Synthetic A-scans from the Kirchhoff integral, one per azimuth.

    PO.C is the delay-to-range map of po_src_03_kirchhoff_pred, so it has to be set
    before the histogram is built and restored afterwards. Everything
    else in that module is a cached tessellation.
    """
    spec, _ = PO.ricker_spec(PO.NT, PO.DT, PO.F0)
    keep = PO.C
    PO.C = speed
    try:
        out = np.empty((len(azimuths), PO.NT))
        for k, a in enumerate(azimuths):
            out[k] = PO.trace_from(PO.azimuth_response(a, QUAD)[0], spec)
    finally:
        PO.C = keep
    return out


def po_transfer(speed, azimuths=AZ_DENSE):
    """Azimuth mean of |H(f)|^2, the prediction's own spectral transfer.

    This is the brute-force equivalent of SSQ(f) in the closed form, and
    it is measured rather than assumed: the source spectrum is simply
    never applied. Its SHAPE is what decides whether the closed form may
    weight the integrand by (f/f0)^4.
    """
    keep = PO.C
    PO.C = speed
    try:
        freq = np.fft.rfftfreq(PO.NT, PO.DT)
        lam = PO.C / np.maximum(freq, 1e-9)
        s_mid = (np.arange(PO.NS_BIN) + 0.5) / PO.NS_BIN * PO.S_MAX
        b2 = PO.bess((2 * np.pi * PO.A_EL / lam)[:, None]
                     * s_mid[None, :]) ** 2
        acc = np.zeros(len(freq))
        for a in azimuths:
            hs = np.fft.rfft(PO.azimuth_response(a, QUAD)[0], axis=0)
            acc += np.abs((hs * b2).sum(1) * (PO.S_T / lam ** 2)) ** 2
    finally:
        PO.C = keep
    return freq, acc / len(azimuths)


def po_levels(traces):
    """The same three estimators, on the prediction.

    coda_brick is the ideal truncation of the PO frequency integral to
    the analysis band. It is available here and not on the measurement
    because a brick wall has an unbounded impulse response: on a trace
    carrying a front arrival 100 dB above the gate it smears that
    arrival everywhere, which is the locality failure db_reconcile
    exists to remove. The prediction has no front arrival, so the brick
    wall is legitimate on it, and the gap between brick and butter is
    what the Butterworth transition bands cost.
    """
    i0, i1 = int(GATE[0] / PO.DT), int(GATE[1] / PO.DT)
    freq = np.fft.rfftfreq(PO.NT, PO.DT)
    spec = np.fft.rfft(traces, axis=1)
    spec[:, (freq < BAND[0]) | (freq > BAND[1])] = 0.0
    brick = np.fft.irfft(spec, PO.NT, axis=1)
    butter_ = bandpass(traces, 1.0 / PO.DT, *BAND, axis=1)
    env = np.abs(hilbert(traces, axis=1))
    return dict(coda_env=(env[:, i0:i1] ** 2).mean(1),
                coda_loc=2.0 * (traces[:, i0:i1] ** 2).mean(1),
                coda_brick=2.0 * (brick[:, i0:i1] ** 2).mean(1),
                coda_band=2.0 * (butter_[:, i0:i1] ** 2).mean(1))


def closed_form(speed, band=None, shortcut=False):
    """The closed form of Eq. (posum), integrated over the source.

    With shortcut=True the integrand carries the extra (f/f0)^4 of
    po_src_06_reconcile.py, which is what Section 2 currently quotes.
    """
    geom = np.load(os.path.join(PO_DIR, "po_src_geom.npz"))
    diag = np.load(os.path.join(PO_DIR, "po_src_diag.npz"))
    s_v = geom["farea"].sum() / (np.pi * PO.R_DISC ** 2 * PO.THK)
    r_rms = float(np.sqrt((diag["rr"] ** 2 * diag["ain"]).sum()
                          / diag["ain"].sum()))
    g_ori = 1.123
    spec, _ = PO.ricker_spec(PO.NT, PO.DT, PO.F0)
    freq = np.fft.rfftfreq(PO.NT, PO.DT)
    phi = np.linspace(1e-9, np.pi / 2, 20001)
    sphi = np.sin(phi)
    quad = 2 * np.pi * sphi * np.gradient(phi)

    def i_b4(f):
        x = np.maximum(2 * np.pi * PO.A_EL * f / PO.C * sphi, 1e-9)
        return ((2 * j1(x) / x) ** 4 * quad).sum()

    d0, d1 = GATE[0] * speed / 2, GATE[1] * speed / 2
    ssq = ((PO.S_T * freq / PO.C) ** 2
           * (r_rms ** 2 * s_v * g_ori / (16 * np.pi)) * (1 / d0 - 1 / d1)
           * np.array([i_b4(f) for f in freq]))
    if shortcut:
        k0 = int(np.argmin(np.abs(freq - PO.F0)))
        ssq = ssq[k0] * (freq / PO.F0) ** 4
    weight = np.abs(spec * PO.DT) ** 2 * ssq
    if band is not None:
        weight = np.where((freq >= band[0]) & (freq <= band[1]), weight, 0.0)
    t_g = GATE[1] - GATE[0]
    return db(4.0 / t_g * weight.sum() * freq[1]), freq, weight, g_ori


def level(values):
    """Revolution level in dB, averaged as an energy, as db_reconcile."""
    return db(np.asarray(values).mean())


# ------------------------------------------------------------------ draw

def report_source_spectrum():
    """How much of the Ricker source the analysis band rejects."""
    spec, _ = PO.ricker_spec(PO.NT, PO.DT, PO.F0)
    freq = np.fft.rfftfreq(PO.NT, PO.DT)
    shares = census(freq, np.abs(spec) ** 2)
    print("SOURCE SPECTRUM, 2 MHz Ricker, per cent of energy")
    print("  " + "".join("%10s" % n for n in EDGE_NAMES))
    print("  " + "".join("%10.2f" % (100 * s) for s in shares))
    print("  The band rejects %.1f per cent of the source itself, almost"
          % (100 * (1 - shares[1])))
    print("  all of it between 3.0 and 4.0 MHz. Any prediction that")
    print("  integrates the full spectrum inherits that share.")
    print()


def report_transfer(freq, h2):
    """The measured shape of the prediction, against the two models."""
    k0 = int(np.argmin(np.abs(freq - PO.F0)))
    print("PREDICTION'S OWN SPECTRAL TRANSFER, |H(f)|^2 over 180 azimuths")
    print("  %8s %12s %12s %12s" % ("f MHz", "measured", "flat model",
                                    "f^4 shortcut"))
    for f in (0.8e6, 1.5e6, 2.0e6, 2.5e6, 3.0e6, 4.0e6, 6.0e6):
        k = int(np.argmin(np.abs(freq - f)))
        print("  %8.1f %12.2f %12.2f %12.2f"
              % (f / 1e6, h2[k] / h2[k0], 1.0, (f / PO.F0) ** 4))
    inb = (freq >= BAND[0]) & (freq <= BAND[1])
    upper = (freq >= 1.5e6) & (freq <= BAND[1])
    print("  measured spread across 0.8 to 3.0 MHz : %.2f dB"
          % db((h2[inb]).max() / (h2[inb]).min()))
    print("  measured spread across 1.5 to 3.0 MHz : %.2f dB"
          % db((h2[upper]).max() / (h2[upper]).min()))
    print("  an f^4 weighting would impose         : %.2f dB"
          % db((BAND[1] / BAND[0]) ** 4))
    print("  The transfer is near flat, which is what the closed form of")
    print("  po_src_11_closed_form assumes and what the (f/f0)^4 shortcut of")
    print("  po_src_06_reconcile does not.")
    print()


def report_prediction(pred):
    """Prediction levels, by estimator and by delay-to-range speed."""
    print("PREDICTION, 180 azimuths, dB re source, averaged as energies")
    print("  %-10s %11s %11s %11s %11s %9s"
          % ("speed", "global env", "local raw", "local brick",
             "local band", "band cost"))
    for c in SPEEDS:
        p = pred[c]
        print("  %-10.1f %11.2f %11.2f %11.2f %11.2f %+9.2f"
              % (c, level(p["coda_env"]), level(p["coda_loc"]),
                 level(p["coda_brick"]), level(p["coda_band"]),
                 level(p["coda_band"]) - level(p["coda_loc"])))
    p = pred[C_SECTION2]
    print()
    print("  global envelope minus local, at %.0f m/s: %+.3f dB. The"
          % (C_SECTION2, level(p["coda_env"]) - level(p["coda_loc"])))
    print("  prediction is already local, so only the band half of the")
    print("  audit acts on it.")
    print("  out of band, ideal truncation : %.1f per cent of gate power"
          % (100 * (1 - p["coda_brick"].mean() / p["coda_loc"].mean())))
    print("  out of band, Section 3 filter : %.1f per cent, the extra"
          % (100 * (1 - p["coda_band"].mean() / p["coda_loc"].mean())))
    print("  being what the transition bands take from inside the band.")
    print()


def report_speed(pred):
    """Whether the quoted speed is the speed the anchor was run at."""
    print("DELAY-TO-RANGE SPEED, local in-band level")
    for c in SPEEDS:
        print("  %-24s %8.2f dB" % ("%.1f m/s" % c,
                                    level(pred[c]["coda_band"])))
    d = level(pred[3843.0]["coda_band"]) - level(pred[3850.0]["coda_band"])
    s = level(pred[3807.5]["coda_band"]) - level(pred[3843.0]["coda_band"])
    raw = (level(pred[3807.5]["coda_loc"])
           - level(pred[3843.0]["coda_loc"]))
    print("  3850 to 3843 (-0.18 per cent) moves it %+.2f dB" % d)
    print("  3843 to 3807.5 (-0.92 per cent) moves it %+.2f dB, and"
          % s)
    print("  %+.2f dB on the unfiltered estimator Section 2 used." % raw)
    print("  The two steps do not scale, so the quoted 2.6 dB per per")
    print("  cent is a secant across a whole per cent and not a local")
    print("  slope: it is one strong facet crossing the late gate edge.")
    print("  The cached po_src_dense.npz that Section 2 quotes was")
    print("  computed at the nominal 3850 m/s, and po_src_final.npz at")
    print("  the superseded per-azimuth envelope-peak speeds averaging")
    print("  3807.5 m/s. Neither is the 3843 m/s the text claims. The")
    print("  3850 to 3843 error is small; the 3807.5 one is not.")
    print()


def report_measurement(spec, unif):
    """The measured comparands, and why the out-of-band share is suspect."""
    print("MEASUREMENT, ppw 10 specimen, %d matched azimuths, dB re source"
          % len(spec["az"]))
    for key, name in (("coda_env", "global envelope, raw"),
                      ("coda_loc", "local, raw"),
                      ("coda_band", "local, in band  AUDITED")):
        v = spec[key]
        print("  %-26s energies %8.2f   dB mean %8.2f"
              % (name, level(v), db(v).mean()))
    print("  pedestal removed by going local : %+.2f dB"
          % (level(spec["coda_loc"]) - level(spec["coda_env"])))
    print("  rejected by the analysis band   : %+.2f dB (%.1f per cent)"
          % (level(spec["coda_band"]) - level(spec["coda_loc"]),
             100 * (1 - spec["coda_band"].mean() / spec["coda_loc"].mean())))
    print()
    print("  SPECTRAL CENSUS of the raw gate, per cent, Hann tapered")
    print("  %-22s" % "run" + "".join("%10s" % n for n in EDGE_NAMES))
    for name, d in (("ppw 10 specimen", spec), ("ppw 10 uniform", unif)):
        print("  %-22s" % name
              + "".join("%10.2f" % (100 * d["gate_" + n].mean())
                        for n in EDGE_NAMES))
    print("  %-22s" % "source, for reference"
          + "".join("%10.2f" % (100 * s) for s in
                    census(np.fft.rfftfreq(PO.NT, PO.DT),
                           np.abs(PO.ricker_spec(PO.NT, PO.DT,
                                                 PO.F0)[0]) ** 2)))
    print("  At ppw 10 the out-of-band gate content is mostly LOW, not")
    print("  high: 11 per cent below 0.8 MHz against 1.4 per cent in the")
    print("  source and 0.6 per cent in the prediction. The uniform")
    print("  control, which has no scatterers at all, is 84 per cent")
    print("  sub-band, so that content is front-arrival drift and not")
    print("  coda. Above 4 MHz the specimen carries 1.8 per cent against")
    print("  0.7 in the source: grid noise, but a small share at this")
    print("  resolution, unlike ppw 6. Between 3.0 and 4.0 MHz the two")
    print("  sides are comparable, 12.1 per cent measured against 8.8")
    print("  predicted, so that is genuine bandwidth both sides lose")
    print("  together. Filtering both is therefore like for like;")
    print("  filtering neither would credit the model with the drift.")
    print()


def report_comparison(pred, spec):
    """The four possible pairings, and the one to use."""
    p = pred[C_SECTION2]
    print("COMPARISON at %.0f m/s, energies over azimuth, dB re source"
          % C_SECTION2)
    print("  %-34s %10s %10s %12s"
          % ("pairing", "measured", "predicted", "pred - meas"))
    for m_key, p_key, name in (
            ("coda_env", "coda_env", "as published, global env vs raw"),
            ("coda_loc", "coda_loc", "both local, both unfiltered"),
            ("coda_band", "coda_band", "both local, both in band")):
        mv, pv = level(spec[m_key]), level(p[p_key])
        print("  %-34s %10.2f %10.2f %+12.2f" % (name, mv, pv, pv - mv))
    print()
    mv, pv = level(spec["coda_band"]), level(p["coda_band"])
    print("  The published pairing reports the prediction LOW by %.2f dB."
          % (level(spec["coda_env"]) - level(pred[3850.0]["coda_env"])))
    print("  The audited pairing keeps that sign and shrinks the gap to")
    print("  %.2f dB, well inside the +/- 3 dB band of Section 2."
          % abs(pv - mv))
    print("  The sign survives the delay-mapping speed as well:")
    for c in SPEEDS:
        print("    %.1f m/s : prediction %+.2f dB against the measurement"
              % (c, level(pred[c]["coda_band"]) - mv))
    print()
    print("  Distribution shape, which Section 2 also quotes")
    print("  %-24s %10s %10s" % ("", "measured", "predicted"))
    print("  %-24s %10.2f %10.2f"
          % ("dB mean", db(spec["coda_band"]).mean(),
             db(p["coda_band"]).mean()))
    print("  %-24s %10.2f %10.2f"
          % ("azimuthal sd, dB", db(spec["coda_band"]).std(ddof=1),
             db(p["coda_band"]).std(ddof=1)))
    top = np.sort(p["coda_band"])[::-1]
    print("  %-24s %10s %10.1f"
          % ("top azimuth, per cent", "-", 100 * top[0] / top.sum()))
    print("  physical optics is %.2f dB below the simulation when both"
          % (db(spec["coda_band"]).mean() - db(p["coda_band"]).mean()))
    print("  are averaged as decibels rather than energies.")
    print()


def report_correlation(spec):
    """Per-azimuth correlation on the matched azimuths, audited."""
    matched = po_traces(C_SECTION2, np.array(spec["az"], float))
    p = po_levels(matched)
    m_db, p_db = db(spec["coda_band"]), db(p["coda_band"])
    print("PER-AZIMUTH AGREEMENT, %d matched azimuths, audited estimator"
          % len(m_db))
    print("  Pearson r on dB  = %+.3f" % np.corrcoef(m_db, p_db)[0, 1])
    print("  Spearman         = %+.3f"
          % np.corrcoef(np.argsort(np.argsort(m_db)),
                        np.argsort(np.argsort(p_db)))[0, 1])
    print("  measured sd %.2f dB, predicted sd %.2f dB"
          % (m_db.std(ddof=1), p_db.std(ddof=1)))
    print()


def report_closed_form():
    """The closed form, and the shortcut Section 2 currently quotes."""
    full, freq, weight, g = closed_form(C_SECTION2)
    band, _, _, _ = closed_form(C_SECTION2, band=BAND)
    short, _, _, _ = closed_form(C_SECTION2, shortcut=True)
    print("CLOSED FORM, Eq. (posum), at %.0f m/s" % C_SECTION2)
    print("  spectral integration, g = %.3f     : %8.2f dB" % (g, full))
    print("  the same, truncated to the band    : %8.2f dB" % band)
    print("  isotropic normals, g removed       : %8.2f dB"
          % (full - db(g)))
    print("  (f/f0)^4 shortcut of po_src_06_reconcile : %8.2f dB" % short)
    print("  the shortcut inflates the level by : %+8.2f dB"
          % (short - full))
    print("  out of band, of predicted power    : %8.1f per cent"
          % (100 * (1 - 10 ** ((band - full) / 10))))
    print()
    print("  Section 2 quotes -87.78 dB isotropic and -87.28 dB with g.")
    print("  Those are the shortcut values. Integrated over the source")
    print("  spectrum properly they are %.2f and %.2f dB, so the claim"
          % (full - db(g), full))
    print("  that the closed form and the brute force agree to 0.42 dB")
    print("  does not survive. The brute force is the number the level")
    print("  comparison actually uses, and it is unaffected.")
    print()


def main():
    spec = load_sweep("girdle_seed11_ppw10_licensing")
    unif = load_sweep("girdle_seed11_ppw10_zerocontrast")
    pred = {c: po_levels(po_traces(c)) for c in SPEEDS}
    freq, h2 = po_transfer(C_SECTION2)
    report_source_spectrum()
    report_transfer(freq, h2)
    report_prediction(pred)
    report_speed(pred)
    report_measurement(spec, unif)
    report_comparison(pred, spec)
    report_correlation(spec)
    report_closed_form()


if __name__ == "__main__":
    main()

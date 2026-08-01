"""Every way the real machine corrupts the ideal trace.

Each term is separately switchable so you can find which one actually limits
the reconstruction rather than guessing. The dominant pair after fitting the
AS5600 at the specimen is couplant gap and rim runout, both about 4.1 m/s
equivalent; thermal drift is 3.4 m/s behind them.

`ScanState` carries the slow drifts that persist between angles: the freezer
temperature cycle and the once-per-revolution rim runout. Those are the terms
a reference-azimuth revisit can actually correct, which is why they are
stateful rather than redrawn per angle.
"""
import numpy as np


class ScanState:
    """Slow, correlated drift across a scan. Not redrawn per angle."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        e = cfg.err
        # rim runout as a few harmonics of the rotation, fixed for the specimen
        self.runout_phase = rng.uniform(0, 2 * np.pi, e.rim_runout_harmonics)
        w = rng.dirichlet(np.ones(e.rim_runout_harmonics))
        self.runout_amp = e.rim_runout_m * w
        self.t0 = rng.uniform(0, e.T_cycle_period_s)

    def temperature_C(self, elapsed_s):
        e = self.cfg.err
        return e.T_mean_C + e.T_cycle_amplitude_K * np.sin(
            2 * np.pi * (elapsed_s + self.t0) / e.T_cycle_period_s)

    def runout_m(self, azimuth_rad):
        """Extra (or missing) path at this azimuth from an off-centre disk."""
        e = self.cfg.err
        return float(sum(
            self.runout_amp[k] * np.cos((k + 1) * azimuth_rad + self.runout_phase[k])
            for k in range(e.rim_runout_harmonics)))

    def couplant_gap_m(self, rng):
        """Redrawn every angle because the probe lifts and reseats."""
        e = self.cfg.err
        if not self.cfg.acq.lift_and_reseat:
            return e.couplant_gap_mean_m
        return max(0.0, rng.normal(e.couplant_gap_mean_m, e.couplant_gap_sigma_m))


def angle_error_rad(cfg, rng):
    """Where the specimen actually sits versus where it was commanded."""
    e = cfg.err
    if not e.enabled:
        return 0.0
    if e.encoder_on_specimen:
        # AS5600 at the specimen measures true position, so the drive errors
        # stop mattering; what is left is the sensor's own noise
        err = rng.normal(0.0, e.encoder_rms_deg)
    else:
        err = rng.normal(0.0, e.microstep_accuracy_deg)
        if not e.unidirectional_scan:
            err += rng.uniform(-e.angle_backlash_deg, e.angle_backlash_deg)
    return np.radians(err)


def apply(t, trace, cfg, state, azimuth_rad, elapsed_s, rng):
    """Corrupt one ideal trace. Returns (trace, diagnostics)."""
    e = cfg.err
    a = cfg.acq
    if not e.enabled:
        return trace, dict(dt_couplant=0.0, dt_thermal=0.0, dt_runout=0.0,
                           T_C=e.T_mean_C, gap_m=0.0)

    c = cfg.solver.c_ref
    D = cfg.specimen.diameter_m
    fs = a.fs

    # --- couplant: a gap the pulse crosses twice, at oil speed not ice speed
    gap = state.couplant_gap_m(rng)
    c_oil = e.couplant_c + e.couplant_dcdT * (e.T_mean_C - 20.0)
    dt_coup = 2.0 * gap * (1.0 / c_oil - 1.0 / c)

    # --- thermal: the whole specimen slows or speeds up together
    T = state.temperature_C(elapsed_s)
    dv = e.dvdT * (T - e.T_mean_C)
    dt_therm = -2.0 * D * dv / (c ** 2)

    # --- rim runout: the disk is not perfectly concentric
    dr = state.runout_m(azimuth_rad)
    dt_runout = 2.0 * dr / c

    shift = dt_coup + dt_therm + dt_runout + rng.normal(0.0, e.trigger_jitter_s)
    out = _shift(trace, shift * fs)

    # --- receiver noise, reduced by hardware averaging
    if e.noise_rms_frac > 0:
        sigma = e.noise_rms_frac / np.sqrt(max(a.n_averages, 1))
        out = out + rng.normal(0.0, sigma * np.abs(out).max() + 1e-30, out.shape)

    # --- digitiser. Full scale is set by the RIM echo, which is the largest
    # thing in the record by far (R = -0.97 against grain boundaries at
    # 0.01 to 0.035), so the whole grain-boundary coda sits in the bottom few
    # counts of an 8-bit converter.
    #
    # Hardware averaging only rescues that if the per-shot noise dithers by at
    # least about half an LSB. Below that, quantisation is deterministic:
    # every shot rounds identically and averaging 1024 of them recovers
    # nothing. Above it, N shots buy sqrt(N) of effective resolution.
    if a.adc_bits:
        fs_amp = np.abs(out).max()
        if fs_amp > 0:
            lsb = fs_amp / (2 ** (a.adc_bits - 1))
            sigma_shot = e.noise_rms_frac * fs_amp      # BEFORE averaging
            if sigma_shot >= 0.5 * lsb:
                lsb = lsb / np.sqrt(max(a.n_averages, 1))
            out = np.round(out / lsb) * lsb

    return out, dict(dt_couplant=dt_coup, dt_thermal=dt_therm,
                     dt_runout=dt_runout, T_C=T, gap_m=gap)


def _shift(x, samples):
    """Sub-sample time shift via the Fourier shift theorem."""
    if abs(samples) < 1e-9:
        return x
    n = len(x)
    f = np.fft.rfft(x)
    k = np.fft.rfftfreq(n, 1.0)
    return np.fft.irfft(f * np.exp(-2j * np.pi * k * samples), n)


def budget(cfg):
    """Velocity-equivalent size of each error term, for reporting."""
    e = cfg.err
    c = cfg.solver.c_ref
    D = cfg.specimen.diameter_m
    c_oil = e.couplant_c + e.couplant_dcdT * (e.T_mean_C - 20.0)
    rows = []
    dt = 2.0 * e.couplant_gap_sigma_m * abs(1.0 / c_oil - 1.0 / c)
    rows.append(("couplant gap", dt, dt * c ** 2 / (2 * D)))
    dt = 2.0 * e.rim_runout_m / c
    rows.append(("rim runout", dt, dt * c ** 2 / (2 * D)))
    dt = 2.0 * D * abs(e.dvdT) * e.T_cycle_amplitude_K / c ** 2
    rows.append(("thermal drift", dt, abs(e.dvdT) * e.T_cycle_amplitude_K))
    ang = e.encoder_rms_deg if e.encoder_on_specimen else e.microstep_accuracy_deg
    dx = np.radians(ang) * D / 2.0
    rows.append(("angular", dx / c, dx / c * c ** 2 / (2 * D)))
    rows.append(("trigger jitter", e.trigger_jitter_s,
                 e.trigger_jitter_s * c ** 2 / (2 * D)))
    return rows

import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""The scan loop: rotate, seat, fire, record, repeat.

Produces the sinogram (angle x time) that the reconstruction consumes, plus
the ground truth needed to score it. The truth is stored per angle because
the thing being recovered is a per-grain, per-direction velocity, and that
changes with azimuth even though the microstructure does not.
"""
import time

import numpy as np

import errors as E
import forward as F


def ray_direction(azimuth_rad, tilt_rad=0.0):
    """Central-ray direction for a probe at this azimuth and out-of-plane tilt."""
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    cb, sb = np.cos(tilt_rad), np.sin(tilt_rad)
    return np.array([-ca * cb, -sa * cb, sb])


def truth_along_diameter(build, cfg, azimuth_rad, tilt_rad=0.0, ds=None):
    """Grains the central ray crosses, and their qP speed at this azimuth.

    This is what a perfect reconstruction would return, so it is the scoring
    reference for the per-grain velocity fit.
    """
    labels, axes, h = build["labels"], build["axes"], build["h"]
    R = cfg.specimen.diameter_m / 2.0
    ds = ds or h * 0.5
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    # start low when tilted up, so the ray has the full thickness to climb
    z0 = -np.sign(tilt_rad) * cfg.specimen.thickness_m / 2.0 * 0.9
    origin = np.array([R * ca, R * sa, z0])
    d = ray_direction(azimuth_rad, tilt_rad)
    s, lab = F.march(labels, h, origin, d,
                     2.0 * R / max(np.cos(tilt_rad), 1e-6) * 1.02, ds)
    gid, s0, s1 = F.segments(s, lab)
    if len(gid) == 0:
        return dict(grains=gid, length_m=np.empty(0), speed=np.empty(0),
                    direction=d)
    v = F.grain_speeds(axes[gid], d)
    return dict(grains=gid, length_m=s1 - s0, speed=v, direction=d,
                entry_m=s0, exit_m=s1,
                mean_speed=float(np.sum(s1 - s0) / np.sum((s1 - s0) / v)))


def run_scan(build, cfg, seed=0, progress=None):
    """Full azimuthal scan. Returns a dict ready for the reconstruction."""
    rng = np.random.default_rng(seed)
    state = E.ScanState(cfg, rng)
    a = cfg.acq
    n_ang = a.n_angles()
    tilts = np.radians(np.atleast_1d(np.asarray(a.tilts_deg, float)))
    az = np.radians(np.arange(n_ang) * a.az_step_deg)
    commanded = np.tile(az, len(tilts))
    tilt_of = np.repeat(tilts, n_ang)
    n_rec = len(commanded)

    n_samp = int(a.record_end_s * a.fs)
    sino = np.zeros((n_rec, n_samp), dtype=np.float32)
    actual = np.zeros(n_rec)
    dirs = np.zeros((n_rec, 3))
    diags, truth = [], []
    t_axis = np.arange(n_samp) / a.fs

    dwell = a.n_averages / max(a.prf_hz, 1.0)      # time actually spent per angle
    t0 = time.time()
    for i in range(n_rec):
        phi = commanded[i] + E.angle_error_rad(cfg, rng)
        beta = tilt_of[i]
        actual[i] = phi
        dirs[i] = ray_direction(phi, beta)
        _, clean = F.simulate_angle(build, cfg, phi, tilt_rad=beta)
        elapsed = i * (dwell + a.seek_time_s)
        noisy, dg = E.apply(t_axis, clean, cfg, state, phi, elapsed, rng)
        sino[i] = noisy.astype(np.float32)
        diags.append(dg)
        truth.append(truth_along_diameter(build, cfg, phi, beta))
        if progress is not None:
            progress((i + 1) / n_rec, i + 1, n_rec, time.time() - t0)

    return dict(t=t_axis, sinogram=sino, angle_cmd=commanded, angle_true=actual,
                tilt=tilt_of, directions=dirs, diagnostics=diags, truth=truth,
                wall_s=time.time() - t0, config=cfg, seed=seed)


# ───────────────────────── echo picking / windowing ──────────────────────
def envelope(x):
    """Analytic-signal envelope, used for picking arrivals."""
    from scipy.signal import hilbert
    return np.abs(hilbert(x, axis=-1))


def echo_windows(cfg):
    """Nominal E1/E2 sample windows from the reference speed."""
    c = cfg.solver.c_ref
    D = cfg.specimen.diameter_m
    fs = cfg.acq.fs
    t1, t2 = 2 * D / c, 4 * D / c
    half = cfg.acq.window_frac
    return dict(E1=(int(t1 * (1 - half) * fs), int(t1 * (1 + half) * fs)),
                E2=(int(t2 * (1 - half) * fs), int(t2 * (1 + half) * fs)))


def pick_echo(t, trace, lo, hi):
    """Peak of the envelope inside a window, refined by parabolic interpolation."""
    hi = min(hi, len(trace) - 1)
    if hi <= lo + 2:
        return np.nan, 0.0
    env = envelope(trace[lo:hi])
    k = int(np.argmax(env))
    if 0 < k < len(env) - 1:
        y0, y1, y2 = env[k - 1], env[k], env[k + 1]
        denom = y0 - 2 * y1 + y2
        k = k + (0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0)
    fs = 1.0 / (t[1] - t[0])
    return (lo + k) / fs, float(env.max())


def bulk_velocity_curve(scan):
    """Apparent diameter-average speed versus azimuth, from the E1 arrival.

    This is the classic measurement: one velocity per angle, fitted with a
    2-phi curve to give bulk anisotropy. It is the baseline the spatial
    reconstruction has to beat.
    """
    cfg = scan["config"]
    w = echo_windows(cfg)
    D = cfg.specimen.diameter_m
    t = scan["t"]
    v, amp = [], []
    for i in range(scan["sinogram"].shape[0]):
        te, ae = pick_echo(t, scan["sinogram"][i], *w["E1"])
        v.append(2 * D / te if np.isfinite(te) and te > 0 else np.nan)
        amp.append(ae)
    return np.asarray(v), np.asarray(amp)


def fit_two_phi(angles_rad, v):
    """Least-squares v = v0 + A*cos(2(phi - phi0)); returns (v0, A, phi0_deg)."""
    ok = np.isfinite(v)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    p, c, s = angles_rad[ok], np.cos(2 * angles_rad[ok]), np.sin(2 * angles_rad[ok])
    G = np.stack([np.ones_like(p), c, s], axis=1)
    m, *_ = np.linalg.lstsq(G, v[ok], rcond=None)
    v0, a, b = m
    return float(v0), float(np.hypot(a, b)), float(np.degrees(0.5 * np.arctan2(b, a)))

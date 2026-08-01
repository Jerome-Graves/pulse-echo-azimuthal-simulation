"""Recover per-grain c-axis orientation from the echo train.

The chain, and why each step is possible:

1. INTERVAL TIMES -> PER-GRAIN SPEED.
   The optical thin section gives the grain geometry, so the boundary
   positions along each diameter are KNOWN. Only the speeds are unknown.
   Between two consecutive boundary echoes the pulse has crossed exactly one
   grain of known length L, so v = 2L / (t_i - t_(i-1)) directly. No inversion
   needed for this step, it is arithmetic. The hard part is picking the
   echoes, not solving for v.

2. SPEED -> ANGLE FROM THE C-AXIS.
   Ice qP speed depends only on the angle psi between propagation and the
   c-axis. Inverting v(psi) puts the c-axis on a CONE about the ray. It is a
   cone and not a direction because v(psi) is even and non-monotonic (minimum
   at 50.4 deg), so one measurement is never enough.

3. MANY AZIMUTHS -> THE ACTUAL 3D AXIS.
   A grain near the centre is crossed at many azimuths. Each gives its own
   cone. The c-axis is where the cones intersect, which is a 2-parameter fit
   (colatitude, azimuth) against many measurements. This is the "recursive
   modelling of the 3D orientations and fitting" step: it is what lifts the
   result from one bulk number per specimen to an orientation per grain.

Scoring is against the true axes, using disorientation angle, and grains are
reported with their crossing count so you can see the fit degrade as coverage
drops toward the rim.
"""
import numpy as np

import forward as F


# ───────────────────────── 1. picking the echoes ─────────────────────────
def pick_arrivals(t, trace, t_expected, search_frac=0.35, min_prom=0.0,
                  t_start=0.0):
    """Find the echo nearest each expected arrival.

    `t_expected` comes from the known geometry at a reference speed, so the
    search window only has to cover the anisotropy spread (about +/-4%), not
    the whole record. Returns picked times with NaN where nothing was found.
    """
    from scipy.signal import hilbert
    env = np.abs(hilbert(trace))
    fs = 1.0 / (t[1] - t[0])
    peak = env.max() + 1e-30
    out = np.full(len(t_expected), np.nan)
    conf = np.zeros(len(t_expected))
    for i, te in enumerate(t_expected):
        if not np.isfinite(te):
            continue
        half = max(search_frac * te / max(len(t_expected), 1), 3.0 / fs)
        lo = max(int((te - half) * fs), 0)
        hi = min(int((te + half) * fs), len(env) - 1)
        if hi <= lo + 2:
            continue
        seg = env[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] / peak < min_prom:
            continue
        if 0 < k < len(seg) - 1:                      # parabolic refinement
            y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
            d = y0 - 2 * y1 + y2
            k = k + (0.5 * (y0 - y2) / d if abs(d) > 1e-30 else 0.0)
        out[i] = (lo + k) / fs
        conf[i] = seg.max() / peak
    return out, conf


V_MIN, V_MAX = 3776.0, 4046.0      # full qP range of ice Ih, nothing outside


def pick_arrivals_dp(t, trace, lengths_m, t_start=0.0, max_cand=400,
                     v_min=V_MIN, v_max=V_MAX, slack=1.03,
                     anchor_start=False, t_expected=None, prior_frac=0.45):
    """Pick the whole boundary sequence at once, as a constrained path.

    Picking each boundary independently is what makes the reconstruction
    fail: nothing stops two boundaries selecting the same peak, or selecting
    them out of order, and a weak boundary echo (two grains of similar
    orientation give R ~ 0.005) is routinely lost under a neighbouring strong
    one. Measured against truth, roughly one interval in five came back a
    gross mispick.

    The physics rules most of that out. qP speed in ice spans only
    3776 to 4046 m/s, so given boundary i-1 the next arrival must fall in a
    window barely 7% wide. Enforcing that across the whole sequence turns
    picking into a shortest-path problem over candidate peaks, solved exactly
    by dynamic programming, instead of a set of independent guesses.

    MEASURED RESULT: this is WORSE than independent picking, and it is not
    close. Against truth, with errors on and dithered:

        independent   34% rejected, median |err| 16.7 m/s, RMS 67.9
        dp anchored   38% rejected, median |err| 22.5 m/s, RMS 87.6
        dp free start 31% rejected, median |err| 25.7 m/s, RMS 95.3
        dp + prior    51% rejected, median |err| 24.3 m/s, RMS 91.5

    The reason is that the DP demands a COMPLETE path through every boundary.
    Weak boundary echoes are routinely absent from the candidate peaks
    entirely, because the 8-bit record quantises them away, and when a true
    peak is missing the path is forced through a wrong one and the interval
    constraint then propagates that error along the chain. Independent picking
    fails locally; this fails globally.

    Kept, not deleted, because the fix is visible from the failure: the state
    space needs an explicit "boundary not detected" option so the path can
    skip an undetectable interface and span two grains instead of mis-seating
    every later pick. That is the next thing to try.
    """
    from scipy.signal import hilbert, find_peaks
    env = np.abs(hilbert(trace))
    fs = 1.0 / (t[1] - t[0])
    L = np.asarray(lengths_m, float)
    n_b = len(L)
    if n_b == 0:
        return np.empty(0), np.empty(0)

    pk, _ = find_peaks(env)
    if len(pk) == 0:
        return np.full(n_b, np.nan), np.zeros(n_b)
    if len(pk) > max_cand:                       # keep the strongest
        pk = np.sort(pk[np.argsort(env[pk])[::-1][:max_cand]])
    tc = pk / fs
    score = np.log(env[pk] / (env.max() + 1e-30) + 1e-6)

    dt_lo = 2.0 * L / (v_max * slack)
    dt_hi = 2.0 * L * slack / v_min
    NEG = -1e18

    # Geometric prior. Amplitude alone is not enough to identify a boundary:
    # maximising it along any feasible path just threads the strongest peaks
    # in the record, which are rarely the true boundaries. The expected
    # arrival from the known geometry is a strong prior and the common-mode
    # timing error (tens of ns) is negligible against a several-microsecond
    # interval, so the prior stays valid even with errors on.
    if t_expected is not None:
        te = np.asarray(t_expected, float)
        win = prior_frac * np.maximum(2.0 * L / v_min, 1e-9)
        allow = np.abs(tc[None, :] - te[:, None]) <= win[:, None]
    else:
        allow = np.ones((n_b, len(pk)), bool)

    dp = np.full((n_b, len(pk)), NEG)
    back = np.zeros((n_b, len(pk)), dtype=np.int32)
    # The FIRST arrival is left free. Couplant gap, thermal drift and trigger
    # jitter shift the whole record by an unknown common-mode amount, so
    # anchoring boundary 0 to t=0 forces every later window off the arrivals
    # it is hunting for and the path collapses. Intervals are immune to that
    # shift, so only they are constrained. The cost is grain 0's speed, which
    # has no reference and is discarded downstream.
    if anchor_start:
        gap0 = tc - t_start
        dp[0] = np.where((gap0 >= dt_lo[0]) & (gap0 <= dt_hi[0]), score, NEG)
    else:
        dp[0] = score
    dp[0] = np.where(allow[0], dp[0], NEG)

    D = tc[:, None] - tc[None, :]                # D[j, k] = t_j - t_k
    for i in range(1, n_b):
        feas = (D >= dt_lo[i]) & (D <= dt_hi[i])
        prev = np.where(feas, dp[i - 1][None, :], NEG)
        back[i] = prev.argmax(axis=1)
        dp[i] = np.where(allow[i], prev.max(axis=1) + score, NEG)

    if dp[-1].max() <= NEG / 2:                  # no consistent path exists
        return np.full(n_b, np.nan), np.zeros(n_b)

    out = np.empty(n_b, int)
    out[-1] = int(dp[-1].argmax())
    for i in range(n_b - 1, 0, -1):
        out[i - 1] = back[i, out[i]]
    return tc[out], env[pk[out]] / (env.max() + 1e-30)


def speeds_from_intervals(t_picked, lengths_m, t_start=0.0):
    """v_i = 2 L_i / (t_i - t_(i-1)). Pure arithmetic once the picks exist."""
    tp = np.concatenate(([t_start], t_picked))
    dt = np.diff(tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 2.0 * np.asarray(lengths_m) / dt
    v[~np.isfinite(v)] = np.nan
    # anything outside the physical range for ice is a mispick, not a grain
    v[(v < 3600) | (v > 4200)] = np.nan
    return v


# ─────────────────── 2 and 3. speed cones -> the c-axis ──────────────────
def fit_caxis(directions, speeds, n_seed=6000, n_refine=6):
    """Least-squares c-axis from speeds measured along several directions.

    The cost surface is genuinely multi-modal because v(psi) is non-monotonic
    (minimum at 50.4 deg), so two different angles can give the same speed and
    local descent lands in whichever basin it started in. A sparse seed grid
    misses the true basin badly: at colatitude 50 deg, right on the velocity
    minimum, a 64-point grid returned answers 86 deg from the truth on
    NOISELESS data while the true axis scored an exact zero residual.

    So: evaluate a dense hemisphere all at once (vectorised, so thousands of
    candidates cost the same as one loop iteration), then shrink a local grid
    around the winner.
    """
    d = np.asarray(directions, float)
    v = np.asarray(speeds, float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.array([0.0, 0.0, 1.0]), np.inf, int(ok.sum())
    d, v = d[ok], v[ok]

    cand = _fib_hemisphere(n_seed)
    cost = _cost_many(cand, d, v)
    best = cand[int(np.argmin(cost))]
    best_cost = float(cost.min())

    span = np.pi / 16
    for _ in range(n_refine):                      # shrinking local grid
        loc = _local_grid(best, span, 7)
        c = _cost_many(loc, d, v)
        j = int(np.argmin(c))
        if c[j] < best_cost:
            best, best_cost = loc[j], float(c[j])
        span *= 0.45
    return best, float(best_cost / len(v)), int(ok.sum())


HUBER_DELTA = 15.0     # m/s


def _cost_many(N, d, v, delta=HUBER_DELTA):
    """Robust speed misfit for MANY candidate axes at once.

    Huber, not least squares, because echo picking is bimodal: measured
    against truth, the good picks land within about 4 m/s (2% of the 267 m/s
    signal range) but roughly one pick in five is a gross mispick. Squared
    error lets that minority dominate a fit built mostly from good data, and
    the recovered axis follows the outliers.
    """
    psi = np.arccos(np.clip(np.abs(d @ np.asarray(N).T), 0.0, 1.0))   # (k, m)
    r = np.abs(F.qp_speed_at(psi) - v[:, None])
    quad = np.minimum(r, delta)
    return (0.5 * quad ** 2 + delta * (r - quad)).sum(axis=0)


def _cost(n, d, v):
    psi = np.arccos(np.clip(np.abs(d @ n), 0.0, 1.0))
    return float(np.sum((F.qp_speed_at(psi) - v) ** 2))


def _local_grid(n, span, k):
    """Small tangent-plane grid of candidate axes around `n`."""
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    g = np.linspace(-span, span, k)
    U, V = np.meshgrid(g, g, indexing="ij")
    M = (n[None, None, :] + U[..., None] * e1[None, None, :]
         + V[..., None] * e2[None, None, :]).reshape(-1, 3)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def _fib_hemisphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - i / n)          # upper hemisphere only (axes are +/-)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def _local_perturbations(n, step):
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    out = []
    for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        m = n + step * (du * e1 + dv * e2)
        out.append(m / np.linalg.norm(m))
    return out


# ─────────────────────────── the whole pipeline ──────────────────────────
def _peak_in(t, trace, lo, hi):
    """Envelope peak inside a sample window, parabolically refined."""
    from scipy.signal import hilbert
    lo, hi = max(lo, 0), min(hi, len(trace) - 1)
    if hi <= lo + 2:
        return np.nan, 0.0
    env = np.abs(hilbert(trace[lo:hi]))
    k = int(np.argmax(env))
    if 0 < k < len(env) - 1:
        y0, y1, y2 = env[k - 1], env[k], env[k + 1]
        d = y0 - 2 * y1 + y2
        k = k + (0.5 * (y0 - y2) / d if abs(d) > 1e-30 else 0.0)
    fs = 1.0 / (t[1] - t[0])
    return (lo + k) / fs, float(env.max())


def run(scan, build, use_true_picks=False, min_views=3, rim_reference=True,
        picker="independent", progress=None):
    """Reconstruct a c-axis for every grain the beam crossed often enough.

    `use_true_picks=True` bypasses echo picking and feeds the exact arrival
    times. That separates the two failure modes: if the fit is good with true
    picks but bad with real ones, the limit is PICKING, not the inversion.
    """
    cfg = scan["config"]
    t = scan["t"]
    axes_true = build["axes"]
    c_ref = cfg.solver.c_ref

    per_grain = {}
    n_ang = scan["sinogram"].shape[0]
    for i in range(n_ang):
        tr = scan["truth"][i]
        gid, L = tr["grains"], tr["length_m"]
        if len(gid) == 0:
            continue
        d = (scan["directions"][i] if "directions" in scan
             else tr.get("direction",
                         np.array([-np.cos(scan["angle_true"][i]),
                                   -np.sin(scan["angle_true"][i]), 0.0])))

        if use_true_picks:
            v = tr["speed"]
        else:
            t_exp = 2.0 * np.cumsum(L) / c_ref          # geometry at c_ref
            trace = scan["sinogram"][i]
            # Reference to the rim echo, not the trigger. Couplant gap,
            # thermal drift and trigger jitter all shift the whole trace
            # together, so they cancel in a ratio against E1 and the search
            # windows stop walking off the arrivals they are looking for.
            t0_ref = 0.0
            if rim_reference:
                t_e1_nom = 2.0 * float(np.sum(L)) / c_ref
                lo = int(t_e1_nom * (1 - 0.12) / (t[1] - t[0]))
                hi = int(t_e1_nom * (1 + 0.12) / (t[1] - t[0]))
                te1, _ = _peak_in(t, trace, lo, hi)
                if np.isfinite(te1) and te1 > 0:
                    t_exp = t_exp * (te1 / t_e1_nom)    # common-mode removed
            if picker == "dp":
                picked, _ = pick_arrivals_dp(t, trace, L, t_start=t0_ref,
                                             t_expected=t_exp)
                v = speeds_from_intervals(picked, L)
                v[0] = np.nan          # no anchor for the first segment
            elif picker == "fit":
                picked, _ = pick_arrivals(t, trace, t_exp, t_start=t0_ref)
                v = fit_trace_velocities(t, trace, L, cfg,
                                         v0=speeds_from_intervals(picked, L))
            else:
                picked, _ = pick_arrivals(t, trace, t_exp, t_start=t0_ref)
                v = speeds_from_intervals(picked, L)

        for k, g in enumerate(gid):
            if np.isfinite(v[k]):
                per_grain.setdefault(int(g), {"d": [], "v": []})
                per_grain[int(g)]["d"].append(d)
                per_grain[int(g)]["v"].append(float(v[k]))
        if progress is not None:
            progress((i + 1) / n_ang, i + 1, n_ang)

    gids, rec, err, err_mir, nviews, rms = [], [], [], [], [], []
    for g, rec_d in sorted(per_grain.items()):
        if len(rec_d["v"]) < min_views:
            continue
        n_hat, r, nv = fit_caxis(np.array(rec_d["d"]), np.array(rec_d["v"]))
        gids.append(g)
        rec.append(n_hat)
        nviews.append(nv)
        rms.append(r)
        tru = axes_true[g]
        err.append(_disorientation(n_hat, tru))
        err_mir.append(min(_disorientation(n_hat, tru),
                           _disorientation(n_hat, mirror(tru))))

    gi = np.array(gids, int)
    return dict(grains=gi,
                axes_rec=np.array(rec).reshape(-1, 3),
                axes_true=axes_true[gi] if len(gi) else np.empty((0, 3)),
                error_deg=np.array(err),
                error_deg_modulo_mirror=np.array(err_mir),
                n_views=np.array(nviews), rms_ms=np.array(rms),
                used_true_picks=use_true_picks,
                tilted=bool(np.any(np.abs(scan.get("tilt", [0])) > 1e-6)))


def mirror(n):
    """Reflection of an axis in the scan plane, the one thing an in-plane
    monostatic scan provably cannot distinguish from `n`."""
    return np.array([n[0], n[1], -n[2]])


def ambiguity_deg(n):
    """How far the mirror twin sits from `n`: arccos|cos 2*theta|.

    Zero for c-axes in the scan plane or along the disk axis, and a full
    90 deg at 45 deg colatitude. This is the price of a purely in-plane scan.
    """
    return _disorientation(np.asarray(n, float), mirror(np.asarray(n, float)))


def _disorientation(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def score(res):
    """Headline numbers. Reports error BOTH ways so the mirror ambiguity is
    visible as a cost rather than quietly absorbed."""
    if len(res["error_deg"]) == 0:
        return dict(n=0, median=np.nan, median_mod_mirror=np.nan,
                    mean=np.nan, p90=np.nan, within10=np.nan, within20=np.nan,
                    random_baseline=57.3)
    e, em = res["error_deg"], res["error_deg_modulo_mirror"]
    return dict(n=len(e), median=float(np.median(e)),
                median_mod_mirror=float(np.median(em)),
                mean=float(e.mean()), p90=float(np.percentile(e, 90)),
                within10=float((em < 10).mean() * 100),
                within20=float((em < 20).mean() * 100),
                random_baseline=57.3)


# ══════════════════ 4. trace fitting: no picking at all ══════════════════
"""Fit the whole waveform instead of identifying individual echoes.

Picking asks "where is boundary i?" and needs every boundary to be findable.
The 8-bit record quantises the weakest boundary echoes away entirely, so that
question sometimes has no answer and any assignment-based method is forced
into a wrong one.

Fitting asks a different question: which set of per-grain velocities, pushed
through the forward model, reproduces the measured trace? An undetectable
boundary simply contributes almost nothing to the misfit, so it costs a
measurement instead of corrupting its neighbours.

Two practical details make it work:
  * SPEED. The reflectivity series is sparse (about a dozen spikes), so the
    model is built by superposing shifted copies of the pulse rather than by
    convolving a mostly-zero array. That is ~1000x cheaper and is what makes
    a per-trace optimisation affordable.
  * CYCLE SKIPPING. The pulse oscillates at 5 MHz (0.2 us period) and a 1.3%
    velocity error already shifts an arrival by half a period, so a direct
    waveform misfit has local minima everywhere. Standard multiscale FWI
    remedy: fit the ENVELOPE first, which is smooth and cannot cycle-skip,
    then refine on the waveform.

MEASURED RESULT: this LOSES to independent picking, in every regime tested.

    end-to-end c-axis error, errors on and dithered:
        independent picking   median 31.0 deg   (19.0 mod mirror)
        trace fitting         median 58.5 deg   (37.0 mod mirror)
        exact arrival times   median  0.0 deg

    per-velocity error against truth, by forward physics:
        forward physics                       fit      pick
        full (beam+multiples+errors)      22.6 m/s   16.0 m/s
        single ray, multiples, errors     22.8 m/s   15.7 m/s
        single ray, no multiples, errors  22.2 m/s   15.7 m/s
        single ray, no multiples, NO err   9.3 m/s    3.3 m/s

The obvious suspect was model mismatch, this synth being single-ray primaries
only while the data carries beam spreading, Laguerre obliquity and multiples.
That is NOT the limit: stripping those from the forward model moves the fit by
0.4 m/s. Removing the ERROR MODEL moves it by 13 m/s.

Fitting does produce more usable measurements (nothing is rejected) and fewer
catastrophic outliers (RMS 58 against 75), but each measurement is less
precise, and the downstream Huber loss already tolerated picking's outliers.
More mediocre data lost to less, better data.

The real conclusion is not about either algorithm. Three independent methods
(independent picking, constrained-path DP, waveform fitting) all converge to
roughly the same wall once the error model is on, and all three are near
perfect with it off. The limit is the ACQUISITION, not the inversion: the
8-bit record quantises the weak boundary echoes away and no amount of
processing recovers information that was never digitised. Time-gated gain,
or gating the rim echo out of the record, is the fix.
"""
RHO = 917.0


def _pulse_bank(cfg, fs, sub=8):
    """Pulse pre-shifted by sub-sample fractions, for sub-sample arrivals."""
    _, p = F.gaussian_pulse(cfg.probe.f0, cfg.probe.frac_bw, fs * sub)
    return [p[k::sub] for k in range(sub)], sub


def synth_trace(v, L, cfg, bank, sub, n, fs):
    """Model trace for a set of per-grain velocities. Superposition, not convolution."""
    v = np.asarray(v, float)
    tcum = 2.0 * np.cumsum(L / v)              # two-way time to each interface
    Z = RHO * v
    amps = np.empty(len(v))
    amps[:-1] = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
    amps[-1] = cfg.err.rim_reflection          # far rim closes the sequence
    trans = np.concatenate(([1.0], np.cumprod(1.0 - amps[:-1] ** 2)))
    amps = amps * trans

    out = np.zeros(n)
    half = len(bank[0]) // 2
    for tt, a in zip(tcum, amps):
        x = tt * fs
        k = int(np.floor(x))
        f = int(round((x - k) * sub)) % sub
        p = bank[f]
        lo, hi = k - half, k - half + len(p)
        if hi <= 0 or lo >= n:
            continue
        s0, s1 = max(lo, 0), min(hi, n)
        out[s0:s1] += a * p[s0 - lo:s1 - lo]
    return out


def _resid(v, data, L, cfg, bank, sub, n, fs, envelope_mode):
    m = synth_trace(v, L, cfg, bank, sub, n, fs)
    if envelope_mode:
        from scipy.signal import hilbert
        m = np.abs(hilbert(m))
    scale = float(m @ data) / float(m @ m + 1e-30)    # gain solved analytically
    return scale * m - data


def fit_trace_velocities(t, trace, L, cfg, v0=None, window_pad=1.10):
    """Recover per-grain velocities from one trace by waveform fitting."""
    from scipy.optimize import least_squares
    from scipy.signal import hilbert
    fs = 1.0 / (t[1] - t[0])
    L = np.asarray(L, float)
    if len(L) < 2:
        return np.full(len(L), np.nan)

    n = min(int(2.0 * L.sum() / V_MIN * window_pad * fs), len(trace))
    data = np.asarray(trace[:n], float)
    bank, sub = _pulse_bank(cfg, fs)
    v0 = np.full(len(L), cfg.solver.c_ref) if v0 is None else np.clip(
        np.nan_to_num(np.asarray(v0, float), nan=cfg.solver.c_ref), V_MIN, V_MAX)

    common = dict(L=L, cfg=cfg, bank=bank, sub=sub, n=n, fs=fs)
    try:
        env = np.abs(hilbert(data))
        r1 = least_squares(_resid, v0, bounds=(V_MIN, V_MAX), method="trf",
                           x_scale=100.0, max_nfev=120, diff_step=1e-3,
                           kwargs=dict(data=env, envelope_mode=True, **common))
        r2 = least_squares(_resid, r1.x, bounds=(V_MIN, V_MAX), method="trf",
                           x_scale=50.0, max_nfev=120, diff_step=5e-4,
                           kwargs=dict(data=data, envelope_mode=False, **common))
        return r2.x
    except Exception:
        return np.full(len(L), np.nan)

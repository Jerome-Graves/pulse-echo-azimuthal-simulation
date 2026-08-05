"""FW sweep tab: resumable sweep control + living fabric inversion.

The live panel is a st.fragment with run_every=5 s while a runner or a
fit is active: ONLY the panel reruns, nothing else. This replaces the
old sleep+st.rerun full-page cycle, which - combined with the app's
anti-flicker CSS (stale elements stay opaque) - showed TWO copies of
every chart while the new page was mid-render. Fragments rerun in
place, so there is nothing stale to ghost.

All heavy work happens in DETACHED subprocesses (sweep_runner.py /
fit_sweep.py): the GUI process never touches CUDA (it segfaults
Streamlit's worker thread) and never blocks.
"""
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import specimen as S                           # noqa: E402
import sweep_runner as SW                      # noqa: E402
import viz                                     # noqa: E402


# ─────────────────────────── small helpers ────────────────────────────
def _read_json(path):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _stamp_grid_status(d):
    """Freshen grid_status.json at daemon spawn time so the next
    rerender (or the daemon's own single-flight check seeing a rival)
    doesn't double-spawn during the daemon's slow import."""
    try:
        with open(os.path.join(d, "grid_status.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"phase": "starting", "t": time.time()}, fh)
    except OSError:
        pass


def _spawn(args, log_path):
    """Detached child that survives GUI reruns and closes."""
    log = open(log_path, "a")
    return subprocess.Popen(
        [sys.executable] + args, stdout=log, stderr=subprocess.STDOUT,
        cwd=os.path.join(ROOT, "sim"),
        creationflags=(subprocess.CREATE_NO_WINDOW
                       | subprocess.CREATE_NEW_PROCESS_GROUP))


def _runner_active(stat):
    return (time.time() - stat.get("heartbeat", 0) < 120
            and stat.get("phase") not in ("stopped", "complete",
                                          "NO CUDA - cannot run"))


def _fit_active(d):
    lock = os.path.join(d, "fit.lock")
    return (os.path.exists(lock)
            and time.time() - os.path.getmtime(lock) < 600)


def _map_active(d):
    lock = os.path.join(d, "map.lock")
    return (os.path.exists(lock)
            and time.time() - os.path.getmtime(lock) < 600)


def _grid_active(d):
    """True while EITHER grid process is live: the batch tool (lock
    file) or the continuous daemon. Daemon liveness = the status
    HEARTBEAT (beats even during minutes-long builds), falling back to
    the state file for pre-status daemons."""
    lock = os.path.join(d, "grid.lock")
    if (os.path.exists(lock)
            and time.time() - os.path.getmtime(lock) < 600):
        return True
    gstat = _read_json(os.path.join(d, "grid_status.json"))
    if gstat and time.time() - gstat.get("t", 0) < 30:
        return True
    gs = os.path.join(d, "grid_state.npz")
    return (os.path.exists(gs)
            and time.time() - os.path.getmtime(gs) < 30)


# ────────────────── specimen & fabric stats for a sweep ───────────────
@st.cache_data(show_spinner="Building specimen preview (CPU, cached)...")
def _spec_stats(cfg_json):
    """The Specimen & Fabric tab's stats, computed for a SWEEP's own
    specimen (same seed/grains/kappa/axis as the runner uses). Built at
    preview resolution (ppw 1.5) on the CPU: grain counts, sizes and
    orientation statistics do not need the solver grid, and the GUI
    process must never touch CUDA (Streamlit segfault)."""
    cfg = json.loads(cfg_json)

    class _CPUSpec(S.DiskSpecimen):
        @staticmethod
        def _label_grid_gpu(x, z, R, seeds, weights):
            return None                      # force the CPU label path

    sp = _CPUSpec(diameter_m=cfg["diameter_mm"] * 1e-3,
                  thickness_m=cfg["thickness_mm"] * 1e-3,
                  n_grains=cfg["n_grains"], size_cv=cfg["size_cv"],
                  concentration=cfg["concentration"],
                  spatial_corr=cfg["spatial_corr"],
                  fabric_axis=tuple(cfg["fabric_axis"]),
                  seed=cfg["seed"])
    b = sp.build(3850.0 / (cfg["f0_mhz"] * 1e6) / 1.5)
    labels, axes = b["labels"], np.asarray(b["axes"])
    w, _ = S.orientation_tensor(axes)
    pr = [(u, v) for u, ns in b["adjacency"].items() for v in ns if u < v]
    mis = S.disorientation_deg(axes[[p[0] for p in pr]],
                               axes[[p[1] for p in pr]])
    cnt = np.bincount(labels[labels >= 0].ravel(),
                      minlength=len(axes)).astype(float)
    d_eq = 2 * (3 * cnt[cnt > 0] * b["h"] ** 3 / (4 * np.pi)) ** (1 / 3) * 1e3
    # volume-weighted SAMPLE fabric axis: what this particular grain
    # draw actually points at (differs from the nominal axis by the
    # 100-grain sampling scatter - the fit is scored against reality,
    # so show reality)
    vw = cnt / max(cnt.sum(), 1.0)
    T = np.einsum("i,ij,ik->jk", vw, axes, axes)
    ev, evec = np.linalg.eigh(T)
    a = evec[:, -1]
    samp_deg = float(np.degrees(np.arctan2(a[1], a[0])) % 180.0)
    ax = np.asarray(cfg["fabric_axis"], float)
    nom_deg = float(np.degrees(np.arctan2(ax[1], ax[0])) % 180.0)
    # ── region statistics: whole volume vs what the BEAM actually
    # samples. The transducer interrogates a slab of ~element diameter
    # around the mid-plane, and a lab thin section is a single plane -
    # both are DIFFERENT statistical samples of the same specimen
    # (fewer grains, area- not volume-weighted). This is the
    # thin-section-vs-beam-volume mismatch config.py flags as a
    # first-class error term, made visible per sweep.
    def _region(counts, area_equiv=False):
        cw = counts.astype(float)
        tot = cw.sum()
        if tot <= 0 or (cw > 0).sum() < 2:
            return None
        Tr = np.einsum("i,ij,ik->jk", cw / tot, axes, axes)
        evr, evecr = np.linalg.eigh(Tr)
        ar = evecr[:, -1]
        deg = float(np.degrees(np.arctan2(ar[1], ar[0])) % 180.0)
        nz_c = cw[cw > 0]
        if area_equiv:      # thin section: equivalent-circle diameter
            d = 2 * np.sqrt(nz_c * b["h"] ** 2 / np.pi) * 1e3
        else:               # volume: equivalent-sphere diameter
            d = 2 * (3 * nz_c * b["h"] ** 3 / (4 * np.pi)) ** (1 / 3) * 1e3
        return dict(n=int((cw > 0).sum()), a1=float(evr[-1]),
                    axis_deg=deg, size_mm=float(d.mean()))
    nzz = labels.shape[2]
    kmid = nzz // 2
    er = max(int(cfg["element_d_mm"] * 1e-3 / 2 / b["h"]), 1)
    z0, z1 = max(kmid - er, 0), min(kmid + er + 1, nzz)
    sec = labels[:, :, kmid]
    slab = labels[:, :, z0:z1]
    cnt_sec = np.bincount(sec[sec >= 0].ravel(), minlength=len(axes))
    cnt_slab = np.bincount(slab[slab >= 0].ravel(), minlength=len(axes))
    regions = dict(
        volume=_region(cnt),
        beam_slab=_region(cnt_slab),
        section=_region(cnt_sec, area_equiv=True))
    return dict(n=int((cnt > 0).sum()), size_mm=float(d_eq.mean()),
                w_desc=[float(w[0]), float(w[1]), float(w[2])],
                mis=float(mis.mean()), samp_deg=samp_deg,
                nom_deg=nom_deg, samp_a1=float(ev[-1]),
                nom_a1=float(S.watson_eigenvalue(cfg["concentration"])),
                axes=axes, labels=labels, h=float(b["h"]),
                regions=regions, slab_mm=float((z1 - z0) * b["h"] * 1e3))


@st.cache_data(show_spinner="Meshing grains...")
def _sw_poly3d(labels, vals, h, title, vmax, cap, step, lbl):
    import viz as _viz
    return _viz.polycrystal_3d(labels, vals, h, title, vmin=0.0,
                               vmax=vmax, max_grains=cap, step=step,
                               value_label=lbl)


@st.cache_data(show_spinner="Building c-axis field...")
def _sw_field3d(labels, axes, h):
    import viz as _viz
    return _viz.caxis_field_3d(labels, axes, h)


def _spec_stats_panel(cfg):
    s = _spec_stats(json.dumps(cfg, sort_keys=True))
    labels, axes, h = s["labels"], s["axes"], s["h"]
    key = f"ss_{cfg['name']}"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grains", f"{s['n']}")
    m2.metric("Mean grain size", f"{s['size_mm']:.1f} mm")
    m3.metric("Fabric eigenvalue 1", f"{s['w_desc'][0]:.3f}",
              help="1/3 random, 1.0 single max")
    m4.metric("Mean disorientation", f"{s['mis']:.1f}°",
              help="Drives scattering strength")

    colat = viz.colatitude_deg(axes)
    azim = viz.azimuth_deg(axes)
    sv1, sv2, sv3, sv4, sv5 = st.tabs(["3D grains", "c-axis field",
                                       "Pole figure", "2D slice",
                                       "Plane vs volume"])
    with sv1:
        k1, k2, k3 = st.columns([2, 2, 3])
        colour_by = k1.selectbox("Colour by", ["c-axis colatitude",
                                               "in-plane azimuth"],
                                 key=key + "_col")
        cap = k2.slider("Max grains drawn", 25, 1000, 400, 25,
                        key=key + "_cap",
                        help="Largest grains are kept. Lower this if "
                             "the browser struggles.")
        dstep = k3.select_slider("Mesh detail", [1, 2, 3], value=2,
                                 key=key + "_step", format_func=lambda v:
                                 {1: "full", 2: "half", 3: "coarse"}[v])
        vals, vmax, lbl = ((colat, 90.0, "colat")
                           if colour_by.startswith("c-axis")
                           else (azim, 180.0, "azim"))
        st.plotly_chart(_sw_poly3d(labels, vals, h, colour_by, vmax,
                                   cap, dstep, lbl), width="stretch")
        st.caption("Each grain is an isosurface coloured by its own "
                   "c-axis. Drag to rotate. This is the fabric the "
                   "ultrasound sees.")
    with sv2:
        st.plotly_chart(_sw_field3d(labels, axes, h), width="stretch")
        st.caption("Bars are double-ended because +c and -c are the "
                   "same crystal direction. Bar length scales with "
                   "grain size.")
    with sv3:
        p1, p2 = st.columns([3, 2])
        p1.plotly_chart(viz.pole_figure(axes), width="stretch")
        p2.metric("Nominal axis / a₁",
                  f"{s['nom_deg']:.0f}° / {s['nom_a1']:.2f}")
        p2.metric("SAMPLE axis / a₁ (this draw)",
                  f"{s['samp_deg']:.1f}° / {s['samp_a1']:.2f}",
                  help="Volume-weighted orientation-tensor axis of "
                       "this particular grain draw. The fit should be "
                       "judged against THIS, not the nominal: a "
                       "100-grain draw scatters several degrees from "
                       "nominal, and the beam's chord sampling adds "
                       "more (measured floor ~8-19° across seeds).")
        p2.caption("Eigenvalues "
                   f"{s['w_desc'][0]:.3f} / {s['w_desc'][1]:.3f} / "
                   f"{s['w_desc'][2]:.3f}. Equal-area (Schmidt) lower "
                   "hemisphere; 1/3,1/3,1/3 random, 1,0,0 single max.")
    with sv4:
        nz = labels.shape[2]
        q1, q2 = st.columns(2)
        kz = q1.slider("z slice", 0, nz - 1, nz // 2, key=key + "_kz")
        sl = labels[:, :, kz].astype(float)
        idx = np.clip(sl.astype(int), 0, None)
        lut = np.random.default_rng(0).permutation(len(axes))
        with q1:
            fig, ax = plt.subplots(figsize=(4.4, 4.4))
            ax.imshow(np.where(sl >= 0, lut[idx], np.nan).T,
                      cmap="tab20", origin="lower",
                      interpolation="nearest")
            ax.set_title("grain labels")
            ax.set_xticks([]); ax.set_yticks([])
            st.pyplot(fig, use_container_width=True)
        with q2:
            st.write("")
            fig2, ax2 = plt.subplots(figsize=(4.4, 4.4))
            im = ax2.imshow(np.where(sl >= 0, azim[idx], np.nan).T,
                            cmap="hsv", origin="lower", vmin=0,
                            vmax=180, interpolation="nearest")
            ax2.set_title("in-plane c-axis azimuth (deg)")
            ax2.set_xticks([]); ax2.set_yticks([])
            fig2.colorbar(im, ax=ax2, fraction=0.046)
            st.pyplot(fig2, use_container_width=True)
    with sv5:
        rg = s["regions"]
        rows, names = [], []
        for tag, label in (("volume", "whole specimen (3D volume)"),
                           ("beam_slab",
                            f"beam slab (mid ± element, "
                            f"{s['slab_mm']:.1f} mm thick)"),
                           ("section", "mid thin-section (single plane)")):
            r = rg.get(tag)
            if r is None:
                continue
            names.append(label)
            rows.append([r["n"], f"{r['a1']:.3f}",
                         f"{r['axis_deg']:.1f}", f"{r['size_mm']:.1f}"])
        import pandas as pd
        st.dataframe(pd.DataFrame(
            rows, index=names,
            columns=["grains", "a₁ (weighted)", "fabric axis (°)",
                     "mean grain size (mm)"]), width="stretch")
        vol, slab_ = rg.get("volume"), rg.get("beam_slab")
        if vol and slab_:
            d_ax = abs(vol["axis_deg"] - slab_["axis_deg"])
            d_ax = min(d_ax, 180 - d_ax)
            st.caption(
                f"The ultrasound only samples the beam slab: its fabric "
                f"axis differs from the whole-volume axis by "
                f"{d_ax:.1f}° and its a₁ by "
                f"{abs(vol['a1'] - slab_['a1']):.3f} here. A lab thin "
                "section is yet another sample (area-weighted, "
                "equivalent-circle sizes). Judge the fit against the "
                "region the instrument actually measures - this "
                "beam-volume vs thin-section mismatch is the "
                "first-class error term flagged in config.py.")


# ─────────────────────────────── the tab ──────────────────────────────
def render():
    st.subheader("Full-wave azimuth sweep")
    st.caption("Each azimuth is saved the moment it completes. Stop any "
               "time; Continue picks up exactly where it left off, in "
               "this session or any future one. The fabric fit re-runs "
               "itself as new azimuths arrive.")
    names = SW.list_sweeps()
    sel = st.selectbox("Sweep session", ["« new sweep »"] + names,
                       index=(1 if names else 0))
    if sel == "« new sweep »":
        _new_sweep_form()
        return
    d = SW.sweep_dir(sel)
    with st.expander("Specimen & fabric stats (this sweep's specimen)"):
        if st.toggle("Compute", key=f"specstats_{sel}",
                     help="One-time CPU preview build of the sweep's "
                          "specimen (cached afterwards)."):
            _spec_stats_panel(SW.load(sel))
    live = _runner_active(_read_json(os.path.join(d, "status.json"))) \
        or _fit_active(d) or _map_active(d) or _grid_active(d)
    # dynamic cadence: poll only while something is actually running
    st.fragment(run_every=(5 if live else None))(lambda: _panel(sel))()


def _new_sweep_form():
    with st.form("new_sweep"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", "sweep5mhz_seed7")
        f0_mhz = c2.number_input("f0 (MHz)", 0.5, 10.0, 5.0, 0.5)
        ppw = c3.number_input("points per wavelength", 4.0, 8.0, 6.0, 1.0)
        az0 = c1.number_input("azimuth start (deg)", 0, 359, 0)
        az1 = c2.number_input("azimuth stop (deg)", 1, 360, 360)
        azs = c3.number_input("azimuth step (deg)", 1, 90, 1)
        seed = c1.number_input("specimen seed", 0, 9999, 7)
        grains = c2.number_input("grains", 10, 2000, 100)
        kappa = c3.number_input("fabric concentration k", 0.0, 50.0, 3.93)
        ax_choice = c1.selectbox(
            "fabric axis",
            ["in-plane x (1,0,0)", "in-plane y (0,1,0)",
             "vertical (0,0,1) - null case", "tilted (0,0.5,0.87)"])
        ax_map = {"in-plane x (1,0,0)": [1.0, 0.0, 0.0],
                  "in-plane y (0,1,0)": [0.0, 1.0, 0.0],
                  "vertical (0,0,1) - null case": [0.0, 0.0, 1.0],
                  "tilted (0,0.5,0.87)": [0.0, 0.5, 0.87]}
        st.caption("ppw 6 is required for amplitude work; ppw 4 is "
                   "timing-only. 5 MHz at ppw 6 is ~18-20 min/azimuth "
                   "on the RTX 4070 (records now run to ~70 us for the "
                   "tail observable); azimuths are visited in a spread-"
                   "maximising order, so a partial sweep is always "
                   "usable - a fit is worthwhile from ~12 azimuths "
                   "(~4 h) and solid by ~40 (overnight).")
        if st.form_submit_button("Create sweep", type="primary"):
            SW.create(name, f0_mhz=float(f0_mhz), ppw=float(ppw),
                      az_start=int(az0), az_stop=int(az1),
                      az_step=int(azs), seed=int(seed),
                      n_grains=int(grains), concentration=float(kappa),
                      fabric_axis=ax_map[ax_choice])
            st.rerun()


@st.cache_data(show_spinner=False)
def _sweep_levels(name, n_done):
    """(azimuths, coda_db, E1) from the saved files; recomputes for
    legacy/imported traces that lack the runner's keys."""
    cfg = SW.load(name)
    dd = SW.sweep_dir(name)
    az, coda, e1 = [], [], []
    for a in SW.done_azimuths(cfg):
        # with-block: linger a handle and every writer's os.replace
        # on this dir starts losing races (Windows)
        with np.load(os.path.join(dd, f"az{a:03d}.npz")) as z:
            az.append(a)
            if "coda_db" in z.files and "E1" in z.files:
                coda.append(float(z["coda_db"]))
                e1.append(float(z["E1"]))
            else:
                E1v, _t1, cv, _tl = SW._analyse(
                    np.asarray(z["trace"], float).ravel(),
                    float(z["dt"]))
                coda.append(cv)
                e1.append(E1v)
    return np.array(az), np.array(coda), np.array(e1)


def _panel(sel):
    cfg = SW.load(sel)
    d = SW.sweep_dir(sel)
    done = SW.done_azimuths(cfg)
    total = len(SW.azimuths(cfg))
    stat = _read_json(os.path.join(d, "status.json"))
    running = _runner_active(stat)
    fitting = _fit_active(d)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("azimuths saved", f"{len(done)}/{total}")
    m2.metric("state", ("RUNNING" if running else
                        "complete" if len(done) == total else "idle"))
    sec_per_az = stat.get("sec_per_az")
    m3.metric("per azimuth",
              f"{sec_per_az:.0f} s" if sec_per_az else "-")
    m4.metric("ETA", (f"{sec_per_az*(total-len(done))/3600:.1f} h"
                      if sec_per_az and len(done) < total else "-"))
    if running:
        st.info(f"runner: {stat.get('phase', '?')}")
    st.progress(len(done) / max(total, 1))
    with st.expander("configuration"):
        st.json(cfg)

    b1, b2, b3 = st.columns(3)
    if b1.button("▶ Start / Continue", disabled=running, type="primary"):
        _spawn([os.path.join(ROOT, "sim", "sweep_runner.py"),
                "--run", sel], os.path.join(d, "run.log"))
        time.sleep(1.5)
        st.rerun()                       # full rerun -> cadence goes live
    if b2.button("⏹ Stop", disabled=not running):
        open(os.path.join(d, "STOP"), "w").close()
        st.warning("Stop requested. Every completed azimuth is already "
                   "saved; the in-flight one is redone on Continue.")
    if b3.button("↻ Refresh"):
        st.rerun()

    if not done:
        return
    az, coda, e1 = _sweep_levels(sel, len(done))
    p1, p2 = st.columns(2)
    fig_c = go.Figure(go.Scatterpolar(
        theta=az, r=coda, mode="markers",
        marker=dict(size=5, color=coda, colorscale="Viridis")))
    fig_c.update_layout(template="plotly_dark", height=380,
                        title="coda RMS 24-36 us (dB re own E1)",
                        margin=dict(l=30, r=30, t=40, b=20))
    p1.plotly_chart(fig_c, width="stretch", key="sweep_coda_polar")
    e1_db = 20 * np.log10(e1 / e1.max() + 1e-30)
    fig_e = go.Figure(go.Scatterpolar(
        theta=az, r=e1_db, mode="markers",
        marker=dict(size=5, color=e1_db, colorscale="Inferno")))
    fig_e.update_layout(template="plotly_dark", height=380,
                        title="E1 fade pattern (dB re max)",
                        margin=dict(l=30, r=30, t=40, b=20))
    p2.plotly_chart(fig_e, width="stretch", key="sweep_e1_polar")

    _fit_section(sel, d, done, fitting)

    with st.expander("runner log (tail)"):
        lp = os.path.join(d, "run.log")
        if os.path.exists(lp):
            with open(lp, encoding="utf-8", errors="replace") as f:
                st.code("".join(f.readlines()[-25:]))


def _fit_section(sel, d, done, fitting):
    st.divider()
    st.markdown("**Fabric inversion**: Watson ODF fit (axis + "
                "concentration) on this sweep's azimuthal coda + E1 ToF "
                "observables. Runs in the background; refits itself as "
                "azimuths arrive.")
    fit = _read_json(os.path.join(d, "fit_result.json"))
    n_prev = int(fit.get("n_azimuths", 0)) if fit else 0

    f1, f2 = st.columns([1, 2])
    auto = f2.checkbox("auto-refit as new azimuths arrive", value=True)
    manual = f1.button(f"⚙ Fit fabric now ({len(done)} azimuths)",
                       disabled=len(done) < 6 or fitting)
    if manual or (auto and not fitting and len(done) >= 6
                  and len(done) > n_prev):
        _spawn([os.path.join(ROOT, "sim", "fit_sweep.py"), sel],
               os.path.join(d, "fit.log"))
        fitting = True
    if fitting:
        st.info(f"fit running on {len(done)} azimuths "
                "(result updates below when done)")
    elif len(done) < 6:
        f2.caption("needs at least 6 azimuths")
    if not fit:
        return

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("fabric axis: recovered", f"{fit['alpha_deg']:.1f}°",
              f"true value = {fit['truth_alpha_deg']:.0f}° "
              f"(off by {fit['alpha_err_deg']:.1f}°)",
              delta_color="off")
    r2.metric("κ concentration: recovered", f"{fit['kappa']:.2f}",
              f"true value = {fit['truth_kappa']:.2f}",
              delta_color="off")
    r3.metric("a₁ eigenvalue: recovered", f"{fit['a1']:.3f}",
              f"true value = {fit['truth_a1']:.3f}", delta_color="off")
    r4.metric("misfit χ²", f"{fit['chi2']:.1f}",
              f"from {fit['n_azimuths']} azimuths "
              f"({2*fit['n_azimuths']} data values)", delta_color="off")
    st.caption(f"fitted {fit['fitted_at']}; grey lines under each "
               "number show the specimen's TRUE value (what the sweep "
               "was built with), not a difference")
    if fit.get("c2t_used"):
        st.caption(f"axis evidence: coherent-2θ φ₂ = "
                   f"{fit['c2t_phi2_deg']:.1f}° ± "
                   f"{fit['c2t_se_deg']:.1f}° "
                   f"(A₂ {fit['c2t_a2_db']:.2f} dB, calibration-free) "
                   "combined with the ToF template in stage 1")
    elif "c2t_used" in fit:
        st.caption("coherent-2θ axis channel: declined for this sweep "
                   "(needs ≥48 azimuths + rigid2 convention)")

    excluded = set(fit.get("excluded_rots", []))
    rots_f = fit["rots"]
    used = [i for i, r in enumerate(rots_f) if r not in excluded]
    excl = [i for i, r in enumerate(rots_f) if r in excluded]
    fr = go.Figure()
    fr.add_bar(x=[rots_f[i] for i in used],
               y=[fit["coda_res_db"][i] for i in used],
               name="coda residual (dB), fitted")
    fr.add_bar(x=[rots_f[i] for i in excl],
               y=[fit["coda_res_db"][i] for i in excl],
               name="coda residual, EXCLUDED (null / walk-off "
                    "bands, known unmodeled physics)",
               marker_color="rgba(160,160,160,0.45)")
    fr.add_scatter(x=rots_f, y=fit["tof_res_us"],
                   name="ToF shape residual (µs)", yaxis="y2",
                   mode="markers")
    fr.update_layout(
        template="plotly_dark", height=320,
        xaxis_title="probe azimuth (deg)",
        yaxis_title="coda residual (dB, measured - model)",
        yaxis2=dict(title="ToF residual (µs)", overlaying="y",
                    side="right"),
        title="per-azimuth fit residuals; grey bars were excluded "
              "from the fit by the physics doctrine, not failures "
              "of it",
        margin=dict(l=30, r=30, t=40, b=20))
    st.plotly_chart(fr, width="stretch", key="sweep_fit_residuals")

    # ── THE CONTINUOUS GRID FITTER (Jerome's architecture: a daemon
    # that mods the grid non-stop to fit the measurements, rather than
    # a batch solve chained after each fit). The global fit above
    # remains its background/prior; grid_daemon.py owns the state.
    gs_path = os.path.join(d, "grid_state.npz")
    frp2 = os.path.join(d, "fit_result.json")
    stop_g = os.path.join(d, "GRID_STOP")
    # liveness comes from the status HEARTBEAT (refreshed even during
    # long blocking stages), not the state file (which goes stale for
    # minutes during the one-off specimen build - looked dead)
    gstat = _read_json(os.path.join(d, "grid_status.json"))
    daemon_live = bool(gstat) and time.time() - gstat.get("t", 0) < 30
    if not daemon_live:                        # pre-status fallback
        daemon_live = (os.path.exists(gs_path)
                       and time.time() - os.path.getmtime(gs_path) < 30)
    # auto-start once a fit exists; an explicit Stop leaves GRID_STOP
    # behind, which suppresses auto-restart until Start is pressed.
    # Gate on _grid_active (NOT just daemon_live) so a CLI batch
    # grid_inversion run doesn't get a daemon spawned on top of it.
    if (auto and os.path.exists(frp2) and not daemon_live
            and not _grid_active(d) and not os.path.exists(stop_g)):
        _stamp_grid_status(d)   # blocks a 5s-later double-spawn while
        _spawn([os.path.join(ROOT, "sim", "grid_daemon.py"), sel],
               os.path.join(d, "grid_daemon.log"))
        st.rerun()   # cadence was computed pre-spawn: go live NOW
    gb1, gb2 = st.columns([1, 3])
    if daemon_live:
        if gb1.button("⏹ Stop grid fitter"):
            with open(stop_g, "w", encoding="utf-8") as fh:
                fh.write("stop")
            st.rerun()
        phase = (gstat or {}).get("phase", "solving")
        gb2.caption(f"continuous grid fitter RUNNING ({phase}); it "
                    "re-solves every few seconds and folds in new "
                    "azimuths and fit updates as they land")
    else:
        if gb1.button("▶ Start grid fitter",
                      disabled=(not os.path.exists(frp2)
                                or _grid_active(d))):
            if os.path.exists(stop_g):
                os.remove(stop_g)
            _stamp_grid_status(d)
            _spawn([os.path.join(ROOT, "sim", "grid_daemon.py"), sel],
                   os.path.join(d, "grid_daemon.log"))
            time.sleep(0.5)
            st.rerun()
        gb2.caption("grid fitter stopped. State is kept; Start "
                    "resumes from the last iteration")
    if os.path.exists(gs_path):
        # copy out and CLOSE: a lingering NpzFile handle makes the
        # daemon's os.replace lose its save race (Windows)
        with np.load(gs_path) as zf:
            z = {k: np.asarray(zf[k]) for k in zf.files}
        n_g = int(z["n"])
        cell = float(z["cell_mm"])
        g1_, g2_, g3_, g4_ = st.columns(4)
        g1_.metric("coda rms at state",
                   f"{float(z['coda_rms']):.2f} dB")
        g2_.metric("ToF rms at state",
                   f"{float(z['tof_rms']):.3f} µs")
        if "texture" in z:
            _th = np.asarray(z["theta_grid"], float)
            g3_.metric("θ median vs truth",
                       (f"{float(np.nanmedian(_th)):.1f}°"
                        if np.isfinite(_th).any() else "no truth"))
            g4_.metric("grain walls",
                       f"{100 * float(z['wall_frac']):.0f}%")
        else:
            g3_.metric("truth corr: scatter",
                       f"{float(z['corr_scatter']):+.2f}")
            g4_.metric("truth corr: slowness",
                       f"{float(z['corr_slow']):+.2f}")
        _age = time.time() - os.path.getmtime(gs_path)
        st.caption(f"iteration {int(z['iteration'])}, updated "
                   f"{_age:.0f}s ago; {n_g}x{n_g} cells of "
                   f"{cell:.1f} mm; the state IS the texture: one "
                   "c-axis per cell, scatter predicted at its grain "
                   "boundaries, ToF from its phase delays; robust-TV "
                   "prior favours grains, ray-cast coda edge evidence "
                   "cheapens walls where scatter demands them")
        # measured vs grid-model overlays (Jerome's preferred display)
        if "rots" in z:
            srt = np.argsort(np.asarray(z["rots"], float))
            r_s = np.asarray(z["rots"], float)[srt]
            charts = [
                ("grid_coda_overlay",
                 np.asarray(z["coda_meas"], float)[srt],
                 np.asarray(z["coda_pred"], float)[srt],
                 "coda level (dB)",
                 "coda level: measured vs grid model"),
                ("grid_tof_overlay",
                 np.asarray(z["tof_meas"], float)[srt],
                 np.asarray(z["tof_pred"], float)[srt],
                 "ToF (µs)",
                 "ToF: measured vs grid model")]
            if "fade_meas" in z:
                charts.append(
                    ("grid_fade_overlay",
                     np.asarray(z["fade_meas"], float)[srt],
                     np.asarray(z["fade_pred"], float)[srt],
                     "E1 amplitude (dB)",
                     "E1 fade: measured vs grid model"))
            cols_ov = st.columns(len(charts))
            for col, (key_, meas, pred, ytitle, title) in zip(
                    cols_ov, charts):
                of = go.Figure()
                of.add_scatter(x=r_s, y=meas, mode="markers",
                               name="measured",
                               marker=dict(size=5, opacity=0.75))
                of.add_scatter(x=r_s, y=pred, mode="lines",
                               name="grid model",
                               line=dict(color="#ef553b", width=2))
                of.update_layout(
                    template="plotly_dark", height=340,
                    xaxis_title="probe azimuth (deg)",
                    yaxis_title=ytitle, title=title,
                    margin=dict(l=30, r=30, t=40, b=20))
                col.plotly_chart(of, width="stretch", key=key_)
        mh = np.asarray(z["misfit"], float)
        if mh.size > 1:
            mfig = go.Figure(go.Scatter(y=mh, mode="lines",
                                        line=dict(width=1.5)))
            mfig.update_layout(
                template="plotly_dark", height=160,
                yaxis_title="misfit", xaxis_title="iteration (recent)",
                title="misfit descent; flat means converged for the "
                      "data it has; steps are new azimuths or fit "
                      "updates arriving",
                margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(mfig, width="stretch", key="grid_misfit")
        ext = n_g * cell / 2
        ctr = (np.arange(n_g) + 0.5) * cell - ext
        yy, xx = np.meshgrid(ctr, ctr, indexing="ij")
        mask = (xx ** 2 + yy ** 2) <= ext ** 2
        if "texture" in z:
            # ONE product field: the recovered texture itself
            fld = np.where(mask, np.asarray(z["alpha_grid"]), np.nan)
            gfig = go.Figure(go.Heatmap(
                z=fld, x=ctr, y=ctr, colorscale="Phase",
                zmin=0.0, zmax=180.0,
                colorbar=dict(title="c-axis (deg)")))
            gfig.add_shape(type="circle", x0=-ext, y0=-ext, x1=ext,
                           y1=ext, line=dict(color="white", width=1.5))
            gfig.update_layout(
                template="plotly_dark", height=470,
                xaxis_title="x (mm)", yaxis_title="y (mm)",
                yaxis_scaleanchor="x",
                title="recovered TEXTURE: per-cell c-axis angle "
                      "(cyclic colours; uniform patches = grains)",
                margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(gfig, width="stretch", key="grid_scatter")
        else:
            h1, h2 = st.columns(2)
            second = (("kappa_grid",
                       "fabric concentration κ per cell; ONE "
                       "inversion from ToF + coda jointly")
                      if "kappa_grid" in z else
                      ("d_slow",
                       "recovered SLOWNESS field (µs/mm vs "
                       "background)"))
            for col, key_, field, title in (
                    (h1, "grid_scatter", "d_scatter",
                     "recovered SCATTERING field (dB vs global "
                     "model)"),
                    (h2, "grid_slow", second[0], second[1])):
                fld = np.where(mask, np.asarray(z[field]), np.nan)
                if field == "kappa_grid":
                    gfig = go.Figure(go.Heatmap(
                        z=fld, x=ctr, y=ctr, colorscale="Viridis",
                        zmin=0.0))
                else:
                    gfig = go.Figure(go.Heatmap(
                        z=fld, x=ctr, y=ctr, colorscale="RdBu_r",
                        zmid=0.0))
                gfig.add_shape(type="circle", x0=-ext, y0=-ext,
                               x1=ext, y1=ext,
                               line=dict(color="white", width=1.5))
                gfig.update_layout(template="plotly_dark", height=430,
                                   xaxis_title="x (mm)",
                                   yaxis_title="y (mm)",
                                   yaxis_scaleanchor="x", title=title,
                                   margin=dict(l=30, r=30, t=40,
                                               b=20))
                col.plotly_chart(gfig, width="stretch", key=key_)

    # THE spatial map - one figure, one number per square: the absolute
    # c-axis error theta = arccos|a_true . a_inv|. When the coupled
    # grid daemon is publishing per-cell axes, THIS map is its live
    # output at the current iteration (Jerome: "it should show the
    # output of the grid model inversion at the current iteration");
    # the static global-fit version below is the fallback.
    tm_path = os.path.join(d, "cof_truth_map.npz")
    live_theta = None
    if os.path.exists(gs_path):
        try:
            th_g = np.asarray(z["theta_grid"], float)
            if np.isfinite(th_g).any():
                live_theta = th_g
        except (KeyError, NameError):
            pass
    if live_theta is not None:
        n_g = int(z["n"])
        cell = float(z["cell_mm"])
        ext = n_g * cell / 2
        ctr = (np.arange(n_g) + 0.5) * cell - ext
        hm = go.Figure(go.Heatmap(
            z=live_theta, x=ctr, y=ctr,
            colorscale="Viridis", zmin=0, zmax=90,
            colorbar=dict(title="θ (deg)")))
        hm.add_shape(type="circle", x0=-ext, y0=-ext, x1=ext,
                     y1=ext, line=dict(color="white", width=1.5))
        hm.update_layout(
            template="plotly_dark", height=520,
            xaxis_title="x (mm)", yaxis_title="y (mm)",
            yaxis_scaleanchor="x",
            title="absolute c-axis error per cell: θ = arccos|a_true "
                  "· a_cell|, LIVE from the texture inversion "
                  f"at iteration {int(z['iteration'])}",
            margin=dict(l=30, r=30, t=40, b=20))
        st.plotly_chart(hm, width="stretch", key="caxis_error_map")
        _cap = (f"median θ {float(np.nanmedian(live_theta)):.1f}°. "
                "THE product: per-cell inverted c-axis vs the true "
                "local crystals, live each iteration.")
        if "theta_gauge" in z:
            _tg = float(np.nanmedian(np.asarray(z["theta_gauge"],
                                                float)))
            _cap += (f" Gauge-aware median {_tg:.1f}° (best of the "
                     "cell's mirror pair; diameters-only geometry "
                     "cannot distinguish a cell axis from its "
                     "reflection about the local beam line, so raw θ "
                     "partly grades the initialisation; the gauge "
                     "number grades what the data determined).")
        _cap += (" Out-of-plane truth tilt sets an honest floor: "
                 "the in-plane rig cannot see tilt.")
        st.caption(_cap)
        return
    # auto-update: whenever a NEWER fit exists than the map (the fit
    # auto-refits as traces arrive), rebuild the map - cheap after the
    # first build thanks to the per-sweep tensor cache
    frp = os.path.join(d, "fit_result.json")
    # NOTE: no 'not fitting' gate - with continuous refits there is
    # never a fit-free moment, which starved the map for 79 minutes;
    # map and fit only touch each other's files atomically, so
    # concurrency is safe
    if (auto and os.path.exists(frp)
            and not _map_active(d)
            and (not os.path.exists(tm_path)
                 or os.path.getmtime(tm_path) < os.path.getmtime(frp))):
        _spawn([os.path.join(ROOT, "sim", "fit_sweep.py"),
                "--map", sel], os.path.join(d, "fit.log"))
    g1, g2 = st.columns([1, 3])
    if g1.button("🗺 Build / refresh c-axis error map",
                 disabled=_map_active(d)):
        _spawn([os.path.join(ROOT, "sim", "fit_sweep.py"),
                "--map", sel], os.path.join(d, "fit.log"))
        time.sleep(0.5)
        st.rerun()
    if _map_active(d):
        g2.info("map building; it will appear here automatically")
    if os.path.exists(tm_path):
        with np.load(tm_path) as zf:
            ang = np.asarray(zf["angle_err_deg"])
            edges = np.asarray(zf["edges_mm"])
            alpha_fit = float(zf["alpha_fit_deg"])
        R_mm = float(edges[-1])
        # plotly heatmap x/y are cell CENTERS; feeding edges shifts the
        # whole grid half a cell off the circle
        centers = 0.5 * (edges[:-1] + edges[1:])
        hm = go.Figure(go.Heatmap(
            z=ang, x=centers, y=centers,
            colorscale="Viridis", zmin=0, zmax=90,
            colorbar=dict(title="θ (deg)")))
        hm.add_shape(type="circle", x0=-R_mm, y0=-R_mm, x1=R_mm,
                     y1=R_mm, line=dict(color="white", width=1.5))
        hm.update_layout(
            template="plotly_dark", height=520,
            xaxis_title="x (mm)", yaxis_title="y (mm)",
            yaxis_scaleanchor="x",
            title="absolute c-axis error per square: "
                  "θ = arccos|a_true · a_inverted|  "
                  "(dark = inversion matches the local crystals)",
            margin=dict(l=30, r=30, t=40, b=20))
        st.plotly_chart(hm, width="stretch", key="caxis_error_map")
        _age = time.time() - os.path.getmtime(tm_path)
        st.caption(f"map rebuilt {_age:.0f}s ago against the fitted "
                   f"axis {alpha_fit:.2f}°; a static "
                   "image means the axis estimate is stable, not that "
                   "updates stopped. "
                   f"median θ {float(np.nanmedian(ang)):.1f}°. For a "
                   "homogeneous fabric this is the crystals' own "
                   "Watson scatter about the (correctly) recovered "
                   "axis; localized bright patches on a structured "
                   "specimen would be real fabric anomalies.")

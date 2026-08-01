"""Pulse-Echo COF Studio - rotational pulse-echo ultrasound on an ice disk.

Layout deliberately mirrors OpenUSCT Studio: tabbed workflow across the top,
configuration tabs first, then a gated ACQUISITION tab, then reconstruction
tabs that unlock only once acquired data exists.
"""
import sys, os, time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")

import specimen as S                       # noqa: E402
import viz                                 # noqa: E402
import acquisition as ACQ                  # noqa: E402
import errors as ERR                       # noqa: E402
import reconstruct as RECON                # noqa: E402
from specimen import DiskSpecimen          # noqa: E402
from config import Config                  # noqa: E402

# Do NOT probe the GPU here. Streamlit runs this script in a worker thread and
# initialising CUDA off the main thread segfaults the whole server with no
# traceback. The ray forward model is pure numpy anyway; the GPU is only needed
# by the full-wave reference solver, which probes lazily when it is actually
# run. Detect by inspection, never by touching the driver.
def _backend_name():
    import importlib.util
    if importlib.util.find_spec("cupy") is None:
        return "CPU (numpy)"
    have_cuda = any(os.environ.get(v) for v in ("CUDA_PATH", "CUDA_HOME"))
    return "numpy forward, CUDA available for full-wave" if have_cuda \
        else "CPU (numpy), cupy present but no CUDA_PATH"


BACKEND_NAME = _backend_name()

st.set_page_config(page_title="Pulse-Echo COF Studio", layout="wide",
                   page_icon="\U0001f9ca")

st.markdown(
    '<div style="display:flex;align-items:center;gap:0.8rem;padding:0.2rem 0 0.6rem 0;">'
    '<svg width="52" height="52" viewBox="0 0 512 512" style="flex:0 0 auto;" '
    'xmlns="http://www.w3.org/2000/svg"><g transform="translate(256,256)">'
    '<circle r="200" fill="#eaf4fa" stroke="#1b4f72" stroke-width="16"/>'
    '<path d="M-200 0 L200 0" stroke="#2f7fb8" stroke-width="12"/>'
    '<circle r="120" fill="none" stroke="#54a8d6" stroke-width="8" '
    'stroke-dasharray="26 20"/><circle r="60" fill="none" stroke="#54a8d6" '
    'stroke-width="8" stroke-dasharray="18 14"/>'
    '<rect x="-250" y="-34" width="56" height="68" rx="8" fill="#1b4f72"/>'
    '<circle r="22" fill="#2f7fb8"/></g></svg>'
    '<span style="font-size:2.4rem;font-weight:700;">Pulse-Echo COF Studio</span>'
    "</div>", unsafe_allow_html=True)
st.caption("Rotational single-transducer pulse-echo on an ice disk: forward "
           f"simulation, error model and COF reconstruction. Backend: {BACKEND_NAME}.")
st.markdown("""<style>
/* keep stale elements fully opaque during reruns */
[data-stale="true"] { opacity: 1 !important; }
.element-container.stale-element { opacity: 1 !important; }
div[data-testid="stAppViewContainer"] .element-container {
  transition: none !important; }
</style>""", unsafe_allow_html=True)

# ─────────────────── workflow-gated tabs (OpenUSCT pattern) ───────────────────
_built = "spec" in st.session_state
_acquired = "scan" in st.session_state

_tab_names = ["Specimen & Fabric", "Probe & Excitation", "Error model",
              ("Acquisition ✅" if _acquired else "**Acquisition**")]
if _acquired:
    _tab_names += ["Backscatter", "COF reconstruction"]
else:
    st.info("Set up the specimen, probe and error model, then run the "
            "ACQUISITION tab. The backscatter and reconstruction tabs "
            "unlock once data exists.")
_tab_names += ["FW sweep"]
_tabs = st.tabs(_tab_names)
tab_spec, tab_probe, tab_err, tab_acq = _tabs[0], _tabs[1], _tabs[2], _tabs[3]
tab_bs = _tabs[4] if _acquired else None
tab_cof = _tabs[5] if _acquired else None
tab_sweep = _tabs[-1]

CFG = Config()


@st.cache_data(show_spinner=False)
def _build(dia, thk, n, cv, kap, cor, axis, sd, ppw):
    sp = DiskSpecimen(diameter_m=dia * 1e-3, thickness_m=thk * 1e-3, n_grains=n,
                      size_cv=cv, concentration=kap, spatial_corr=cor,
                      fabric_axis=axis, seed=sd)
    t0 = time.time()
    b = sp.build(3850.0 / 5e6 / ppw)
    return sp, b, time.time() - t0


@st.cache_data(show_spinner="Meshing grains...")
def _poly3d(labels, vals, h, title, vmax, cap, step, lbl):
    return viz.polycrystal_3d(labels, vals, h, title, vmin=0.0, vmax=vmax,
                              max_grains=cap, step=step, value_label=lbl)


@st.cache_data(show_spinner="Building c-axis field...")
def _field3d(labels, axes, h):
    return viz.caxis_field_3d(labels, axes, h)


# ══════════════════════════ 1. SPECIMEN & FABRIC ══════════════════════════
with tab_spec:
    cs1, cs2 = st.columns(2)
    with cs1:
        st.subheader("Disk geometry")
        g1, g2 = st.columns(2)
        dia = g1.number_input("Diameter (mm)", 20.0, 300.0, 100.0, 5.0)
        thk = g2.number_input("Thickness (mm)", 2.0, 200.0, 35.0, 1.0)
        n_grains = st.slider("Grain count", 50, 2000, 100, 50)
        size_cv = st.slider("Grain size CV", 0.05, 0.90, 0.35, 0.05,
                            help="Spread of grain sizes about the mean.")
        seed = st.number_input("Random seed", 0, 9999, 7, 1)
    with cs2:
        st.subheader("Crystal orientation fabric")
        a1 = st.slider("Fabric strength, eigenvalue a₁ (BULK)",
                       0.34, 0.99, 0.70, 0.01,
                       help="The number thin sections actually report. "
                            "0.33 = random, 1.0 = perfect single maximum. "
                            "Drives the bulk 2-phi velocity anisotropy.")
        kappa = S.kappa_for_eigenvalue(a1)
        st.caption(f"a₁ = {a1:.2f}  ->  Watson κ = {kappa:.2f}"
                   f"   ({'random' if a1 < 0.40 else 'weak' if a1 < 0.55 else 'moderate' if a1 < 0.75 else 'strong' if a1 < 0.90 else 'very strong'} single maximum)")
        corr = st.slider("Spatial correlation (LOCAL disorder)", 0.0, 0.95, 0.0, 0.05,
                         help="How alike neighbouring grains are. Drives "
                              "misorientation, hence scattering strength.")
        # default IN-PLANE: the azimuthal method reads the c-axis through
        # the 2-phi patterns, which a vertical (disk-normal) fabric does
        # not produce - vertical is available but it is the null case
        ax_choice = st.selectbox("Predominant c-axis",
                                 ["in-plane x (1,0,0)", "vertical (0,0,1)",
                                  "in-plane y (0,1,0)", "tilted (0,0.5,0.87)"])
        AXES = {"vertical (0,0,1)": (0, 0, 1), "in-plane x (1,0,0)": (1, 0, 0),
                "in-plane y (0,1,0)": (0, 1, 0), "tilted (0,0.5,0.87)": (0, 0.5, 0.87)}
        ppw_prev = st.select_slider("Preview cells per wavelength",
                                    [1, 1.5, 2, 3, 4, 6], value=1.5,
                                    help="Preview only. The solver uses "
                                         "config.Solver.ppw, currently "
                                         f"{CFG.solver.ppw}.")

    if st.button("Build specimen", type="primary"):
        with st.spinner("Building grains..."):
            st.session_state.spec = _build(dia, thk, n_grains, size_cv, kappa,
                                           corr, AXES[ax_choice], seed, ppw_prev)
        st.rerun()

    if _built:
        sp, b, dt = st.session_state.spec
        labels, axes = b["labels"], b["axes"]
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        w, V = S.orientation_tensor(axes)
        pr = [(u, v) for u, ns in b["adjacency"].items() for v in ns if u < v]
        mis = S.disorientation_deg(axes[[p[0] for p in pr]], axes[[p[1] for p in pr]])
        cnt = np.bincount(labels[labels >= 0].ravel(), minlength=len(axes))
        d_eq = 2 * (3 * cnt[cnt > 0] * b["h"] ** 3 / (4 * np.pi)) ** (1 / 3) * 1e3
        m1.metric("Grains", f"{int((cnt>0).sum())}")
        m2.metric("Mean grain size", f"{d_eq.mean():.1f} mm")
        m3.metric("Fabric eigenvalue 1", f"{w[0]:.3f}", help="1/3 random, 1.0 single max")
        m4.metric("Mean disorientation", f"{mis.mean():.1f}°",
                  help="Drives scattering strength")

        sv1, sv2, sv3, sv4 = st.tabs(["3D grains", "c-axis field",
                                      "Pole figure", "2D slice"])
        colat = viz.colatitude_deg(axes)
        azim = viz.azimuth_deg(axes)

        with sv1:
            k1, k2, k3 = st.columns([2, 2, 3])
            colour_by = k1.selectbox("Colour by", ["c-axis colatitude",
                                                   "in-plane azimuth"])
            cap = k2.slider("Max grains drawn", 25, 1000, 400, 25,
                            help="Largest grains are kept. Lower this if the "
                                 "browser struggles.")
            dstep = k3.select_slider("Mesh detail", [1, 2, 3],
                                     value=1, format_func=lambda s:
                                     {1: "full", 2: "half", 3: "coarse"}[s])
            vals, vmax, lbl = ((colat, 90.0, "colat") if colour_by.startswith("c-axis")
                               else (azim, 180.0, "azim"))
            st.plotly_chart(_poly3d(labels, vals, b["h"], colour_by, vmax, cap,
                                    dstep, lbl), width="stretch")
            st.caption("Each grain is an isosurface coloured by its own c-axis. "
                       "Drag to rotate. This is the fabric the ultrasound sees.")

        with sv2:
            st.plotly_chart(_field3d(labels, axes, b["h"]), width="stretch")
            st.caption("Bars are double-ended because +c and -c are the same "
                       "crystal direction. Bar length scales with grain size.")

        with sv3:
            pf1, pf2 = st.columns([3, 2])
            pf1.plotly_chart(viz.pole_figure(axes), width="stretch")
            pf2.metric("Eigenvalue 1", f"{w[0]:.3f}")
            pf2.metric("Eigenvalue 2", f"{w[1]:.3f}")
            pf2.metric("Eigenvalue 3", f"{w[2]:.3f}")
            pf2.caption("Equal-area (Schmidt) lower hemisphere. "
                        "1/3,1/3,1/3 is random; 1,0,0 is a single maximum.")

        with sv4:
            nz = labels.shape[2]
            q1, q2 = st.columns(2)
            kz = q1.slider("z slice", 0, nz - 1, nz // 2)
            sl = labels[:, :, kz].astype(float)
            idx = np.clip(sl.astype(int), 0, None)
            rng = np.random.default_rng(0)
            lut = rng.permutation(len(axes))
            with q1:
                fig, ax = plt.subplots(figsize=(4.4, 4.4))
                ax.imshow(np.where(sl >= 0, lut[idx], np.nan).T, cmap="tab20",
                          origin="lower", interpolation="nearest")
                ax.set_title("grain labels"); ax.set_xticks([]); ax.set_yticks([])
                st.pyplot(fig, use_container_width=True)
            with q2:
                st.write("")
                fig2, ax2 = plt.subplots(figsize=(4.4, 4.4))
                im = ax2.imshow(np.where(sl >= 0, azim[idx], np.nan).T, cmap="hsv",
                                origin="lower", vmin=0, vmax=180,
                                interpolation="nearest")
                ax2.set_title("in-plane c-axis azimuth (deg)")
                ax2.set_xticks([]); ax2.set_yticks([])
                fig2.colorbar(im, ax=ax2, fraction=0.046)
                st.pyplot(fig2, use_container_width=True)

        with st.expander("Full specimen diagnostics"):
            st.code(sp.diagnose(b), language=None)
            st.caption(f"built in {dt:.1f}s at preview resolution, grid {b['shape']}")

# ══════════════════════════ 2. PROBE & EXCITATION ══════════════════════════
with tab_probe:
    p = CFG.probe
    cp1, cp2 = st.columns(2)
    with cp1:
        st.subheader("Transducer")
        f0 = st.number_input("Centre frequency (MHz)", 0.5, 20.0, p.f0 / 1e6, 0.5)
        el = st.number_input("Element diameter (mm)", 1.0, 25.0,
                             p.element_d_m * 1e3, 0.05)
        bw = st.slider("Fractional bandwidth", 0.2, 1.0, p.frac_bw, 0.05)
        st.caption("Evident V3XX broadband planar contact, single element [thesis]")
    with cp2:
        st.subheader("Pulser (OPLabBox / OPTEL)")
        st.number_input("@PAMP pulser amplitude", 0.0, 255.0, p.pamp, 1.0,
                        help="Units undocumented in the control code.")
        pw = st.number_input("@PWDT pulse width (us)", 0.02, 2.0,
                             p.pulse_width_s * 1e6, 0.02,
                             help="Half-period at f0 is optimal. The thesis used "
                                  "0.2/0.5 us, which suit 2.5/1.0 MHz.")
        st.number_input("@GAIN receiver gain (dB)", 0.0, 80.0, p.gain_db, 1.0)
        if abs(pw - 0.5 / f0) > 0.25 * (0.5 / f0):
            st.warning(f"Pulse width {pw:.2f} us is mismatched to {f0:.1f} MHz "
                       f"(optimal {0.5/f0:.2f} us) - this costs bandwidth, "
                       "hence range resolution.")

    from config import Probe as _P
    pv = _P(f0=f0 * 1e6, element_d_m=el * 1e-3, frac_bw=bw)
    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Wavelength", f"{pv.wavelength()*1e3:.2f} mm")
    r2.metric("Range resolution", f"{pv.range_res_m()*1e3:.2f} mm", help="c/2B")
    r3.metric("Cross-range", f"{pv.cross_range_m()*1e3:.2f} mm",
              help="D/4.88 - set by aperture only, independent of frequency")
    r4.metric("Near field", f"{pv.near_field_m()*1e3:.1f} mm")
    st.caption(f"Beam half-angle {np.degrees(pv.beam_half_angle()):.1f}°; "
               f"beam diameter at disk centre {pv.beam_diameter_at(dia/2*1e-3)*1e3:.1f} mm "
               f"versus {thk:.0f} mm disk thickness.")

# ══════════════════════════ 3. ERROR MODEL ══════════════════════════
with tab_err:
    e = CFG.err
    st.subheader("Realistic error sources")
    st.caption("Every term degrades the synthetic data. Disable all to get the "
               "ideal case and isolate which error actually limits the result.")
    on = st.toggle("Errors enabled", value=True)
    ce1, ce2, ce3 = st.columns(3)
    with ce1:
        st.markdown("**Angular**")
        enc = st.checkbox("AS5600 at specimen", value=True,
                          help="Confirmed build. Measures true position, so belt "
                               "and microstep errors stop mattering.")
        enc_rms = st.number_input("Encoder RMS (deg)", 0.0, 1.0,
                                  e.encoder_rms_deg, 0.005,
                                  disabled=not enc, format="%.3f")
        ustep = st.number_input("Open-loop microstep accuracy (deg)", 0.0, 3.0,
                                e.microstep_accuracy_deg, 0.01, disabled=enc)
        st.caption("Thesis quoted ±2.3° (k=2) for the OLD open-loop rig, "
                   "explicitly 'specification-derived, no calibrated budget'.")
    with ce2:
        st.markdown("**Couplant & thermal**")
        gap_um = st.number_input("Couplant gap sigma (um)", 0.0, 200.0,
                                 e.couplant_gap_sigma_m * 1e6, 5.0)
        dT = st.number_input("Temperature swing (±K)", 0.0, 5.0,
                             e.T_cycle_amplitude_K, 0.1)
        st.caption(f"dv/dT = {e.dvdT} m/s/K -> {abs(e.dvdT)*dT:.1f} m/s")
        oiled = st.checkbox("Far rim oiled", value=e.rim_is_oiled)
    with ce3:
        st.markdown("**Specimen & electronics**")
        runout_um = st.number_input("Rim runout (um)", 0.0, 500.0,
                                    e.rim_runout_m * 1e6, 5.0)
        jit_ns = st.number_input("Trigger jitter (ns)", 0.0, 50.0,
                                 e.trigger_jitter_s * 1e9, 0.5)
        noise = st.number_input("Receiver noise (frac. full scale)", 0.0, 0.05,
                                0.005, 0.0005, format="%.4f",
                                help="Counter-intuitive but real: MORE noise "
                                     "helps. Below ~0.39% of full scale the "
                                     "8-bit ADC rounds every shot identically, "
                                     "so averaging recovers nothing.")
        _lsb = 1.0 / 2 ** (CFG.acq.adc_bits - 1)
        if noise >= 0.5 * _lsb:
            st.success(f"Dithered: noise > 0.5 LSB ({50*_lsb:.2f}%), so "
                       f"{CFG.acq.n_averages} averages buy "
                       f"{np.log2(np.sqrt(CFG.acq.n_averages)):.1f} extra bits.")
        else:
            st.error(f"NOT dithered: noise < 0.5 LSB ({50*_lsb:.2f}%). "
                     "Quantisation is deterministic and averaging gains "
                     "nothing. This is the single biggest limit on the "
                     "grain-boundary coda.")
        st.caption("Ground-truth mismatch: optical section "
                   f"{CFG.optical.section_thickness_m*1e3:.1f} mm vs beam band "
                   f"~{pv.beam_diameter_at(dia/2*1e-3)*1e3:.0f} mm")

# ══════════════════════════ 4. ACQUISITION ══════════════════════════
with tab_acq:
    a = CFG.acq
    ca1, ca2 = st.columns(2)
    with ca1:
        st.subheader("Scan")
        az_step = st.number_input("Azimuth step (deg)", 0.5, 10.0,
                                  a.az_step_deg, 0.5)
        n_avg = st.number_input("Averages", 1, 4096, a.n_averages, 1,
                                help="DSOX3024A does 1024 in hardware")
        reseat = st.checkbox("Lift and reseat each angle", value=a.lift_and_reseat,
                             help="Thesis: Z retracts to clearance after every "
                                  "angle. This redraws the couplant gap, which "
                                  "is why it costs precision.")
        scan_seed = st.number_input("Scan noise seed", 0, 9999, 1, 1)
    with ca2:
        st.subheader("Record")
        rec_us = st.number_input("Record end (us)", 10.0, 500.0,
                                 a.record_end_s * 1e6, 5.0,
                                 help="Must exceed E2 to capture it")
        fs_msps = st.number_input("Sample rate (MS/s)", 10.0, 4000.0,
                                  a.fs / 1e6, 10.0)
        n_rev = st.slider("Rim round trips (E1, E2, ...)", 1, 4, a.n_reverb)
        c = CFG.solver.c_ref
        st.caption(f"E1 at {2*dia*1e-3/c*1e6:.1f} us, E2 at "
                   f"{4*dia*1e-3/c*1e6:.1f} us at the reference speed")
        tilt_opt = st.selectbox(
            "Out-of-plane probe tilt",
            ["none (0)", "0, +8", "-8, 0, +8", "-9.5, -5, 0, +5, +9.5"], index=2,
            help="With every ray in the scan plane, n_z never multiplies "
                 "anything, so the c-axis is only recoverable up to reflection "
                 "in that plane. Tilting breaks that degeneracy.")
        TILTS = {"none (0)": (0.0,), "0, +8": (0.0, 8.0),
                 "-8, 0, +8": (-8.0, 0.0, 8.0),
                 "-9.5, -5, 0, +5, +9.5": (-9.5, -5.0, 0.0, 5.0, 9.5)}
        tmax = np.degrees(np.arctan(thk / 2 / dia))
        st.caption(f"Geometric ceiling for this disk: ±{tmax:.1f}° from "
                   f"mid-height (the ray must not exit the flat face).")

    def _cfg_from_ui():
        """Collect every widget on every tab into one Config."""
        cf = Config()
        cf.specimen.diameter_m, cf.specimen.thickness_m = dia * 1e-3, thk * 1e-3
        cf.specimen.n_grains = n_grains
        cf.probe.f0, cf.probe.element_d_m = f0 * 1e6, el * 1e-3
        cf.probe.frac_bw = bw
        cf.acq.az_step_deg, cf.acq.n_averages = az_step, int(n_avg)
        cf.acq.lift_and_reseat, cf.acq.n_reverb = reseat, int(n_rev)
        cf.acq.record_end_s, cf.acq.fs = rec_us * 1e-6, fs_msps * 1e6
        cf.acq.tilts_deg = TILTS[tilt_opt]
        cf.err.enabled = on
        cf.err.encoder_on_specimen, cf.err.encoder_rms_deg = enc, enc_rms
        cf.err.microstep_accuracy_deg = ustep
        cf.err.couplant_gap_sigma_m = gap_um * 1e-6
        cf.err.T_cycle_amplitude_K, cf.err.rim_is_oiled = dT, oiled
        cf.err.rim_runout_m = runout_um * 1e-6
        cf.err.trigger_jitter_s, cf.err.noise_rms_frac = jit_ns * 1e-9, noise
        return cf

    st.divider()
    bb1, bb2 = st.columns([3, 2])
    with bb2:
        st.markdown("**Error budget at these settings**")
        rows = ERR.budget(_cfg_from_ui())
        st.dataframe({"source": [r[0] for r in rows],
                      "ns": [round(r[1] * 1e9, 1) for r in rows],
                      "m/s": [round(r[2], 2) for r in rows]},
                     hide_index=True, width="stretch")

    with bb1:
        if not _built:
            st.warning("Build a specimen first (Specimen & Fabric tab).")
        else:
            n_ang = int(round(360.0 / az_step))
            st.caption(f"{n_ang} angles, one A-scan each. "
                       f"Rays: {a.n_rays_inplane} in-plane x "
                       f"{a.n_rays_outplane} out-of-plane per angle.")
            if st.button("Run acquisition", type="primary"):
                cfa = _cfg_from_ui()
                bar = st.progress(0.0, text="scanning...")

                def _prog(frac, i, n, el_s):
                    bar.progress(frac, text=f"angle {i}/{n}  ({el_s:.1f}s)")

                st.session_state.scan = ACQ.run_scan(
                    st.session_state.spec[1], cfa, seed=int(scan_seed),
                    progress=_prog)
                bar.empty()
                st.rerun()

    if _acquired:
        sc = st.session_state.scan
        st.success(f"Scan complete: {sc['sinogram'].shape[0]} angles x "
                   f"{sc['sinogram'].shape[1]} samples in {sc['wall_s']:.1f}s")

# ══════════════════════ 5-6. unlocked after acquisition ══════════════════════
if tab_bs is not None:
    with tab_bs:
        import plotly.graph_objects as go
        sc = st.session_state.scan
        cfs, tt, sino = sc["config"], sc["t"], sc["sinogram"]
        deg = np.degrees(sc["angle_cmd"])
        st.subheader("Backscatter")

        d1, d2 = st.columns([3, 2])
        with d1:
            env = ACQ.envelope(sino)
            db = 20 * np.log10(env / (env.max() + 1e-30) + 1e-6)
            floor = st.slider("Display floor (dB)", -120, -10, -60, 5)
            fig = go.Figure(go.Heatmap(
                z=db, x=tt * 1e6, y=deg, zmin=floor, zmax=0,
                colorscale="Inferno", colorbar=dict(title="dB")))
            c0 = cfs.solver.c_ref
            for k in range(1, cfs.acq.n_reverb + 1):
                fig.add_vline(x=2 * k * cfs.specimen.diameter_m / c0 * 1e6,
                              line=dict(color="cyan", width=1, dash="dot"),
                              annotation_text=f"E{k}")
            fig.update_layout(template="plotly_dark", height=520,
                              title="B-scan: every A-scan versus azimuth",
                              xaxis_title="time (us)", yaxis_title="azimuth (deg)",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")
        with d2:
            ia = st.slider("Azimuth index", 0, len(deg) - 1, 0)
            fa = go.Figure()
            fa.add_trace(go.Scatter(x=tt * 1e6, y=sino[ia], mode="lines",
                                    name="A-scan", line=dict(width=1)))
            fa.add_trace(go.Scatter(x=tt * 1e6, y=ACQ.envelope(sino[ia]),
                                    mode="lines", name="envelope",
                                    line=dict(width=2)))
            fa.update_layout(template="plotly_dark", height=520,
                             title=f"A-scan at {deg[ia]:.0f} deg",
                             xaxis_title="time (us)", yaxis_title="amplitude",
                             margin=dict(l=0, r=0, t=40, b=0),
                             legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fa, width="stretch")

        st.caption("Everything before E1 is grain-boundary backscatter. That "
                   "coda is the signal carrying spatial information; the E1 "
                   "arrival alone only gives one bulk number per angle.")

if tab_cof is not None:
    with tab_cof:
        import plotly.graph_objects as go
        sc = st.session_state.scan
        cfs = sc["config"]
        st.subheader("COF reconstruction")

        v_meas, amp = ACQ.bulk_velocity_curve(sc)
        v_true = np.array([t["mean_speed"] for t in sc["truth"]])
        ang = sc["angle_true"]
        v0, A2, p2 = ACQ.fit_two_phi(ang, v_meas)
        v0t, A2t, p2t = ACQ.fit_two_phi(ang, v_true)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean speed", f"{v0:.1f} m/s", f"{v0 - v0t:+.1f} vs truth")
        m2.metric("2-phi amplitude", f"{A2:.2f} m/s", f"{A2 - A2t:+.2f} vs truth")
        m3.metric("Fast axis", f"{p2:.1f}°", f"{p2 - p2t:+.1f}° vs truth")
        m4.metric("Residual scatter", f"{np.nanstd(v_meas - v_true):.1f} m/s",
                  help="How much the error model costs you")

        fg = go.Figure()
        fg.add_trace(go.Scatter(x=np.degrees(ang), y=v_true, mode="lines",
                                name="true diameter-average",
                                line=dict(width=3)))
        fg.add_trace(go.Scatter(x=np.degrees(ang), y=v_meas, mode="markers",
                                name="measured from E1", marker=dict(size=4)))
        gg = np.linspace(0, 2 * np.pi, 361)
        fg.add_trace(go.Scatter(
            x=np.degrees(gg), y=v0 + A2 * np.cos(2 * (gg - np.radians(p2))),
            mode="lines", name="2-phi fit", line=dict(dash="dash")))
        fg.update_layout(template="plotly_dark", height=430,
                         title="Bulk anisotropy: the measurement you already have",
                         xaxis_title="azimuth (deg)", yaxis_title="speed (m/s)",
                         margin=dict(l=0, r=0, t=40, b=0),
                         legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fg, width="stretch")

        st.divider()
        st.markdown("#### Per-grain velocities along one diameter")
        st.caption("This is the target. Interval times between internal echoes "
                   "give d/(t_i - t_i-1) per grain, because the optical geometry "
                   "is known. The inversion that fits these is the next module.")
        ig = st.slider("Azimuth for the per-grain view", 0,
                       len(ang) - 1, 0, key="cof_ang")
        tr = sc["truth"][ig]
        if len(tr["grains"]):
            fb = go.Figure(go.Bar(
                x=[f"g{g}" for g in tr["grains"]], y=tr["speed"],
                marker=dict(color=tr["speed"], colorscale="twilight",
                            colorbar=dict(title="m/s")),
                text=[f"{L*1e3:.1f} mm" for L in tr["length_m"]],
                textposition="outside"))
            fb.update_layout(template="plotly_dark", height=360,
                             title=f"True per-grain qP speed at "
                                   f"{np.degrees(ang[ig]):.0f} deg "
                                   f"({len(tr['grains'])} grains crossed)",
                             yaxis_title="speed (m/s)",
                             yaxis_range=[tr["speed"].min() - 20,
                                          tr["speed"].max() + 30],
                             margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fb, width="stretch")
            st.caption(f"Harmonic mean {tr['mean_speed']:.1f} m/s is all the E1 "
                       f"arrival can tell you. The {len(tr['grains'])} individual "
                       "values are what the coda has to recover.")

        st.divider()
        st.markdown("#### Per-grain c-axis inversion")
        rc1, rc2, rc3 = st.columns([2, 2, 3])
        true_picks = rc1.checkbox("Use exact arrival times", value=False,
                                  help="Bypasses echo picking. If the fit is "
                                       "good here but bad with real picks, the "
                                       "limit is PICKING, not the inversion.")
        min_v = rc2.slider("Minimum views per grain", 2, 12, 3)
        if rc3.button("Run reconstruction", type="primary"):
            pb = st.progress(0.0, text="fitting c-axes...")
            st.session_state.recon = RECON.run(
                sc, st.session_state.spec[1], use_true_picks=true_picks,
                min_views=min_v,
                progress=lambda f, i, n: pb.progress(f, text=f"trace {i}/{n}"))
            pb.empty()

        if "recon" in st.session_state:
            res = st.session_state.recon
            s = RECON.score(res)
            if s["n"] == 0:
                st.warning("No grain had enough views. Lower the minimum, "
                           "or add tilts to increase coverage.")
            else:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Grains solved", f"{s['n']}")
                q2.metric("Median error", f"{s['median']:.1f}°",
                          f"{s['median']-57.3:.1f} vs random",
                          delta_color="inverse")
                q3.metric("Modulo mirror", f"{s['median_mod_mirror']:.1f}°",
                          help="Allowing the reflection twin an in-plane scan "
                               "provably cannot distinguish.")
                q4.metric("Within 20°", f"{s['within20']:.0f}%")

                gap = s["median"] - s["median_mod_mirror"]
                if gap > 3 and not res["tilted"]:
                    st.warning(f"{gap:.1f}° of the error is the mirror "
                               "degeneracy alone. Every ray lies in the scan "
                               "plane, so n_z never enters. Add out-of-plane "
                               "tilt on the Acquisition tab to remove it.")

                e1, e2 = st.columns(2)
                with e1:
                    fh = go.Figure(go.Histogram(x=res["error_deg_modulo_mirror"],
                                                nbinsx=30))
                    fh.add_vline(x=57.3, line=dict(color="red", dash="dash"),
                                 annotation_text="random")
                    fh.update_layout(template="plotly_dark", height=340,
                                     title="Disorientation error (mod mirror)",
                                     xaxis_title="degrees", yaxis_title="grains",
                                     margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fh, width="stretch")
                with e2:
                    fv = go.Figure(go.Scatter(
                        x=res["n_views"], y=res["error_deg_modulo_mirror"],
                        mode="markers",
                        marker=dict(size=7, color=res["error_deg_modulo_mirror"],
                                    colorscale="Inferno_r", showscale=False)))
                    fv.update_layout(template="plotly_dark", height=340,
                                     title="Error versus how often the grain was crossed",
                                     xaxis_title="views", yaxis_title="error (deg)",
                                     margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fv, width="stretch")

                pc1, pc2 = st.columns(2)
                pc1.plotly_chart(viz.pole_figure(res["axes_true"],
                                                 "TRUE c-axes (solved grains)"),
                                 width="stretch")
                pc2.plotly_chart(viz.pole_figure(res["axes_rec"],
                                                 "RECOVERED c-axes"),
                                 width="stretch")


# ────────────────────── FW azimuth sweep (resumable batch) ──────────────────
# The whole tab lives in sweep_tab.py (fragment-based live panel - see
# its docstring for why: the old sleep+rerun cycle ghost-duplicated
# charts under this app's anti-flicker CSS).
with tab_sweep:
    import sweep_tab
    sweep_tab.render()

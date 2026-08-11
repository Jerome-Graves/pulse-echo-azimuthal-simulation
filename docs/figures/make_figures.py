"""Regenerates every generated figure in docs/figures/.

Run from this directory:  python make_figures.py

All figures come from tracked data or seed-repeatable specimen builds,
so this script reproduces them exactly:

    specimen-3d.png        3D grain rendering (README hero, manual title)
    specimen-fabric.png    mid-plane slice: grain labels + azimuth field
    pole-figure.png        equal-area pole figure of the same fabric
    trace-anatomy.png      annotated clean 2 MHz reference trace
    sweep-observables.png  ToF and coda vs azimuth for the 360-az sweep
    fe-arbiter.png         FDTD vs FE grain-scatter envelopes (round 5)

Requirements: the pinned analysis environment plus matplotlib; the two
plotly figures (specimen-3d, pole-figure) additionally need plotly and
kaleido. CPU only; the specimen build takes a few minutes.

arc-backprojection.png is not generated here: it is a copy of the arc
image from the legacy demo sweep (out/sweeps/singlemax_seed7_ppw6_
fittest_legacy/arc_image.png).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path[:0] = [os.path.join(sys.path[0], d)
                for d in ("core", "model", "pipeline")]
sys.path.insert(0, os.path.join(ROOT, "vendor"))

import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import specimen as S
from specimen import DiskSpecimen
import viz


def build_specimen():
    """The reference 300-grain specimen used by all fabric figures."""
    kappa = S.kappa_for_eigenvalue(0.70)
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=300,
                      size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    return sp.build(3850.0 / 5e6 / 2.0)


def fig_specimen_3d(b):
    from PIL import Image
    azim = viz.azimuth_deg(b["axes"])
    fig = viz.polycrystal_3d(b["labels"], azim, b["h"], "",
                             vmin=0.0, vmax=180.0, max_grains=400, step=1,
                             value_label="azim")
    fig.update_layout(width=1600, height=1100,
                      margin=dict(l=0, r=0, t=0, b=0), annotations=[],
                      scene_camera=dict(eye=dict(x=1.15, y=1.15, z=0.65)))
    fig.update_scenes(annotations=[])
    out = os.path.join(HERE, "specimen-3d.png")
    fig.write_image(out, scale=2)
    img = Image.open(out)
    w, h = img.size
    img.crop((int(w * 0.01), int(h * 0.06),
              int(w * 0.99), int(h * 0.99))).save(out)


def fig_specimen_fabric(b):
    labels, axes = b["labels"], b["axes"]
    azim = viz.azimuth_deg(axes)
    sl = labels[:, :, labels.shape[2] // 2].astype(float)
    idx = np.clip(sl.astype(int), 0, None)
    lut = np.random.default_rng(0).permutation(len(axes))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4))
    ax.imshow(np.where(sl >= 0, lut[idx], np.nan).T, cmap="tab20",
              origin="lower", interpolation="nearest")
    ax.set_title("grain structure")
    ax.set_xticks([]); ax.set_yticks([])
    im = ax2.imshow(np.where(sl >= 0, azim[idx], np.nan).T, cmap="hsv",
                    origin="lower", vmin=0, vmax=180,
                    interpolation="nearest")
    ax2.set_title("in-plane c-axis azimuth (deg)")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "specimen-fabric.png"), dpi=150,
                facecolor="white", bbox_inches="tight")


def fig_pole(b):
    pf = viz.pole_figure(b["axes"])
    pf.update_layout(width=800, height=760,
                     margin=dict(l=10, r=10, t=30, b=10))
    pf.write_image(os.path.join(HERE, "pole-figure.png"), scale=2)


def fig_trace_anatomy():
    d = np.load(os.path.join(ROOT, "sim", "ref", "fw_fabric00.npz"))
    tr, dt = d["trace"], float(d["dt"])
    t_us = np.arange(len(tr)) * dt * 1e6
    env = np.abs(hilbert(tr - tr.mean()))
    sel = t_us > 40
    e1_i = int(np.where(sel)[0][np.argmax(env[sel])])
    env_db = 20 * np.log10(np.maximum(env / env[e1_i], 1e-8))
    src_i = int(np.argmax(env[t_us < 5]))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(t_us, env_db, lw=0.9, color="#1b4f72")
    ax.annotate("source\nfeed-through", (t_us[src_i], env_db[src_i]),
                xytext=(4, 15), fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7))
    ax.axvline(t_us[e1_i], color="#c0392b", lw=1, ls="--")
    ax.annotate("back-wall echo E1", (t_us[e1_i], env_db[e1_i]),
                xytext=(t_us[e1_i] - 21, 18), fontsize=9, color="#c0392b")
    ax.axvspan(8, 46, color="#54a8d6", alpha=0.15)
    ax.annotate("grain-scattered backscatter (coda)", (27, -18),
                fontsize=9, color="#1b4f72", ha="center")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("envelope (dB re E1)")
    ax.set_ylim(-60, 32); ax.set_xlim(0, t_us[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "trace-anatomy.png"), dpi=150,
                facecolor="white", bbox_inches="tight")


def fig_sweep_observables():
    sw = os.path.join(ROOT, "out", "sweeps", "singlemax_seed11_ppw6_rigid2")
    az, tof, coda = [], [], []
    for f in sorted(os.listdir(sw)):
        if f.startswith("az") and f.endswith(".npz"):
            a = np.load(os.path.join(sw, f))
            az.append(float(a["az_deg"]))
            tof.append(float(a["t1_s"]) * 1e6)
            coda.append(float(a["coda_db"]))
    az, tof, coda = map(np.asarray, (az, tof, coda))
    o = np.argsort(az)
    az, tof, coda = az[o], tof[o], coda[o]
    alpha = json.load(open(os.path.join(sw, "fit_result.json")))["alpha_deg"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
    a1.plot(az, tof, ".", ms=3, color="#1b4f72")
    a1.set_ylabel("E1 time of flight (µs)")
    a2.plot(az, coda, ".", ms=3, color="#2f7fb8")
    a2.set_ylabel("coda level (dB re E1)")
    a2.set_xlabel("azimuth (deg)")
    for ax_ in (a1, a2):
        for k in (0, 1):
            ax_.axvline((alpha + 180 * k) % 360, color="#c0392b",
                        lw=1, ls="--")
    a1.set_title(f"360-azimuth sweep, fitted fast axis {alpha:.1f}° "
                 f"(dashed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sweep-observables.png"), dpi=150,
                facecolor="white", bbox_inches="tight")


def fig_fe_arbiter():
    d = np.load(os.path.join(ROOT, "sim", "fe_crosscheck",
                             "fe_p2plus_ab5_traces.npz"))

    def env_db(x, ref):
        e = np.abs(hilbert(x - x.mean()))
        return 20 * np.log10(np.maximum(e / ref, 1e-9))

    tf = np.arange(len(d["fa"])) * float(d["dtf"]) * 1e6
    te = np.arange(len(d["ea"])) * float(d["dte"]) * 1e6
    fd = env_db(d["fa"] - d["fb"], np.abs(hilbert(d["fa"])).max())
    fe = env_db(d["ea"] - d["eb"], np.abs(hilbert(d["ea"])).max())
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(tf, fd, lw=1.1, color="#1b4f72", label="FDTD (staircase grid)")
    ax.plot(te, fe, lw=1.1, color="#c0392b",
            label="finite elements (conforming mesh)")
    ax.set_xlim(4.9, 6.5); ax.set_ylim(-75, -30)
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("grain-scatter envelope (dB re direct)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fe-arbiter.png"), dpi=150,
                facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    fig_trace_anatomy()
    print("trace-anatomy.png")
    fig_sweep_observables()
    print("sweep-observables.png")
    fig_fe_arbiter()
    print("fe-arbiter.png")
    b = build_specimen()
    fig_specimen_fabric(b)
    print("specimen-fabric.png")
    fig_pole(b)
    print("pole-figure.png")
    fig_specimen_3d(b)
    print("specimen-3d.png")

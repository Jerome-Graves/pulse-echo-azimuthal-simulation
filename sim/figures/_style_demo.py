r"""Proof that figstyle.py works. Produces figures/_style_demo.pdf.

This is not a manuscript figure and it supports no claim in the paper. It
exists so that a change to the house style can be seen and checked in one
place: it uses every facility figstyle exposes exactly once, on the six
kinds of panel the paper contains (convergence, ensemble, magnitude
field, series comparison, signed field, schematic).

Reads no data. Every curve and every image is a closed-form function
evaluated here, with a seeded generator where a spread is needed, so the
output is reproducible and nothing in it is a measurement.

Writes <GitHub>/pulse-echo-cof-paper/figures/_style_demo.pdf and prints
three checks that are computed, not quoted:
  1. the relative luminance and mono grey of every palette entry, and the
     worst-case greyscale gap as series are added,
  2. the luminance span and monotonicity of each colormap constant,
  3. the assigned size of all fourteen manuscript figures, with the
     plotting area the default margins leave at that size.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator

import figstyle as fs

# The fourteen figures of the manuscript, in the order they appear.
FIG_ORDER = ["regime", "geometry", "microstructure", "trace", "tiltmethod",
             "staircase", "tilterror", "dispersion", "ensemble", "tofaz",
             "codafield", "identify", "scales", "concept"]

# Staggered-grid first-derivative coefficients, orders 2, 4 and 6. These
# are the exact Taylor coefficients, not fitted, so the phase error below
# is closed form and its slope on log-log is the scheme order.
STAGGERED = {
    2: [1.0],
    4: [9 / 8, -1 / 24],
    6: [75 / 64, -25 / 384, 3 / 640],
}

SEED = 11               # the specimen seed used throughout the sweeps
GATE_US = (24.0, 36.0)  # coda gate, in microseconds, as in the analysis


# --------------------------------------------------------------- compute

def fd_phase_error(coeffs, ppw):
    """Fractional phase-velocity error of a staggered FD derivative.

    The Fourier symbol of the operator is D(k) = (2/h) sum c_n
    sin((n - 1/2) k h) against an exact k, and kh = 2 pi / ppw. The error
    falls as ppw**-order, which is what makes this a convergence panel.
    """
    kh = 2.0 * np.pi / np.asarray(ppw, float)
    n = np.arange(1, len(coeffs) + 1) - 0.5
    d = 2.0 * (np.asarray(coeffs)[None, :] * np.sin(np.outer(kh, n))).sum(1)
    return np.abs(d / kh - 1.0)


def _smooth(x, width):
    """Gaussian smoothing with a closed-form kernel, width in samples."""
    t = np.arange(-3 * width, 3 * width + 1)
    k = np.exp(-0.5 * (t / width) ** 2)
    return np.convolve(x, k / k.sum(), mode="same")


def coda_ensemble(rng, t_us, n_real=24, tau_us=18.0, onset_us=6.0):
    """Decaying coda envelopes, one per realisation, linear amplitude.

    A single exponential decay modulated by smoothed noise: the point of
    the panel is the spread between realisations, not the physics, so the
    simplest generator that produces a realistic spread is the honest
    choice here.
    """
    decay = np.exp(-np.clip(t_us - onset_us, 0.0, None) / tau_us)
    env = np.empty((n_real, t_us.size))
    for k in range(n_real):
        env[k] = decay * (0.35 + np.abs(_smooth(rng.normal(size=t_us.size),
                                                12.0)) * 9.0)
    return env


def speckle_field(rng, shape, width):
    """Unit-peak speckle, smoothed to a correlation length of `width`."""
    f = rng.normal(size=shape)
    f = np.apply_along_axis(_smooth, 1, f, width)
    f = np.apply_along_axis(_smooth, 0, f, width)
    return np.abs(f) / np.abs(f).max()


def backscatter_field(rng, x_mm, z_mm, decay_mm=26.0, flaw=(12.0, 34.0)):
    """Backscatter level in dB, with one planted reflector to point at.

    Normalised to its own peak so the colourbar is a level relative to
    the strongest return, which is how the manuscript quotes coda levels.
    """
    xx, zz = np.meshgrid(x_mm, z_mm)
    amp = speckle_field(rng, xx.shape, 3.0) * np.exp(-zz / decay_mm)
    amp += 0.9 * np.exp(-(((xx - flaw[0]) ** 2 + (zz - flaw[1]) ** 2)
                          / (2.0 * 2.2 ** 2)))
    return 20.0 * np.log10(np.maximum(amp / amp.max(), 1e-3))


def ricker_ring(x_mm, z_mm, radius_mm, wavelength_mm):
    """Signed snapshot of an outgoing Ricker pulse on a circular front."""
    xx, zz = np.meshgrid(x_mm, z_mm)
    r = np.sqrt(xx ** 2 + zz ** 2) - radius_mm
    a = (np.pi * r / wavelength_mm) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def staircase(x_mm, cell_mm, tilt_deg, z0_mm):
    """Interface height, exact and sampled to the nearest grid node.

    This is the whole of the staircase problem in two lines: the sampled
    interface can only take the values the grid provides, so a tilted
    boundary becomes a flight of steps whose riser is one cell.
    """
    exact = z0_mm + x_mm * np.tan(np.radians(tilt_deg))
    return exact, np.round(exact / cell_mm) * cell_mm


# ------------------------------------------------------------------ draw

def draw_convergence(ax):
    """Log-log order study: the one panel licensed to carry a grid."""
    ppw = np.geomspace(4.0, 16.0, 60)
    for i, (order, coeffs) in enumerate(sorted(STAGGERED.items())):
        ax.plot(ppw, fd_phase_error(coeffs, ppw), label="order %d" % order,
                **fs.series(i, n=ppw.size))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.7, 17.5)
    ax.set_ylim(2e-8, 3e-1)
    fs.convergence_grid(ax)
    fs.plain_log_ticks(ax, "x", [4, 6, 8, 10, 12, 16])
    fs.log_minor_off(ax)
    ax.set_xlabel(r"points per wavelength, $n_\lambda$")
    ax.set_ylabel("phase velocity error")
    # In the empty upper right, parallel to the order 2 curve rather than
    # on top of it, which is where Huang puts an asymptote.
    fs.reference_slope(ax, 7.0, 15.0, 1.6e-1, -2.0, r"slope $-2$")
    fs.legend(ax, loc="lower left")


def draw_ensemble(ax, rng):
    """Thin realisations under a thick mean, with the gate marked."""
    t_us = np.linspace(0.0, 60.0, 1200)
    env = coda_ensemble(rng, t_us)
    ref = env.max()
    ax.axvspan(*GATE_US, color=fs.GREY_LIGHT, alpha=0.35, linewidth=0,
               zorder=0)
    for e in env:
        ax.plot(t_us, 20 * np.log10(e / ref), color=fs.GREY_LIGHT,
                linewidth=fs.LW["realisation"], zorder=1)
    mean, sd = env.mean(0), env.std(0)
    fs.confidence_band(ax, t_us, 20 * np.log10(np.maximum(mean - sd, 1e-6)
                                               / ref),
                       20 * np.log10((mean + sd) / ref), 1)
    kw = fs.series(1, emphasis=True)
    kw["marker"] = "none"
    ax.plot(t_us, 20 * np.log10(mean / ref), zorder=3, **kw)
    ax.set_xlim(0, 60)
    ax.set_ylim(-52, 4)
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("envelope level (dB)")
    fs.condition(ax, "girdle, $n_\\lambda = 6$")
    fs.direct_label(ax, 40.0, -20.0, "ensemble mean", colour=fs.BLUE)
    fs.direct_label(ax, 40.0, -46.0, "realisations", colour=fs.GREY)
    fs.direct_label(ax, np.mean(GATE_US), 1.0, "coda gate", colour=fs.GREY,
                    ha="center")


def draw_field(ax, rng):
    """Magnitude field: sequential colormap, colourbar, scale bar, arrow."""
    x_mm = np.linspace(-30.0, 30.0, 200)
    z_mm = np.linspace(0.0, 60.0, 200)
    level = backscatter_field(rng, x_mm, z_mm)
    # rasterized on the mesh body only. Every axis, tick, label and the
    # colourbar outline stay vector.
    im = ax.pcolormesh(x_mm, z_mm, level, cmap=fs.cmap(fs.CMAP_SEQ),
                       vmin=-40, vmax=0, shading="auto", rasterized=True)
    ax.invert_yaxis()
    ax.set_xlabel("lateral position (mm)")
    ax.set_ylabel("depth (mm)")
    fs.colourbar(im, ax, "backscatter level (dB)")
    fs.scale_bar(ax, -26.0, 55.0, 10.0, "10 mm", colour="white")
    fs.arrow_label(ax, "planted reflector", xy=(12.0, 34.0),
                   xytext=(-8.0, 20.0), colour="white")
    # Shifted right of the default: the panel letter is inside this one.
    fs.condition(ax, "single maximum", x=0.17, box=True)
    return im


def azimuthal_level(i, az_deg):
    """Level of series i against azimuth, two-lobed and offset by series.

    The series are offset by 4.2 dB and share a phase, so the vertical
    gap between neighbours is the same everywhere and a 7 pt direct
    label placed just above any curve clears the one above it wherever
    it is put.
    """
    return (-26.0 - 4.2 * i
            + (2.4 - 0.2 * i) * np.cos(2 * np.radians(az_deg)))


def draw_series(ax, rng):
    """Six series, direct labelled, plus a measured set with 3 sigma bars."""
    az = np.linspace(0.0, 360.0, 361)
    names = ["black", "blue", "orange", "purple", "vermilion", "bluegreen"]
    for i, name in enumerate(names):
        kw = fs.series(i, n=az.size, dense=True)
        if i == 0:
            kw["marker"] = "none"        # this one carries the sampled set
        ax.plot(az, azimuthal_level(i, az), **kw)
        x_lab = 45.0 + 55.0 * i
        fs.direct_label(ax, x_lab, azimuthal_level(i, x_lab) + 0.9, name,
                        colour=fs.PALETTE[i], ha="center")
    # Sampled means of the first series against its model line, the
    # measured-points-versus-theory idiom, with a 3 sigma interval taken
    # from 40 realisations of a 1.2 dB spread.
    az_s = np.arange(0.0, 361.0, 45.0)
    draws = rng.normal(0.0, 1.2, size=(40, az_s.size))
    fs.errorbar(ax, az_s, azimuthal_level(0, az_s) + draws.mean(0),
                3.0 * draws.std(0), 0, linestyle="none")
    ax.xaxis.set_major_locator(FixedLocator([0, 120, 240, 360]))
    ax.set_xlim(0, 360)
    ax.set_ylim(-52, -20)
    ax.set_xlabel("azimuth (degrees)")
    ax.set_ylabel("backscatter level (dB)")


def draw_signed(ax):
    """Signed snapshot: the zero level is drawn, not left to the colour."""
    x_mm = np.linspace(-30.0, 30.0, 240)
    z_mm = np.linspace(0.0, 60.0, 240)
    u = ricker_ring(x_mm, z_mm, 34.0, 9.0)
    clip = 0.6 * np.abs(u).max()
    im = ax.pcolormesh(x_mm, z_mm, u, cmap=fs.cmap(fs.CMAP_SIGNED),
                       vmin=-clip, vmax=clip, shading="auto",
                       rasterized=True)
    ax.contour(x_mm, z_mm, u, levels=[0.0], colors=fs.GREY,
               linewidths=fs.LW["annotation"])
    ax.invert_yaxis()
    ax.set_xlabel("lateral position (mm)")
    ax.set_ylabel("depth (mm)")
    fs.colourbar(im, ax, "particle velocity (a.u.)")
    fs.condition(ax, "zero level drawn", box=True)


def draw_schematic(ax, cell_mm=2.0, tilt_deg=20.0, wavelength_mm=8.0):
    """Two media, a cell grid, a staircased boundary and the truth over it."""
    fs.schematic(ax)
    x_edge = np.arange(0.0, 40.0 + cell_mm, cell_mm)
    x_cell = 0.5 * (x_edge[:-1] + x_edge[1:])
    exact, sampled = staircase(x_cell, cell_mm, tilt_deg, 9.0)

    ax.add_patch(Rectangle((0.0, 0.0), 40.0, 24.0, facecolor=fs.TINT_A,
                           edgecolor="none", zorder=1))
    ax.fill_between(x_cell, sampled, 24.0, step="mid", color=fs.TINT_B,
                    linewidth=0, zorder=2)
    for xe in x_edge:
        ax.plot([xe, xe], [0.0, 24.0], color=fs.GREY_LIGHT,
                linewidth=fs.LW["cell_grid"], zorder=3)
    for ze in np.arange(0.0, 24.0 + cell_mm, cell_mm):
        ax.plot([0.0, 40.0], [ze, ze], color=fs.GREY_LIGHT,
                linewidth=fs.LW["cell_grid"], zorder=3)

    # The cell the sampling gets most wrong is the whole argument of the
    # figure, so it is filled rather than described.
    k = int(np.argmax(np.abs(sampled - exact)))
    ax.add_patch(Rectangle((x_edge[k], sampled[k] - cell_mm), cell_mm,
                           cell_mm, facecolor=fs.HIGHLIGHT,
                           edgecolor="none", zorder=4))
    ax.plot(np.array([0.0, 40.0]),
            9.0 + np.array([0.0, 40.0]) * np.tan(np.radians(tilt_deg)),
            color="black", linewidth=fs.LW["interface"], zorder=5)

    ax.set_xlim(-2.0, 46.0)
    ax.set_ylim(-6.0, 26.0)
    # The wavelength is drawn to scale against the cells, which is the
    # only way the reader can judge whether the sampling is adequate.
    fs.scale_bar(ax, 3.0, 2.0, wavelength_mm, r"$\lambda$")
    fs.axes_triad(ax, 34.0, -4.0, 5.0, labels=("x", "z"))
    fs.direct_label(ax, x_edge[k] + 3.0, sampled[k] - 3.5,
                    "worst cell", colour=fs.GREY)
    fs.condition(ax, "sampled at 2 mm", y=0.99)


# --------------------------------------------------------------- reports

def report_palette():
    """Greyscale behaviour of the palette, computed rather than quoted."""
    names = ["black", "blue", "orange", "purple", "vermilion", "bluegreen"]
    y = [fs.luminance(c) for c in fs.PALETTE]
    print("series palette, Rec. 709 luminance and mono grey")
    for name, colour, yy in zip(names, fs.PALETTE, y):
        print("  %-10s %-8s Y = %.3f  mono %s"
              % (name, colour, yy, fs.greyscale(colour)))
    print("  worst-case greyscale gap as series are added:")
    for n in range(2, len(y) + 1):
        gap = min(abs(y[i] - y[j]) for i in range(n) for j in range(i + 1, n))
        print("    %d series  %.3f" % (n, gap))


def report_colormaps():
    """Luminance span and monotonicity of every colormap constant."""
    print("colormaps, luminance over 65 samples")
    for name in (fs.CMAP_SEQ, fs.CMAP_SEQ_ALT, fs.CMAP_SEQ_DARK,
                 fs.CMAP_MONO, fs.CMAP_SIGNED):
        y = np.array([fs.luminance(fs.cmap(name)(s))
                      for s in np.linspace(0.0, 1.0, 65)])
        d = np.diff(y)
        mono = bool(np.all(d > -1e-9) or np.all(d < 1e-9))
        print("  %-9s span %.3f  %s" % (name, y.max() - y.min(),
                                        "monotonic" if mono
                                        else "NOT monotonic"))


def report_assigned_sizes():
    """Every assigned figure size, and the plotting area it leaves."""
    print("assigned figure sizes and the area the default margins leave")
    for name in FIG_ORDER:
        fig = fs.canvas(name)
        w_in, h_in = fig.get_size_inches()
        m = fs.mm_margins(fig)
        print("  %-15s %6.2f x %5.1f mm   area %5.1f x %4.1f mm"
              % (name, w_in / fs.MM, h_in / fs.MM,
                 (m["right"] - m["left"]) * w_in / fs.MM,
                 (m["top"] - m["bottom"]) * h_in / fs.MM))
        plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    rng = np.random.default_rng(SEED)

    fig, ax = fs.figure("double", 120.0, nrows=2, ncols=3,
                        wspace=fs.WSPACE_COLOURBAR, hspace=0.34,
                        left=11.0, right=12.0, bottom=8.5, top=5.0)
    draw_convergence(ax[0, 0])
    draw_ensemble(ax[0, 1], rng)
    draw_field(ax[0, 2], rng)
    draw_series(ax[1, 0], rng)
    draw_signed(ax[1, 1])
    draw_schematic(ax[1, 2])

    # Panel letters last: they are placed in figure coordinates and read
    # ax.get_position(), which the colourbars have just changed.
    for a, s in zip((ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]), "abde"):
        fs.panel(a, s)
    fs.panel(ax[0, 2], "c", inside=True)
    fs.panel(ax[1, 2], "f", dx=fs.DX_NO_YLABEL)

    report_palette()
    report_colormaps()
    report_assigned_sizes()
    fs.save(fig, "_style_demo",
            expect=["points per wavelength", "azimuth (degrees)",
                    "backscatter level (dB)", "depth (mm)"])


if __name__ == "__main__":
    main()

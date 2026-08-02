r"""Figure 'staircase': one grain boundary voxelised at n_lambda 6, 8 and 10.

Supports the claim of Section 4.3 that a boundary inclined to the grid is
represented as a flight of steps whose height falls in proportion to the
grid spacing but stays a finite fraction of the wavelength at every
resolution used in this work. That finite fraction is what makes the
discretisation error of Table tab:tilterror first order in h rather than
negligible.

Nothing here is a cartoon. The three panels show the label field the
solver is actually handed: the supersampled volume fraction of
sim/validation/tilt_testbed.py, quantised to its 21 material bins and
then thresholded by the 'naive' treatment, which gives a cell wholly to
whichever grain fills more than half of it. That is the nearest-point
rule up to the bin quantisation, and the script prints how many cells of
the slice the two rules disagree on.

Reads   ../validation/tilt_testbed.py  (C_REF, F0, L, DEPTH, SPONGE,
                                        NBIN, rot_y, frac_field)
Writes  <GitHub>/pulse-echo-cof-paper/figures/staircase.pdf

Prints, for each resolution, the grid spacing, the peak-to-peak
perpendicular offset of the represented boundary from the true plane, and
that offset in wavelengths, which are the numbers the caption claims.

CPU only. frac_field is pure numpy; tilt_testbed.run(), which is the part
that touches the GPU, is never called.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

import figstyle as fs

# The mid tilt of the default sweep in tilt_testbed (0, 15, 22.5, 30, 45).
# At 30 degrees the tread is h/tan(30) = 1.73 h, so several whole steps fit
# across a window a few wavelengths wide.
TILT_DEG = 30.0

# The three resolutions of Table tab:tilterror.
PPWS = (6.0, 8.0, 10.0)

# Half-size of the window, in millimetres, about the foot of the
# perpendicular from the source. The same physical window is used at all
# three resolutions, so the wavelength bar is the same length in every
# panel and only the cells shrink.
HALF_Z_MM = 3.0
HALF_X_MM = 2.8

# Grain A (the near half-space, on the source side) and grain B.
MEDIA = ListedColormap([fs.TINT_B, fs.TINT_A])


# --------------------------------------------------------------- compute

def load_testbed():
    """Import tilt_testbed from its own location, not from sys.path.

    sim/validation is not importable as a package and the module is
    normally run as a script, so it is loaded by file path relative to
    this file. The import itself is safe on a machine whose GPU is busy:
    the CuPy import inside fdtd is lazy and nothing at module level in
    either file allocates a device.
    """
    path = (Path(__file__).resolve().parents[1]
            / "validation" / "tilt_testbed.py")
    spec = importlib.util.spec_from_file_location("tilt_testbed", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def voxelise(tb, ppw, tilt_deg=TILT_DEG):
    """Label field of one x-z slice through the tilted bicrystal.

    Follows run() in tilt_testbed exactly: same grid spacing, same domain
    size including the sponge pad, same stand-off snapped to a cell face,
    same supersampled volume fraction. Only one cell thick in y, which
    costs nothing because the plane normal has no y component and the
    field is therefore constant along y.
    """
    h = tb.C_REF / tb.F0 / ppw
    n_grid = int(round(tb.L / h)) + 2 * (tb.SPONGE + 3)
    centre = np.full(3, n_grid * h / 2)             # the source cell
    normal = tb.rot_y(np.radians(tilt_deg)) @ np.array([1.0, 0.0, 0.0])
    # Snapping the stand-off to a whole number of cells puts the theta = 0
    # interface on a cell face, which is what makes that case exact truth.
    stand_off = round((centre[0] + tb.DEPTH) / h) * h - centre[0]
    origin = centre + stand_off * normal
    frac = tb.frac_field((n_grid, 1, n_grid), h, normal, origin)[:, 0, :]
    bins = np.rint(frac * (tb.NBIN - 1))
    return dict(h=h, n_grid=n_grid, normal=normal, origin=origin,
                stand_off=stand_off,
                near=bins / (tb.NBIN - 1) > 0.5)    # True where grain A


def nearest_point(v):
    """The strict nearest-point rule: the cell centre alone decides."""
    coord = (np.arange(v["n_grid"]) + 0.5) * v["h"]
    signed = (v["normal"][2] * (coord - v["origin"][2])[:, None]
              + v["normal"][0] * (coord - v["origin"][0])[None, :])
    return signed < 0


def step_offset(v):
    """Perpendicular offset of the represented boundary from the plane.

    Down each grid column the material flips at exactly one cell face. The
    signed distance from that face to the true plane, measured along the
    normal, is the representation error, and its peak-to-peak value is the
    step height the wave sees. Columns whose transition falls in the
    sponge pad are excluded.
    """
    first_far = np.argmax(~v["near"], axis=1)
    z = (np.arange(v["n_grid"]) + 0.5) * v["h"]
    x_true = (v["origin"][0]
              - v["normal"][2] / v["normal"][0] * (z - v["origin"][2]))
    offset = (first_far * v["h"] - x_true) * v["normal"][0]
    pad = v["n_grid"] // 8
    interior = (first_far > pad) & (first_far < v["n_grid"] - pad)
    return np.ptp(offset[interior])


# ------------------------------------------------------------------ draw

def draw_panel(ax, v, ppw, lam_mm, step_mm):
    """One resolution: the voxelised media, the cell grid, the true plane."""
    fs.schematic(ax)
    h_mm = v["h"] * 1e3
    z0_mm, x0_mm = v["origin"][2] * 1e3, v["origin"][0] * 1e3

    # Whole cells only, so the drawn grid is the real grid and not a
    # resampling of it. Coordinates are shifted to put the foot of the
    # perpendicular at the origin of every panel.
    iz = np.arange(int(np.floor((z0_mm - HALF_Z_MM) / h_mm)),
                   int(np.ceil((z0_mm + HALF_Z_MM) / h_mm)) + 1)
    ix = np.arange(int(np.floor((x0_mm - HALF_X_MM) / h_mm)),
                   int(np.ceil((x0_mm + HALF_X_MM) / h_mm)) + 1)
    ax.pcolormesh(iz * h_mm - z0_mm, ix * h_mm - x0_mm,
                  v["near"][iz[0]:iz[-1], ix[0]:ix[-1]].T.astype(float),
                  cmap=MEDIA, vmin=0.0, vmax=1.0, edgecolors=fs.GREY_LIGHT,
                  linewidth=fs.LW["cell_grid"], zorder=1)

    slope = -v["normal"][2] / v["normal"][0]        # true plane, x against z
    edge = np.array([-HALF_Z_MM, HALF_Z_MM])
    ax.plot(edge, slope * edge, color="black", linewidth=fs.LW["interface"],
            zorder=5)

    ax.set_xlim(-HALF_Z_MM, HALF_Z_MM)
    ax.set_ylim(HALF_X_MM, -HALF_X_MM)              # beam axis points down
    # High enough that the box clears the interface where it leaves the
    # window at the top left corner.
    fs.condition(ax, "$n_\\lambda = %g$\n$h = %.3f$ mm" % (ppw, h_mm),
                 y=0.975, box=True)
    fs.scale_bar(ax, -HALF_Z_MM + 0.28, HALF_X_MM - 0.42, lam_mm,
                 r"$\lambda$")
    # The measured step, in the units the caption argues in.
    fs.direct_label(ax, HALF_Z_MM - 0.22, HALF_X_MM - 0.30,
                    r"step $%.3f\,\lambda$" % (step_mm / lam_mm),
                    ha="right", va="bottom")


def annotate_first(ax, tilt_deg=TILT_DEG):
    """Orientation, tilt angle and the two grains, on the first panel only.

    The construction is identical in the other two, so repeating the
    labels there would only compete with the cells, which are the whole
    point of the figure.
    """
    phi = np.radians(np.linspace(0.0, tilt_deg, 60))
    ax.plot(1.15 * np.cos(phi), 1.15 * np.sin(phi), color=fs.GREY,
            linewidth=fs.LW["annotation"], zorder=6)
    ax.plot([0.0, 1.75], [0.0, 0.0], color=fs.GREY, linestyle=(0, (2.0, 1.5)),
            linewidth=fs.LW["annotation"], zorder=6)
    half = np.radians(0.5 * tilt_deg)
    fs.direct_label(ax, 1.42 * np.cos(half), 1.42 * np.sin(half),
                    r"$\theta = %g^\circ$" % tilt_deg, colour=fs.GREY,
                    va="center")
    fs.direct_label(ax, 1.10, -1.32, "grain A")
    fs.direct_label(ax, -2.72, 1.35, "grain B")
    fs.axes_triad(ax, 1.80, -2.46, 0.50, labels=("z", "x"))


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    tb = load_testbed()
    lam_mm = tb.C_REF / tb.F0 * 1e3

    fig, axes = fs.figure("staircase", nrows=1, ncols=3, wspace=0.05,
                          left=2.0, right=2.0, bottom=2.0, top=5.0)
    print("tilted grain boundary at %g degrees, wavelength %.3f mm"
          % (TILT_DEG, lam_mm))
    print("%6s %9s %9s %11s %10s %9s"
          % ("n_lam", "h (mm)", "h/lambda", "step (mm)", "step/lam", "cells"))
    for ax, ppw in zip(np.ravel(axes), PPWS):
        v = voxelise(tb, ppw)
        step = step_offset(v)
        draw_panel(ax, v, ppw, lam_mm, step * 1e3)
        print("%6g %9.4f %9.4f %11.4f %10.4f %9d"
              % (ppw, v["h"] * 1e3, v["h"] / (lam_mm * 1e-3), step * 1e3,
                 step / (lam_mm * 1e-3),
                 int(np.ceil(2 * HALF_Z_MM / (v["h"] * 1e3)))))
        # The deployed treatment thresholds a 21-bin volume fraction, so it
        # can differ from the strict cell-centre rule where the fraction
        # lands within half a bin of one half. Reported, not assumed.
        print("       differs from the nearest-point rule on %d of %d cells"
              % (int((v["near"] != nearest_point(v)).sum()), v["n_grid"] ** 2))
    annotate_first(np.ravel(axes)[0])

    # dx = 0: these panels carry no y axis at all, so the letter sits over
    # the left edge of the drawing rather than out in the margin, where at
    # a 2 mm margin it would run off the page.
    for ax, letter in zip(np.ravel(axes), "abc"):
        fs.panel(ax, letter, dx=0.0)
    fs.save(fig, "staircase", expect=["grain A", "grain B", "mm"])


if __name__ == "__main__":
    main()

r"""Figure `concept`: the proposed single-transducer acquisition (Sec. 6).

Supports the argument of Section 6.1, that a monostatic instrument needs
one prepared contact surface rather than two, sends the outgoing and
returning waves through the same coupling layer, and leaves the specimen
on its rotation stage between azimuths.

A schematic, but not a free-hand one: the core section, the element and
their ratio are drawn to scale from the same configuration the
simulations used, so the figure cannot disagree with Fig. geometry.

Reads (path resolved relative to this file):
  ../../out/sweeps/girdle_seed11_ppw8_dev/config.json   specimen and probe sizes

Writes ../../../pulse-echo-cof-paper/figures/concept.pdf and prints the
dimensions it draws.
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "out" / "sweeps" / "girdle_seed11_ppw8_dev" / "config.json"

# Where the drawing sits in its own millimetre world. The core is drawn
# at true scale about (CORE_X, base to top); everything to the right of
# it is instrumentation and has no physical size.
CORE_X = -30.0
CORE_BASE = -8.0
ELLIPSE_B = 7.0                 # foreshortening of a circle seen at an
                                # angle: a drawing convention, not a length
XLIM = (-94.0, 70.0)
YLIM = (-46.0, 48.0)


# --------------------------------------------------------------- compute

def load_dimensions():
    """Core diameter, section length, element size and azimuth step."""
    with open(CONFIG) as fh:
        cfg = json.load(fh)
    return dict(diameter_mm=cfg["diameter_mm"],
                length_mm=cfg["thickness_mm"],
                element_mm=cfg["element_d_mm"],
                az_step_deg=cfg["az_step"])


# ------------------------------------------------------------------ draw

def _axes_mm(fig, left, bottom, width, height):
    """An axes placed by millimetres on the page, not by fractions."""
    w_in, h_in = fig.get_size_inches()
    return fig.add_axes([left * fs.MM / w_in, bottom * fs.MM / h_in,
                         width * fs.MM / w_in, height * fs.MM / h_in])


def _cylinder(ax, x0, y_base, y_top, radius, b, face, top_face, zorder):
    """A right circular cylinder seen slightly from above.

    Drawn as base ellipse, then body, then top ellipse, so that the
    hidden half of the base is covered rather than dashed.
    """
    ax.add_patch(Ellipse((x0, y_base), 2 * radius, 2 * b, facecolor=face,
                         edgecolor="black", linewidth=fs.LW["schematic"],
                         zorder=zorder))
    ax.add_patch(Rectangle((x0 - radius, y_base), 2 * radius, y_top - y_base,
                           facecolor=face, edgecolor="none",
                           zorder=zorder + 1))
    for x in (x0 - radius, x0 + radius):
        ax.plot([x, x], [y_base, y_top], color="black",
                linewidth=fs.LW["schematic"], zorder=zorder + 2)
    ax.add_patch(Ellipse((x0, y_top), 2 * radius, 2 * b, facecolor=top_face,
                         edgecolor="black", linewidth=fs.LW["schematic"],
                         zorder=zorder + 2))


def _box(ax, x0, y0, width, height, text):
    """One stage of the signal path."""
    ax.add_patch(FancyBboxPatch((x0, y0), width, height,
                                boxstyle="round,pad=0,rounding_size=2.5",
                                facecolor="white", edgecolor="black",
                                linewidth=fs.LW["schematic"], zorder=5))
    ax.text(x0 + 0.5 * width, y0 + 0.5 * height, text, ha="center",
            va="center", fontsize=fs.FS["annotation"], zorder=6)


def _rotation_arrow(ax, x0, y_top, radius, b):
    """A turn arrow drawn on the top face, about the core axis."""
    t = np.radians(np.linspace(200.0, 340.0, 60))
    x, y = x0 + radius * np.cos(t), y_top + b * np.sin(t)
    ax.plot(x, y, color=fs.GREY, linewidth=fs.LW["schematic"], zorder=10)
    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-6], y[-6]),
                arrowprops=dict(arrowstyle="-|>", color=fs.GREY,
                                linewidth=fs.LW["schematic"],
                                mutation_scale=5.0), zorder=10)


def draw(ax, d):
    """The whole sketch: specimen, stage, probe and signal path."""
    fs.schematic(ax)
    radius = 0.5 * d["diameter_mm"]
    core_top = CORE_BASE + d["length_mm"]
    y_probe = 0.5 * (CORE_BASE + core_top)

    # Rotation stage: a turntable the core stands on, over its drive.
    ax.add_patch(Rectangle((CORE_X - 25.0, -32.0), 50.0, 18.0,
                           facecolor="white", edgecolor="black",
                           linewidth=fs.LW["schematic"], zorder=1))
    _cylinder(ax, CORE_X, -16.0, CORE_BASE, radius + 4.0, 8.0,
              fs.GREY_LIGHT, fs.GREY_LIGHT, zorder=2)

    _cylinder(ax, CORE_X, CORE_BASE, core_top, radius, ELLIPSE_B,
              fs.TINT_A, fs.TINT_B, zorder=6)
    _rotation_arrow(ax, CORE_X, core_top, 0.68 * radius, 0.68 * ELLIPSE_B)

    # Pulse echo: one aperture, one coupling layer, out and back along a
    # diameter. Dashed because the path is inside the specimen.
    ax.annotate("", xy=(CORE_X - radius + 1.0, y_probe),
                xytext=(CORE_X + radius - 1.0, y_probe),
                arrowprops=dict(arrowstyle="<->", color=fs.GREY,
                                linewidth=fs.LW["annotation"],
                                linestyle=(0, (3.0, 1.6)), shrinkA=0.0,
                                shrinkB=0.0, mutation_scale=6.0), zorder=9)
    probe_x = CORE_X + radius
    ax.add_patch(Rectangle((probe_x, y_probe - 0.5 * d["element_mm"]), 5.0,
                           d["element_mm"], facecolor="black",
                           edgecolor="none", zorder=9))

    # Signal path. The pulser and receiver are one instrument because the
    # aperture is one aperture, which is the point of the geometry.
    _box(ax, 34.0, 0.0, 34.0, 16.0, "pulser /\nreceiver")
    _box(ax, 34.0, -28.0, 34.0, 16.0, "digitiser")
    ax.annotate("", xy=(34.0, y_probe), xytext=(probe_x + 5.0, y_probe),
                arrowprops=dict(arrowstyle="<->", color="black",
                                linewidth=fs.LW["annotation"],
                                shrinkA=0.0, shrinkB=0.0,
                                mutation_scale=6.0), zorder=5)
    ax.annotate("", xy=(51.0, -12.0), xytext=(51.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                linewidth=fs.LW["annotation"],
                                shrinkA=0.0, shrinkB=0.0,
                                mutation_scale=6.0), zorder=5)
    fs.direct_label(ax, 51.0, -31.0, "one trace\nper azimuth", colour=fs.GREY,
                    ha="center", va="top")

    # Dimensions, both to scale in the drawing.
    y_dim = core_top + ELLIPSE_B + 5.0
    ax.annotate("", xy=(CORE_X - radius, y_dim), xytext=(CORE_X + radius,
                                                         y_dim),
                arrowprops=dict(arrowstyle="<->", color="black",
                                linewidth=fs.LW["annotation"],
                                shrinkA=0.0, shrinkB=0.0), zorder=6)
    for x in (CORE_X - radius, CORE_X + radius):
        ax.plot([x, x], [y_dim - 4.0, y_dim + 2.0], color="black",
                linewidth=fs.LW["annotation"], zorder=6)
    fs.direct_label(ax, CORE_X, y_dim + 1.5, "%g mm core" % d["diameter_mm"],
                    ha="center", va="bottom")

    x_dim = CORE_X - radius - 7.0
    ax.annotate("", xy=(x_dim, CORE_BASE), xytext=(x_dim, core_top),
                arrowprops=dict(arrowstyle="<->", color="black",
                                linewidth=fs.LW["annotation"],
                                shrinkA=0.0, shrinkB=0.0), zorder=6)
    for y in (CORE_BASE, core_top):
        ax.plot([x_dim - 2.0, x_dim + 2.5], [y, y], color="black",
                linewidth=fs.LW["annotation"], zorder=6)
    fs.direct_label(ax, x_dim - 2.5, y_probe, "%g mm" % d["length_mm"],
                    ha="center", va="center", rotation=90)

    # Above the turn arrow and on the top face, so the two read together.
    fs.direct_label(ax, CORE_X, core_top + 3.4, "%d° steps"
                    % d["az_step_deg"], colour=fs.GREY, ha="center",
                    va="center", zorder=11)
    fs.direct_label(ax, CORE_X, -35.0, "rotation stage", colour=fs.GREY,
                    ha="center", va="top")
    fs.arrow_label(ax, "%.2f mm element" % d["element_mm"],
                   xy=(probe_x + 5.2, y_probe + 2.4), xytext=(48.0, 24.0),
                   colour="black", ha="center", va="bottom")

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    d = load_dimensions()
    print("proposed acquisition, drawn from the simulated configuration")
    print("  core section    %g mm diameter, %g mm long"
          % (d["diameter_mm"], d["length_mm"]))
    print("  element         %.2f mm, %.1f per cent of the diameter"
          % (d["element_mm"], 100 * d["element_mm"] / d["diameter_mm"]))
    print("  azimuth step    %d deg, %d per revolution"
          % (d["az_step_deg"], round(360 / d["az_step_deg"])))

    fig = fs.canvas("concept")
    width_mm = 86.0
    height_mm = width_mm * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
    ax = _axes_mm(fig, 1.4, 0.5 * (55.0 - height_mm), width_mm, height_mm)
    draw(ax, d)
    fs.save(fig, "concept",
            expect=["pulser /", "digitiser", "rotation stage", "100 mm core",
                    "35 mm"])


if __name__ == "__main__":
    main()

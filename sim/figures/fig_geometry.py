r"""Figure `geometry`: model geometry and acquisition (Section 3.1).

Supports the claim that a single 6.35 mm element on the rim of a 100 mm
disc insonifies a diameter, that the 24 to 36 us coda gate lies wholly
between the near field and the backwall arrival, and that the whole
specimen sits inside a padded grid whose outer ten cells absorb.

Everything drawn is to scale and every length is derived here, either
from the run configuration on disk or in closed form. Nothing is copied
out of the manuscript.

Reads (paths resolved relative to this file):
  ../../out/sweeps/girdle_perp_ppw8/config.json   run configuration
  ../../out/sweeps/girdle_perp_ppw8/az000.npz     one recorded trace

Writes ../../../pulse-echo-cof-paper/figures/geometry.pdf and prints the
lengths that appear in the figure and in the surrounding text.

Panels: (a) plan view through the mid-plane of the disc, (b) section
through the beam axis showing the thickness, (c) the recorded trace of
panel (b)'s geometry with the coda gate marked.
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from scipy.signal import butter, hilbert, sosfiltfilt

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "out" / "sweeps" / "girdle_perp_ppw8"

# Restated rather than imported: sim/analysis/_t2_common.py, which owns
# these three constants, pulls in the openUSCT forward model on import
# and is far more machinery than a figure needs.
C_REF = 3850.0                  # m/s, reference qP speed (_t2_common.C_REF)
CODA_W = (24e-6, 36e-6)         # coda gate, s (_t2_common.CODA_W)
BAND = (0.8e6, 3.0e6)           # coda passband, Hz (_t2_common.BAND)

# The solver pads the specimen bounding box by sponge + 3 cells on every
# face (sim/rotation_test.py, rotated_grid: m = sponge + 3). The extra
# three cells keep the finite-difference stencil off the sponge.
PAD_EXTRA_CELLS = 3

# -6 dB half-angle of a circular piston, sin(theta) = 0.514 lambda / D.
PISTON_6DB = 0.514


# --------------------------------------------------------------- compute

def load_config():
    """Run configuration of the sweep the figure describes."""
    with open(SWEEP / "config.json") as fh:
        return json.load(fh)


def load_trace(name="az000.npz"):
    """(time in us, band-passed envelope in dB re backwall, backwall time)."""
    with np.load(SWEEP / name) as z:
        trace = np.asarray(z["trace"], float).ravel()
        dt_s = float(z["dt"])
        backwall_amp = float(z["E1"])       # envelope peak of the backwall
        backwall_s = float(z["t1_s"])       # measured, not the nominal 2D/c
    rate = 1.0 / dt_s
    sos = butter(4, [BAND[0] / (rate / 2), BAND[1] / (rate / 2)],
                 btype="band", output="sos")
    # Filter the complete record and only then gate, as the analysis does:
    # filtering a truncated segment rings at its own edges.
    envelope = np.abs(hilbert(sosfiltfilt(sos, trace)))
    t_us = np.arange(trace.size) * dt_s * 1e6
    return t_us, 20 * np.log10(envelope / backwall_amp), backwall_s * 1e6


def geometry(cfg):
    """Every length in the drawing, in millimetres, from the configuration.

    The grid spacing follows the solver's own rule h = c / (f0 n_lambda),
    so the absorbing layer is drawn at the thickness it actually has at
    this resolution rather than at a nominal one.
    """
    f0_hz = cfg["f0_mhz"] * 1e6
    cell_mm = C_REF / f0_hz / cfg["ppw"] * 1e3
    pad_cells = cfg["sponge"] + PAD_EXTRA_CELLS
    n_disc = round(cfg["diameter_mm"] / cell_mm)
    n_thick = round(cfg["thickness_mm"] / cell_mm)

    wavelength_mm = C_REF / f0_hz * 1e3
    element_mm = cfg["element_d_mm"]
    half_angle = np.arcsin(PISTON_6DB * wavelength_mm / element_mm)

    return dict(
        radius_mm=0.5 * cfg["diameter_mm"],
        diameter_mm=cfg["diameter_mm"],
        half_thickness_mm=0.5 * cfg["thickness_mm"],
        thickness_mm=cfg["thickness_mm"],
        element_radius_mm=0.5 * element_mm,
        element_mm=element_mm,
        cell_mm=cell_mm,
        sponge_mm=cfg["sponge"] * cell_mm,
        domain_half_xy_mm=0.5 * (n_disc + 2 * pad_cells) * cell_mm,
        domain_half_z_mm=0.5 * (n_thick + 2 * pad_cells) * cell_mm,
        wavelength_mm=wavelength_mm,
        half_angle=half_angle,
        # Near-field length of the element. At 5.2 mm it is 5 per cent of
        # the path, which is why the beam is drawn as a plain cone.
        near_field_mm=element_mm ** 2 / (4.0 * wavelength_mm),
        # The coda gate as a one-way distance along the chord.
        gate_mm=tuple(C_REF * t / 2.0 * 1e3 for t in CODA_W),
        backwall_us=cfg["diameter_mm"] * 1e-3 / C_REF * 2e6,
        az_step_deg=cfg["az_step"],
        n_azimuth=int(round((cfg["az_stop"] - cfg["az_start"])
                            / cfg["az_step"])),
    )


def beam_edge(g, s_mm, sign=1.0):
    """A point on the -6 dB beam boundary, s_mm along the beam axis.

    The boundary is the far-field asymptote translated out to the edge of
    the element, so it starts at the true aperture and opens at the true
    divergence. The element sits at +x on the rim and fires towards -x.
    """
    return np.stack([g["radius_mm"] - s_mm * np.cos(g["half_angle"]),
                     sign * (g["element_radius_mm"]
                             + s_mm * np.sin(g["half_angle"]))], axis=-1)


def beam_exit_range(g):
    """Path length at which the beam boundary reaches the curved rim.

    Solves |beam_edge(s)| = R, whose positive root is where the cone
    leaves the disc. The beam is wide enough that it does so before the
    axis reaches the backwall, which is why the drawing is clipped.
    """
    b = (-2.0 * g["radius_mm"] * np.cos(g["half_angle"])
         + 2.0 * g["element_radius_mm"] * np.sin(g["half_angle"]))
    c = g["element_radius_mm"] ** 2
    return 0.5 * (-b + np.sqrt(b * b - 4.0 * c))


def beam_polygon_plan(g):
    """The insonified region in plan, clipped by the curved rim."""
    s_exit = beam_exit_range(g)
    s = np.linspace(0.0, s_exit, 60)
    upper, lower = beam_edge(g, s, +1.0), beam_edge(g, s, -1.0)
    theta_exit = np.arctan2(upper[-1, 1], upper[-1, 0])
    arc = np.linspace(theta_exit, 2 * np.pi - theta_exit, 40)
    rim = g["radius_mm"] * np.stack([np.cos(arc), np.sin(arc)], axis=-1)
    return np.vstack([upper, rim, lower[::-1]])


def beam_polygon_section(g):
    """The insonified region in section, clipped by the two flat faces.

    The beam half-width reaches the half thickness before the axis
    reaches the backwall, so in section the cone is truncated by the
    faces rather than by the far wall.
    """
    z_max = g["half_thickness_mm"]
    s_face = ((z_max - g["element_radius_mm"]) / np.sin(g["half_angle"]))
    s_back = 2.0 * g["radius_mm"] / np.cos(g["half_angle"])
    s = np.linspace(0.0, min(s_face, s_back), 40)
    upper, lower = beam_edge(g, s, +1.0), beam_edge(g, s, -1.0)
    corner = [[-g["radius_mm"], z_max], [-g["radius_mm"], -z_max]]
    return np.vstack([upper, corner, lower[::-1]])


# ------------------------------------------------------------------ draw

def _axes_mm(fig, left, bottom, width, height):
    """An axes placed by millimetres on the page, not by fractions."""
    w_in, h_in = fig.get_size_inches()
    return fig.add_axes([left * fs.MM / w_in, bottom * fs.MM / h_in,
                         width * fs.MM / w_in, height * fs.MM / h_in])


def _domain(ax, half_x, half_z, sponge):
    """Computational domain: outer box, absorbing band, undamped interior."""
    ax.add_patch(Rectangle((-half_x, -half_z), 2 * half_x, 2 * half_z,
                           facecolor=fs.GREY_LIGHT, edgecolor=fs.GREY,
                           linewidth=fs.LW["schematic"], zorder=1))
    ax.add_patch(Rectangle((-half_x + sponge, -half_z + sponge),
                           2 * (half_x - sponge), 2 * (half_z - sponge),
                           facecolor="white", edgecolor=fs.GREY,
                           linewidth=fs.LW["annotation"], zorder=2))


def _gate_marks(ax, g, extend_mm):
    """The coda gate as a slab of the beam, with its two boundaries."""
    s0, s1 = g["gate_mm"]
    s = np.linspace(s0, s1, 2)
    slab = np.vstack([beam_edge(g, s, +1.0), beam_edge(g, s, -1.0)[::-1]])
    ax.add_patch(Polygon(slab, closed=True, facecolor=fs.BLUE, alpha=0.20,
                         edgecolor="none", zorder=5))
    for s_gate in (s0, s1):
        x = g["radius_mm"] - s_gate * np.cos(g["half_angle"])
        y = (g["element_radius_mm"] + s_gate * np.sin(g["half_angle"])
             + extend_mm)
        ax.plot([x, x], [-y, y], color=fs.BLUE, linewidth=fs.LW["reference"],
                linestyle=(0, (2.4, 1.4)), zorder=6)


def draw_plan(ax, g):
    """Plan view: disc, element, beam, coda gate, rotation."""
    fs.schematic(ax)
    _domain(ax, g["domain_half_xy_mm"], g["domain_half_xy_mm"], g["sponge_mm"])
    ax.add_patch(Circle((0, 0), g["radius_mm"], facecolor=fs.TINT_A,
                        edgecolor="black", linewidth=fs.LW["schematic"],
                        zorder=3))

    ax.add_patch(Polygon(beam_polygon_plan(g), closed=True,
                         facecolor=fs.HIGHLIGHT, alpha=0.55, edgecolor="none",
                         zorder=4))
    s_exit = beam_exit_range(g)
    for sign in (+1.0, -1.0):
        edge = beam_edge(g, np.array([0.0, s_exit]), sign)
        ax.plot(edge[:, 0], edge[:, 1], color=fs.GREY,
                linewidth=fs.LW["annotation"], zorder=5)
    ax.plot([g["radius_mm"], -g["radius_mm"]], [0, 0], color=fs.GREY,
            linestyle=(0, (3.0, 1.5)), linewidth=fs.LW["annotation"],
            zorder=5)

    # The element is the set of rim cells within its own radius of the
    # axis, so it is drawn as a block on the arc it actually occupies.
    half_deg = np.degrees(np.arcsin(g["element_radius_mm"] / g["radius_mm"]))
    ax.add_patch(Wedge((0, 0), g["radius_mm"] + 2.2, -half_deg, half_deg,
                       width=2.2, facecolor="black", edgecolor="none",
                       zorder=7))
    # The stretch of rim the beam actually reaches, which is what returns
    # the backwall echo.
    lit = np.arctan2(*beam_edge(g, s_exit, +1.0)[::-1])
    lit_arc = np.linspace(lit, 2 * np.pi - lit, 40)
    ax.plot(g["radius_mm"] * np.cos(lit_arc), g["radius_mm"] * np.sin(lit_arc),
            color="black", linewidth=fs.LW["interface"], zorder=7)

    _gate_marks(ax, g, extend_mm=3.0)
    fs.direct_label(ax, -20.0, -24.0,
                    "coda gate\n%.0f to %.0f mm" % g["gate_mm"],
                    colour=fs.BLUE, ha="center", va="top")

    # Diameter, which is also the element-to-backwall path length.
    y_dim = g["domain_half_xy_mm"] + 3.0
    ax.annotate("", xy=(-g["radius_mm"], y_dim),
                xytext=(g["radius_mm"], y_dim),
                arrowprops=dict(arrowstyle="<->", color="black",
                                linewidth=fs.LW["annotation"],
                                shrinkA=0.0, shrinkB=0.0), zorder=6)
    # Extension lines rather than short ticks: at the two ends of a
    # diameter they are the vertical tangents to the rim, so the reader
    # can see that the dimension is the disc and not the grid.
    for x in (-g["radius_mm"], g["radius_mm"]):
        ax.plot([x, x], [3.0, y_dim + 2.0], color=fs.GREY,
                linewidth=fs.LW["annotation"], zorder=6)
    fs.direct_label(ax, 0.0, y_dim + 1.5, "%g mm" % g["diameter_mm"],
                    ha="center", va="bottom")

    fs.direct_label(ax, -g["domain_half_xy_mm"] - 4.0, 0.0, "backwall",
                    colour=fs.GREY, ha="center", va="center", rotation=90)
    fs.arrow_label(ax, "%.2f mm\nelement" % g["element_mm"],
                   xy=(g["radius_mm"] + 1.0, 3.0), xytext=(20.0, 30.0),
                   colour="black", ha="center")
    fs.arrow_label(ax, "%.2f° beam\nhalf-angle" % np.degrees(g["half_angle"]),
                   xy=tuple(beam_edge(g, 72.0, +1.0)), xytext=(-16.0, 24.0),
                   colour=fs.GREY, ha="center")

    # Rotation detail: the specimen turns under a fixed probe, and the
    # azimuth index advances clockwise in this frame (the sign convention
    # of rotated_grid, sim/rotation_test.py).
    r_arc = 46.0
    arc = np.radians(np.linspace(-40.0, -82.0, 60))
    ax.plot(r_arc * np.cos(arc), r_arc * np.sin(arc), color=fs.GREY,
            linewidth=fs.LW["schematic"], zorder=6)
    for a in np.radians(np.arange(-78.0, -41.0, g["az_step_deg"])):
        ax.plot([(r_arc - 1.2) * np.cos(a), (r_arc + 1.2) * np.cos(a)],
                [(r_arc - 1.2) * np.sin(a), (r_arc + 1.2) * np.sin(a)],
                color=fs.GREY, linewidth=fs.LW["schematic"], zorder=6)
    ax.annotate("", xy=(r_arc * np.cos(arc[-1]), r_arc * np.sin(arc[-1])),
                xytext=(r_arc * np.cos(arc[-6]), r_arc * np.sin(arc[-6])),
                arrowprops=dict(arrowstyle="-|>", color=fs.GREY,
                                linewidth=fs.LW["schematic"],
                                mutation_scale=5.0), zorder=6)
    fs.direct_label(ax, 26.0, -16.0, "%d° steps" % g["az_step_deg"],
                    colour=fs.GREY, ha="center", va="top")

    fs.axes_triad(ax, -46.0, 42.0, 7.0, labels=("x", "y"), colour=fs.GREY)
    ax.set_xlim(-62.0, 58.0)
    ax.set_ylim(-58.0, 64.0)


def draw_section(ax, g):
    """Section through the beam axis: thickness and absorbing layer."""
    fs.schematic(ax)
    _domain(ax, g["domain_half_xy_mm"], g["domain_half_z_mm"], g["sponge_mm"])
    ax.add_patch(Rectangle((-g["radius_mm"], -g["half_thickness_mm"]),
                           g["diameter_mm"], g["thickness_mm"],
                           facecolor=fs.TINT_A, edgecolor="black",
                           linewidth=fs.LW["schematic"], zorder=3))
    # Outlined, not just tinted: a yellow fill on a blue one is a weak
    # contrast once the page is printed in mono.
    ax.add_patch(Polygon(beam_polygon_section(g), closed=True,
                         facecolor=fs.HIGHLIGHT, alpha=0.55,
                         edgecolor=fs.GREY, linewidth=fs.LW["annotation"],
                         zorder=4))
    ax.add_patch(Rectangle((g["radius_mm"], -g["element_radius_mm"]), 2.2,
                           g["element_mm"], facecolor="black",
                           edgecolor="none", zorder=6))

    x_dim = -g["domain_half_xy_mm"] - 4.0
    ax.annotate("", xy=(x_dim, -g["half_thickness_mm"]),
                xytext=(x_dim, g["half_thickness_mm"]),
                arrowprops=dict(arrowstyle="<->", color="black",
                                linewidth=fs.LW["annotation"],
                                shrinkA=0.0, shrinkB=0.0), zorder=6)
    for y in (-g["half_thickness_mm"], g["half_thickness_mm"]):
        ax.plot([x_dim - 2.0, x_dim + 3.5], [y, y], color="black",
                linewidth=fs.LW["annotation"], zorder=6)
    fs.direct_label(ax, x_dim - 2.5, 0.0, "%g mm" % g["thickness_mm"],
                    ha="center", va="center", rotation=90)

    fs.arrow_label(ax, "%d-cell absorbing layer"
                   % round(g["sponge_mm"] / g["cell_mm"]),
                   xy=(-26.0, g["domain_half_z_mm"] - 0.5 * g["sponge_mm"]),
                   xytext=(14.0, g["domain_half_z_mm"] + 2.0),
                   colour=fs.GREY, ha="center", va="bottom")
    fs.axes_triad(ax, -62.0, -27.0, 5.0, labels=("x", "z"), colour=fs.GREY)
    ax.set_xlim(-66.0, 56.0)
    ax.set_ylim(-31.0, 28.0)


def draw_trace(ax, t_us, level_db, backwall_us):
    """The recorded trace of that geometry, with the gate marked."""
    gate_us = tuple(t * 1e6 for t in CODA_W)
    ax.axvspan(*gate_us, color=fs.GREY_LIGHT, alpha=0.45, linewidth=0,
               zorder=0)
    ax.plot(t_us, level_db, color=fs.BLACK, linewidth=fs.LW["data"],
            zorder=3)
    ax.axvline(backwall_us, color=fs.GREY, linestyle=(0, (2.4, 1.4)),
               linewidth=fs.LW["reference"], zorder=2)

    ax.set_xlim(0.0, 62.0)
    ax.set_ylim(-92.0, 34.0)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, -40, -80])
    ax.set_yticks([20, -20, -60], minor=True)
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("envelope level (dB)")
    fs.direct_label(ax, np.mean(gate_us), 28.0, "coda gate", colour=fs.GREY,
                    ha="center", va="top")
    fs.direct_label(ax, backwall_us - 1.5, 12.0, "backwall", colour=fs.GREY,
                    ha="right", va="center")
    fs.direct_label(ax, 9.0, 12.0, "source", colour=fs.GREY, ha="left",
                    va="center")


# --------------------------------------------------------------- reports

def report(g, backwall_us):
    """The lengths the figure asserts, so they can be checked against 3.1."""
    print("acquisition geometry, all derived from the run configuration")
    print("  disc                %.0f mm diameter, %.0f mm thick"
          % (g["diameter_mm"], g["thickness_mm"]))
    print("  element             %.2f mm on the rim, fires along a diameter"
          % g["element_mm"])
    print("  azimuths            %d at %d deg"
          % (g["n_azimuth"], g["az_step_deg"]))
    print("  wavelength          %.3f mm at %.0f m/s"
          % (g["wavelength_mm"], C_REF))
    print("  beam half-angle     %.2f deg (-6 dB piston, near field %.2f mm)"
          % (np.degrees(g["half_angle"]), g["near_field_mm"]))
    print("  beam at the far rim %.1f mm half-width, against %.1f mm "
          "half-thickness"
          % (g["element_radius_mm"]
             + 2 * g["radius_mm"] * np.tan(g["half_angle"]),
             g["half_thickness_mm"]))
    print("  coda gate           %.1f to %.1f us, %.1f to %.1f mm along "
          "the chord"
          % (CODA_W[0] * 1e6, CODA_W[1] * 1e6, g["gate_mm"][0],
             g["gate_mm"][1]))
    print("  backwall            %.2f us nominal, %.2f us measured"
          % (g["backwall_us"], backwall_us))
    print("  grid                %.4f mm cell, %d-cell absorbing layer "
          "= %.2f mm"
          % (g["cell_mm"], round(g["sponge_mm"] / g["cell_mm"]),
             g["sponge_mm"]))
    print("  domain              %.2f x %.2f x %.2f mm"
          % (2 * g["domain_half_xy_mm"], 2 * g["domain_half_xy_mm"],
             2 * g["domain_half_z_mm"]))


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    cfg = load_config()
    g = geometry(cfg)
    t_us, level_db, backwall_us = load_trace()
    report(g, backwall_us)

    # Placed by hand, in millimetres: the three panels are an equal-aspect
    # square, an equal-aspect strip and a plotted axes, which no gridspec
    # arranges without leaving one of them adrift in its cell.
    fig = fs.canvas("geometry")
    ax_plan = _axes_mm(fig, 1.0, 0.8, 56.0, 56.0 * 122.0 / 120.0)
    ax_sec = _axes_mm(fig, 64.0, 30.6, 62.0, 62.0 * 59.0 / 122.0)
    ax_tr = _axes_mm(fig, 77.0, 8.0, 57.0, 20.0)

    draw_plan(ax_plan, g)
    draw_section(ax_sec, g)
    draw_trace(ax_tr, t_us, level_db, backwall_us)

    fs.panel(ax_plan, "a", inside=True)
    fs.panel(ax_sec, "b", inside=True)
    fs.panel(ax_tr, "c", inside=True)
    fs.save(fig, "geometry",
            expect=["coda gate", "backwall", "envelope level (dB)",
                    "100 mm", "35 mm"])


if __name__ == "__main__":
    main()

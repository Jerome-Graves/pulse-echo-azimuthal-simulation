r"""Figure 'tiltmethod': the rotation-invariance construction, drawn.

Supports Section 4.3. A point source insonifies a plane interface at
perpendicular distance d whose normal is n(theta). The reflected field is
that of an image source at 2d, so the echo returns to the source at 2d/v
for every theta. Rotating both c-axes by the same R(theta) holds the
angle between each axis and the propagation direction fixed, so the
reflection physics is invariant by rigid rotation and the only thing that
changes with theta is the angle the interface makes with the grid. Any
variation of the measured echo is therefore discretisation error, and it
can be measured without knowing the correct answer.

The three panels are three tilts of the same configuration. Under each
one sit the two quantities the construction holds fixed, the echo time
and the exact quasi-P reflection coefficient, both computed here. They do
not change, which is the whole argument.

Reads
  ../validation/tilt_testbed.py  geometry (DEPTH, C_REF, F0) and the qP
                                 speed along the beam
  ../validation/exact_rc.py      exact anisotropic interface solver, used
                                 to confirm that |R_PP| does not depend on
                                 the tilt
Writes
  <GitHub>/pulse-echo-cof-paper/figures/tiltmethod.pdf

Prints the invariants at all five tilts of the testbed's default sweep,
which is where the manuscript's twelve significant figures come from.

CPU only. Both source modules are pure numpy and scipy; no FDTD run is
started and the GPU is never touched.
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon

import figstyle as fs

# c-axis angles of the two grains, measured from the interface normal
# before the rotation. These are the --psi defaults of tilt_testbed and
# the pair the manuscript quotes: psi_b = 51 degrees is one longitudinal
# normal of ice Ih and the minimum of its quasi-P speed.
PSI_A_DEG, PSI_B_DEG = 0.0, 51.0

# Tilts drawn, taken from the testbed's default sweep 0, 15, 22.5, 30, 45.
# Three well separated ones make the rotation visible in one glance.
TILTS_DEG = (0.0, 22.5, 45.0)
SWEEP_DEG = (0.0, 15.0, 22.5, 30.0, 45.0)

# Drawing window, in millimetres, with the source at the origin. The
# horizontal axis is the grid axis z and the vertical axis is the beam
# axis x, drawn downwards so the reflector sits below the source in the
# usual pulse-echo sense.
HALF_Z_MM = 13.5
TOP_X_MM = 6.5
BOT_X_MM = 18.5
FOOT_X_MM = 24.5            # blank strip under the drawing, for the text

# Production resolution of the sweeps (key 'ppw' in every config.json).
# Used only to say how many cells the drawn reference grid is worth.
PPW_REF = 8.0

ARC_HALF_DEG = 40.0         # angular half-extent of the drawn wavefronts
CAXIS_HALF_MM = 1.8         # half-length of a c-axis symbol
CAXIS_AT_MM = 6.6           # its stand-off along the interface, so that
                            # the symbols never sit on the perpendicular

# The two c-axes: angle to the normal, colour, name, and stand-off from
# the interface along the normal, which also decides which grain each one
# is drawn in. Colour is paired with a name in every panel, so neither a
# greyscale print nor a colourblind reader loses the distinction.
CAXES = ((PSI_A_DEG, fs.BLUE, "$c_A$", -3.2),
         (PSI_B_DEG, fs.ORANGE, "$c_B$", 3.5))


# --------------------------------------------------------------- compute

def _load(name):
    """Import a validation module by file path, relative to this script."""
    here = Path(__file__).resolve().parents[1]
    path = here / "validation" / ("%s.py" % name)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_sources():
    """The testbed and the exact interface solver.

    exact_rc.py is a script as well as a module: it prints its own
    consistency checks at import, including a cross-check of its Bond
    rotation against ringfwi. Those are not this figure's output, so they
    are captured and dropped; only its solver functions are wanted here.
    """
    testbed = _load("tilt_testbed")
    with contextlib.redirect_stdout(io.StringIO()):
        exact = _load("exact_rc")
    return testbed, exact


def reflection_coefficient(exact, tilt_deg):
    """|R| for quasi-P at normal incidence on the tilted interface.

    Solved by matching displacement and traction across the boundary using
    the Christoffel eigenvectors of each grain, with the whole
    configuration rotated by the tilt. The answer must not depend on the
    tilt: that independence is the premise the figure illustrates.
    """
    ca, cb, _, _, _ = exact.setup(PSI_A_DEG, PSI_B_DEG, tilt_deg=tilt_deg)
    return abs(exact.exact_normal_RT(ca, cb, exact.RHO)["R_by"]["qP"])


def invariants(testbed, exact, tilts_deg):
    """Echo time and reflection coefficient at each tilt.

    The speed is that of quasi-P in the near grain along the beam. Both
    the propagation direction and that grain's c-axis are rotated by
    R(theta), so the angle between them stays psi_a and the speed is
    tilt-independent for the same reason the reflection coefficient is.
    """
    speed = testbed.vA(PSI_A_DEG)
    return dict(speed=speed,
                depth=testbed.DEPTH,
                time_s=2.0 * testbed.DEPTH / speed,
                r_pp=[reflection_coefficient(exact, t) for t in tilts_deg])


def geometry(tilt_deg, depth_mm):
    """Interface frame and the three points, in drawing coordinates.

    The lab normal is R_y(theta) x_hat = (cos t, 0, -sin t), which in the
    (z, x) plane of the drawing is (-sin t, cos t).
    """
    t = np.radians(tilt_deg)
    normal = np.array([-np.sin(t), np.cos(t)])
    along = np.array([np.cos(t), np.sin(t)])
    return dict(normal=normal, along=along, source=np.zeros(2),
                foot=depth_mm * normal, image=2.0 * depth_mm * normal)


def caxis_direction(psi_deg, tilt_deg):
    """A c-axis at psi to the normal, after the rigid rotation by theta.

    R_y(theta) (cos psi, 0, sin psi) = (cos(psi - theta), 0,
    sin(psi - theta)), so in drawing coordinates the axis is
    (sin(psi - theta), cos(psi - theta)) and its angle to the normal stays
    psi whatever theta is. That is the point of the construction.
    """
    a = np.radians(psi_deg - tilt_deg)
    return np.array([np.sin(a), np.cos(a)])


# ------------------------------------------------------------- geometry

def clip_slab(point, direction, box):
    """Parameter range over which a line stays inside the drawing box."""
    lo, hi = -np.inf, np.inf
    for i, (mn, mx) in enumerate(box):
        d = direction[i]
        if abs(d) < 1e-12:
            continue
        ends = sorted(((mn - point[i]) / d, (mx - point[i]) / d))
        lo, hi = max(lo, ends[0]), min(hi, ends[1])
    return lo, hi


def clip_half_plane(poly, normal, point):
    """Sutherland-Hodgman clip of a convex polygon to n.(p - point) >= 0."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        sa, sb = normal @ (a - point), normal @ (b - point)
        if sa >= 0.0:
            out.append(a)
        if (sa > 0.0) != (sb > 0.0):
            out.append(a + (b - a) * sa / (sa - sb))
    return np.array(out)


def bearing(vec):
    """Direction of a drawing vector, in degrees from the z axis."""
    return np.degrees(np.arctan2(vec[1], vec[0]))


def arc(centre, radius, deg0, deg1, n=120):
    """Points on a circular arc, in drawing coordinates."""
    a = np.radians(np.linspace(deg0, deg1, n))
    return centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)


# ------------------------------------------------------------------ draw

def draw_media(ax, g):
    """The two grains, split by the interface and cut to the window."""
    box = np.array([[-HALF_Z_MM, -TOP_X_MM], [HALF_Z_MM, -TOP_X_MM],
                    [HALF_Z_MM, BOT_X_MM], [-HALF_Z_MM, BOT_X_MM]])
    ax.add_patch(Polygon(box, facecolor=fs.TINT_A, edgecolor="none",
                         zorder=1))
    far = clip_half_plane(box, g["normal"], g["foot"])
    ax.add_patch(Polygon(far, facecolor=fs.TINT_B, edgecolor="none",
                         zorder=1))


def draw_grid(ax, pitch_mm):
    """Reference grid, aligned with the finite-difference grid.

    Drawn one wavelength apart rather than one cell apart: a cell is
    1/PPW_REF of this and the lines would close up into grey. It is
    axis-aligned and anchored on the source cell, which is what matters
    here, because the source sits at the same grid location at every tilt
    and only the interface moves against it.
    """
    for k in range(int(-HALF_Z_MM / pitch_mm), int(HALF_Z_MM / pitch_mm) + 1):
        ax.plot([k * pitch_mm] * 2, [-TOP_X_MM, BOT_X_MM], color=fs.GREY_LIGHT,
                linewidth=fs.LW["cell_grid"], zorder=2)
    for k in range(int(-TOP_X_MM / pitch_mm), int(BOT_X_MM / pitch_mm) + 1):
        ax.plot([-HALF_Z_MM, HALF_Z_MM], [k * pitch_mm] * 2,
                color=fs.GREY_LIGHT, linewidth=fs.LW["cell_grid"], zorder=2)


def draw_construction(ax, g, depth_mm):
    """Interface, perpendicular, source, image source and the wavefronts."""
    box = ((-HALF_Z_MM, HALF_Z_MM), (-TOP_X_MM, BOT_X_MM))
    lo, hi = clip_slab(g["foot"], g["along"], box)
    ends = np.array([g["foot"] + lo * g["along"], g["foot"] + hi * g["along"]])
    ax.plot(ends[:, 0], ends[:, 1], color="black",
            linewidth=fs.LW["interface"], zorder=5)

    ax.plot([g["source"][0], g["image"][0]], [g["source"][1], g["image"][1]],
            color="black", linestyle=(0, (2.2, 1.6)),
            linewidth=fs.LW["annotation"], zorder=5)
    # An arrowhead on the same line, just short of the source: the
    # reflected field is the field of the image source, so this ray is the
    # echo, and it arrives here.
    ax.annotate("", xy=0.06 * g["image"], xytext=0.20 * g["image"],
                arrowprops=dict(arrowstyle="-|>", color="black",
                                linewidth=fs.LW["annotation"],
                                mutation_scale=5.0, shrinkA=0.0,
                                shrinkB=0.0), zorder=5)

    # The incident front at t = d/v, just touching the plane, and the
    # reflected front at t = 2d/v, which is the same sphere seen from the
    # image source and which is back at the source exactly then.
    out = bearing(g["normal"])
    ax.plot(*arc(g["source"], depth_mm, out - ARC_HALF_DEG,
                 out + ARC_HALF_DEG), color=fs.GREY,
            linewidth=fs.LW["annotation"], zorder=4)
    back = bearing(-g["normal"])
    ax.plot(*arc(g["image"], 2.0 * depth_mm, back - ARC_HALF_DEG,
                 back + ARC_HALF_DEG), color=fs.GREY,
            linestyle=(0, (2.6, 1.4)), linewidth=fs.LW["annotation"],
            zorder=4)

    ax.plot(*g["source"], marker="o", color="black", markersize=3.4,
            zorder=6)
    ax.plot(*g["image"], marker="o", markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=0.6, markersize=3.4,
            zorder=6)


def caxis_centre(g, offset_mm):
    """Where a c-axis symbol goes: beside the perpendicular, in one grain."""
    return g["foot"] + offset_mm * g["normal"] + CAXIS_AT_MM * g["along"]


def draw_caxes(ax, g, tilt_deg):
    """The two c-axes, rigidly rotated with the plane.

    Drawn as double-headed arrows because a c-axis is an axis and not a
    direction, and placed on the same side of the perpendicular at every
    tilt so the reader can see them turn with the interface. The angle
    from the normal to the far grain's axis is arced in every panel, not
    just the first: it is the same 51 degrees at every tilt, and that is
    the invariance the whole test rests on.
    """
    for psi, colour, name, offset in CAXES:
        centre = caxis_centre(g, offset)
        direction = caxis_direction(psi, tilt_deg)
        ax.annotate("", xy=centre + CAXIS_HALF_MM * direction,
                    xytext=centre - CAXIS_HALF_MM * direction,
                    arrowprops=dict(arrowstyle="<|-|>", color=colour,
                                    linewidth=fs.LW["data"],
                                    mutation_scale=5.0,
                                    shrinkA=0.0, shrinkB=0.0), zorder=6)
        # Set beside the symbol along the interface, not off one of its
        # ends: the two axes point differently at every tilt, and only a
        # sideways offset clears both heads at all of them.
        label = centre + 2.5 * g["along"]
        fs.direct_label(ax, label[0], label[1], name, colour=colour,
                        ha="left", va="center")

    centre = caxis_centre(g, CAXES[1][3])
    ref = centre + 2.6 * g["normal"]
    ax.plot([centre[0], ref[0]], [centre[1], ref[1]], color=fs.GREY,
            linestyle=(0, (1.6, 1.4)), linewidth=fs.LW["annotation"],
            zorder=5)
    ax.plot(*arc(centre, 1.5, bearing(caxis_direction(CAXES[1][0], tilt_deg)),
                 bearing(g["normal"])), color=fs.GREY,
            linewidth=fs.LW["annotation"], zorder=5)


def draw_tilt_angle(ax, g, tilt_deg):
    """The angle between the interface normal and the grid axis."""
    ax.plot([0.0, 0.0], [0.0, 5.6], color=fs.GREY, linestyle=(0, (1.6, 1.4)),
            linewidth=fs.LW["annotation"], zorder=5)
    ax.plot(*arc(g["source"], 4.2, 90.0, 90.0 + tilt_deg), color=fs.GREY,
            linewidth=fs.LW["annotation"], zorder=5)
    tip = 5.2 * np.array([np.cos(np.radians(90.0 + 0.5 * tilt_deg)),
                          np.sin(np.radians(90.0 + 0.5 * tilt_deg))])
    fs.direct_label(ax, tip[0], tip[1], r"$\theta$", colour=fs.GREY,
                    ha="center", va="center")


def annotate_first(ax, g, depth_mm):
    """Names and dimensions, on the reference panel only.

    The other two panels repeat the same construction, so labelling them
    again would bury the one thing they are there to show, which is that
    the picture turns.
    """
    fs.direct_label(ax, 0.8, -0.8, "source", va="bottom")
    fs.direct_label(ax, g["image"][0] + 0.9, g["image"][1] + 0.7,
                    "image source", va="top")
    for frac in (0.5, 1.5):
        mid = frac * depth_mm * g["normal"] + 1.0 * g["along"]
        fs.direct_label(ax, mid[0], mid[1], "$d$", ha="left", va="center")

    out = bearing(g["normal"])
    near = arc(g["source"], depth_mm, out + ARC_HALF_DEG, out + ARC_HALF_DEG)
    fs.direct_label(ax, near[0][0] - 0.5, near[1][0] - 0.3, "$d/v$",
                    colour=fs.GREY, ha="right", va="center")
    back = bearing(-g["normal"])
    far = arc(g["image"], 2.0 * depth_mm, back - 30.0, back - 30.0)
    fs.direct_label(ax, far[0][0] - 0.5, far[1][0], "$2d/v$", colour=fs.GREY,
                    ha="right", va="center")

    base = g["foot"] - 5.4 * g["along"]
    ax.annotate("", xy=base + 3.0 * g["normal"], xytext=base,
                arrowprops=dict(arrowstyle="-|>", color="black",
                                linewidth=fs.LW["schematic"],
                                mutation_scale=5.0), zorder=6)
    tip = base + 3.4 * g["normal"]
    fs.direct_label(ax, tip[0] - 0.4, tip[1], r"$\mathbf{n}(\theta)$",
                    ha="right", va="center")

    # The arc itself is drawn in every panel, since holding it constant is
    # the point. Only its name belongs on the reference panel.
    centre = caxis_centre(g, CAXES[1][3])
    mid = np.radians(0.5 * (bearing(caxis_direction(CAXES[1][0], 0.0))
                            + bearing(g["normal"])))
    fs.direct_label(ax, centre[0] + 2.2 * np.cos(mid),
                    centre[1] + 2.2 * np.sin(mid), r"$\psi_b$",
                    colour=fs.GREY, ha="left", va="center")


def draw_panel(ax, tilt_deg, inv, index, lam_mm, first=False):
    """One tilt of the construction, with its two invariants beneath it."""
    fs.schematic(ax)
    depth_mm = inv["depth"] * 1e3
    g = geometry(tilt_deg, depth_mm)
    draw_media(ax, g)
    draw_grid(ax, lam_mm)
    draw_construction(ax, g, depth_mm)
    draw_caxes(ax, g, tilt_deg)
    if tilt_deg > 0.0:
        draw_tilt_angle(ax, g, tilt_deg)
    if first:
        annotate_first(ax, g, depth_mm)

    ax.set_xlim(-HALF_Z_MM, HALF_Z_MM)
    ax.set_ylim(FOOT_X_MM, -TOP_X_MM)       # beam axis points down the page
    # Top right, not the usual top left: the reflected wavefront sweeps
    # into the top left corner as the tilt grows and would sit under it.
    fs.condition(ax, r"$\theta = %g^\circ$" % tilt_deg, x=0.66, y=0.965,
                 box=True)
    fs.scale_bar(ax, -3.0 * lam_mm, BOT_X_MM - 1.1, lam_mm, r"$\lambda$")
    # The two quantities the construction fixes. They are identical under
    # all three panels, which is the argument the figure is making.
    ax.text(0.0, BOT_X_MM + 1.4,
            "$2d/v$ = %.3f µs\n$|R_{PP}|$ = %.7f"
            % (inv["time_s"] * 1e6, inv["r_pp"][index]),
            fontsize=fs.FS["annotation"], ha="center", va="top",
            linespacing=1.5)


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    testbed, exact = load_sources()
    lam_mm = testbed.C_REF / testbed.F0 * 1e3
    sweep = invariants(testbed, exact, SWEEP_DEG)

    print("point source, plane reflector at d = %.3f mm, c-axes %g/%g deg"
          % (sweep["depth"] * 1e3, PSI_A_DEG, PSI_B_DEG))
    print("qP speed along the beam in the near grain %.2f m/s, "
          "echo at 2d/v = %.4f us" % (sweep["speed"], sweep["time_s"] * 1e6))
    print("exact |R_PP| against tilt")
    for tilt, r in zip(SWEEP_DEG, sweep["r_pp"]):
        print("  %6.1f deg   %.12f" % (tilt, r))
    print("  spread over the sweep %.2e" % np.ptp(sweep["r_pp"]))
    # The drawn grid is one wavelength, not one cell. State the ratio so a
    # reader cannot mistake it for the finite-difference grid.
    print("reference grid drawn at lambda = %.3f mm, which is %g cells of "
          "%.3f mm at n_lambda = %g"
          % (lam_mm, PPW_REF, lam_mm / PPW_REF, PPW_REF))

    fig, axes = fs.figure("tiltmethod", nrows=1, ncols=3, wspace=0.05,
                          left=2.0, right=2.0, bottom=1.5, top=5.0)
    # The drawn tilts are three of the five in the sweep, so the numbers
    # under the panels are the same ones printed above.
    for i, (ax, tilt) in enumerate(zip(np.ravel(axes), TILTS_DEG)):
        draw_panel(ax, tilt, sweep, SWEEP_DEG.index(tilt), lam_mm,
                   first=(i == 0))
    for ax, letter in zip(np.ravel(axes), "abc"):
        fs.panel(ax, letter, dx=0.0)
    fs.save(fig, "tiltmethod", expect=["source", "image source"])


if __name__ == "__main__":
    main()

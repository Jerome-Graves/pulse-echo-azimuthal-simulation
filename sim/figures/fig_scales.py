r"""Figure `scales`: the coincidence of grain size and coherence length.

This is the figure that explains the negative result of
Section~\ref{sec:results}: the coda carries no usable crystallographic
information because the first Fresnel zone at the coda gate is the same
size as a grain, so the beam interrogates about one grain boundary at a
time and the return is set by the geometry of that single facet rather
than by an average over many.

Five quantities are drawn to one common length scale, all of them
computed here from the acquisition parameters and the tessellation, none
of them copied from the manuscript:

    mean grain diameter          D_g = (6 V / pi N)^(1/3)
    first Fresnel zone           D_F = sqrt(2 lambda L)   (two-way)
    specular lobe of one facet   lambda / D_g
    wavelength                   lambda = c / f0
    near-field extent            D_e^2 / (4 lambda)

Reads
    data/po_src_geom.npz
        Exact Laguerre (power) tessellation of the reference specimen,
        seed 11: 104 seed positions, their power weights, and the 536
        grain-boundary polygons. Written on the CPU by the physical-optics
        pipeline (po_src_2geom.py) from DiskSpecimen(seed=11), and copied
        here so this figure is reproducible from the repository alone.
    ../../out/sweeps/girdle_perp_ppw8/config.json
        Acquisition: centre frequency, element diameter, disc dimensions.

Writes
    <paper>/figures/scales.pdf

The script prints the numbers it draws, together with the realised
grain-size distribution and the boundary spacing along the beam, so that
every length quoted in Sections 3 and 5 can be checked against it.

No GPU and no simulation call. The heaviest step is a coarse (1 mm) CPU
power-diagram labelling, which serves only the printed check that the
closed-form mean grain diameter matches the realised tessellation.
"""
import json
import os

import numpy as np
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import MultipleLocator

import figstyle as S

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.join(HERE, "data", "po_src_geom.npz")
SWEEP = os.path.join(HERE, "..", "..", "out", "sweeps", "girdle_perp_ppw8")

# Isotropic-equivalent qP speed of the simulated ice, m/s. This is the
# reference speed used throughout the analysis (C_REF in
# sim/analysis/_t2_common.py, and the speed master_run.py sizes every
# grid with), so the wavelength quoted in the paper follows from it.
C_REF = 3850.0

# Coda analysis gate, seconds. CODA_W in sim/analysis/_t2_common.py and
# Section 3 of the paper. The Fresnel zone is evaluated at its centre.
CODA_GATE_S = (24e-6, 36e-6)

# -6 dB half-angle constant of a circular piston: its first directivity
# zero is at 1.22 lambda / D and its -6 dB point at 0.514 lambda / D.
PISTON_HALF_ANGLE_C = 0.514


# ────────────────────────────── computation ──────────────────────────────
def load_geometry():
    """Tessellation and acquisition parameters, in SI units."""
    with open(os.path.join(SWEEP, "config.json")) as fh:
        cfg = json.load(fh)
    with np.load(GEOM) as z:
        geom = dict(seeds=z["seeds"], weights=z["weights"],
                    facet_area=z["farea"])
    geom.update(f0=cfg["f0_mhz"] * 1e6,
                element_d=cfg["element_d_mm"] * 1e-3,
                disc_d=cfg["diameter_mm"] * 1e-3,
                thickness=cfg["thickness_mm"] * 1e-3)
    return geom


def length_scales(geom):
    """The five quantities of the figure, in metres and degrees.

    The Fresnel zone is the two-way (monostatic) one: a scatterer at
    transverse offset r from the axis lengthens the path by
    2(sqrt(L^2 + r^2) - L), and the first zone ends where that reaches
    lambda/2, giving r = sqrt(lambda L / 2) and a diameter
    sqrt(2 lambda L). The one-way zone would overstate the coherently
    illuminated patch by a factor sqrt(2).
    """
    lam = C_REF / geom["f0"]
    n_grains = len(geom["seeds"])
    disc_volume = np.pi * (geom["disc_d"] / 2) ** 2 * geom["thickness"]

    # Volume-equivalent mean diameter: the diameter of the sphere whose
    # volume is the mean grain volume. This is the definition behind D in
    # the paper's table of dimensionless parameters.
    d_grain = (6.0 * disc_volume / (np.pi * n_grains)) ** (1 / 3)

    gate_centre_s = 0.5 * (CODA_GATE_S[0] + CODA_GATE_S[1])
    range_m = 0.5 * C_REF * gate_centre_s        # one-way range at the gate
    d_fresnel = np.sqrt(2.0 * lam * range_m)

    return dict(lam=lam, d_grain=d_grain, d_fresnel=d_fresnel,
                near_field=geom["element_d"] ** 2 / (4.0 * lam),
                lobe_deg=np.degrees(lam / d_grain),
                range_m=range_m, n_grains=n_grains,
                beam_half_deg=np.degrees(np.arcsin(
                    PISTON_HALF_ANGLE_C * lam / geom["element_d"])),
                ka=(2 * np.pi / lam) * (d_grain / 2),
                d_over_lam=d_grain / lam)


def power_labels(geom, points):
    """Grain index of each point, by the power (Laguerre) metric.

    |x - s|^2 - w expanded, so a block of points costs one matrix
    product instead of an (n_points, n_seeds, 3) temporary.
    """
    s, w = geom["seeds"], geom["weights"]
    d = (-2.0 * (points @ s.T) + (s ** 2).sum(1) - w)
    return np.argmin(d, axis=1)


def boundary_spacing(geom):
    """Boundary area per unit volume, and the spacing along the beam.

    A line through a random microstructure crosses S_v / 2 boundaries per
    unit length, so 2 / S_v is the mean distance between boundary
    crossings along the beam. It is the range-direction counterpart of
    the transverse coincidence the figure draws.
    """
    volume = np.pi * (geom["disc_d"] / 2) ** 2 * geom["thickness"]
    s_v = geom["facet_area"].sum() / volume
    return s_v, 2.0 / s_v


def grain_diameters(geom, h=1.0e-3):
    """Volume-equivalent diameter of every grain, by CPU labelling.

    A power cell clipped to the disc has no closed-form volume, so the
    cells are counted on a coarse grid. At 1 mm the grid is fifteen
    times finer than the smallest grain and returns the mean of the
    production 0.241 mm labelling to better than one percent, which is
    ample for the spread drawn here.
    """
    r_disc = geom["disc_d"] / 2
    nx = int(np.ceil(geom["disc_d"] / h))
    nz = int(np.ceil(geom["thickness"] / h))
    x = (np.arange(nx) - (nx - 1) / 2) * h
    z = (np.arange(nz) - (nz - 1) / 2) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    inside = (X ** 2 + Y ** 2) <= r_disc ** 2
    pxy = np.stack([X[inside], Y[inside]], axis=1)

    counts = np.zeros(len(geom["seeds"]), int)
    for zz in z:
        pts = np.column_stack([pxy, np.full(len(pxy), zz)])
        counts += np.bincount(power_labels(geom, pts),
                              minlength=len(geom["seeds"]))
    vol = counts * h ** 3
    return (6.0 * vol[vol > 0] / np.pi) ** (1 / 3)


def slice_cells(geom, window, z_plane=0.0):
    """Grain polygons in the plane z = z_plane, exactly.

    The restriction of a 3D power diagram to a plane is a 2D power
    diagram of the projected seeds with weights w - (z_i - z)^2, so every
    cell is an intersection of half-planes and follows from clipping the
    drawing window with one half-plane per other seed. Exact, and it
    resamples no voxel grid.
    """
    p = geom["seeds"][:, :2]
    w = geom["weights"] - (geom["seeds"][:, 2] - z_plane) ** 2
    x0, x1, y0, y1 = window
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    cells = []
    for i in range(len(p)):
        poly = box
        for j in range(len(p)):
            if j == i or len(poly) == 0:
                continue
            a = 2.0 * (p[j] - p[i])
            c = p[j] @ p[j] - w[j] - p[i] @ p[i] + w[i]
            poly = _clip_halfplane(poly, a, c)
        if len(poly) >= 3:
            cells.append((i, poly))
    return cells


def _clip_halfplane(poly, a, c):
    """Sutherland-Hodgman clip of a convex polygon to a.x <= c."""
    out = []
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        dp, dq = a @ p - c, a @ q - c
        if dp <= 0:
            out.append(p)
        if (dp < 0) != (dq < 0):
            out.append(p + dp / (dp - dq) * (q - p))
    return np.array(out) if out else np.zeros((0, 2))


# ─────────────────────────────── drawing ─────────────────────────────────
# Scene and ruler share one axes at equal aspect, so a length drawn in
# the scene and the same length drawn as a bar are the same number of
# millimetres on the page. That is the whole point of the figure.
X_SPAN_MM = 52.0        # width of the common length axis
BAR_PITCH_MM = 3.0      # vertical spacing of the ruler bars
BAR_BASE_MM = 2.2       # centre of the lowest bar, above the axis line
SCENE_GAP_MM = 3.2      # clear space between the ruler and the scene
LABEL_BOX = dict(fc="white", ec="none", alpha=0.85, pad=1.0)


def axes_span(fig, margins):
    """Data span in millimetres that exactly fills the axes box.

    At equal aspect matplotlib shrinks the box to whatever the data
    limits ask for, leaving white space and pulling the drawing away
    from the panel edge. Taking the y span from the box aspect instead
    makes the axes fill its cell exactly.
    """
    w_in, h_in = fig.get_size_inches()
    box_w = (margins["right"] - margins["left"]) * w_in
    box_h = (margins["top"] - margins["bottom"]) * h_in
    return X_SPAN_MM, X_SPAN_MM * box_h / box_w


def draw_scene(ax, geom, sc, y_lo, y_hi):
    """The beam plane at the coda gate, drawn to scale.

    Real grain boundaries of the reference specimen, with the mean grain
    and the first Fresnel zone as concentric circles centred where the
    gate centre falls. The thinness of the annulus between them is the
    result.
    """
    mm = 1e3
    y_beam = 0.5 * (y_lo + y_hi)
    half_h = 0.5 * (y_hi - y_lo)
    # Signed position of the gate centre along the beam, from the centre
    # of the disc: the element sits on the rim, one radius behind it.
    gate_x = mm * (sc["range_m"] - geom["disc_d"] / 2)
    x_shift = 0.5 * X_SPAN_MM - gate_x           # centre the scene on it
    win = np.array([-x_shift, X_SPAN_MM - x_shift,
                    -half_h, half_h]) / mm

    cells = slice_cells(geom, win)
    for _, poly in cells:
        ax.add_patch(Polygon(poly * mm + [x_shift, y_beam], closed=True,
                             fc="none", ec=S.GREY, lw=S.LW["annotation"],
                             zorder=2))
    # Tint the grain the gate centre falls in, so that the polygons read
    # as grains and the circles below can be judged against a real one.
    here = int(power_labels(geom, np.array([[gate_x / mm, 0.0, 0.0]]))[0])
    for i, poly in cells:
        if i == here:
            ax.add_patch(Polygon(poly * mm + [x_shift, y_beam], closed=True,
                                 fc=S.HIGHLIGHT, ec="none", alpha=0.35,
                                 zorder=1))

    cx = gate_x + x_shift
    r_f, r_g = 0.5 * mm * sc["d_fresnel"], 0.5 * mm * sc["d_grain"]

    # Specular lobe: the echo from a facet of extent D is confined to
    # lambda/D about the mirror direction, so the facet is seen at all
    # only if its normal points back within this cone.
    half = np.radians(0.5 * sc["lobe_deg"])
    ax.add_patch(Polygon([[cx, y_beam],
                          [0.0, y_beam + cx * np.tan(half)],
                          [0.0, y_beam - cx * np.tan(half)]], closed=True,
                         fc=S.GREY, ec="none", alpha=0.35, zorder=3))
    ax.annotate("", xy=(cx - r_f - 0.8, y_beam), xytext=(1.5, y_beam),
                zorder=4, arrowprops=dict(arrowstyle="-|>", color="black",
                                          linewidth=S.LW["annotation"],
                                          mutation_scale=5))
    S.direct_label(ax, 1.5, y_beam + 0.8, "beam", colour="black",
                   bbox=LABEL_BOX)
    S.arrow_label(ax, "specular lobe of one facet, %.1f$\\degree$ wide"
                  % sc["lobe_deg"],
                  xy=(0.30 * cx, y_beam - 0.30 * cx * np.tan(half)),
                  xytext=(1.5, y_lo + 0.6), colour=S.GREY, va="bottom",
                  bbox=LABEL_BOX)

    ax.add_patch(Circle((cx, y_beam), r_g, fc="none", ec="black",
                        lw=S.LW["emphasis"], ls=(0, (3.2, 1.6)), zorder=6))
    ax.add_patch(Circle((cx, y_beam), r_f, fc="none", ec=S.BLUE,
                        lw=S.LW["emphasis"], zorder=6))
    S.arrow_label(ax, "mean grain", xy=(cx + 0.78 * r_g, y_beam + 0.68 * r_g),
                  xytext=(cx + r_g + 1.6, y_beam + 0.86 * r_g),
                  colour="black", bbox=LABEL_BOX)
    S.arrow_label(ax, "first Fresnel zone",
                  xy=(cx + 0.72 * r_f, y_beam - 0.72 * r_f),
                  xytext=(cx + r_f + 0.8, y_beam - r_f - 1.4),
                  colour=S.BLUE, va="top", bbox=LABEL_BOX)
    S.arrow_label(ax, "%.0f%% apart" % (100 * (1 - r_f / r_g)),
                  xy=(cx + 0.5 * (r_f + r_g), y_beam),
                  xytext=(cx + r_g + 2.4, y_beam - 1.6), colour="black",
                  va="center", bbox=LABEL_BOX)
    S.direct_label(ax, 1.5, y_hi - 0.5,
                   "the first Fresnel zone and the grain are the same size,\n"
                   "so the beam sees about one boundary at a time",
                   colour="black", va="top", bbox=LABEL_BOX,
                   linespacing=1.25)


def draw_ruler(ax, sc):
    """The four lengths as bars from a common origin on the length axis."""
    mm = 1e3
    rows = [("mean grain diameter", mm * sc["d_grain"], "black"),
            ("first Fresnel zone at the gate", mm * sc["d_fresnel"], S.BLUE),
            ("near-field extent", mm * sc["near_field"], S.GREY),
            ("wavelength", mm * sc["lam"], S.GREY)]
    for k, (name, value, colour) in enumerate(rows):
        y = BAR_BASE_MM + (len(rows) - 1 - k) * BAR_PITCH_MM
        ax.plot([0.0, value], [y, y], color=colour, lw=2.0,
                solid_capstyle="butt", zorder=4)
        S.direct_label(ax, value + 1.4, y, "%s, %.3g mm" % (name, value),
                       colour=colour, va="center")
    return BAR_BASE_MM + (len(rows) - 1) * BAR_PITCH_MM


def draw_coincidence(ax, sc, y_lo, y_hi):
    """Drop lines so the two large bars can be read against each other.

    Two bars of nearly equal length are only convincing if their ends can
    be compared exactly, and 2.2 mm at this scale is 3.5 mm on the page.
    """
    for value in (1e3 * sc["d_grain"], 1e3 * sc["d_fresnel"]):
        ax.plot([value, value], [y_lo, y_hi + 0.9], color=S.GREY,
                lw=S.LW["annotation"], ls=(0, (1.0, 1.4)), zorder=3)


# ──────────────────────────────── figure ─────────────────────────────────
def main():
    geom = load_geometry()
    sc = length_scales(geom)
    d_eq = grain_diameters(geom)
    s_v, spacing = boundary_spacing(geom)

    print("wavelength                      %6.3f mm" % (1e3 * sc["lam"]))
    print("mean grain diameter, %d grains  %6.2f mm"
          % (sc["n_grains"], 1e3 * sc["d_grain"]))
    print("  individual grains, mean       %6.2f mm, 10 to 90%% %.1f to %.1f"
          % ((1e3 * d_eq.mean(),)
             + tuple(1e3 * np.percentile(d_eq, [10, 90]))))
    print("first Fresnel zone at %4.1f us   %6.2f mm, at range %.1f mm"
          % (1e6 * 0.5 * sum(CODA_GATE_S), 1e3 * sc["d_fresnel"],
             1e3 * sc["range_m"]))
    print("near-field extent               %6.2f mm"
          % (1e3 * sc["near_field"]))
    print("specular lobe, lambda / D       %6.2f deg" % sc["lobe_deg"])
    print("beam half-angle at -6 dB        %6.2f deg" % sc["beam_half_deg"])
    print("D / lambda %.2f, ka %.1f" % (sc["d_over_lam"], sc["ka"]))
    print("boundary area per unit volume   %6.1f /m, so boundaries every "
          "%.1f mm along the beam" % (s_v, 1e3 * spacing))

    fig = S.canvas("scales", 75.0)
    margins = S.mm_margins(fig, left=3.0, right=3.0, bottom=8.5, top=2.0)
    ax = fig.add_axes([margins["left"], margins["bottom"],
                       margins["right"] - margins["left"],
                       margins["top"] - margins["bottom"]])
    x_span, y_span = axes_span(fig, margins)
    ax.set_xlim(0.0, x_span)
    ax.set_ylim(0.0, y_span)
    ax.set_aspect("equal")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(top=False, right=False)
    # Minor ticks every 2 mm: the axis is a ruler and is meant to be read
    # against the drawing, not just to carry four labelled decades.
    ax.xaxis.set_minor_locator(MultipleLocator(2.0))
    ax.set_xlabel("length (mm)")

    y_ruler_top = draw_ruler(ax, sc)
    draw_coincidence(ax, sc, BAR_BASE_MM - 1.4, y_ruler_top)
    draw_scene(ax, geom, sc, y_ruler_top + SCENE_GAP_MM, y_span)

    S.save(fig, "scales", expect=("length (mm)", "mean grain diameter",
                                  "first Fresnel zone", "wavelength",
                                  "near-field extent", "beam"))


if __name__ == "__main__":
    main()

r"""Figure `microstructure`: the simulated specimen and its fabric (Sec. 3.2).

Supports the claims that the specimen is a Laguerre tessellation of about
100 approximately equiaxed grains with a size coefficient of variation of
0.35 and a volume-equivalent mean diameter near 17 mm, that the grain
boundaries are planar by construction so their normals are available in
closed form, and that the two fabrics simulated are a girdle and a single
maximum.

The specimen is not read from a saved label volume: it is rebuilt here by
the same DiskSpecimen the sweeps use, from the run configuration on disk,
at a coarse grid. The build is deterministic in its seed, and the seeds,
weights and c-axes it returns were checked element by element against the
production ppw 8 build of the same sweep and agree exactly. Every number
printed and drawn is therefore computed here, not copied out of the
manuscript.

Panels (a) and (b) use no voxels at all. A Laguerre cell is an
intersection of half-spaces, so its intersection with any plane is a
convex polygon that can be clipped exactly; the cut-away surfaces and the
mid-plane slice are drawn from the seeds and weights as exact vector
polygons. The label volume is used only where a volume is needed, which
is the grain-size distribution of panel (c).

Reads (paths resolved relative to this file):
  ../../out/sweeps/girdle_seed11_ppw8_dev/config.json   girdle run configuration
  ../../out/sweeps/singlemax_seed11_ppw8_twin/config.json     single-maximum ditto
and imports ../specimen.py, which generates the specimen.

Writes ../../../pulse-echo-cof-paper/figures/microstructure.pdf and
prints the grain statistics and fabric eigenvalues that appear in the
figure and in Section 3.2.

Panels: (a) cut-away of the tessellation with a third of the disc
removed, so the reader sees interior grains and not only the surface
layer; (b) the mid-thickness slice, which is the plane the beam travels
in, with the in-plane trace of each c-axis drawn as a tick; (c) the
distribution of equivalent grain diameters; (d, e) equal-area pole
figures of the two fabrics.

Colour is a CYCLIC scale throughout, because a c-axis is an axial
direction: c and -c are the same orientation, so the colour must return
to itself after 180 degrees. On a linear scale the two ends of the scale
would be the same orientation in two very different colours, and a false
seam would run through the middle of the fabric.
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
SWEEPS = REPO / "out" / "sweeps"

sys.path.insert(0, str(REPO / "sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import specimen as sp                                    # noqa: E402

# The production build labels the grid on the GPU. A figure has no
# business competing with a running sweep for the card, and the CPU
# fall-back takes about ten seconds at the grid used here, so the GPU
# path is switched off. Returning None is exactly the "CUDA unavailable"
# signal DiskSpecimen.build is already written to handle.
sp.DiskSpecimen._label_grid_gpu = staticmethod(lambda *a, **k: None)

# The two sweeps whose specimens the manuscript calls the girdle and the
# single maximum. Their configurations carry every specimen parameter.
GIRDLE_RUN = "girdle_seed11_ppw8_dev"
SINGLEMAX_RUN = "singlemax_seed11_ppw8_twin"

# Grid for the label volume. The production grid of 0.241 mm is set by
# the wave, and nothing drawn here is a wave. At 1 mm a mean grain is two
# thousand cells, which fixes its volume to well under a per cent; only
# the two or three smallest grains are coarsely resolved, and the mean
# and coefficient of variation printed below sit within 0.03 mm and 0.002
# of the values the production grid gives for the same tessellation.
BUILD_H = 1.0e-3                # m

# Restated rather than imported: sim/analysis/_t2_common.py owns this
# constant but pulls in the openUSCT forward model on import, which is
# far more machinery than a figure needs.
C_REF = 3850.0                  # m/s, reference qP speed (_t2_common.C_REF)

# The rim of the disc is drawn as an inscribed polygon so that every
# surface of the cut-away is planar and can be clipped exactly. At 120
# facets the silhouette departs from a circle by 0.02 mm on a 100 mm
# disc, which is a hundredth of the width of the thinnest line here.
N_RIM = 120

# Azimuths of the wedge removed from the disc, in degrees. A quarter
# would do, but a third exposes half again as much interior for the same
# silhouette, and the interior is the whole reason for cutting it open.
NOTCH_DEG = (0.0, 120.0)

# Orthographic camera. The azimuth looks into the removed wedge, offset
# from its bisector so the two cut faces are seen at different
# foreshortening and the block reads as a solid rather than a symmetric
# emblem.
ELEV_DEG, AZIM_DEG = 25.0, 52.0

# Direction the scene is lit from, for depth cueing only. The face
# colour carries meaning, so the shading is kept to a narrow range: just
# enough to separate a vertical face from a horizontal one.
LIGHT = np.array([0.30, 0.45, 0.84])
SHADE_FLOOR = 0.55

CMAP_CYCLIC = "twilight"        # perceptually uniform and cyclic; the
                                # sequential maps in figstyle cannot
                                # carry a periodic quantity

TICK_LENGTH_MM = 9.0            # full length of a c-axis trace in panel (b)

# Edge tags. Only the first two are drawn: a "seam" is an edge of the
# construction, not of the specimen, and drawing it would put a false
# grain boundary on the picture.
GRAIN, REAL, SEAM = "grain", "real", "seam"


# --------------------------------------------------------------- compute

def load_config(run):
    """Specimen and acquisition parameters of one sweep."""
    with open(SWEEPS / run / "config.json") as fh:
        return json.load(fh)


def build(cfg, h=BUILD_H):
    """The specimen of one sweep, rebuilt exactly as the sweep built it.

    Only `concentration` and `fabric_axis` differ between the two runs,
    and neither is touched until the c-axes are drawn, so the two builds
    return the same tessellation with different fabrics.
    """
    spec = sp.DiskSpecimen(
        diameter_m=cfg["diameter_mm"] * 1e-3,
        thickness_m=cfg["thickness_mm"] * 1e-3,
        n_grains=cfg["n_grains"], size_cv=cfg["size_cv"],
        concentration=cfg["concentration"],
        spatial_corr=cfg["spatial_corr"],
        fabric_axis=tuple(cfg["fabric_axis"]), seed=cfg["seed"])
    return spec.build(h)


def grain_diameters(built):
    """Volume-equivalent diameter of every realised grain, in mm."""
    labels, h = built["labels"], built["h"]
    counts = np.bincount(labels[labels >= 0].ravel(),
                         minlength=len(built["seeds"]))
    volume = counts * h ** 3
    return 2.0 * (3.0 * volume[counts > 0] / (4.0 * np.pi)) ** (1 / 3) * 1e3


def nominal_diameter(cfg):
    """Design volume-equivalent mean diameter, in mm.

    The disc volume shared equally between the requested number of
    grains. This is the 17.4 mm the manuscript quotes; the realised
    arithmetic mean of panel (c) is smaller because the tessellation
    delivers a few more grains than requested and because the mean of a
    set of diameters is not the diameter of the mean volume.
    """
    radius = 0.5 * cfg["diameter_mm"] * 1e-3
    volume = np.pi * radius ** 2 * cfg["thickness_mm"] * 1e-3
    per_grain = volume / cfg["n_grains"]
    return 2.0 * (3.0 * per_grain / (4.0 * np.pi)) ** (1 / 3) * 1e3


def caxis_angle(axes):
    """Cyclic colour scalar: c-axis orientation about x, in degrees.

    x is the girdle axis of the specimen drawn in panels (a) and (b), so
    the c-axes lie close to the y-z plane and this angle says where on
    the girdle each grain sits. It is folded onto [0, 180) because a
    c-axis is axial, and it is that folding, not a matter of taste, that
    requires a cyclic colormap.
    """
    return np.degrees(np.arctan2(axes[:, 2], axes[:, 1])) % 180.0


# ------------------------------------------- exact Laguerre cross sections

def tagged(points, tags):
    """Polygon as a list of (point, tag), tag naming the edge that starts
    at that point."""
    return [(np.asarray(p, float), t) for p, t in zip(points, tags)]


def clip_halfplane(poly, normal, offset, tag):
    """Sutherland-Hodgman clip of a tagged polygon to normal.p <= offset.

    Tags are carried so that the drawing can tell a grain boundary from
    an edge of the region the cell was clipped to. A vertex created on
    the way out of the half-plane starts an edge that lies in the
    clipping plane and takes its tag; one created on the way in
    continues the edge it was cut from and keeps that edge's tag.
    """
    out = []
    n = len(poly)
    for i in range(n):
        p, tag_p = poly[i]
        q = poly[(i + 1) % n][0]
        dp = offset - float(normal @ p)
        dq = offset - float(normal @ q)
        if dp >= 0.0:
            out.append((p, tag_p))
        if (dp >= 0.0) != (dq >= 0.0):
            cut = p + dp / (dp - dq) * (q - p)
            out.append((cut, tag if dp >= 0.0 else tag_p))
    return out


def polygon_area(poly):
    """Unsigned area of a tagged polygon."""
    pts = np.array([p for p, _ in poly])
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) -
                           np.dot(y, np.roll(x, -1))))


def power_cells(seeds2, weights2, boundary, min_area):
    """The 2D power (Laguerre) diagram clipped to `boundary`.

    Cell i is the set of points whose power distance |p - s_i|^2 - w_i is
    least, so it is the intersection of one half-plane per other seed:
    2 (s_j - s_i).p <= q_j - q_i with q = |s|^2 - w. Half-planes are
    applied nearest seed first, which kills an empty cell in two or three
    clips instead of a hundred.
    """
    q = (seeds2 ** 2).sum(1) - weights2
    cells = {}
    for i in range(len(seeds2)):
        offsets = seeds2 - seeds2[i]
        order = np.argsort((offsets ** 2).sum(1))
        poly = boundary
        for j in order:
            if j == i:
                continue
            poly = clip_halfplane(poly, 2.0 * offsets[j], q[j] - q[i],
                                  GRAIN)
            if len(poly) < 3:
                break
        if len(poly) >= 3 and polygon_area(poly) > min_area:
            cells[i] = poly
    return cells


class Surface:
    """A planar face of the drawn body, with its own 2D coordinates.

    A Laguerre cell restricted to a plane is again a power diagram, with
    the seed projected into the plane and its weight reduced by the
    squared distance from it, so one 2D routine draws the mid-plane
    slice, the flat faces of the rim and the cut faces alike.
    """

    def __init__(self, origin, e1, e2, boundary):
        self.origin = np.asarray(origin, float)
        self.e1 = np.asarray(e1, float)
        self.e2 = np.asarray(e2, float)
        self.normal = np.cross(self.e1, self.e2)
        self.boundary = boundary

    def cells(self, seeds, weights, min_area):
        """{grain: tagged polygon} of the grains that reach this face."""
        offset = seeds - self.origin
        seeds2 = np.column_stack([offset @ self.e1, offset @ self.e2])
        weights2 = weights - (offset @ self.normal) ** 2
        return power_cells(seeds2, weights2, self.boundary, min_area)

    def to_3d(self, poly):
        """Vertices of a polygon of this face, back in specimen space."""
        pts = np.array([p for p, _ in poly])
        return (self.origin + pts[:, :1] * self.e1
                + pts[:, 1:] * self.e2)


def rectangle(u0, u1, v0, v1, tags):
    """Tagged rectangle, edges tagged from the u0-v0 corner anticlockwise."""
    return tagged([(u0, v0), (u1, v0), (u1, v1), (u0, v1)], tags)


def arc_points(radius, t0, t1):
    """Vertices of the inscribed rim polygon from azimuth t0 to t1."""
    k0 = int(round(t0 / (2 * np.pi) * N_RIM))
    k1 = int(round(t1 / (2 * np.pi) * N_RIM))
    ang = np.arange(k0, k1 + 1) * 2 * np.pi / N_RIM
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def sector(radius, t0, t1, tag_first, tag_last):
    """Tagged polygon of the sector from azimuth t0 to t1.

    The two straight radii are tagged separately because one may be an
    exposed cut face and the other only a seam between two pieces of the
    same flat surface.
    """
    pts = arc_points(radius, t0, t1)
    return ([(np.zeros(2), tag_first)]
            + [(p, REAL) for p in pts[:-1]]
            + [(pts[-1], tag_last)])


def cutaway_surfaces(radius, half_thickness):
    """Every planar face of the disc with a wedge of it removed.

    The removed wedge is what makes the panel worth drawing: an opaque
    disc shows only the grains that happen to touch its surface, whereas
    the two cut faces are a section through the interior. What is left is
    covered by the top face, the two cut rectangles and the rim facets.
    The top is built in sectors narrower than 180 degrees because the
    notched disc is not convex and the clipper needs a convex region.

    The underside is not built at all. The camera looks down on the
    specimen, so it can never be seen, and half the clipping work would
    be spent on faces that are discarded.
    """
    z = np.eye(3)[2]
    a1, a2 = np.radians(NOTCH_DEG)
    kept = 2 * np.pi - (a2 - a1)
    n_piece = int(np.ceil(kept / np.radians(170.0)))
    cuts = a2 + np.arange(n_piece + 1) * kept / n_piece

    surfaces = [Surface(
        (0, 0, half_thickness), np.eye(3)[0], np.eye(3)[1],
        sector(radius, cuts[i], cuts[i + 1],
               REAL if i == 0 else SEAM,
               REAL if i == n_piece - 1 else SEAM))
        for i in range(n_piece)]

    # The two cut faces. Each outward normal is the radial direction of
    # its own edge turned a quarter turn into the removed wedge, which is
    # what the two orderings of (radial, z) below produce.
    for angle, radial_first in ((a1, False), (a2, True)):
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        if radial_first:
            surfaces.append(Surface((0, 0, 0), radial, z, rectangle(
                0.0, radius, -half_thickness, half_thickness,
                [REAL] * 4)))
        else:
            surfaces.append(Surface((0, 0, 0), z, radial, rectangle(
                -half_thickness, half_thickness, 0.0, radius,
                [REAL] * 4)))

    # Rim facets, skipping the wedge that was removed. Their vertical
    # edges are seams between adjacent facets; the horizontal ones are
    # the real junction with the top and bottom faces.
    step = 2 * np.pi / N_RIM
    apothem = radius * np.cos(0.5 * step)
    half_chord = radius * np.sin(0.5 * step)
    first = int(round(a2 / step))
    for k in range(first, first + N_RIM - int(round((a2 - a1) / step))):
        centre = (k + 0.5) * step
        normal = np.array([np.cos(centre), np.sin(centre), 0.0])
        tangent = np.array([-np.sin(centre), np.cos(centre), 0.0])
        surfaces.append(Surface(
            apothem * normal, tangent, z,
            rectangle(-half_chord, half_chord,
                      -half_thickness, half_thickness,
                      [REAL, SEAM, REAL, SEAM])))
    return surfaces


# ------------------------------------------------------------ projection

def camera(elev_deg, azim_deg):
    """Orthographic (view, right, up) unit vectors."""
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    view = np.array([np.cos(elev) * np.cos(azim),
                     np.cos(elev) * np.sin(azim), np.sin(elev)])
    right = np.array([-np.sin(azim), np.cos(azim), 0.0])
    return view, right, np.cross(view, right)


def render_cutaway(built, radius, half_thickness, min_area):
    """(faces, shades, grain index, grain edges, structural edges) in mm.

    Faces pointing away from the camera are dropped before anything is
    clipped, so only the surfaces the reader can see are computed. What
    survives is the boundary of a solid seen from outside, so the faces
    tile the picture without overlapping and a depth sort is enough to
    order them.

    Grain boundaries and edges of the solid are returned separately
    because they carry different weight on the page: without a heavier
    line where one face of the body meets another, a cut face and the
    top face read as one continuous surface.
    """
    view, right, up = camera(ELEV_DEG, AZIM_DEG)
    seeds, weights = built["seeds"], built["weights"]

    faces, shades, owners, depths = [], [], [], []
    grain_edges, structure_edges = [], []
    for surf in cutaway_surfaces(radius, half_thickness):
        if surf.normal @ view <= 0.0:
            continue
        shade = SHADE_FLOOR + (1.0 - SHADE_FLOOR) * max(
            float(surf.normal @ LIGHT), 0.0)
        for grain, poly in surf.cells(seeds, weights, min_area).items():
            pts = surf.to_3d(poly)
            flat = np.column_stack([pts @ right, pts @ up]) * 1e3
            faces.append(flat)
            shades.append(shade)
            owners.append(grain)
            depths.append(float((pts @ view).mean()))
            for i, (_, tag) in enumerate(poly):
                if tag == GRAIN:
                    grain_edges.append(flat[[i, (i + 1) % len(poly)]])
                elif tag == REAL:
                    structure_edges.append(flat[[i, (i + 1) % len(poly)]])

    order = np.argsort(depths)
    return ([faces[i] for i in order], np.array(shades)[order],
            np.array(owners)[order], grain_edges, structure_edges)


# ---------------------------------------------------------------- drawing

def draw_triad(ax, origin, length):
    """The specimen frame, projected by the same camera as the body.

    figstyle's axes_triad draws two axis-aligned arrows, which is right
    for a plane schematic and wrong here: under a projection none of the
    three axes is axis-aligned on the page, and all three are needed
    because the colour scale is an angle measured about x.
    """
    _, right, up = camera(ELEV_DEG, AZIM_DEG)
    for axis, name in zip(np.eye(3), ("x", "y", "z")):
        step = length * np.array([axis @ right, axis @ up])
        tip = origin + step
        ax.annotate("", xy=tuple(tip), xytext=tuple(origin),
                    arrowprops=dict(arrowstyle="->", color="black",
                                    linewidth=fs.LW["schematic"]),
                    zorder=6)
        # Labels sit clear of the arrow rather than beyond its tip: the
        # tip of the z arrow is a couple of millimetres below the solid.
        away = (-0.32, 0.0) if name == "z" else (0.0, -0.20)
        ax.text(tip[0] + away[0] * length, tip[1] + away[1] * length,
                name, color="black", fontsize=fs.FS["annotation"],
                ha="right" if name == "z" else "center",
                va="center" if name == "z" else "top", zorder=6)


def draw_cutaway(ax, built, cfg, cmap, norm):
    """Panel (a): the tessellation with a wedge of the disc cut away."""
    radius = 0.5 * cfg["diameter_mm"] * 1e-3
    half_thickness = 0.5 * cfg["thickness_mm"] * 1e-3
    angle = caxis_angle(built["axes"])

    # A face smaller than a twentieth of a square millimetre is a sliver
    # of the construction and not a grain any reader can see.
    faces, shades, owners, grain_edges, structure_edges = render_cutaway(
        built, radius, half_thickness, min_area=5e-8)

    # Each face is given a hairline outline in its own colour. Without it
    # the antialiased gap between two abutting polygons shows as a white
    # thread, and a rim of thirty facets acquires thirty false seams.
    rgb = np.array([cmap(norm(angle[g]))[:3] for g in owners])
    rgb = rgb * shades[:, None]
    ax.add_collection(PolyCollection(
        faces, facecolors=rgb, edgecolors=rgb, linewidths=0.3, zorder=2))
    ax.add_collection(LineCollection(
        grain_edges, colors="black", linewidths=fs.LW["cell_grid"],
        zorder=3))
    ax.add_collection(LineCollection(
        structure_edges, colors="black", linewidths=fs.LW["schematic"],
        zorder=4))

    span = np.concatenate(faces)
    x0, x1 = span[:, 0].min(), span[:, 0].max()
    y0, y1 = span[:, 1].min(), span[:, 1].max()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0 - 0.13 * (y1 - y0), y1)
    fs.schematic(ax)
    ax.set_anchor("N")

    # The triad hangs below the corner where the two cut faces meet, in
    # the wedge of empty page that the cut opens up, so it costs the
    # drawing almost no height.
    _, right, up = camera(ELEV_DEG, AZIM_DEG)
    corner = np.array([0.0, 0.0, -half_thickness]) * 1e3
    draw_triad(ax, np.array([corner @ right, corner @ up])
               - [0.0, 0.19 * (y1 - y0)], 0.09 * (x1 - x0))
    return len(faces)


def draw_slice(ax, built, cfg, cmap, norm):
    """Panel (b): the mid-thickness plane, which is the beam plane."""
    radius = 0.5 * cfg["diameter_mm"] * 1e-3
    axes, angle = built["axes"], caxis_angle(built["axes"])
    x, y, _ = np.eye(3)
    rim = arc_points(radius, 0.0, 2 * np.pi)[:-1]
    surf = Surface((0, 0, 0), x, y, tagged(rim, [REAL] * len(rim)))
    cells = surf.cells(built["seeds"], built["weights"], min_area=1e-8)

    polys = [np.array([p for p, _ in poly]) * 1e3
             for poly in cells.values()]
    ax.add_collection(PolyCollection(
        polys, facecolors=[cmap(norm(angle[g])) for g in cells],
        edgecolors="black", linewidths=fs.LW["cell_grid"], zorder=2))

    # The trace of each c-axis in this plane, drawn as a tick through the
    # cell centroid. It pairs the colour with a second, achromatic
    # channel, so the fabric survives a greyscale print: the ticks of a
    # girdle about x all lie along y, and shorten as the c-axis tips out
    # of the plane.
    ticks = []
    for grain, poly in cells.items():
        if polygon_area(poly) < 3e-5:            # smaller than 30 mm^2
            continue                             # a sliver has no room
        centre = np.array([p for p, _ in poly]).mean(axis=0) * 1e3
        step = 0.5 * TICK_LENGTH_MM * axes[grain, :2]
        ticks.append([centre - step, centre + step])
    ax.add_collection(LineCollection(
        ticks, colors="black", linewidths=fs.LW["annotation"], zorder=3))

    # Clear strip above the disc for the panel letter and the condition
    # label, and one below it for the scale bar.
    ax.set_xlim(-1.03 * radius * 1e3, 1.03 * radius * 1e3)
    ax.set_ylim(-1.13 * radius * 1e3, 1.19 * radius * 1e3)
    fs.schematic(ax)
    ax.set_anchor("N")
    fs.scale_bar(ax, 24.0, -1.07 * radius * 1e3, 20.0, "20 mm")


def draw_sizes(ax, diameters, nominal):
    """Panel (c): the distribution of equivalent grain diameters.

    Both the design diameter and the realised arithmetic mean are drawn.
    They differ by 2 mm and the manuscript quotes both, so leaving either
    off would invite a reader to measure the histogram and find the wrong
    one.
    """
    bins = np.arange(0.0, np.ceil(diameters.max() / 2.5) * 2.5 + 2.5, 2.5)
    counts, _, _ = ax.hist(diameters, bins=bins, color=fs.GREY_LIGHT,
                           edgecolor="black", linewidth=fs.LW["spine"],
                           zorder=2)
    top = 1.45 * counts.max()

    # The lines stop short of the labels above them, which is why the
    # bars are given headroom in the first place.
    ax.plot([nominal] * 2, [0, 0.72 * top], color=fs.BLACK,
            linewidth=fs.LW["emphasis"], zorder=4)
    ax.plot([diameters.mean()] * 2, [0, 0.72 * top], color=fs.BLUE,
            linestyle=fs.SERIES[1][1], linewidth=fs.LW["data"], zorder=4)

    # The two lines are 2 mm apart on a 35 mm axis, so they cannot each
    # be labelled where they stand. They are named instead in the clear
    # strip above the bars, each in its own colour and each also carrying
    # a dash pattern, so the pairing survives a greyscale print.
    for i, (text, colour) in enumerate((
            ("nominal %.1f mm" % nominal, fs.BLACK),
            ("mean %.1f mm" % diameters.mean(), fs.BLUE),
            ("CV %.2f" % (diameters.std() / diameters.mean()), fs.BLACK))):
        fs.direct_label(ax, 0.96, 0.985 - 0.085 * i, text, colour=colour,
                        ha="right", va="top", transform=ax.transAxes)

    ax.set_xlim(0.0, bins[-1])
    ax.set_ylim(0.0, top)
    ax.set_xlabel("grain diameter (mm)")
    ax.set_ylabel("number of grains")


def draw_pole(ax, axes, cmap, norm, caption, girdle_axis=None,
              single_axis=None):
    """Panel (d) or (e): equal-area pole figure, upper hemisphere.

    Viewed down the disc axis z, so the page is the plane the beam
    travels in and the figure can be read directly against panel (b). A
    c-axis is axial, so each is folded onto the upper hemisphere; a
    fabric that lies in the plane of the disc therefore appears as two
    antipodal patches on the primitive circle rather than one.
    """
    sign = np.where(axes[:, 2] < 0.0, -1.0, 1.0)
    folded = axes * sign[:, None]
    horizontal = np.maximum(np.hypot(folded[:, 0], folded[:, 1]), 1e-12)
    # Lambert equal-area: r = sqrt(2) sin(theta/2), unity on the equator.
    r = np.sqrt(1.0 - folded[:, 2])
    pts = (r / horizontal)[:, None] * folded[:, :2]

    ang = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(ang), np.sin(ang), color="black",
            linewidth=fs.LW["spine"], zorder=3)
    if girdle_axis is not None:
        # The girdle plane is normal to the fabric axis, so it projects
        # to the diameter perpendicular to that axis. Drawing it names
        # the fabric without relying on colour.
        d = np.array([-girdle_axis[1], girdle_axis[0]])
        d = 1.08 * d / np.linalg.norm(d)
        ax.plot([-d[0], d[0]], [-d[1], d[1]], color=fs.GREY,
                linestyle=fs.SERIES[1][1], linewidth=fs.LW["reference"],
                zorder=5)
    if single_axis is not None:
        # The fabric axis is horizontal, so it meets the primitive circle
        # at two antipodal points. Ticks outside the circle mark them
        # without adding a symbol on top of the data.
        u = np.array(single_axis[:2]) / np.linalg.norm(single_axis[:2])
        for s in (1.0, -1.0):
            ax.plot([s * u[0], s * 1.16 * u[0]], [s * u[1], s * 1.16 * u[1]],
                    color="black", linewidth=fs.LW["interface"],
                    solid_capstyle="butt", zorder=5)

    ax.scatter(pts[:, 0], pts[:, 1], s=5.5,
               c=cmap(norm(caxis_angle(axes))), edgecolors="black",
               linewidths=fs.LW["cell_grid"], zorder=4)
    ax.text(1.07, 0.0, "x", fontsize=fs.FS["annotation"], ha="left",
            va="center")
    ax.text(0.0, 1.07, "y", fontsize=fs.FS["annotation"], ha="center",
            va="bottom")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.91, 1.15)
    fs.schematic(ax)
    ax.set_anchor("N")
    for i, line in enumerate(caption):
        fs.direct_label(ax, 0.0, -1.10 - 0.26 * i, line, ha="center",
                        va="top")


# ----------------------------------------------------------------- layout

# Panel boxes in millimetres from the bottom left of the page, so that
# the cut-away gets the width it needs and the two pole figures stack in
# one column instead of costing a fifth of the page each.
BOXES = dict(
    cutaway=(0.5, 9.0, 64.0, 50.5),
    slice=(65.0, 8.5, 46.0, 47.0),
    sizes=(122.0, 9.0, 31.5, 45.5),
    pole_girdle=(156.0, 32.5, 27.5, 29.0),
    pole_single=(156.0, 2.0, 27.5, 29.0),
)


def axes_mm(fig, box):
    """Axes at a position given in millimetres on the page."""
    width, height = (v / fs.MM for v in fig.get_size_inches())
    x, y, w, h = box
    return fig.add_axes([x / width, y / height, w / width, h / height])


def pole_caption(name, cfg, axes):
    """Fabric type, its Watson concentration, and the realised strength.

    a1 is the largest eigenvalue of the orientation tensor, which is the
    number a thin section actually reports: 1/3 is random and 1 is a
    perfect single maximum.
    """
    a1 = sp.orientation_tensor(axes)[0][0]
    return (name, "$\\kappa = %+g$" % cfg["concentration"],
            "$a_1$ = %.2f" % a1)


def report(girdle, single, diameters, nominal, wavelength_mm, n_faces):
    """The numbers of Section 3.2, recomputed from the specimen drawn."""
    print("grains realised       : %d" % diameters.size)
    print("equivalent diameter   : mean %.2f mm, sd %.2f mm, CV %.3f"
          % (diameters.mean(), diameters.std(),
             diameters.std() / diameters.mean()))
    print("                        extremes %.2f and %.2f mm"
          % (diameters.min(), diameters.max()))
    print("nominal diameter      : %.2f mm (disc volume shared equally)"
          % nominal)
    print("wavelength            : %.3f mm, D/lambda = %.2f"
          % (wavelength_mm, nominal / wavelength_mm))
    for name, built in (("girdle", girdle), ("single maximum", single)):
        w, _ = sp.orientation_tensor(built["axes"])
        print("%-22s: eigenvalues %.3f %.3f %.3f"
              % (name, w[0], w[1], w[2]))
    print("cut-away faces drawn  : %d" % n_faces)


def main():
    girdle_cfg = load_config(GIRDLE_RUN)
    single_cfg = load_config(SINGLEMAX_RUN)
    girdle = build(girdle_cfg)
    single = build(single_cfg)

    diameters = grain_diameters(girdle)
    nominal = nominal_diameter(girdle_cfg)
    wavelength_mm = C_REF / (girdle_cfg["f0_mhz"] * 1e6) * 1e3

    cmap, norm = fs.cmap(CMAP_CYCLIC), Normalize(0.0, 180.0)

    fig = fs.canvas("microstructure")
    ax_cut = axes_mm(fig, BOXES["cutaway"])
    ax_slice = axes_mm(fig, BOXES["slice"])
    ax_size = axes_mm(fig, BOXES["sizes"])
    ax_girdle = axes_mm(fig, BOXES["pole_girdle"])
    ax_single = axes_mm(fig, BOXES["pole_single"])

    n_faces = draw_cutaway(ax_cut, girdle, girdle_cfg, cmap, norm)
    draw_slice(ax_slice, girdle, girdle_cfg, cmap, norm)
    draw_sizes(ax_size, diameters, nominal)
    draw_pole(ax_girdle, girdle["axes"], cmap, norm,
              pole_caption("girdle", girdle_cfg, girdle["axes"]),
              girdle_axis=np.array(girdle_cfg["fabric_axis"]))
    draw_pole(ax_single, single["axes"], cmap, norm,
              pole_caption("single maximum", single_cfg, single["axes"]),
              single_axis=np.array(single_cfg["fabric_axis"]))

    # aspect 23 rather than the default 20 so the bar fills the width it
    # was given instead of stopping short of the drawing above it.
    bar = fs.colourbar(ScalarMappable(norm=norm, cmap=cmap), ax_cut,
                       label="c-axis angle about x (deg)",
                       location="bottom", aspect=23)
    bar.set_ticks([0, 45, 90, 135, 180])

    fs.condition(ax_slice, "beam plane, z = 0", x=0.17)
    for ax, letter in ((ax_cut, "a"), (ax_slice, "b"), (ax_size, "c"),
                       (ax_girdle, "d"), (ax_single, "e")):
        fs.panel(ax, letter, inside=True)

    report(girdle, single, diameters, nominal, wavelength_mm, n_faces)
    fs.save(fig, "microstructure",
            expect=("grain diameter (mm)", "number of grains",
                    "c-axis angle about x (deg)", "beam plane"))


if __name__ == "__main__":
    main()

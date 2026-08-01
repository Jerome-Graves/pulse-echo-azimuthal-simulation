"""3D renders of the specimen, matching the OpenUSCT Studio look.

`polycrystal_3d` is the same picture as ringfwi.render3d.polycrystal_figure
(one marching-cubes mesh per grain, coloured by a per-grain scalar through a
matplotlib colormap) but each grain is meshed inside its own bounding box
instead of over the whole volume. openUSCT does `labels == k` across the full
array once per grain, which is O(n_grains * n_cells); at 400+ grains on a
200^3 grid that is ~1e10 element tests. Cropping first makes it tractable.
"""
import numpy as np

CMAP = "twilight"


def grain_colors(values, vmin=0.0, vmax=90.0, cmap_name=CMAP):
    """Map per-grain scalars to plotly rgb strings via a matplotlib cmap."""
    from matplotlib import colormaps
    cmap = colormaps[cmap_name]
    t = (np.asarray(values, float) - vmin) / (vmax - vmin + 1e-30)
    return {k: "rgb({},{},{})".format(*(int(255 * c) for c in cmap(float(tv))[:3]))
            for k, tv in enumerate(np.clip(t, 0.0, 1.0))}


def colatitude_deg(axes):
    """Angle of each c-axis from vertical, folded to 0-90 deg (axes are +/- equivalent)."""
    return np.degrees(np.arccos(np.clip(np.abs(np.asarray(axes)[:, 2]), 0.0, 1.0)))


def azimuth_deg(axes):
    """In-plane c-axis azimuth, folded to 0-180 deg."""
    a = np.asarray(axes)
    return np.degrees(np.arctan2(a[:, 1], a[:, 0])) % 180.0


def polycrystal_3d(labels, grain_values, h, title="", vmin=0.0, vmax=90.0,
                   cmap_name=CMAP, opacity=0.55, max_grains=None, step=1,
                   value_label="deg", height=560):
    """Interactive 3D polycrystal: one coloured mesh per grain."""
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes
    from scipy import ndimage

    lab = np.asarray(labels)[::step, ::step, ::step]
    hs = float(h) * step
    colors = grain_colors(grain_values, vmin, vmax, cmap_name)

    counts = np.bincount(lab[lab >= 0].ravel(),
                         minlength=int(len(np.asarray(grain_values))))
    ids = np.flatnonzero(counts > 0)
    n_total = len(ids)
    if max_grains and n_total > max_grains:           # keep the largest grains
        ids = ids[np.argsort(counts[ids])[::-1][:max_grains]]

    # find_objects indexes labels 1..N, so shift background (-1) to 0 first
    boxes = ndimage.find_objects(lab + 1)

    fig = go.Figure()
    shown = 0
    for k in ids:
        sl = boxes[int(k)]
        if sl is None:
            continue
        mask = np.pad((lab[sl] == k).astype(np.float32), 1)   # close border surfaces
        if mask.sum() < 2:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, 0.5, spacing=(hs, hs, hs))
        except (ValueError, RuntimeError):
            continue
        verts = verts - hs                                    # undo the pad shift
        verts += np.array([s.start for s in sl]) * hs         # back to global coords
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0] * 1e3, y=verts[:, 1] * 1e3, z=verts[:, 2] * 1e3,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=colors[int(k)], opacity=opacity, flatshading=True,
            name=f"grain {k}", showlegend=False,
            hovertext=f"grain {k}: {float(grain_values[k]):.1f} {value_label}"))
        shown += 1

    sub = f"{shown} of {n_total} grains" + (f", every {step}th cell" if step > 1 else "")
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{sub}</sub>"),
        scene_aspectmode="data", template="plotly_dark",
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
        margin=dict(l=0, r=0, t=50, b=0), height=height)
    return fig


def caxis_field_3d(labels, axes, h, values=None, vmin=0.0, vmax=90.0,
                   cmap_name=CMAP, title="c-axis orientation field",
                   length_frac=0.8, height=560, max_grains=None):
    """C-axes drawn as line segments at grain centroids.

    Each c-axis is a double-ended bar because +c and -c are physically
    identical: a cone or arrow would imply a polarity the crystal does not have.
    """
    import plotly.graph_objects as go
    from scipy import ndimage
    from matplotlib import colormaps

    lab = np.asarray(labels)
    axes = np.asarray(axes, float)
    if values is None:
        values = colatitude_deg(axes)

    counts = np.bincount(lab[lab >= 0].ravel(), minlength=len(axes))
    ids = np.flatnonzero(counts > 0)
    if max_grains and len(ids) > max_grains:
        ids = ids[np.argsort(counts[ids])[::-1][:max_grains]]

    cen = np.array(ndimage.center_of_mass(np.ones_like(lab, dtype=np.uint8),
                                          labels=lab + 1,
                                          index=(ids + 1).tolist())) * h * 1e3
    # bar length scales with grain size so big grains read as more important
    d_eq = 2.0 * (3.0 * counts[ids] * h ** 3 / (4 * np.pi)) ** (1 / 3) * 1e3
    L = length_frac * d_eq

    cmap = colormaps[cmap_name]
    fig = go.Figure()
    xs, ys, zs, cs = [], [], [], []
    for n, k in enumerate(ids):
        v = axes[k] * (L[n] / 2.0)
        c0, c1 = cen[n] - v, cen[n] + v
        xs += [c0[0], c1[0], None]
        ys += [c0[1], c1[1], None]
        zs += [c0[2], c1[2], None]
        t = np.clip((values[k] - vmin) / (vmax - vmin + 1e-30), 0, 1)
        rgb = "rgb({},{},{})".format(*(int(255 * c) for c in cmap(float(t))[:3]))
        cs += [rgb, rgb, rgb]

    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color=cs, width=5), showlegend=False,
                               hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=cen[:, 0], y=cen[:, 1], z=cen[:, 2], mode="markers",
        marker=dict(size=2.5, color=values[ids], colorscale="twilight",
                    cmin=vmin, cmax=vmax,
                    colorbar=dict(title="colat (deg)", thickness=12)),
        showlegend=False,
        text=[f"grain {k}: {values[k]:.1f} deg" for k in ids], hoverinfo="text"))
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{len(ids)} c-axes, bar length "
                        f"proportional to grain size</sub>"),
        scene_aspectmode="data", template="plotly_dark",
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
        margin=dict(l=0, r=0, t=50, b=0), height=height)
    return fig


def pole_figure(axes, title="c-axis pole figure", height=520):
    """Equal-area lower-hemisphere projection, the standard glaciology view."""
    import plotly.graph_objects as go
    a = np.asarray(axes, float)
    a = a * np.sign(a[:, 2:3] + 1e-12)                 # fold to upper hemisphere
    r = np.sqrt(2.0 * (1.0 - np.abs(a[:, 2])))         # Schmidt (equal-area)
    th = np.arctan2(a[:, 1], a[:, 0])
    colat = colatitude_deg(a)
    fig = go.Figure(go.Scatterpolar(
        r=r, theta=np.degrees(th), mode="markers",
        marker=dict(size=6, color=colat, colorscale="twilight", cmin=0, cmax=90,
                    colorbar=dict(title="colat (deg)", thickness=12), opacity=0.85),
        hovertext=[f"{c:.1f} deg" for c in colat], hoverinfo="text"))
    fig.update_layout(
        title=title, template="plotly_dark", height=height,
        polar=dict(radialaxis=dict(range=[0, np.sqrt(2.0)], showticklabels=False),
                   angularaxis=dict(direction="counterclockwise")),
        margin=dict(l=30, r=30, t=50, b=30))
    return fig

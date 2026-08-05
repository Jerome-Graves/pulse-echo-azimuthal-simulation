"""Ice disk specimen: grain geometry + c-axis fabric.

Three controls, deliberately INDEPENDENT of one another, because they drive
different observables:

  size_cv          grain size spread          -> scattering length scale
  concentration    how tightly c-axes cluster -> BULK 2-phi anisotropy
  spatial_corr     how alike NEIGHBOURS are   -> LOCAL disorientation, hence
                                                 scattering strength (alpha)

You can therefore have a strong bulk fabric with high local disorder, or a
weak bulk fabric with locally similar neighbours. Those look identical in a
bulk measurement and completely different in backscatter, which is the whole
point of the experiment.
"""
from dataclasses import dataclass
import os
import sys
import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from ringfwi import anisotropy as an   # noqa: E402

# single-crystal ice Ih, c-axis along z  [Gammon et al. 1983, via openUSCT]
ICE_TI = dict(C11=an.ICE_C11, C33=an.ICE_C33, C44=an.ICE_C44,
              C12=an.ICE_C12, C13=an.ICE_C13, rho=an.ICE_RHO)


def ice_voigt6():
    """6x6 Voigt stiffness of single-crystal ice with the c-axis along +z."""
    C11, C33, C44 = ICE_TI["C11"], ICE_TI["C33"], ICE_TI["C44"]
    C12, C13 = ICE_TI["C12"], ICE_TI["C13"]
    C66 = 0.5 * (C11 - C12)
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
    C[3, 3] = C[4, 4] = C44
    C[5, 5] = C66
    return C


# ─────────────────────────── fabric sampling ─────────────────────────
def sample_watson(n, axis, kappa, rng):
    """Watson distribution on the sphere: axial, clustered about `axis`.

    kappa = 0   -> uniform (random fabric)
    kappa > 0   -> single maximum about axis (girdle if kappa < 0)
    """
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    if abs(kappa) < 1e-9:
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    # sample cos(theta) by rejection against exp(kappa * t^2)
    t = np.empty(n)
    got = 0
    while got < n:
        u = rng.uniform(-1, 1, n)
        keep = rng.uniform(0, 1, n) < np.exp(kappa * (u ** 2 - (1 if kappa > 0 else 0)))
        k = u[keep][: n - got]
        t[got:got + k.size] = k
        got += k.size
    phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(np.clip(1 - t ** 2, 0, None))
    local = np.stack([s * np.cos(phi), s * np.sin(phi), t], axis=1)
    R = an._rotation_z_to(axis)
    return local @ R.T


def renormalise_fabric(axes, target_a1, n_iter=60):
    """Restore a target bulk eigenvalue after neighbour smoothing.

    Smoothing each grain toward its neighbours lowers LOCAL disorientation,
    which is what we want, but it also drags every axis toward the mean and so
    raises the BULK eigenvalue, which we do not want: the two controls stop
    being independent and you cannot simulate a weak overall fabric that
    happens to be locally aligned.

    Fix: rescale each axis's angle from the dominant eigenvector until the
    eigenvalue is back on target. Azimuth about that axis is untouched, so the
    spatial pattern the smoothing created survives intact.
    """
    w, V = orientation_tensor(axes)
    if abs(w[0] - target_a1) < 1e-4:
        return axes
    e1 = V[:, 0]
    sgn = np.sign(axes @ e1)
    sgn[sgn == 0] = 1.0
    a = axes * sgn[:, None]                      # fold onto the e1 hemisphere
    cos = np.clip(a @ e1, -1.0, 1.0)
    psi = np.arccos(cos)
    perp = a - cos[:, None] * e1[None, :]
    pn = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = np.divide(perp, np.maximum(pn, 1e-12), out=np.zeros_like(perp),
                     where=pn > 1e-12)

    def rebuild(k):
        p2 = np.minimum(psi * k, np.pi / 2)
        return np.cos(p2)[:, None] * e1[None, :] + np.sin(p2)[:, None] * perp

    lo, hi = 0.0, 6.0
    for _ in range(n_iter):
        k = 0.5 * (lo + hi)
        if orientation_tensor(rebuild(k))[0][0] > target_a1:
            lo = k                               # still too concentrated
        else:
            hi = k
    out = rebuild(0.5 * (lo + hi))
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def watson_eigenvalue(kappa, n=4001):
    """Largest orientation-tensor eigenvalue a1 of a Watson distribution.

    a1 = <cos^2 theta> = int_0^1 x^2 e^(k x^2) dx / int_0^1 e^(k x^2) dx
    Evaluated as e^(k(x^2-1)) so large kappa cannot overflow.
    a1 = 1/3 at kappa=0 (random) and tends to 1 as kappa -> infinity.
    """
    x = np.linspace(0.0, 1.0, n)
    wgt = np.exp(kappa * (x ** 2 - 1.0))
    return float(np.trapezoid(x ** 2 * wgt, x) / np.trapezoid(wgt, x))


def kappa_for_eigenvalue(a1, lo=0.0, hi=400.0, tol=1e-6):
    """Invert `watson_eigenvalue`: the kappa giving a target eigenvalue a1.

    a1 is the number thin-section glaciology actually reports, so this lets
    a fabric be specified the same way a real specimen is measured.
    """
    a1 = float(np.clip(a1, 1.0 / 3.0 + 1e-9, 1.0 - 1e-6))
    if a1 <= 1.0 / 3.0 + 1e-9:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if watson_eigenvalue(mid) < a1:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def orientation_tensor(axes):
    """Second-order orientation tensor and its eigenvalues (standard COF descriptor).

    Eigenvalues sum to 1. (1/3,1/3,1/3)=random; (1,0,0)=perfect single maximum.
    """
    a = np.asarray(axes, float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    T = (a[:, :, None] * a[:, None, :]).mean(axis=0)
    w, V = np.linalg.eigh(T)
    return np.sort(w)[::-1], V[:, np.argsort(w)[::-1]]


def disorientation_deg(a, b):
    """Angle between two c-axes, folded to [0,90] because c-axes are axial."""
    d = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(np.arccos(np.clip(d, 0, 1)))


# ───────────────────────────── the specimen ──────────────────────────
@dataclass
class DiskSpecimen:
    diameter_m: float = 100e-3
    thickness_m: float = 35e-3
    n_grains: int = 400

    size_cv: float = 0.35        # grain-size coefficient of variation
    concentration: float = 4.0   # Watson kappa -> bulk fabric strength
    spatial_corr: float = 0.0    # 0 = neighbours independent (max local disorder)
                                 # 1 = neighbours identical (no local disorder)
    fabric_axis: tuple = (0.0, 0.0, 1.0)
    seed: int = 7

    def build(self, h):
        """Voronoi grains on a grid of spacing h. Returns a dict."""
        rng = np.random.default_rng(self.seed)
        R = self.diameter_m / 2
        nx = int(np.ceil(self.diameter_m / h))
        nz = int(np.ceil(self.thickness_m / h))

        # A Laguerre diagram with a broad weight spread legitimately produces
        # EMPTY cells. Real ice has no zero-volume grains, so we over-seed,
        # let the weighting eliminate some, then drop empties and relabel.
        # The outer loop tunes the seed count to land on n_grains realised.
        mc = self._mc_points(R, self.thickness_m, 400000, rng)
        base = (R / self.n_grains ** (1 / 3)) ** 2
        sigma = np.sqrt(np.log(1 + self.size_cv ** 2))
        n_seed = self.n_grains

        for _outer in range(4):
            seeds = self._seed_points(R, self.thickness_m, n_seed, rng)
            radii = rng.lognormal(-0.5 * sigma ** 2, sigma, n_seed)
            wshape = base * (radii ** 2 - 1.0)

            # bisect the weight scale on the realised size CV
            lo, hi, scale = 0.0, 8.0, 0.0
            for _ in range(20):
                trial = 0.5 * (lo + hi)
                cnt = np.bincount(self._assign(mc, seeds, trial * wshape),
                                  minlength=n_seed).astype(float)
                live = cnt[cnt > 0]
                cv = (live ** (1 / 3)).std() / (live ** (1 / 3)).mean()
                if cv < self.size_cv:
                    lo = trial
                else:
                    hi = trial
                scale = trial
            n_live = int((np.bincount(self._assign(mc, seeds, scale * wshape),
                                      minlength=n_seed) > 0).sum())
            if n_live >= self.n_grains or _outer == 3:
                break
            n_seed = int(np.ceil(n_seed * self.n_grains / max(n_live, 1)))

        weights = scale * wshape
        self._weight_scale, self._n_seed = scale, n_seed

        # --- label the grid (KD-tree shortlist, then exact power metric).
        # At 5 MHz resolution the all-points-at-once CPU path materialises a
        # (131M, 12, 3) float64 temporary ~ 37 GB and the machine SWAPS: the
        # "6-8 minute build" was mostly paging. The GPU path labels the grid
        # plane by plane with an exact all-seeds power argmin in float64
        # (a handful of GB/s of trivial arithmetic), and falls back to the
        # CPU path when CUDA is unavailable.
        x = (np.arange(nx) - (nx - 1) / 2) * h
        z = (np.arange(nz) - (nz - 1) / 2) * h
        raw = self._label_grid_gpu(x, z, R, seeds, weights)
        if raw is None:
            X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
            inside = (X ** 2 + Y ** 2) <= R ** 2
            raw = np.full(X.shape, -1, np.int32)
            P = np.stack([X[inside], Y[inside], Z[inside]], axis=1)
            raw[inside] = self._assign(P, seeds, weights).astype(np.int32)

        # drop empty cells, relabel contiguously, keep the surviving seeds
        inside = raw >= 0
        keep = np.unique(raw[inside])
        remap = np.full(len(seeds), -1, np.int32)
        remap[keep] = np.arange(len(keep))
        labels = np.full(raw.shape, -1, np.int16)
        labels[inside] = remap[raw[inside]].astype(np.int16)
        seeds = seeds[keep]
        n_real = len(keep)

        # --- c-axes: bulk fabric from the Watson concentration
        axes = sample_watson(n_real, self.fabric_axis, self.concentration, rng)

        # --- spatial correlation: blend each grain toward its neighbours'
        #     mean, which lowers LOCAL disorientation without touching the
        #     bulk fabric much.
        adj = self._adjacency(labels)
        if self.spatial_corr > 0:
            target_a1 = orientation_tensor(axes)[0][0]   # before smoothing
            for _ in range(3):
                nb = np.zeros_like(axes)
                for g, ns in adj.items():
                    if ns:
                        v = axes[list(ns)]
                        v = v * np.sign(v @ axes[g])[:, None]   # axial: fold
                        nb[g] = v.mean(axis=0)
                    else:
                        nb[g] = axes[g]
                axes = (1 - self.spatial_corr) * axes + self.spatial_corr * nb
                axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            # put the bulk fabric back where the user asked for it, so that
            # spatial_corr changes LOCAL disorder only
            axes = renormalise_fabric(axes, target_a1)

        # weights are returned alongside the surviving seeds so callers
        # can EVALUATE the tessellation at arbitrary points instead of
        # resampling this label volume - see rotation_test.rotated_grid,
        # whose nearest-neighbour resample was rasterising twice.
        return dict(labels=labels, axes=axes, seeds=seeds,
                    weights=weights[keep], adjacency=adj,
                    h=h, inside=inside, shape=labels.shape)

    @staticmethod
    def _seed_points(R, thickness, n, rng):
        pts = []
        while len(pts) < n:
            p = rng.uniform(-1, 1, (n * 3, 3))
            pts.extend(p[(p[:, 0] ** 2 + p[:, 1] ** 2) <= 1.0].tolist())
        s = np.array(pts[:n]); s[:, :2] *= R; s[:, 2] *= thickness / 2
        return s

    @staticmethod
    def _mc_points(R, thickness, n, rng):
        p = rng.uniform(-1, 1, (int(n * 1.4), 3))
        p = p[(p[:, 0] ** 2 + p[:, 1] ** 2) <= 1.0][:n]
        p[:, :2] *= R
        p[:, 2] *= thickness / 2
        return p

    @staticmethod
    def _label_grid_gpu(x, z, R, seeds, weights):
        """Label the full grid on the GPU, one z-plane at a time.

        Exact power-diagram argmin over ALL seeds in float64 (the CPU path
        shortlists the 12 Euclidean-nearest first, which is exact only for
        modest weight spreads; all-seeds is exact always). MEASURED against
        the CPU path on the 2 MHz protocol grid: 3 differing cells of
        10.7M, all cases where the shortlist missed the true power minimum,
        i.e. the GPU labels are the more exact ones. Three boundary cells
        are far below every quantified noise floor in this project, so the
        GPU labels are the standard going forward; traces saved before
        2026-07-25 used the CPU labels. Returns None when CUDA is
        unavailable so the caller can fall back.
        """
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                return None
        except Exception:
            return None
        nx, nz = len(x), len(z)
        xd = cp.asarray(x)
        X, Y = cp.meshgrid(xd, xd, indexing="ij")
        inside = (X ** 2 + Y ** 2) <= R ** 2
        Pxy = cp.stack([X[inside], Y[inside]], axis=1)      # (npts, 2) f64
        sd = cp.asarray(seeds)                              # (ns, 3)
        wd = cp.asarray(weights)
        # xy part of the power distance is z-independent: compute once
        d_xy = (((Pxy[:, None, :] - sd[None, :, :2]) ** 2).sum(-1)
                - wd[None, :])                              # (npts, ns)
        raw = np.full((nx, nx, nz), -1, np.int32)
        ins_h = cp.asnumpy(inside)
        for iz in range(nz):
            d2 = d_xy + (z[iz] - sd[:, 2])[None, :] ** 2
            lab = cp.asnumpy(d2.argmin(axis=1).astype(cp.int32))
            plane = np.full((nx, nx), -1, np.int32)
            plane[ins_h] = lab
            raw[:, :, iz] = plane
        return raw

    @staticmethod
    def _assign(P, seeds, weights, k=12):
        """Laguerre (power-diagram) assignment, KD-tree shortlisted.

        Power distance is |x-s|^2 - w. Taking the k Euclidean-nearest seeds
        and minimising the power distance among them is exact whenever the
        weight spread is modest, which the calibration enforces.
        """
        from scipy.spatial import cKDTree
        k = min(k, len(seeds))
        _, idx = cKDTree(seeds).query(P, k=k, workers=-1)
        idx = np.atleast_2d(idx.T).T
        d2 = ((P[:, None, :] - seeds[idx]) ** 2).sum(-1) - weights[idx]
        return idx[np.arange(len(P)), d2.argmin(1)].astype(np.int16)

    @staticmethod
    def _adjacency(labels):
        adj = {}
        for ax in range(3):
            a = np.take(labels, np.arange(labels.shape[ax] - 1), axis=ax)
            b = np.take(labels, np.arange(1, labels.shape[ax]), axis=ax)
            m = (a != b) & (a >= 0) & (b >= 0)
            for u, v in np.unique(np.stack([a[m], b[m]], 1), axis=0):
                adj.setdefault(int(u), set()).add(int(v))
                adj.setdefault(int(v), set()).add(int(u))
        return adj

    # ---------------------------------------------------------------- report
    def diagnose(self, built):
        labels, axes, adj = built["labels"], built["axes"], built["adjacency"]
        h = built["h"]
        cnt = np.bincount(labels[labels >= 0].ravel(), minlength=len(built['axes']))
        vol = cnt * h ** 3
        d_eq = 2 * (3 * vol / (4 * np.pi)) ** (1 / 3) * 1e3          # mm
        d_eq = d_eq[cnt > 0]
        pairs = [(u, v) for u, ns in adj.items() for v in ns if u < v]
        mis = disorientation_deg(axes[[p[0] for p in pairs]],
                                 axes[[p[1] for p in pairs]]) if pairs else np.array([0.])
        w, V = orientation_tensor(axes)
        L = []
        L.append(f"grains realised      : {int((cnt>0).sum())} of {self.n_grains} target "
                 f"(seeded {getattr(self,'_n_seed',self.n_grains)})")
        L.append(f"grain size           : mean {d_eq.mean():.2f} mm, "
                 f"CV {d_eq.std()/d_eq.mean():.2f} (target {self.size_cv:.2f}), "
                 f"range {d_eq.min():.1f}-{d_eq.max():.1f} mm")
        L.append(f"neighbour pairs      : {len(pairs)}")
        L.append(f"DISORIENTATION       : mean {mis.mean():.1f} deg, "
                 f"median {np.median(mis):.1f}, 90th {np.percentile(mis,90):.1f}")
        L.append(f"fabric eigenvalues   : {w[0]:.3f} {w[1]:.3f} {w[2]:.3f}  "
                 f"(1/3 each = random)")
        L.append(f"predominant axis     : [{V[0,0]:+.2f} {V[1,0]:+.2f} {V[2,0]:+.2f}]")
        # what the disorientation means acoustically
        vmax, vmin = an.ice_qp_vs_caxis(0.0), an.ice_qp_vs_caxis(np.radians(50.4))
        L.append(f"qP range             : {vmin:.0f}-{vmax:.0f} m/s "
                 f"({100*(vmax-vmin)/vmax:.1f}% anisotropy)")
        return "\n".join(L)


if __name__ == "__main__":
    import time
    sp = DiskSpecimen()
    h = 3850.0 / 5e6 / 6 * 4      # coarse grid just for a quick geometry check
    t0 = time.time()
    b = sp.build(h)
    print(f"grid {b['shape']}  h={h*1e3:.3f} mm   ({time.time()-t0:.1f} s)\n")
    print(sp.diagnose(b))

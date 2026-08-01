"""SKEPTIC step 5: an INDEPENDENT predictor.

Their predictor marches rays through the RASTERISED int16 label grid.  Mine
never touches the label grid: it labels each sample point analytically from the
Laguerre power distance |x-s_k|^2 - w_k, finds the boundary crossings, and takes
the facet normal from the seed pair.  Different code path, no rasterisation,
constant velocity (so no forward._VQP either).  If their r is real this must
reproduce it; if it is an artefact of the grid it must not.
"""
import os
import sys

import numpy as np
from scipy.special import j1

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from specimen import DiskSpecimen                      # noqa: E402
from sk1 import load, centre_harm, all_shifts, EPS      # noqa: E402

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
LAM = C_REF / F0
K = 2 * np.pi / LAM
A_EL = 6.35e-3 / 2
DT = 20e-9
T0 = 1.2 / F0
TG = np.arange(20e-6, 42e-6, DT)
EDGES = np.concatenate([TG - DT / 2, [TG[-1] + DT / 2]])
GATE = (TG >= 24e-6) & (TG < 36e-6)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skcache")
os.makedirs(CACHE, exist_ok=True)


def kernel():
    """power envelope of the band-passed ricker, on the 20 ns grid"""
    from scipy.signal import butter, sosfiltfilt, hilbert
    fs = 1e9
    tc = np.arange(int(6e-6 * fs)) / fs
    a = (np.pi * F0 * (tc - 3e-6)) ** 2
    w = (1 - 2 * a) * np.exp(-a)
    sos = butter(4, [0.8e6 / (fs / 2), 3.0e6 / (fs / 2)], "band", output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, w))) ** 2
    n = int(round(2e-6 / DT))
    k = np.interp(np.arange(-n, n + 1) * DT + 3e-6, tc, e)
    return k / k.sum()


KERN = kernel()


def piston(theta):
    x = K * A_EL * np.sin(theta)
    b = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x))
                 / np.where(x < 1e-9, 1.0, x))
    return b ** 4


def facet_dir(sinth, dg):
    x = K * dg * sinth
    b = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x))
                 / np.where(x < 1e-9, 1.0, x))
    return b ** 2


def analytic_pred(kap, axis, seed, az_list, dg=17.4e-3, n_ray=200,
                  tmax_deg=18.0, ds=0.15e-3, s_max=0.088, hard=False,
                  tag=""):
    key = (f"a_{kap}_{seed}_{len(az_list)}_{dg*1e3:.1f}_{n_ray}_{tmax_deg}"
           f"_{ds*1e3:.2f}_{hard}{tag}.npy")
    fp = os.path.join(CACHE, key)
    if os.path.exists(fp):
        return np.load(fp)
    b = DiskSpecimen(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                     size_cv=0.35, concentration=kap, spatial_corr=0.0,
                     fabric_axis=tuple(axis), seed=seed).build(3850 / 2e6 / 8)
    sd = np.asarray(b["seeds"], float)
    wt = np.asarray(b["weights"], float)
    rng = np.random.default_rng(7)
    i = np.arange(n_ray) + 0.5
    rr = np.tan(np.radians(tmax_deg)) * np.sqrt(i / n_ray)
    ph = i * np.pi * (3 - np.sqrt(5.0)) + rng.uniform(0, 2 * np.pi)
    wbeam = piston(np.arctan(rr))
    s = (np.arange(int(s_max / ds)) + 0.5) * ds
    rows = []
    for a in az_list:
        th = np.radians(a)
        n = np.array([np.cos(th), np.sin(th), 0.0])
        t1 = np.array([-np.sin(th), np.cos(th), 0.0])
        t2 = np.array([0.0, 0.0, 1.0])
        D = (-n + (rr * np.cos(ph))[:, None] * t1
             + (rr * np.sin(ph))[:, None] * t2)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        P = (DIA / 2) * n + s[None, :, None] * D[:, None, :]   # (nr, ns, 3)
        flat = P.reshape(-1, 3)
        # analytic Laguerre labelling, in chunks
        lab = np.empty(flat.shape[0], np.int32)
        inside = ((flat[:, 0] ** 2 + flat[:, 1] ** 2) < (DIA / 2) ** 2) & \
                 (np.abs(flat[:, 2]) < 0.035 / 2)
        CH = 200000
        for c0 in range(0, flat.shape[0], CH):
            q = flat[c0:c0 + CH]
            d2 = ((q[:, None, :] - sd[None, :, :]) ** 2).sum(-1) - wt[None, :]
            lab[c0:c0 + CH] = np.argmin(d2, 1)
        lab = np.where(inside, lab, -1).reshape(P.shape[:2])
        A, B = lab[:, :-1], lab[:, 1:]
        cr = (A != B) & (A >= 0) & (B >= 0)
        ri, si = np.nonzero(cr)
        if ri.size == 0:
            rows.append(np.zeros(len(TG)))
            continue
        ia, ib = A[ri, si], B[ri, si]
        nm = sd[ib] - sd[ia]
        nm /= np.linalg.norm(nm, axis=1, keepdims=True)
        ct = np.abs(np.einsum("ij,ij->i", nm, D[ri]))
        st = np.sqrt(np.clip(1 - ct ** 2, 0, 1))
        if hard:                      # hard cone instead of the Airy lobe
            Dd = (st < np.sin(np.radians(3.9))).astype(float)
        else:
            Dd = facet_dir(st, dg)
        t = 2 * 0.5 * (s[si] + s[si + 1]) / C_REF + T0
        w = Dd * wbeam[ri]
        rows.append(np.histogram(t, EDGES, weights=w)[0])
    out = np.apply_along_axis(lambda r: np.convolve(r, KERN, "same"), 1,
                              np.array(rows))[:, GATE]
    np.save(fp, out)
    return out


def stats(mc, P):
    pc = centre_harm(10 * np.log10(P + EPS))
    R = all_shifts(mc, pc)
    r0 = float(R[0, 0])
    return r0, float((np.abs(R) >= abs(r0)).mean()), \
        float((np.abs(R[:, 0]) >= abs(r0)).mean())


SW = {"girdle_perp_ppw8": (-8.0, (1.0, 0.0, 0.0), 11, "gp8"),
      "singlemax_ppw8": (3.93, (0.866, 0.5, 0.0), 11, "sm8")}

if __name__ == "__main__":
    for sw, (kap, axis, seed, tag) in SW.items():
        az, tt, M, _ = load(sw, tag)
        mc = centre_harm(10 * np.log10(M + EPS))
        print(f"\n{'='*78}\n{sw}  (their geom r = "
              f"{'0.156' if seed==11 and kap<0 else '0.169'})")
        for lbl, kw in (("analytic, correct seed", dict(seed=seed)),
                        ("analytic, WRONG seed 23", dict(seed=23)),
                        ("analytic, WRONG seed 101", dict(seed=101)),
                        ("analytic, HARD 3.9deg cone", dict(seed=seed,
                                                            hard=True)),
                        ("analytic, no directivity", dict(seed=seed,
                                                          dg=1e-6))):
            P = analytic_pred(kap, axis, az_list=az, **kw)
            r, p2, paz = stats(mc, P)
            print(f"   {lbl:<28} r={r:+.4f}  p2d={p2:.5f}  p_az={paz:.4f}",
                  flush=True)

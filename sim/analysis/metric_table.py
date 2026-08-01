"""Variant x sweep metric table + spectrum-preserving surrogate null.

Metrics per (variant, sweep) on the RAW per-azimuth dB observable:
  std_db   azimuthal standard deviation
  A2_db    k=2 harmonic amplitude  (x ~ m0 + A2 cos(2(th - phi2)))
  phi2     k=2 phase in deg, mod 180
  dax      |phi2 - known sample axis| folded to [0,90]
  p_null   p(A2 >= observed) under a phase-randomised surrogate whose
           smooth red background is fitted to harmonics k=1..20
           EXCLUDING k=2 and k=4 (the fabric lines).

Permutation nulls are invalid here: glint has a 3-7 deg azimuthal
correlation length, so shuffling azimuths destroys the red background
and makes every k=2 line look significant.
"""
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
AXIS = {"rigid_seed11": 28.8, "oos_seed23": 104.3, "iso_gcal": None}
KMAX_BG = 20
K_EXCL = (2, 4)
NSURR = 20000
RNG = np.random.default_rng(20260731)


def harmonics(x, th_deg, kmax):
    """A_k, phi_k for x ~ m0 + sum A_k cos(k*th - k*phi_k)."""
    th = np.radians(th_deg)
    n = len(x)
    xc = x - x.mean()
    A, ph = {}, {}
    for k in range(1, kmax + 1):
        X = (2.0 / n) * np.sum(xc * np.exp(-1j * k * th))
        A[k] = abs(X)
        ph[k] = np.degrees(np.angle(X)) / k % (360.0 / k)
    return A, ph


def bg_model(A, kmax=KMAX_BG, excl=K_EXCL):
    """Smooth red background: power-law fit of P_k = A_k^2 vs k."""
    ks = np.array([k for k in range(1, kmax + 1) if k not in excl])
    P = np.array([A[k] ** 2 for k in ks])
    c = np.polyfit(np.log(ks), np.log(np.maximum(P, 1e-300)), 1)
    return lambda k: float(np.exp(np.polyval(c, np.log(k)))), c


def surrogate_p(A2_obs, P2_bg, nsurr=NSURR):
    """Phase-randomised surrogate: the k=2 Fourier line is redrawn from
    the background law (complex Gaussian -> Rayleigh amplitude with
    E|X|^2 = P2_bg) with a random phase; everything else is preserved."""
    s = np.sqrt(P2_bg / 2.0)
    z = RNG.normal(0, s, nsurr) + 1j * RNG.normal(0, s, nsurr)
    a = np.abs(z)
    return float((a >= A2_obs).mean()), float(np.median(a))


def fold90(d):
    d = abs(d) % 180.0
    return min(d, 180.0 - d)


def load(sweep):
    with np.load(os.path.join(OUT, f"obs_variants_{sweep}.npz")) as z:
        return {k: np.array(z[k]) for k in z.files}


def analyse(sweeps=("rigid_seed11", "oos_seed23", "iso_gcal")):
    D = {s: load(s) for s in sweeps}
    names = [k for k in sorted(D[sweeps[0]]) if k != "az"]
    res = {}
    for s in sweeps:
        az = D[s]["az"]
        for v in names:
            x = 20 * np.log10(D[s][v])
            A, ph = harmonics(x, az, KMAX_BG)
            f_bg, c = bg_model(A)
            p, med = surrogate_p(A[2], f_bg(2))
            ax = AXIS[s]
            res[(v, s)] = dict(
                std=float(x.std()), A2=A[2], phi2=ph[2],
                dax=(fold90(ph[2] - ax) if ax is not None else np.nan),
                p=p, bg2=np.sqrt(f_bg(2)), slope=c[0],
                A4=A[4], phi4=ph[4],
                corr_fd1=float(np.corrcoef(
                    x, 20 * np.log10(D[s]["fd1"]))[0, 1]),
                mean=float(x.mean()))
    return names, sweeps, res


def main():
    names, sweeps, res = analyse()
    order = ["fd1", "gate_only", "broken_rect3",
             "grect2", "ghann2", "ghann2_geo", "ghann2_med",
             "grect3", "ghann3", "ghann3_geo", "ghann3_med",
             "gbutt3", "gbutt3_geo", "butt3",
             "grect5", "ghann5", "ghann5_geo", "ghann5_med",
             "gbutt5", "gbutt5_geo", "butt5",
             "grect8", "ghann8", "ghann8_geo", "ghann8_med"]
    order = [o for o in order if o in names] + \
            [n for n in names if n not in order]
    hdr = (f"{'variant':16s} {'sweep':13s} {'A2_dB':>7s} {'phi2':>7s} "
           f"{'dax':>6s} {'std':>6s} {'bgA2':>6s} {'p':>7s} "
           f"{'rho_fd1':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for v in order:
        for s in sweeps:
            r = res[(v, s)]
            dax = "  -  " if not np.isfinite(r["dax"]) else f"{r['dax']:5.1f}"
            print(f"{v:16s} {s:13s} {r['A2']:7.3f} {r['phi2']:7.1f} "
                  f"{dax:>6s} {r['std']:6.3f} {r['bg2']:6.3f} "
                  f"{r['p']:7.4f} {r['corr_fd1']:8.4f}")
        rr = (res[(v, 'rigid_seed11')]['A2'] / res[(v, 'iso_gcal')]['A2'])
        ro = (res[(v, 'oos_seed23')]['A2'] / res[(v, 'iso_gcal')]['A2'])
        sr = (res[(v, 'rigid_seed11')]['std'] / res[(v, 'iso_gcal')]['std'])
        so = (res[(v, 'oos_seed23')]['std'] / res[(v, 'iso_gcal')]['std'])
        print(f"{'':16s} {'RATIO f/c':13s} A2 rigid/iso {rr:5.2f}  "
              f"oos/iso {ro:5.2f} | std rigid/iso {sr:5.2f}  "
              f"oos/iso {so:5.2f}")
        print()
    np.save(os.path.join(OUT, "metrics.npy"),
            np.array([res], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()

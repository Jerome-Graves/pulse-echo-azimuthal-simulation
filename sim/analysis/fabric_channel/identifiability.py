"""STEPS 1-4: rank / principal-angle identifiability of the project's two
observables against a general l<=4 harmonic ODF.

Noise weighting is the project's own measured scatter, so a singular
value s means "a coefficient perturbation of size delta along this
direction moves the data by s*delta sigmas over the whole sweep".
3/s is therefore the 3-sigma resolution of that coefficient combination,
directly comparable to the fabric's own coefficient magnitudes (a Watson
k=3.93 single maximum has max|c| = 0.92 in this 4pi-normalised basis).
"""
import sys

import numpy as np
from scipy.linalg import subspace_angles

import odf_harm as O

QUAD = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
O.set_quadrature(QUAD)

D = 0.100
SIG_CODA = 1.82                          # dB, fit_sweep.SIG_CODA_SWEEP
SIG_TOF_US = 0.35                        # us, measured sweep ToF residual
SIG_TOF = SIG_TOF_US * 1e-6 / (2 * D)
N_AZ = 60
GAPTOL = 1e-4                            # relative; verified by refinement


def nuisance_project(J, extra=None):
    n = J.shape[0]
    M = [np.ones(n)] + ([np.asarray(extra, float)] if extra is not None
                        else [])
    Q, _ = np.linalg.qr(np.column_stack(M))
    return J - Q @ (Q.T @ J)


def channels(f, beams, scale_free_tof=False):
    Jt, Jc = O.jacobians(f, beams)
    base_t, _ = O.observables(f, beams)
    Jt = nuisance_project(Jt, base_t if scale_free_tof else None) / SIG_TOF
    Jc = nuisance_project(Jc) / SIG_CODA
    return Jt, Jc


def spec(J):
    s = np.linalg.svd(J, compute_uv=False)
    r = int((s > GAPTOL * s.max()).sum())
    return s, r


def rowspace(J):
    U, s, Vt = np.linalg.svd(J)
    return Vt[:int((s > GAPTOL * s.max()).sum())].T


def describe(v, n=3):
    idx = np.argsort(-np.abs(v))[:n]
    return ", ".join(f"{O.LABEL[i]}({v[i]:+.2f})" for i in idx)


FABRICS = [
    ("isotropic", O.ISO),
    ("single max IN-PLANE  k=+3.93 az30", O.watson_f([0.866, 0.5, 0], 3.93)),
    ("single max TILTED 40deg  k=+3.93",
     O.watson_f([0.766 * 0.866, 0.766 * 0.5, 0.643], 3.93)),
    ("girdle normal IN-PLANE  k=-8", O.watson_f([1, 0, 0], -8.0)),
    ("girdle normal AXIAL z   k=-8", O.watson_f([0, 0, 1], -8.0)),
]

beams_ip, _ = O.inplane_beams(N_AZ)
AXIAL = np.array([[0.0, 0.0, 1.0]])


def oblique(n_az, tilt_deg):
    a = np.radians(np.arange(n_az) * 360.0 / n_az)
    t = np.radians(tilt_deg)
    return np.c_[np.cos(a) * np.cos(t), np.sin(a) * np.cos(t),
                 np.full(n_az, np.sin(t))]


print(f"quadrature = {O.N} directions,  {N_AZ} in-plane azimuths,  "
      f"sigma_coda {SIG_CODA} dB, sigma_ToF {SIG_TOF_US} us")
print("\nSTRUCTURALLY BLIND COEFFICIENTS for ANY in-plane beam "
      "(Y_lm vanishes on the equator when l+m is odd):")
Bp = O.basis_at(beams_ip)
blind = [O.LABEL[k] for k in range(O.NC) if np.abs(Bp[k]).max() < 1e-10]
print("   " + ", ".join(blind) + f"   -> {len(blind)} of {O.NC} coefficients")

print("\n" + "=" * 78)
for name, f in FABRICS:
    cmax = np.abs(O.coeffs_of(f)).max()
    print(f"\n### {name}   (max |c_true| = {cmax:.2f})")
    Jt, Jc = channels(f, beams_ip)
    Jb = np.vstack([Jt, Jc])
    for tag, J in (("ToF  ", Jt), ("coda ", Jc), ("BOTH ", Jb)):
        s, r = spec(J)
        print(f"  {tag} rank {r}   s = " +
              " ".join(f"{x:7.2f}" for x in s[:8]))
        print(f"        3sig resolution 3/s = " +
              " ".join(f"{3/x:7.3f}" if x > GAPTOL * s.max() else "      -"
                       for x in s[:8]))
    Rt, Rc = rowspace(Jt), rowspace(Jc)
    ang = np.degrees(subspace_angles(Rc, Rt))
    print(f"  principal angles (coda rowspace vs ToF rowspace), deg: " +
          " ".join(f"{a:6.1f}" for a in ang))
    # coda directions orthogonal to the ToF row space
    resid = Rc - Rt @ (Rt.T @ Rc)
    U, s2, _ = np.linalg.svd(resid, full_matrices=False)
    new = U[:, s2 > 0.5]
    print(f"  NEW directions the coda adds ({new.shape[1]}):")
    for j in range(new.shape[1]):
        v = new[:, j]
        sens = np.linalg.norm(Jc @ v)
        print(f"     {describe(v, 4)}   |  coda s={sens:6.2f}  "
              f"3sig at |dc|={3/max(sens,1e-12):6.3f}")

print("\n" + "=" * 78)
print("STEP 4e: what does ONE extra AXIAL (out-of-plane) beam buy, vs the "
      "coda?")
print(f"{'fabric':<36}{'ToF ip':>8}{'ToF+ax':>8}{'coda ip':>9}"
      f"{'both ip':>9}{'both+ax':>9}{'ToF obl20':>11}")
for name, f in FABRICS:
    b_ax = np.vstack([beams_ip, AXIAL])
    b_ob = np.vstack([beams_ip, oblique(N_AZ, 20.0)])
    r = []
    for beams, which in ((beams_ip, "t"), (b_ax, "t"), (beams_ip, "c"),
                         (beams_ip, "b"), (b_ax, "b"), (b_ob, "t")):
        Jt, Jc = channels(f, beams)
        J = {"t": Jt, "c": Jc, "b": np.vstack([Jt, Jc])}[which]
        r.append(spec(J)[1])
    print(f"{name:<36}{r[0]:8d}{r[1]:8d}{r[2]:9d}{r[3]:9d}{r[4]:9d}"
          f"{r[5]:11d}")

print("\nSTEP 4e detail - which coefficients an oblique ring reaches that "
      "in-plane beams cannot:")
f = O.watson_f([0.866, 0.5, 0], 3.93)
for tilt in (0.0, 10.0, 20.0, 40.0):
    beams = np.vstack([beams_ip, oblique(N_AZ, tilt)]) if tilt else beams_ip
    Jt, Jc = channels(f, beams)
    st, rt = spec(Jt)
    sb, rb = spec(np.vstack([Jt, Jc]))
    print(f"  tilt {tilt:4.0f} deg: ToF rank {rt}, stacked rank {rb}, "
          f"ToF 3sig on its weakest resolved dir "
          f"{3/st[rt-1]:6.3f}")

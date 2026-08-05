"""Part 2: tables, tilt invariance, psi_b sweep, oblique-incidence check."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # siblings
from exact_rc import (MAT, RHO, C0, setup, exact_normal_RT, modes_at_normal,
                      polar_tilt_deg, rot_y, v2t, banner)  # noqa: E402
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from ringfwi import anisotropy as an  # noqa: E402


def vqp(psi_deg):
    return float(an.ice_qp_vs_caxis(np.radians(psi_deg)))


def r_iso(psi_a, psi_b):
    va, vb = vqp(psi_a), vqp(psi_b)
    return (vb - va) / (vb + va), va, vb


banner("1b.  THE 'LEAST LONGITUDINAL' CLAIM, CHECKED ANALYTICALLY")
C11, C33, C44, C13 = MAT['C11'], MAT['C33'], MAT['C44'], MAT['C13']
num, den = C33 - C13 - 2 * C44, C11 - C13 - 2 * C44
psi_LN = np.degrees(np.arctan(np.sqrt(num / den)))
print("Longitudinal-normal condition for TI:  tan^2 psi = "
      "(C33-C13-2C44)/(C11-C13-2C44)")
print(f"   (C33-C13-2C44) = {num/1e9:.4f} GPa   (C11-C13-2C44) = {den/1e9:.4f} GPa")
print(f"   -> psi_LN = {psi_LN:.3f} deg : qP is EXACTLY longitudinal there")
g = np.linspace(0, 90, 90001)
vv = np.array([vqp(p) for p in g])
print(f"   qP speed minimum at psi = {g[np.argmin(vv)]:.3f} deg "
      f"(v = {vv.min():.2f} m/s)")
dl = np.array([polar_tilt_deg(setup(p, p)[0], RHO) for p in np.linspace(0, 90, 3001)])
gg = np.linspace(0, 90, 3001)
print(f"   polarisation tilt at psi=51 deg: {polar_tilt_deg(setup(51,51)[0],RHO):.4f} deg")
print(f"   polarisation tilt maximum      : {dl.max():.4f} deg at psi={gg[np.argmax(dl)]:.2f} deg")


def row(psi_a, psi_b, tilt=0.0, extra=None):
    CA, CB, n, ca, cb = setup(psi_a, psi_b, tilt, extra)
    r = exact_normal_RT(CA, CB, RHO)
    Rpp = r['R_by']['qP']
    Rps = r['R_by']['qSV']
    Rsh = r['R_by']['SH']
    ri, va, vb = r_iso(psi_a, psi_b)
    return dict(Rpp=Rpp, Rps=Rps, Rsh=Rsh, riso=ri, va=va, vb=vb,
                resid=r['resid'], ebal=r['ebal'],
                dA=polar_tilt_deg(CA, RHO), dB=polar_tilt_deg(CB, RHO))


def db(x, y):
    return 20 * np.log10(abs(x) / abs(y)) if y != 0 and x != 0 else np.nan


banner("2.  THE BICRYSTAL LADDER: EXACT vs ISOTROPIC, AT EVERY INTERFACE TILT")
val = np.load((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim", "analysis", "facet_model")) +
              r"\bicrystal_val.npz"))['res']
print("measured validation set (bicrystal_val.npz):")
print(f"{'psi_a':>6}{'psi_b':>7}{'R_iso(file)':>14}{'echo amp':>12}"
      f"{'v_a':>10}{'v_b':>10}")
for r_ in val:
    print(f"{r_[0]:>6.0f}{r_[1]:>7.0f}{r_[2]:>14.6e}{r_[3]:>12.4e}"
          f"{r_[4]:>10.2f}{r_[5]:>10.2f}")

TILTS = [0.0, 15.0, 22.5, 30.0, 45.0]
print("\npsi_a = 0, psi_b = 51 (the ladder pair), swept over interface tilt.")
print("Both c-axes rotate rigidly with the interface, exactly as in "
      "tilt_testbed.py,\nso the exact answer must be tilt-invariant.")
print(f"\n{'tilt':>7}{'|R| exact qP-qP':>19}{'|R_iso|':>14}"
      f"{'exact/iso (dB)':>17}{'R qP->qSV':>14}{'R qP->SH':>12}")
for t in TILTS:
    d = row(0.0, 51.0, t)
    print(f"{t:>7.1f}{abs(d['Rpp']):>19.12f}{abs(d['riso']):>14.9f}"
          f"{db(d['Rpp'], d['riso']):>17.3e}{d['Rps']:>14.3e}{d['Rsh']:>12.1e}")
# a general (non-coplanar-preserving) rotation as a stronger invariance check
th = 0.7
Rg = rot_y(0.4) @ np.array([[np.cos(th), -np.sin(th), 0],
                            [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
d = row(0.0, 51.0, 30.0, Rg)
print(f"{'arb.rot':>7}{abs(d['Rpp']):>19.12f}{abs(d['riso']):>14.9f}"
      f"{db(d['Rpp'], d['riso']):>17.3e}{d['Rps']:>14.3e}{d['Rsh']:>12.1e}")

banner("3.  THE FIVE MEASURED CONTRASTS")
print(f"{'psi_a':>6}{'psi_b':>7}{'delta_A':>9}{'delta_B':>9}"
      f"{'|R| exact':>14}{'|R_iso|':>13}{'diff (dB)':>12}"
      f"{'|R_qP->qSV|':>13}{'meas amp':>11}")
rows = []
for r_ in val:
    pa, pb, riso_file, amp = r_[0], r_[1], r_[2], r_[3]
    d = row(pa, pb)
    assert abs(d['riso'] - riso_file) < 1e-9, (d['riso'], riso_file)
    rows.append((pa, pb, d, amp))
    print(f"{pa:>6.0f}{pb:>7.0f}{d['dA']:>9.4f}{d['dB']:>9.4f}"
          f"{abs(d['Rpp']):>14.9f}{abs(d['riso']):>13.9f}"
          f"{db(d['Rpp'], d['riso']):>12.2e}{abs(d['Rps']):>13.3e}{amp:>11.4e}")

banner("4.  SWEEP psi_b FROM 0 TO 90 AT psi_a = 0 (worst case)")
print(f"{'psi_b':>7}{'delta_B':>9}{'|R| exact':>14}{'|R_iso|':>13}"
      f"{'exact-iso (dB)':>16}{'|R_qP->qSV|':>14}{'conv/R_PP':>11}")
best = (0, None)
for pb in list(range(0, 91, 5)) + [27, 51]:
    d = row(0.0, float(pb))
    dd = db(d['Rpp'], d['riso'])
    if pb not in (27, 51):
        pass
    print(f"{pb:>7.1f}{d['dB']:>9.4f}{abs(d['Rpp']):>14.9f}"
          f"{abs(d['riso']):>13.9f}{dd:>16.3e}{abs(d['Rps']):>14.3e}"
          f"{abs(d['Rps']/d['Rpp']):>11.3f}")
fine = np.linspace(0.25, 90, 3591)
dbs, convs = [], []
for pb in fine:
    d = row(0.0, float(pb))
    dbs.append(abs(db(d['Rpp'], d['riso'])))
    convs.append(abs(d['Rps'] / d['Rpp']))
dbs, convs = np.array(dbs), np.array(convs)
print(f"\nMAXIMUM |exact - iso| over 0 < psi_b <= 90 : "
      f"{dbs.max():.4e} dB at psi_b = {fine[np.argmax(dbs)]:.2f} deg")
print(f"   (as a fraction of R: {10**(dbs.max()/20)-1:.3e})")
print(f"MAXIMUM qP->qSV conversion / R_PP        : {convs.max():.3f} "
      f"at psi_b = {fine[np.argmax(convs)]:.2f} deg")
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    os.pardir, "results", "exact_rc_psib_sweep.npz")
os.makedirs(os.path.dirname(_out), exist_ok=True)
np.savez(_out, psi=fine, dbs=dbs, convs=convs)
print(f"wrote {os.path.normpath(_out)}")

banner("5.  IS THE MEASURED SCATTER EXPLAINED?")
amps = np.array([r_[3] for r_ in val[1:]])
risos = np.array([abs(r_[2]) for r_ in val[1:]])
exs = np.array([abs(row(r_[0], r_[1])['Rpp']) for r_ in val[1:]])
print(f"{'psi_a/psi_b':>13}{'amp/|R_iso|':>14}{'amp/|R_exact|':>15}")
for r_, a, ri, ex in zip(val[1:], amps, risos, exs):
    print(f"{int(r_[0])}/{int(r_[1]):<10}{a/ri:>14.3f}{a/ex:>15.3f}")
for nm, k in (("R_iso", amps / risos), ("R_exact", amps / exs)):
    print(f"\nusing {nm}:  mean {k.mean():.3f}  sd {k.std(ddof=1):.3f}  "
          f"rel.sd {100*k.std(ddof=1)/k.mean():.1f}%  "
          f"max/min {k.max()/k.min():.3f} = {20*np.log10(k.max()/k.min()):.2f} dB")
print("\ncorrelation r(|R|, amp):",
      f"iso {np.corrcoef(risos, amps)[0,1]:.4f}",
      f" exact {np.corrcoef(exs, amps)[0,1]:.4f}")
rr = np.concatenate([[0.0], risos]); aa = np.concatenate([[val[0,3]], amps])
print("correlation including the zero-contrast point:",
      f"iso {np.corrcoef(rr, aa)[0,1]:.4f}")

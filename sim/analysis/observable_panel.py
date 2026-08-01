"""Strand 5: mine the coda waveform for ANY azimuth-dependent structure."""
import os, sys, json
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
import forward as F
from specimen import DiskSpecimen

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
CACHE = r"C:\Users\Jerome\AppData\Local\Temp\claude\C--Users-Jerome\72aa31c0-c0c1-48de-881e-9470fe03e8ba\scratchpad"
C_REF, F0 = 3850.0, 2.0e6
CODA = (24e-6, 36e-6)

# ---------------------------------------------------------------- prediction
def fabric_pred(cfg, rots):
    """E[R^2] dB and mean qP velocity per azimuth, from the ACTUAL grains."""
    key = "pred_%s.npz" % cfg["name"]
    p = os.path.join(CACHE, key)
    if os.path.exists(p):
        z = np.load(p)
        if len(z["rots"]) == len(rots) and np.all(z["rots"] == rots):
            return z["er2"], z["vbar"]
    h = C_REF / F0 / float(cfg["ppw"])
    sp = DiskSpecimen(diameter_m=cfg["diameter_mm"]/1e3, thickness_m=cfg["thickness_mm"]/1e3,
                      n_grains=cfg["n_grains"], size_cv=cfg["size_cv"],
                      concentration=cfg["concentration"], spatial_corr=cfg["spatial_corr"],
                      fabric_axis=tuple(cfg["fabric_axis"]), seed=cfg["seed"])
    b = sp.build(h)
    lab, axes = b["labels"], np.asarray(b["axes"], float)
    ok = lab >= 0
    vol = np.bincount(lab[ok].ravel(), minlength=len(axes)).astype(float); vol /= vol.sum()
    keep = vol > 0
    axes, vol = axes[keep], vol[keep] / vol[keep].sum()
    er2, vb_ = [], []
    for rot in rots:
        a = np.radians(float(rot)); n = np.array([np.cos(a), np.sin(a), 0.0])
        psi = np.arccos(np.clip(np.abs(axes @ n), 0.0, 1.0))
        v = np.interp(psi, F._PSI, F._VQP)
        vb = float(vol @ v); var = float(vol @ (v - vb) ** 2)
        er2.append(10*np.log10(var/(2*vb**2))); vb_.append(vb)
    er2, vb_ = np.array(er2), np.array(vb_)
    np.savez(p, rots=np.asarray(rots), er2=er2, vbar=vb_)
    return er2, vb_

# ---------------------------------------------------------------- observables
def spectrum(x, dt):
    x = x * np.hanning(len(x))
    n = 1 << (int(np.ceil(np.log2(len(x)))) + 2)
    X = np.fft.rfft(x, n); f = np.fft.rfftfreq(n, dt)
    return f, np.abs(X)**2

def observables(tr, dt):
    fs = 1.0/dt
    an = hilbert(tr); e = np.abs(an)
    # backwall reference
    k0 = int(2*0.100/C_REF*fs); w = int(2e-6*fs)
    E1 = e[max(k0-w,0):k0+w].max()
    i0, i1 = int(CODA[0]*fs), int(CODA[1]*fs)
    t = np.arange(i0, i1)/fs
    ec = e[i0:i1]; xc = tr[i0:i1]
    p = ec**2
    o = {}
    rms = np.sqrt(p.mean())
    o["lvl_rms"]   = 20*np.log10(rms/E1)
    im = (i0+i1)//2
    r_e = np.sqrt((e[i0:im]**2).mean()); r_l = np.sqrt((e[im:i1]**2).mean())
    o["lvl_early"] = 20*np.log10(r_e/E1)
    o["lvl_late"]  = 20*np.log10(r_l/E1)
    o["early_late"]= 20*np.log10(r_e/r_l)
    # decay: robust LS slope of dB envelope vs time (smoothed to tame nulls)
    k = max(int(0.25e-6*fs)|1, 3)
    sm = np.convolve(p, np.ones(k)/k, mode="same")
    ldb = 10*np.log10(np.maximum(sm, sm.max()*1e-8))
    o["decay"]     = np.polyfit((t-t.mean())*1e6, ldb, 1)[0]          # dB/us
    # temporal shape
    wsum = p.sum()
    tc = (p*t).sum()/wsum
    o["t_centroid"]= (tc-CODA[0])*1e6
    o["t_spread"]  = np.sqrt((p*(t-tc)**2).sum()/wsum)*1e6
    o["crest"]     = 20*np.log10(ec.max()/rms)
    o["env_kurt"]  = float(kurtosis(ec))
    o["env_skew"]  = float(skew(ec))
    pn = p/wsum
    o["env_entropy"]= float(-(pn*np.log(pn+1e-300)).sum()/np.log(len(pn)))
    thr = rms
    pk = ((ec[1:-1] > ec[:-2]) & (ec[1:-1] >= ec[2:]) & (ec[1:-1] > thr)).sum()
    o["n_peaks"]   = pk/((CODA[1]-CODA[0])*1e6)                        # per us
    # instantaneous frequency, envelope-weighted
    ph = np.unwrap(np.angle(an))
    fi = np.diff(ph)/(2*np.pi*dt)
    fw = fi[i0:i1-1]; pw = p[:-1]
    assert len(fw) == len(pw)
    o["inst_freq"] = float((fw*pw).sum()/pw.sum()/1e6)
    # spectral
    f, P = spectrum(xc, dt)
    m = (f > 0.5e6) & (f < 6e6)
    fm, Pm = f[m], P[m]
    cen = (fm*Pm).sum()/Pm.sum()
    o["f_centroid"]= cen/1e6
    o["f_bandwidth"]= np.sqrt((Pm*(fm-cen)**2).sum()/Pm.sum())/1e6
    def band(a, b):
        s = (f >= a*1e6) & (f < b*1e6); return P[s].sum()
    o["bandratio_lo"] = 10*np.log10(band(1.2,1.8)/band(1.8,2.4))
    o["bandratio_hi"] = 10*np.log10(band(2.4,3.2)/band(1.8,2.4))
    sl = (f >= 1.2e6) & (f <= 3.2e6)
    o["spec_slope"] = np.polyfit(f[sl]/1e6, 10*np.log10(P[sl]+1e-300), 1)[0]
    fl = (f >= 1.0e6) & (f <= 4.0e6)
    o["spec_flat"] = float(np.exp(np.mean(np.log(P[fl]+1e-300))) / np.mean(P[fl]))
    # cepstral comb (facet spacing) on the log spectrum in 1-4 MHz
    lg = np.log(P[fl]+1e-300); lg -= lg.mean()
    C = np.abs(np.fft.rfft(lg*np.hanning(len(lg))))
    dq = 1.0/((f[fl][-1]-f[fl][0]))          # quefrency step, s
    q = np.arange(len(C))*dq
    qm = (q > 0.3e-6) & (q < 4e-6)
    if qm.sum() > 3:
        j = np.argmax(C[qm])
        o["cep_peak"] = float(C[qm][j]/ (C[qm].mean()+1e-300))
        o["cep_quef"] = float(q[qm][j]*1e6)
    else:
        o["cep_peak"] = np.nan; o["cep_quef"] = np.nan
    # envelope autocorrelation width
    d = ec - ec.mean()
    ac = np.correlate(d, d, "full")[len(d)-1:]; ac /= ac[0]
    hw = np.argmax(ac < 0.5) if np.any(ac < 0.5) else len(ac)
    o["acf_width"] = hw/fs*1e6
    # ---- positive controls (declared separately, NOT in the coda family)
    o["_E1_db"] = 20*np.log10(E1)
    o["_bw_seg"] = None
    return o, e, an, E1, fs

def load_sweep(name, d=None):
    d = d or os.path.join(OUT, name)
    cfg = json.load(open(os.path.join(d, "config.json")))
    rots, rows, bws, dts = [], [], [], []
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith("az") and fn.endswith(".npz")): continue
        with np.load(os.path.join(d, fn)) as z:
            tr = np.asarray(z["trace"], float).ravel(); dt = float(z["dt"])
        o, e, an, E1, fs = observables(tr, dt)
        rots.append(int(fn[2:5])); rows.append(o); dts.append(dt)
        k0 = int(2*0.100/C_REF*fs)
        w = int(4e-6*fs)
        bws.append(tr[k0-w:k0+w].copy())
    keys = [k for k in rows[0] if not k.startswith("_")]
    X = {k: np.array([r[k] for r in rows]) for k in keys}
    X["_E1_db"] = np.array([r["_E1_db"] for r in rows])
    L = min(len(b) for b in bws)
    bw = np.array([b[:L] for b in bws])
    return cfg, np.array(rots), X, bw, float(np.mean(dts))

def rel_tof(bw, dt):
    """relative arrival time of the backwall echo, us, by xcorr vs the mean."""
    ref = bw.mean(0)
    out = []
    for x in bw:
        c = np.correlate(x, ref, "full"); j = np.argmax(c)
        if 0 < j < len(c)-1:
            a,b,cc = c[j-1], c[j], c[j+1]
            d = 0.5*(a-cc)/(a-2*b+cc+1e-300)
        else: d = 0.0
        out.append(((j+d)-(len(ref)-1))*dt*1e6)
    return np.array(out)

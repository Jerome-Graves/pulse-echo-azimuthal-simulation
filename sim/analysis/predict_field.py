"""Build predicted azimuth x time coda envelopes and cache them."""
import os
import sys
import time

import numpy as np

import _t2_common as C

TG_LO, TG_HI = 20e-6, 42e-6
TG = np.arange(TG_LO, TG_HI, C.DT_C)
VARIANTS = {"full": (True, True), "geom": (False, True),
            "nodirec": (True, False), "flat": (False, False)}

# (sweep, build tag)  -- build tag may deliberately mismatch (controls)
JOBS = [("girdle_perp_ppw8", "gp8"),
        ("singlemax_ppw8", "sm8"),
        ("girdle_perp_ppw8", "wrongseed8"),
        ("singlemax_ppw8", "wrongseed8"),
        ("iso_gcal", "iso6"),
        ("iso_gcal", "wrongseed6"),
        ("girdle_perp_ppw8", "sm8"),      # same tessellation, WRONG fabric
        ("singlemax_ppw8", "gp8")]        # same tessellation, WRONG fabric


def predict(tag, az_list, variant, n_ray=600, rayseed=0):
    b = np.load(os.path.join(C.HERE, f"t2b_{tag}.npz"))
    lab = b["labels"].astype(np.int32)
    ax, sd, h = b["axes"], b["seeds"], float(b["h"])
    use_dv, use_dir = VARIANTS[variant]
    edges = np.concatenate([TG - C.DT_C / 2, [TG[-1] + C.DT_C / 2]])
    rows = []
    for a in az_list:
        t, w, _ = C.facet_events(lab, ax, sd, h, a, use_dv=use_dv,
                                 use_direc=use_dir, n_ray=n_ray,
                                 rayseed=rayseed)
        rows.append(np.histogram(t, edges, weights=w)[0])
    P = np.array(rows)
    k = C.pulse_power_kernel(C.DT_C)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, P)


if __name__ == "__main__":
    for sweep, tag in JOBS:
        fp = os.path.join(C.HERE, f"t2p_{sweep}__{tag}.npz")
        if os.path.exists(fp):
            print(f"{sweep} <- {tag}: cached")
            continue
        t0 = time.time()
        az, E, e1, tb = C.load_sweep(sweep, TG)
        if len(az) > 90:                                  # iso_gcal has 360
            sel = np.arange(0, len(az), 6)
            az, E = az[sel], E[sel]
        out = {v: predict(tag, az, v) for v in VARIANTS}
        np.savez_compressed(fp, az=az, meas=E, e1=e1, tgrid=TG,
                            **{f"p_{k}": v for k, v in out.items()})
        print(f"{sweep} <- {tag}: n_az={len(az)} "
              f"({time.time()-t0:.0f}s)", flush=True)

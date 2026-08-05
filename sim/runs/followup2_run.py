"""Third simulation programme: the four runs the claims need.

Each of these exists because a claim currently rests on a calculation, on
a design choice nobody varied, or on eight specimens. As before the job
definitions and the solver call come from master_run, which is imported
rather than copied.

  1  BLINDING GEOMETRY               single maximum, axis normal to the
     scan plane, ppw 8, 30 az. The facet-visibility work predicts that a
     specimen whose c-axes all make the same angle to every ray returns
     almost nothing, because two grains at the same angle present the
     same quasi-longitudinal speed however far apart their axes are. Every
     sweep so far has its fabric axis IN the scan plane, so the mechanism
     has never been tested. If the coda drops sharply against the in-plane
     single maximum, the mechanism is established; if it does not, the
     account is wrong. About 0.9 h.

  2  GIRDLE AT A WEAKER CONCENTRATION   kappa = -3, ppw 8, 30 az. The
     claim that the fabric axis is unresolvable across essentially the
     whole girdle range, kappa in [-12.67, -0.20], comes from the
     template's two-fold amplitude against the realisation scatter. It has
     only ever been simulated at kappa = -8. One run at -3 tests the
     interval where the template says the axis should still fail while the
     nominal two-fold term is an order of magnitude larger. About 0.9 h.

  3  SPATIALLY CLUSTERED ORIENTATION    spatial_corr = 0.6 on the seed-11
     girdle, ppw 8, 30 az. Every sweep in the programme was run at
     spatial_corr = 0, so neighbouring grains are independent and local
     disorientation is at its maximum. Nothing in the paper can be
     attributed to clustering, and nothing can be tested against it: that
     is a design gap rather than a null result. Real ice is clustered.
     This pairs against girdle_seed11_ppw8_dev, which shares seed, fabric and
     resolution and differs only in the correlation. About 0.9 h.

  4  FOUR MORE GIRDLE TESSELLATIONS   seeds 13, 29, 47, 67 at ppw 8,
     30 az. The property-observable screen established that at n = 8 the
     between-specimen regime is not a weak discovery engine but not a
     discovery engine at all: 32832 cells returned fewer hits than chance.
     Twelve tessellations will not fix that, but they do carry the
     pre-specified claims that already rest on eight, in particular the
     crossing count against coda level at r = 0.79. About 3.6 h.

Total about 6.3 GPU hours. Resumable: any azimuth whose .npz exists is
skipped. Progress appends to master_status.txt with the other programmes.

NOT included. An ensemble that varies grain size deliberately is still
the experiment the conclusions name as missing, and it is still a
decision about the scope of the paper rather than a repair, so it is not
launched here.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import master_run as M                                     # noqa: E402

# The scan plane is z = const: every production fabric axis lies in it.
# A single maximum along z therefore puts every c-axis at the same angle
# to every ray in the plane, which is the blinding case.
SINGLE_NORMAL = dict(kappa=3.93, axis=(0.0, 0.0, 1.0), tag="singlez")
GIRDLE_WEAK = dict(kappa=-3.0, axis=(1.0, 0.0, 0.0), tag="girdle3")

JOBS = [
    dict(name="singlemax_seed11_ppw8_axis_normal", ppw=8.0, step=12, seed=11,
         mode="normal", **SINGLE_NORMAL),
    dict(name="girdle_seed11_ppw8_weak_kappa3", ppw=8.0, step=12, seed=11,
         mode="normal", **GIRDLE_WEAK),
    dict(name="girdle_seed11_ppw8_clustered", ppw=8.0, step=12, seed=11,
         mode="normal", spatial_corr=0.6, **M.GIRDLE),
]
for sd in (13, 29, 47, 67):
    JOBS.append(dict(name="girdle_seed%d_ppw8_ensemble" % sd, ppw=8.0, step=12,
                     seed=sd, mode="normal", **M.GIRDLE))


if __name__ == "__main__":
    M.note("=" * 60)
    M.note("FOLLOWUP2: %d sweeps (blinding geometry, weak girdle, "
           "clustered, four tessellations)" % len(JOBS))
    t_start = time.time()
    for j, job in enumerate(JOBS, 1):
        M.note("[%d/%d] %s  (ppw %.0f, seed %d, %s, kappa %.2f, corr %.1f)"
               % (j, len(JOBS), job["name"], job["ppw"], job["seed"],
                  job["mode"], job["kappa"], job.get("spatial_corr", 0.0)))
        try:
            M.run_job(job)
        except Exception:
            M.note("  FAILED, continuing:\n" + traceback.format_exc())
    M.note("FOLLOWUP2 DONE in %.1f h" % ((time.time() - t_start) / 3600))

"""Second simulation programme, closing two gaps the audit identified.

Both gaps are places where the paper generalises from a sample that
cannot support the generalisation, and each is closed by running the
existing configuration on more of what it already has rather than by
anything new. The job definitions, the solver call, the config record and
the resumability all come from master_run, which is imported rather than
copied so that the two programmes cannot drift apart.

  1  resolution ladder on a second tessellation   seed 23 girdle at
     ppw 6 and ppw 10, 30 az. The convergence result rests on seed 11
     alone: 4.23, 3.03 and 2.81 dB of per-azimuth scatter at ppw 6, 8
     and 10, and the 2.68 dB level step from ppw 8 to 10 that the
     abstract quotes. One further tessellation at the outer rungs says
     whether that behaviour belongs to the regime or to that specimen,
     which is the difference between a criterion and a demonstration.
     Seed 23 is chosen because it already carries both fabrics at ppw 8,
     so the new rungs join a fully characterised middle. About 2.5 h.

  2  four more single-maximum realisations       seeds 7, 53, 71 and 89
     at ppw 8, 30 az. The fabric separation of 2.83 dB rests on four
     pairs, one of which (seed 17) is a specular outlier carrying
     -7.40 dB of it, and a paired t test on those four returns
     t = -1.46, p = 0.24. The separation is reported but not
     established. These four give eight matched pairs, every girdle seed
     with a single-maximum twin on a bit-identical tessellation, and
     will either establish the separation or retire it. About 3.6 h.

Total about 6.1 GPU hours. Resumable: any azimuth whose .npz exists is
skipped, so this can be stopped and restarted. Progress appends to
master_status.txt alongside the first programme.

NOT included, and deliberately. An ensemble that VARIES grain count,
mean diameter and boundary area is the experiment the conclusions name
as missing, and it is the only one that could turn the coda into a
measurement of grain size. It is about 11 h and it is a decision about
the scope of the paper rather than a repair, so it is not launched here.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import master_run as M                                     # noqa: E402

JOBS = []

# --- 1. resolution ladder on a second tessellation --------------------
# ppw 8 already exists as girdle_seed23_ppw8_ensemble, so only the outer rungs are
# run and the ladder is 6, 8, 10 like seed 11's.
for ppw in (6.0, 10.0):
    JOBS.append(dict(name="girdle_seed23_ppw%d_ladder" % int(ppw), ppw=ppw,
                     step=12, seed=23, mode="normal", **M.GIRDLE))

# --- 2. the remaining single-maximum twins ----------------------------
# Seeds 17, 23 and 41 were run under both fabrics in the first
# programme. These four complete the set, so that every one of the eight
# girdle tessellations has a twin sharing its geometry exactly.
for sd in (7, 53, 71, 89):
    JOBS.append(dict(name="singlemax_seed%d_ppw8_ensemble" % sd, ppw=8.0, step=12,
                     seed=sd, mode="normal", **M.SINGLE))


if __name__ == "__main__":
    M.note("=" * 60)
    M.note("FOLLOWUP RUN: %d sweeps (ladder on seed 23, four single "
           "maxima)" % len(JOBS))
    t_start = time.time()
    for j, job in enumerate(JOBS, 1):
        M.note("[%d/%d] %s  (ppw %.0f, seed %d, %s)"
               % (j, len(JOBS), job["name"], job["ppw"], job["seed"],
                  job["mode"]))
        try:
            M.run_job(job)
        except Exception:
            M.note("  FAILED, continuing:\n" + traceback.format_exc())
    M.note("FOLLOWUP DONE in %.1f h" % ((time.time() - t_start) / 3600))

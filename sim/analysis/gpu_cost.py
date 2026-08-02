"""Measured computational cost of the whole study.

Supports the total quoted in Appendix C. The figure there was written
while the production programme was still running and projected its last
twelve jobs from the rate of those already finished. The programme has
since completed, so the total can be measured instead.

Nothing here launches a simulation. Every number comes from artefacts the
runs left behind: the modification time of each recorded azimuth, and the
per-run durations the two follow-up scripts printed into their own logs.

Method. Each azimuth trace is written the moment that azimuth finishes,
so the interval between consecutive traces of one sweep IS that azimuth's
wall time. The median of those intervals is taken as the sweep's
per-azimuth cost, because the median ignores the gaps where a run was
paused or the machine was doing something else. Two totals follow and
they are not the same number:

  job sum      per-azimuth cost times azimuth count, summed over sweeps.
               This is the work done. It double counts wall time whenever
               two sweeps shared the device, because contention inflates
               the per-azimuth interval of both.
  device busy  the union of the busy intervals on the timeline. This is
               how long the one GPU was occupied, and it is what a reader
               reproducing the study has to budget.

The two agree to under a percent across the production programme, which
ran strictly serially, and diverge by about a fifth over the exploratory
period, where sweeps were deliberately overlapped. The paper quotes
device-busy hours.

Excluded, and deliberately. The finite-element cross-check of Section 4
is a second solver on a conforming mesh and its runs recorded no
durations, so it cannot be measured the same way and is not folded into a
figure described as measured. Analysis, fitting and figure generation run
on the CPU and are not GPU cost.

Reads:  out/sweeps/*/az*.npz          modification times only
        out/sweeps/*/config.json      resolution, frequency, treatment
        sim/results/master_status.txt first line, programme start
        sim/results/bicrystal_ladder.npz end of the resolution ladder
        sim/results/tilt_table.log     per-cell durations
        sim/results/domain_absorber.log per-azimuth durations
        sim/results/two_grain.log      per-run durations
        sim/results/tilt_ppw6.log      per-run durations
Writes: nothing. Prints a table and the totals.
"""
import datetime as dt
import json
import os
import re
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
SWEEPS = os.path.join(_ROOT, 'out', 'sweeps')
RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))

# A sweep interval longer than this multiple of the sweep's own median is
# an interruption, not a slow azimuth, so only the median is charged.
STALL_FACTOR = 1.5


def sweep_records():
    """One record per sweep directory holding at least one azimuth.

    The 'treatment' key is written only by runs/master_run.py, so its
    presence identifies the production programme without a hand-kept list
    that could drift from what was actually run.
    """
    out = []
    for name in sorted(os.listdir(SWEEPS)):
        path = os.path.join(SWEEPS, name)
        if not os.path.isdir(path):
            continue
        stamps = sorted(
            os.path.getmtime(os.path.join(path, f))
            for f in os.listdir(path)
            if f.startswith('az') and f.endswith('.npz'))
        if len(stamps) < 2:
            continue
        cfg_path = os.path.join(path, 'config.json')
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding='utf-8') as fh:
                cfg = json.load(fh)
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        out.append(dict(name=name, stamps=stamps, gaps=gaps,
                        per_az=st.median(gaps), n_az=len(stamps),
                        ppw=cfg.get('ppw'), f0=cfg.get('f0_mhz'),
                        production='treatment' in cfg))
    return out


def log_durations(filename, pattern, scale):
    """Sum the durations a run printed for itself, in hours.

    Duplicate lines are possible: tilt_table_run.py both prints and
    appends, and a resumed run reprints the cells it skipped. Lines are
    therefore counted once each.
    """
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        return 0.0, 0
    seen = set()
    total = 0.0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            match = re.search(pattern, line)
            if not match or line.strip() in seen:
                continue
            seen.add(line.strip())
            total += float(match.group(1)) * scale

    return total / 3600.0, len(seen)


def busy_intervals(records, stall_factor=STALL_FACTOR):
    """[(start, end)] on the wall clock during which the device was busy.

    An azimuth's interval ends when its trace is written and starts one
    azimuth-cost earlier, capped so that a pause between azimuths is not
    charged as compute.
    """
    spans = []
    for rec in records:
        cap = stall_factor * rec['per_az']
        for i, stamp in enumerate(rec['stamps']):
            held = rec['per_az'] if i == 0 else min(rec['gaps'][i - 1], cap)
            spans.append((stamp - held, stamp))
    return spans


def union_hours(spans):
    """Total length of the union of the spans, in hours."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged) / 3600.0


def programme_wall_hours(records):
    """Start of the resolution ladder to the last production trace.

    The ladder leaves no per-azimuth trace of its own, so its start is
    read from the first timestamp in master_status.txt and its end from
    the modification time of the .npz it writes.
    """
    status = os.path.join(RESULTS, 'master_status.txt')
    ladder = os.path.join(RESULTS, 'bicrystal_ladder.npz')
    if not (os.path.exists(status) and os.path.exists(ladder)):
        return None, None
    with open(status, encoding='utf-8') as fh:
        stamp = re.match(r'(\d\d):(\d\d):(\d\d)', fh.readline())
    end = os.path.getmtime(ladder)
    day = dt.datetime.fromtimestamp(end).date()
    start = dt.datetime.combine(
        day, dt.time(*(int(g) for g in stamp.groups()))).timestamp()
    last = max(r['stamps'][-1] for r in records if r['production'])
    return (end - start) / 3600.0, (last - start) / 3600.0


def draw_sweeps(records):
    print('%-24s %5s %5s %5s %9s %9s'
          % ('sweep', 'ppw', 'MHz', 'n az', 'per az s', 'hours'))
    for group, label in ((True, 'PRODUCTION PROGRAMME'),
                         (False, 'EXPLORATORY AND VALIDATION')):
        rows = [r for r in records if r['production'] is group]
        print('  -- %s --' % label)
        for rec in sorted(rows, key=lambda r: r['stamps'][0]):
            print('%-24s %5s %5s %5d %9.1f %9.2f'
                  % (rec['name'], rec['ppw'], rec['f0'], rec['n_az'],
                     rec['per_az'], rec['n_az'] * rec['per_az'] / 3600.0))


def draw_totals(records, ladder_h, wall_h, follow, bench):
    prod = [r for r in records if r['production']]
    expl = [r for r in records if not r['production']]

    def block(label, rows, extra_h=0.0, note=''):
        job = sum(r['n_az'] * r['per_az'] for r in rows) / 3600.0
        busy = union_hours(busy_intervals(rows))
        print('%-32s %8.2f %8.2f  %s'
              % (label, job + extra_h, busy + extra_h, note))
        return busy + extra_h

    tilt_h, dom_h = follow
    two_grain_h, tilt_ppw6_h = bench
    print()
    print('%-32s %8s %8s' % ('', 'job sum', 'device'))
    prod_h = block('production programme', prod, ladder_h,
                   'ladder %.2f h + 17 sweeps' % ladder_h)
    follow_h = tilt_h + dom_h
    print('%-32s %8.2f %8.2f  tilt table %.2f, domain %.2f'
          % ('tilt table and domain check', follow_h, follow_h,
             tilt_h, dom_h))
    bench_h = two_grain_h + tilt_ppw6_h
    expl_h = block('exploratory and validation', expl, bench_h,
                   'sweeps + %.2f h bench runs' % bench_h)
    print('%-32s %8s %8.2f' % ('TOTAL', '', prod_h + follow_h + expl_h))
    print()
    print('%d recorded azimuth simulations, %d production and %d before it'
          % (sum(r['n_az'] for r in records),
             sum(r['n_az'] for r in prod), sum(r['n_az'] for r in expl)))
    print('production programme wall clock, ladder start to last trace '
          '%.2f h' % wall_h)
    biggest = max(records, key=lambda r: r['n_az'] * r['per_az'])
    print('largest single item %s, %.1f h, %d azimuths at %.0f MHz'
          % (biggest['name'], biggest['n_az'] * biggest['per_az'] / 3600.0,
             biggest['n_az'], biggest['f0']))


def draw_sensitivity(records, fixed_h):
    """How much of the total is the stall cap rather than the data."""
    print()
    print('stall cap sensitivity, total device hours')
    for factor in (1.2, 1.5, 2.0, 3.0):
        spans = busy_intervals(records, factor)
        print('  cap %.1f x median   %6.2f h'
              % (factor, union_hours(spans) + fixed_h))


def main():
    records = sweep_records()
    ladder_h, wall_h = programme_wall_hours(records)
    tilt_h, tilt_n = log_durations('tilt_table.log',
                                   r'\s([\d.]+) min$', 60.0)
    dom_h, _n = log_durations('domain_absorber.log', r'\s(\d+) s$', 1.0)
    # Two bench experiments that wrote traces outside out/sweeps: the
    # two-grain and zero-contrast controls, and the first rotated-grid
    # probe that the tilt table later superseded.
    two_grain_h, _n = log_durations('two_grain.log', r'steps, (\d+) s$', 1.0)
    tilt_ppw6_h, _n = log_durations('tilt_ppw6.log', r'\s+(\d+)$', 1.0)
    draw_sweeps(records)
    draw_totals(records, ladder_h, wall_h, (tilt_h, dom_h),
                (two_grain_h, tilt_ppw6_h))
    draw_sensitivity(records,
                     ladder_h + tilt_h + dom_h + two_grain_h + tilt_ppw6_h)
    print('%d tilt-table cells timed' % tilt_n)


if __name__ == '__main__':
    main()

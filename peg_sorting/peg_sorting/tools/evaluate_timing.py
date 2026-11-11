"""Summarise success rate and timing over logged runs (no ROS needed).

    ros2 run peg_sorting evaluate_timing [--dir ~/peg_sorting_logs] [--mode autonomy|teleop] [--last 5]
    python3 scripts/evaluate_timing.py ...        (same thing, from the repo root)

Every autonomous run (task_manager) and teleop session (teleop_keyboard) writes
run_<id>.csv (one row per pick-and-place attempt) and run_<id>.json (summary).
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict


def load_runs(d, mode=None):
    runs = []
    for jp in sorted(glob.glob(os.path.join(d, 'run_*.json'))):
        with open(jp) as f:
            s = json.load(f)
        if mode and s.get('mode') != mode:
            continue
        cp = jp[:-5] + '.csv'
        rows = []
        if os.path.exists(cp):
            with open(cp) as f:
                rows = list(csv.DictReader(f))
        s['rows'] = rows
        runs.append(s)
    return runs


def fmt(v, nd=1):
    return ('%.' + str(nd) + 'f') % v if isinstance(v, (int, float)) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=os.path.expanduser('~/peg_sorting_logs'))
    ap.add_argument('--mode', choices=['autonomy', 'teleop'])
    ap.add_argument('--last', type=int, default=0, help='only the last N runs')
    a = ap.parse_args()
    runs = load_runs(a.dir, a.mode)
    if a.last:
        runs = runs[-a.last:]
    if not runs:
        print('no runs found in %s' % a.dir)
        return 1
    print('%-28s %-9s %8s %8s %9s %8s' % ('run', 'mode', 'placed', 'expected', 'failures', 'time[s]'))
    for r in runs:
        print('%-28s %-9s %8d %8d %9d %8s' % (r['run_id'], r['mode'], r['pegs_placed'], r['pegs_expected'],
                                             r['failures'], fmt(r['total_time_s'])))
    print()
    for mode in sorted({r['mode'] for r in runs}):
        rs = [r for r in runs if r['mode'] == mode]
        placed = sum(r['pegs_placed'] for r in rs)
        expected = sum(max(r['pegs_expected'], 1) for r in rs)
        complete = [r for r in rs if r['pegs_placed'] >= r['pegs_expected'] > 0]
        attempts = [row for r in rs for row in r['rows']]
        ok_att = [row for row in attempts if row['success'] == 'True']
        print('== %s: %d runs ==' % (mode, len(rs)))
        print('  peg success rate      : %d/%d = %.0f%%' % (placed, expected, 100.0 * placed / expected))
        print('  attempt success rate  : %d/%d' % (len(ok_att), len(attempts)))
        print('  fully completed runs  : %d/%d  (zero-failure runs: %d)' % (
            len(complete), len(rs), sum(1 for r in complete if r['failures'] == 0)))
        if complete:
            times = [r['total_time_s'] for r in complete]
            print('  completed run time [s]: mean %.1f  best %.1f  worst %.1f%s' % (
                st.mean(times), min(times), max(times),
                ('  std %.1f' % st.stdev(times)) if len(times) > 1 else ''))
        per = defaultdict(list)
        for row in ok_att:
            if row.get('duration_s'):
                per[row['peg_color']].append(float(row['duration_s']))
        if per:
            print('  per-peg time [s] (successful attempts):')
            for c, v in sorted(per.items()):
                print('    %-8s n=%-3d mean %.1f  best %.1f' % (c, len(v), st.mean(v), min(v)))
        reasons = defaultdict(int)
        for row in attempts:
            if row['success'] != 'True':
                reasons[row.get('reason') or '?'] += 1
        if reasons:
            print('  failure reasons:')
            for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
                print('    %3d x %s' % (v, k))
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

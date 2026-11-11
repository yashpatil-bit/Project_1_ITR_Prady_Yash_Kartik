"""CSV logging of every pick-and-place attempt (autonomy and teleop) for
scripts/evaluate_timing.py."""
import csv
import json
import os
import time

FIELDS = ['run_id', 'mode', 'peg_color', 'shape', 'attempt', 'success', 'verified', 'reason',
          't_start', 't_grasp', 't_place', 't_end', 'duration_s', 'misalignment_deg', 'pitch_deg', 'route']


class RunLogger:
    def __init__(self, log_dir, mode):
        self.dir = os.path.expanduser(log_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.mode = mode
        self.run_id = time.strftime('%Y%m%d_%H%M%S') + '_' + mode
        self.csv_path = os.path.join(self.dir, 'run_%s.csv' % self.run_id)
        self.json_path = os.path.join(self.dir, 'run_%s.json' % self.run_id)
        self._f = open(self.csv_path, 'w', newline='')
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        self._w.writeheader()
        self.t_run_start = None
        self.rows = []

    def start_run(self):
        self.t_run_start = time.time()

    def attempt(self, **kw):
        row = {k: kw.get(k, '') for k in FIELDS}
        row['run_id'] = self.run_id
        row['mode'] = self.mode
        if row['t_start'] != '' and row['t_end'] != '':
            row['duration_s'] = round(float(row['t_end']) - float(row['t_start']), 3)
        for k in ('t_start', 't_grasp', 't_place', 't_end'):
            if row[k] != '' and self.t_run_start is not None:
                row[k] = round(float(row[k]) - self.t_run_start, 3)
        self._w.writerow(row)
        self._f.flush()
        self.rows.append(row)
        return row

    def finish(self, n_pegs_expected, extra=None):
        t_total = (time.time() - self.t_run_start) if self.t_run_start else 0.0
        placed = sorted({r['peg_color'] for r in self.rows if str(r['success']) == 'True'})
        summary = {
            'run_id': self.run_id, 'mode': self.mode, 'total_time_s': round(t_total, 2),
            'pegs_expected': n_pegs_expected, 'pegs_placed': len(placed), 'placed_colors': placed,
            'attempts': len(self.rows),
            'failures': sum(1 for r in self.rows if str(r['success']) != 'True'),
        }
        if extra:
            summary.update(extra)
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self._f.close()
        return summary

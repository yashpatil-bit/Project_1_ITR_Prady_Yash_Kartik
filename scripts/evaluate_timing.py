#!/usr/bin/env python3
"""Success-rate / timing report over the logged runs.  See peg_sorting/tools/evaluate_timing.py.

    python3 scripts/evaluate_timing.py [--dir ~/peg_sorting_logs] [--mode autonomy|teleop] [--last N]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'peg_sorting'))
from peg_sorting.tools.evaluate_timing import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())

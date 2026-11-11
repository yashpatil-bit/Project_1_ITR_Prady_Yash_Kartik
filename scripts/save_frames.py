#!/usr/bin/env python3
"""Save RealSense frames (colour + depth + intrinsics + current calibration) to one file,
so the detection can be debugged/tuned offline.  Stop the vision node first.

    python3 scripts/save_frames.py              -> ~/peg_sorting_logs/frames_<time>.npz
    python3 scripts/save_frames.py --n 10
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'peg_sorting'))
from peg_sorting.utils.config import load_yaml  # noqa: E402
from peg_sorting.utils.transforms import T_from_dict  # noqa: E402
from peg_sorting.vision.camera import RealSenseCamera  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--out', default=os.path.expanduser('~/peg_sorting_logs'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cam = RealSenseCamera(load_yaml('vision.yaml'))
    frames = [cam.get_frame() for _ in range(a.n)]
    cam.stop()
    frames = [f for f in frames if f is not None]
    calib = load_yaml('camera_calib.yaml')
    path = os.path.join(a.out, 'frames_%d.npz' % int(time.time()))
    np.savez_compressed(path, color=np.stack([f.color for f in frames]),
                        depth=np.stack([f.depth for f in frames]).astype(np.float32), K=frames[0].K,
                        T_base_cam=T_from_dict(calib['T_base_cam']),
                        xyz_correction=np.array(calib.get('xyz_correction_m') or [0, 0, 0], float))
    print('saved %d frames to %s (%.1f MB) - send me this file' % (len(frames), path, os.path.getsize(path) / 1e6))


if __name__ == '__main__':
    main()

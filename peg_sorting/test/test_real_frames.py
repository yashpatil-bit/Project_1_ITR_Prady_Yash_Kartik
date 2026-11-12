"""Regression test on real frames from the lab setup (overhead RealSense D435i, 640x480 over
USB 2, shiny metal plate with large depth holes).  Frames + the corrected calibration are in
test/data/real_frames_overhead.npz (depth stored in mm)."""
import math
import os

import numpy as np

from peg_sorting.tools.calibrate_aruco import calibrate
from peg_sorting.utils.config import load_yaml
from peg_sorting.vision.camera import Frame
from peg_sorting.vision.geometry import points_in_base
from peg_sorting.vision.scene import SceneDetector

DATA = os.path.join(os.path.dirname(__file__), 'data', 'real_frames_overhead.npz')


def _frames():
    d = np.load(DATA)
    frames = [Frame(c, z.astype(np.float32) / 1000.0, d['K'], np.zeros(5), 0.0)
              for c, z in zip(d['color'], d['depth'])]
    return frames, d['T_base_cam']


def test_calibration_uses_whole_table_plane():
    frames, T_ref = _frames()
    acfg = {'dictionary': 'auto', 'marker_id': -1, 'marker_length_m': 0.04,
            'marker_in_base': {'x': 0.099, 'y': 0.103, 'z': 0.0, 'yaw_deg': -90}}
    T, info = calibrate(frames, acfg)
    assert np.linalg.norm(T[:3, 3] - T_ref[:3, 3]) < 0.005
    # the plate/board must come out flat (the old local-patch fit was tilted by ~11 deg)
    z = points_in_base(frames[0].depth, frames[0].K, T)[..., 2]
    board = z[100:420, 240:400]
    board = board[np.isfinite(board) & (np.abs(board) < 0.05)]
    assert len(board) > 1000 and abs(np.median(board)) < 0.01


def test_real_pegs_and_sockets():
    frames, T = _frames()
    det = SceneDetector(load_yaml('colors.yaml'), load_yaml('vision.yaml'), T)
    pegs, board, sockets, _ = det.detect_averaged(frames)
    got = {p.color: p for p in pegs}
    assert {c: p.shape for c, p in got.items()} == {
        'yellow': 'triangle', 'blue': 'square', 'orange': 'hexagon', 'green': 'circle'}
    for p in pegs:
        assert 0.08 < p.height < 0.115, (p.color, p.height)
    assert board.found
    assert sorted(s.shape for s in sockets) == ['circle', 'hexagon', 'square', 'triangle']
    ys = [s.y for s in sorted(sockets, key=lambda s: s.y)]
    assert all(0.05 < b - a < 0.09 for a, b in zip(ys[:-1], ys[1:]))   # ~7 cm spacing along the board
    assert abs(math.degrees(board.yaw)) > 75                             # board runs along y


def test_real_pegs_every_single_frame():
    """Each frame on its own must give the right shapes (the live window shows single frames;
    on the real setup the yellow triangle was once read as a square from patchy depth)."""
    frames, T = _frames()
    det = SceneDetector(load_yaml('colors.yaml'), load_yaml('vision.yaml'), T)
    want = {'yellow': 'triangle', 'blue': 'square', 'orange': 'hexagon', 'green': 'circle'}
    for fr in frames:
        pegs, _, _, _ = det.detect(fr, with_sockets=False)
        assert {p.color: p.shape for p in pegs} == want

"""End-to-end vision test on the rendered synthetic scene."""
import math

import numpy as np

from peg_sorting.sim.synthetic_scene import SimWorld, SyntheticCamera
from peg_sorting.tools.calibrate_aruco import calibrate
from peg_sorting.utils.config import load_yaml
from peg_sorting.vision.scene import SceneDetector


def _setup():
    w = SimWorld(load_yaml('sim_scene.yaml'))
    cam = SyntheticCamera(w)
    det = SceneDetector(load_yaml('colors.yaml'), load_yaml('vision.yaml'), w.T_base_cam)
    return w, cam, det


def _ang(a, b, period):
    return abs((a - b + period / 2) % period - period / 2)


def test_pegs_and_sockets_detected():
    w, cam, det = _setup()
    pegs, board, sockets, _ = det.detect_averaged([cam.get_frame() for _ in range(2)])
    assert board.found
    assert sorted(p.color for p in pegs) == sorted(p.color for p in w.pegs)
    for gt in w.pegs:
        p = next(p for p in pegs if p.color == gt.color)
        assert p.shape == gt.shape and p.shape_source == 'geometry'
        assert math.hypot(p.x - gt.x, p.y - gt.y) < 0.004
        assert abs(p.z_top - gt.height) < 0.006
        if gt.shape != 'circle':
            per = 2 * math.pi / {'triangle': 3, 'square': 4, 'hexagon': 6}[gt.shape]
            assert _ang(p.yaw, gt.yaw, per) < math.radians(4)
    assert sorted(s.shape for s in sockets) == sorted(s.shape for s in w.sockets)
    for gt in w.sockets:
        s = next(s for s in sockets if s.shape == gt.shape)
        assert math.hypot(s.x - gt.x, s.y - gt.y) < 0.004
        if gt.shape != 'circle':
            per = 2 * math.pi / {'triangle': 3, 'square': 4, 'hexagon': 6}[gt.shape]
            assert _ang(s.yaw, gt.yaw, per) < math.radians(4)


def test_aruco_calibration_accuracy():
    sc = load_yaml('sim_scene.yaml')
    w = SimWorld(sc)
    cam = SyntheticCamera(w)
    acfg = {'dictionary': 'auto', 'marker_id': -1, 'marker_length_m': sc['aruco']['length'],
            'marker_in_base': {'x': sc['aruco']['x'], 'y': sc['aruco']['y'], 'z': 0.0, 'yaw_deg': 0.0}}
    T, info = calibrate([cam.get_frame() for _ in range(3)], acfg)
    assert info['dictionary'] == 'DICT_4X4_50'
    assert np.linalg.norm(T[:3, 3] - w.T_base_cam[:3, 3]) < 0.005
    R = T[:3, :3].T @ w.T_base_cam[:3, :3]
    assert math.degrees(math.acos(min(1.0, (np.trace(R) - 1) / 2))) < 1.0

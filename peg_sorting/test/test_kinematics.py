import math
import random

import numpy as np

from peg_sorting.planning.kinematics import ArmGeometry, ManipulatorXKinematics
from peg_sorting.planning.trajectory import Limits, cartesian_path, joint_move
from peg_sorting.utils.config import load_yaml


def kin():
    return ManipulatorXKinematics(ArmGeometry.from_config(load_yaml('robot.yaml')))


def test_zero_pose_matches_robotis():
    x, y, z, p = kin().fk([0, 0, 0, 0])
    assert abs(x - (0.024 + 0.124 + 0.126)) < 1e-9
    assert abs(z - (0.0765 + 0.128)) < 1e-9
    assert abs(y) < 1e-12 and abs(p) < 1e-12


def test_ik_roundtrip():
    k = kin()
    rng = random.Random(0)
    n = 0
    for _ in range(3000):
        q = [rng.uniform(lo, hi) for lo, hi in zip(k.g.lower, k.g.upper)]
        x, y, z, p = k.fk(q)
        s = k.ik(x, y, z, p)
        if s is None:
            continue
        n += 1
        assert np.allclose(k.fk(s), (x, y, z, p), atol=1e-6)
    assert n > 1000


def test_top_grasp_reach():
    k = kin()
    q, p = k.ik_search_pitch(0.18, 0.05, 0.07)
    assert q is not None and abs(p - math.pi / 2) < 1e-9


def test_cartesian_path_is_straight_and_vertical_tool():
    k = kin()
    lim = Limits()
    traj = cartesian_path(k, [(0.2, 0.0, 0.15), (0.2, 0.0, 0.08)], math.radians(70), lim)
    for q in traj.q:
        x, y, z, p = k.fk(q)
        assert abs(x - 0.2) < 1e-6 and abs(y) < 1e-6
        assert abs(p - math.radians(70)) < 1e-6
    assert abs(k.fk(traj.q[-1])[2] - 0.08) < 1e-6


def test_joint_move_respects_velocity():
    lim = Limits(joint_vel=1.0, joint_acc=3.0)
    tr = joint_move([0, 0, 0, 0], [1.0, -0.5, 0.2, 0], lim)
    v = np.max(np.abs(np.diff(tr.q, axis=0)) / np.diff(tr.t)[:, None])
    assert v <= 1.0 + 1e-3
    assert np.allclose(tr.q[-1], [1.0, -0.5, 0.2, 0])

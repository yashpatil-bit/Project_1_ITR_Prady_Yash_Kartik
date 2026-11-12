import math
import sys
import types

import numpy as np

# calibrate_arm imports ROS modules at import time; stub them so the maths can be tested anywhere
for name in ('rclpy',):
    sys.modules.setdefault(name, types.ModuleType(name))
for name, attrs in (('peg_sorting.control.robot_interface', ['RobotInterface']),
                    ('peg_sorting.nodes.teleop_keyboard', ['Teleop'])):
    if name not in sys.modules:
        m = types.ModuleType(name)
        for a in attrs:
            setattr(m, a, object)
        sys.modules[name] = m

from peg_sorting.planning.kinematics import ArmGeometry, ManipulatorXKinematics  # noqa: E402
from peg_sorting.tools.calibrate_arm import POSES, fit, tip_z  # noqa: E402


def test_fit_predicts_fingertip_height_at_other_poses():
    """Plate touches only constrain heights, so individual offsets are not unique - but the fitted
    model must predict fingertip heights (and roughly x-y) correctly at other grasp-like poses."""
    rng = np.random.default_rng(1)
    geom = ArmGeometry(tool_length=0.124)
    true = np.array([math.radians(3.0), math.radians(-2.0), math.radians(4.0), 0.150, -0.006])
    kin_true = ManipulatorXKinematics(ArmGeometry(tool_length=true[3]))
    qs = []
    for x, y, pdeg in POSES:
        g = kin_true.ik(x, y, true[4], math.radians(pdeg))   # fingertip on the plate (z = h0)
        assert g is not None
        qs.append([g[0], g[1] - true[0] + rng.normal(0, 0.002), g[2] - true[1] + rng.normal(0, 0.002),
                   g[3] - true[2] + rng.normal(0, 0.002)])
    tip0 = 0.124 + 0.015
    before = np.array([tip_z(geom, q, [0, 0, 0, tip0, 0]) for q in qs])
    p, res = fit(geom, qs, tip0)
    assert np.abs(before).max() > 0.01 and np.sqrt(np.mean(res ** 2)) < 0.001
    kin_fit = ManipulatorXKinematics(ArmGeometry(tool_length=p[3]))
    xy_err = []
    for _ in range(200):
        x, y, z = rng.uniform(0.08, 0.24), rng.uniform(-0.12, 0.09), rng.uniform(0.04, 0.12)
        g = kin_true.ik(x, y, z, math.radians(rng.uniform(35, 90)))
        if g is None:
            continue
        q = [g[0], g[1] - true[0], g[2] - true[1], g[3] - true[2]]
        xf, yf, zf, _ = kin_fit.fk([q[0], q[1] + p[0], q[2] + p[1], q[3] + p[2]])
        assert abs((zf - p[4]) - (z - true[4])) < 0.004   # heights: what the touches calibrate
        xy_err.append(math.hypot(xf - x, yf - y))
    # x-y is not measured by plate touches; the remainder is removed by calibrate_workspace
    assert np.median(xy_err) < 0.005 and max(xy_err) < 0.012


def test_all_poses_reachable_with_our_tool_length():
    kin = ManipulatorXKinematics(ArmGeometry(tool_length=0.124))
    for x, y, pdeg in POSES:
        p = math.radians(pdeg)
        z = 0.03 + 0.015 * math.sin(p)
        assert kin.ik(x, y, z, p) is not None and kin.ik(x, y, z + 0.06, p) is not None, (x, y, pdeg)

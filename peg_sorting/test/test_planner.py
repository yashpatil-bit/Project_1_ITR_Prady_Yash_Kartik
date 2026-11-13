import math

from peg_sorting.planning.kinematics import ArmGeometry, ManipulatorXKinematics
from peg_sorting.planning.path_planner import CircleObstacle, plan_transport
from peg_sorting.planning.pick_place_planner import (PegInfo, PickPlacePlanner, SocketInfo,
                                                     predicted_misalignment)
from peg_sorting.planning.trajectory import Limits
from peg_sorting.utils.config import load_yaml


def make():
    r = load_yaml('robot.yaml')
    k = ManipulatorXKinematics(ArmGeometry.from_config(r))
    task = load_yaml('task.yaml')
    task['pitch'] = {'preferred_deg': 90, 'min_deg': 30}   # planner maths over the whole plate
    task['grasp']['grasp_depth'] = 0.03
    return k, PickPlacePlanner(k, task, Limits.from_config(r['limits']))


def test_plan_keeps_constant_pitch_and_reaches_socket():
    k, pl = make()
    peg = PegInfo('blue', 'square', 0.12, -0.16, 0.10, 0.10, 0.3, 0.059)
    sock = SocketInfo('square', 0.24, 0.055, 0.005, math.radians(45), 0.067)
    plan = pl.plan(peg, sock, load_yaml('robot.yaml')['poses']['observe'], [])
    trajs = [s for s in plan.steps if s.kind == 'traj']
    for s in trajs[1:]:  # all Cartesian segments keep the pitch (peg stays vertical)
        for q in s.traj.q:
            assert abs(k.fk(q)[3] - plan.pitch) < 1e-6
    ins = next(s for s in plan.steps if s.label.startswith('insert'))
    assert ins.guarded
    x, y, z, _ = k.fk(ins.traj.q[-1])
    _, (dx, dy) = pl.grasp_offsets(peg, sock)          # hand-tuned socket_offset_m from task.yaml
    assert math.hypot(x - sock.x - dx, y - sock.y - dy) < 1e-6
    # peg bottom ends just above the table
    h = pl.heights(peg, sock)
    assert abs((z - h['peg_bottom_below_tool']) - float(pl.cfg['place']['bottom_clearance'])) < 1e-6
    gripper = [s.gripper for s in plan.steps if s.kind == 'gripper']
    assert gripper == ['open', 'close', 'release']


def test_transport_avoids_obstacle():
    obs = [CircleObstacle(0.2, 0.0, 0.05)]
    wps, info = plan_transport((0.2, -0.15), (0.2, 0.15), obs)
    assert info['collision_free'] and info['route'] != 'direct'


def test_misalignment_model():
    # square socket aligned with the radial direction -> no error whatever the peg's start yaw
    peg = PegInfo('blue', 'square', 0.15, 0.0, 0.1, 0.1, math.radians(17), 0.06)
    sock = SocketInfo('square', 0.25, 0.0, 0.005, math.radians(45), 0.067)
    assert abs(predicted_misalignment(peg, sock)) < 1e-9
    sock2 = SocketInfo('square', 0.25, 0.08, 0.005, math.radians(45), 0.067)
    assert abs(math.degrees(predicted_misalignment(peg, sock2)) + math.degrees(math.atan2(0.08, 0.25))) < 1e-6


def test_assign_by_shape():
    k, pl = make()
    pegs = [PegInfo('green', 'circle', 0.1, 0.1, 0.1, 0.1), PegInfo('orange', 'hexagon', 0.1, -0.1, 0.1, 0.1)]
    socks = [SocketInfo('hexagon', 0.25, 0, 0.005), SocketInfo('circle', 0.25, 0.1, 0.005)]
    pairs = pl.assign(pegs, socks)
    assert {(p.color, s.shape) for p, s in pairs} == {('green', 'circle'), ('orange', 'hexagon')}


def test_board_placement_search_finds_aligned_pose():
    from peg_sorting.planning.board_placement import board_model, evaluate, search
    _, pl = make()
    # the sim board straight across in front of the robot (as in the lab photo)
    socks = [SocketInfo('circle', 0.24, -0.165, 0.005, 0.0), SocketInfo('hexagon', 0.24, -0.055, 0.005, math.radians(90)),
             SocketInfo('square', 0.24, 0.055, 0.005, math.radians(135)),
             SocketInfo('triangle', 0.24, 0.165, 0.005, math.radians(0))]
    model = board_model(0.24, 0.0, math.radians(90), socks)
    worst_now, _, _ = evaluate(model, 0.24, 0.0, math.radians(90))
    assert worst_now > 10
    cands = search(model, pl, current=(0.24, 0.0, math.radians(90)), step=0.02, yaw_step_deg=2.0, tolerance_deg=2.0)
    assert cands and cands[0][0] <= 2.0


def test_triangle_rotation_hint():
    from peg_sorting.planning.pick_place_planner import peg_rotation_hint
    peg = PegInfo('yellow', 'triangle', 0.06, 0.18, 0.1, 0.1, math.radians(10), 0.058)
    sock = SocketInfo('triangle', 0.1716, 0.1419, 0.005, math.radians(9.8))
    assert abs(math.degrees(predicted_misalignment(peg, sock))) > 50
    h = peg_rotation_hint(peg, sock)
    assert h is not None
    peg.yaw += h
    assert abs(math.degrees(predicted_misalignment(peg, sock))) < 2


def test_planar_fit_recovers_rotation_and_shift():
    import numpy as np
    from peg_sorting.utils.transforms import fit_planar
    src = np.array([[0.10, 0.03], [0.05, 0.10], [0.09, -0.03], [0.09, -0.11]])
    th = math.radians(1.5)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    dst = src @ R.T + np.array([-0.006, 0.004])
    theta, t, res = fit_planar(src, dst)
    assert abs(theta - th) < 1e-9 and np.allclose(t, [-0.006, 0.004]) and np.abs(res).max() < 1e-12


def test_affine_and_similarity_fit_recover_arm_scale_error():
    import numpy as np
    from peg_sorting.utils.transforms import fit_affine_2d, fit_similarity_2d
    cam = np.array([[0.20, 0.14], [0.24, 0.0], [0.08, 0.08], [0.22, -0.12], [0.08, -0.14], [0.16, 0.04]])
    rob = 0.92 * cam + np.array([0.004, -0.003])          # arm reaches 8% less than the model
    s, th, t, res = fit_similarity_2d(cam, rob)
    assert abs(s - 0.92) < 1e-9 and abs(th) < 1e-9 and np.abs(res).max() < 1e-12
    A, b, res = fit_affine_2d(cam, rob)
    assert np.allclose(A, 0.92 * np.eye(2)) and np.allclose(b, [0.004, -0.003]) and np.abs(res).max() < 1e-12

"""Workspace calibration: the robot places one peg at several spots and the camera measures
where it really stands.  Fits the map "camera x-y -> robot command x-y" and saves it to
camera_calib.yaml -> planar_correction (used by the vision node).  It absorbs BOTH small arm
errors (link lengths / joint zero offsets differing from the model) and camera errors.

    ros2 run peg_sorting calibrate_workspace            (driver + vision running)
    ros2 run peg_sorting calibrate_workspace --points 8

Before starting: take the socket BOARD and all pegs except the GREEN cylinder off the plate
(the marker can stay).  Keep a finger on SPACE in the teleop/estop terminal.

1. The arm goes to a hand-over spot with the gripper open, fingers ~3 cm below the peg-top level.
2. Stand the green peg UPRIGHT on the plate between the open fingers (centred), press ENTER.
3. The robot then, for each target spot: picks the peg, places it at the spot, goes to the
   observe pose and the camera measures it.  Hands off the plate while it runs.
"""
import argparse
import math
import time

import numpy as np
import rclpy

from ..control.robot_interface import EmergencyStop, RobotInterface
from ..planning.pick_place_planner import PegInfo, PickPlacePlanner, PlanningError, SocketInfo
from ..planning.trajectory import cartesian_path
from ..utils.config import load_yaml, update_yaml
from ..utils.keyboard import KeyReader
from ..utils.transforms import fit_affine_2d, fit_similarity_2d, planar_correction_from_config
from .check_calibration import safe_approach


def choose_targets(planner, peg, n, marker_xy, start_xy, q_ref, x_range=(0.08, 0.24), y_range=(-0.14, 0.14)):
    """Spread-out spots on the plate that the planner can reach (farthest-point sampling)."""
    cands = []
    for x in np.arange(x_range[0], x_range[1] + 1e-9, 0.02):
        for y in np.arange(y_range[0], y_range[1] + 1e-9, 0.02):
            r = math.hypot(x, y)
            if r < 0.10 or r > 0.26:
                continue
            if marker_xy is not None and math.hypot(x - marker_xy[0], y - marker_xy[1]) < 0.06:
                continue
            if math.hypot(x - start_xy[0], y - start_xy[1]) < 0.04:
                continue
            try:
                planner.plan(peg, SocketInfo('circle', float(x), float(y), planner.table_z), q_ref, [])
            except (PlanningError, ValueError):
                continue
            cands.append((float(x), float(y)))
    chosen = []
    while cands and len(chosen) < n:
        ref = chosen + [tuple(start_xy)]
        best = max(cands, key=lambda c: min(math.hypot(c[0] - r[0], c[1] - r[1]) for r in ref))
        chosen.append(best)
        cands.remove(best)
    return chosen


def run_steps(robot, steps, gr, skip_open=False):
    for st in steps:
        if st.kind == 'traj':
            res = robot.follow(st.traj, guarded=st.guarded)
            if not res and 'final error' not in res.message:
                return False, '%s: %s' % (st.label, res.message)
        elif st.kind == 'wait':
            time.sleep(st.seconds)
        elif st.kind == 'gripper':
            if st.gripper == 'open':
                if not skip_open:
                    robot.gripper(1.0)
            elif st.gripper == 'release':
                robot.gripper(float(gr.get('release_fraction', 0.6)))
            elif st.gripper == 'close':
                res = robot.gripper(0.0)
                if res is None or not (res.stalled or res.position > float(gr.get('grasp_detect_fraction', 0.08))):
                    return False, 'grasp failed (gripper closed on nothing)'
    return True, ''


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', type=int, default=8)
    ap.add_argument('--color', default='green')
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    robot = RobotInterface('calibrate_workspace')
    robot.wait_ready()
    task = load_yaml('task.yaml')
    vcfg = load_yaml('vision.yaml')
    calib = load_yaml('camera_calib.yaml')
    gr = robot.robot_cfg['gripper']
    table_z = float(vcfg.get('table_z', 0.0))
    planner = PickPlacePlanner(robot.kin, task, robot.limits, table_z=table_z)
    height = float(vcfg['pegs'].get('nominal_height', 0.10))
    m = (calib.get('aruco') or {}).get('marker_in_base') or {}
    marker_xy = (float(m['x']), float(m['y'])) if 'x' in m else None
    print(__doc__)
    obs = robot.robot_cfg['poses']['observe']
    cam_pts, rob_pts = [], []
    try:
        with KeyReader() as kr:
            robot.gripper(1.0)
            robot.move_joints(obs, 0.6)
            res = robot.get_scene(5, True)
            if res is not None and res.sockets:
                print('WARNING: the socket board is still on the plate - remove it (pegs would be placed on it).')
            others = [p.color for p in (res.pegs if res else []) if p.color != a.color]
            if others:
                print('WARNING: other pegs on the plate (%s) - remove them.' % ', '.join(others))
            print('Press ENTER to continue (ESC = quit).')
            while True:
                k = kr.get(0.2)
                if k in ('\n', '\r'):
                    break
                if k == 'ESC':
                    return
            # hand-over spot
            start = (0.12, -0.06)
            peg = PegInfo(a.color, 'circle', start[0], start[1], table_z + height, height, 0.0, 0.05)
            targets = choose_targets(planner, peg, a.points, marker_xy, start, obs)
            if len(targets) < 4:
                print('not enough reachable target spots (%d)' % len(targets))
                return
            print('target spots (mm): %s' % ', '.join('(%.0f, %.0f)' % (x * 1e3, y * 1e3) for x, y in targets))
            plan = planner.plan(peg, SocketInfo('circle', targets[0][0], targets[0][1], table_z), robot.joints(), [])
            h = planner.heights(peg, SocketInfo('circle', targets[0][0], targets[0][1], table_z))
            grasp = np.array([start[0], start[1], h['grasp']])
            above = grasp + np.array([0.0, 0.0, 0.10])
            q_hi = robot.kin.ik(*above, plan.pitch)
            if q_hi is None:
                print('hand-over spot not reachable at pitch %.0f deg' % math.degrees(plan.pitch))
                return
            safe_approach(robot, q_hi, above, grasp, plan.pitch)
            print('\nStand the %s peg upright on the plate between the open fingers (centred), then ENTER.'
                  % a.color)
            while True:
                k = kr.get(0.2)
                if k in ('\n', '\r'):
                    break
                if k == 'ESC':
                    robot.follow(cartesian_path(robot.kin, [tuple(grasp), tuple(above)], plan.pitch, robot.limits,
                                                q_seed=robot.joints()))
                    return
            # first move: we are already at the grasp pose -> start from 'close gripper'
            i_close = next(i for i, s in enumerate(plan.steps) if s.gripper == 'close')
            cur = start
            for i, tgt in enumerate(targets):
                if i == 0:
                    steps = plan.steps[i_close:]
                else:
                    peg = PegInfo(a.color, 'circle', cur[0], cur[1], table_z + height, height, 0.0, 0.05)
                    try:
                        steps = planner.plan(peg, SocketInfo('circle', tgt[0], tgt[1], table_z), robot.joints(),
                                             []).steps
                    except PlanningError as e:
                        print('skip (%.0f, %.0f): %s' % (tgt[0] * 1e3, tgt[1] * 1e3, e))
                        continue
                ok, why = run_steps(robot, steps, gr)
                if not ok:
                    print('stopped: %s' % why)
                    break
                cur = tgt
                robot.move_joints(obs, 0.8)
                time.sleep(0.5)
                sc = robot.get_scene(8, False)
                seen = [p for p in (sc.pegs if sc else []) if p.color == a.color]
                if not seen:
                    print('  point %d: peg not seen by the camera - skipped' % (i + 1))
                    continue
                p = seen[0]
                cam_pts.append([p.top_center.x, p.top_center.y])
                rob_pts.append([tgt[0], tgt[1]])
                print('  point %d: robot placed at (%.1f, %.1f) mm, camera sees (%.1f, %.1f) mm, top %.0f mm'
                      % (i + 1, tgt[0] * 1e3, tgt[1] * 1e3, p.top_center.x * 1e3, p.top_center.y * 1e3,
                         p.top_center.z * 1e3))
    except EmergencyStop:
        print('emergency stop')
    finally:
        try:
            if not robot.estopped:
                robot.move_joints(obs, 0.6)
        except Exception:  # noqa: BLE001
            pass

    if len(cam_pts) < 4:
        print('only %d points - nothing saved' % len(cam_pts))
        robot.shutdown()
        return
    cam, rob = np.array(cam_pts), np.array(rob_pts)
    scale, theta, t, res_s = fit_similarity_2d(cam, rob)
    A1, b1, res_a = fit_affine_2d(cam, rob)
    rms_s = float(np.sqrt(np.mean(np.sum(res_s ** 2, axis=1))))
    rms_a = float(np.sqrt(np.mean(np.sum(res_a ** 2, axis=1))))
    rms_0 = float(np.sqrt(np.mean(np.sum((cam - rob) ** 2, axis=1))))
    print('\nbefore correction: camera vs robot RMS %.1f mm' % (rms_0 * 1e3))
    print('similarity fit: scale %.3f, rotation %+.2f deg, shift (%+.1f, %+.1f) mm, RMS %.1f mm'
          % (scale, math.degrees(theta), t[0] * 1e3, t[1] * 1e3, rms_s * 1e3))
    print('affine fit: RMS %.1f mm' % (rms_a * 1e3))
    if rms_a > 0.006:
        print('residual RMS > 6 mm: the points do not agree - NOT saving. (peg tipped over? moved?)')
        robot.shutdown()
        return
    # the vision node already applied the old correction (A0, b0): compose
    A0, b0 = planar_correction_from_config(calib)
    A = A1 @ A0
    b = A1 @ b0 + b1
    path = update_yaml('camera_calib.yaml', {'planar_correction': {
        'A': [[float(v) for v in row] for row in A], 'b': [float(v) for v in b],
        'note': 'calibrate_workspace %s: %d points, residual RMS %.1f mm (similarity scale %.3f)' % (
            time.strftime('%Y-%m-%d %H:%M'), len(cam_pts), rms_a * 1e3, scale)}})
    print('saved %s - RESTART the vision node to use it.' % path)
    robot.shutdown()


if __name__ == '__main__':
    main()

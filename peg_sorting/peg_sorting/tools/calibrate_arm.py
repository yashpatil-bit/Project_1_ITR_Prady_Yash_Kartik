"""Arm (kinematic) calibration by touching the flat plate with the fingertips (driver running).

    ros2 run peg_sorting calibrate_arm

Why: our Manipulator X does not match the stock model exactly (it reached ~8 % less than
commanded and grasped too low).  Small zero-angle errors of joints 2-4 and a different finger
length explain that.  Every fingertip touch on the flat plate gives one exact equation
"fingertip height = plate height", so 8-10 touches at different reaches and gripper angles
determine:  joint 2/3/4 zero offsets, fingertip length, plate height.

For each of the suggested poses:
  - the gripper CLOSES and the arm moves (from above) to ~3 cm above the plate
  - press '-' until "step 2 mm", then 'f' until the closed fingertips JUST touch the plate
    (no pressing - a sheet of paper under the tips should just get pinned)
  - ENTER records it (z = skip this pose); the arm lifts and goes to the next one
At the end the fit is shown; 'y' saves it to robot.yaml, then RESTART the driver (and teleop,
vision, ...) so every program uses the corrected arm model.
"""
import math
import time

import numpy as np
import rclpy

from ..control.robot_interface import RobotInterface
from ..nodes.teleop_keyboard import Teleop
from ..planning.kinematics import ArmGeometry, ManipulatorXKinematics
from ..planning.trajectory import cartesian_path, joint_move
from ..utils.config import update_yaml
from ..utils.keyboard import KeyReader

# (x, y, pitch_deg): spread over reach and gripper angle, inside the plate
POSES = [(0.12, 0.00, 85), (0.16, 0.00, 70), (0.20, 0.00, 55), (0.24, 0.00, 40), (0.13, -0.08, 80),
         (0.20, -0.08, 50), (0.13, 0.06, 65), (0.22, 0.05, 45), (0.10, -0.04, 60), (0.17, 0.04, 85)]


def tip_z(geom, q, params):
    """Fingertip-end height for joints q (as reported now) under calibration params."""
    d2, d3, d4, tip, h0 = params
    g = ArmGeometry(**{k: getattr(geom, k) for k in ('d1', 'base_z_offset', 'l1_forward', 'l1_up', 'l2')},
                    tool_length=tip, lower=geom.lower, upper=geom.upper)
    x, y, z, _ = ManipulatorXKinematics(g).fk([q[0], q[1] + d2, q[2] + d3, q[3] + d4])
    return z - h0


def fit(geom, qs, tip0, prior_rad=math.radians(3.0), prior_tip=0.01):
    """Gauss-Newton least squares with weak priors (keeps the solution sane with few touches)."""
    p = np.array([0.0, 0.0, 0.0, tip0, 0.0])
    p_prior = p.copy()
    w_prior = np.array([0.001 / prior_rad] * 3 + [0.001 / prior_tip, 0.0])  # 1 mm of residual per prior sigma
    for _ in range(30):
        r = np.array([tip_z(geom, q, p) for q in qs] + list(w_prior * (p - p_prior)))
        J = np.zeros((len(r), 5))
        for j in range(5):
            dp = np.zeros(5)
            dp[j] = 1e-6
            r2 = np.array([tip_z(geom, q, p + dp) for q in qs] + list(w_prior * (p + dp - p_prior)))
            J[:, j] = (r2 - r) / 1e-6
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + step
        if np.max(np.abs(step)) < 1e-9:
            break
    res = np.array([tip_z(geom, q, p) for q in qs])
    return p, res


def main(args=None):
    rclpy.init(args=args)
    robot = RobotInterface('calibrate_arm', with_vision=False)
    robot.wait_ready(vision=False)
    cfg = robot.robot_cfg
    geom = ArmGeometry.from_config(cfg)
    tip_off = float(cfg['kinematics'].get('fingertip_offset', 0.015))
    tip0 = geom.tool_length + tip_off
    kin = robot.kin
    tele = Teleop(robot, use_vision=False)
    obs = cfg['poses']['observe']
    qs = []
    print(__doc__)
    robot.gripper(0.0)
    with KeyReader() as kr:
        for i, (x, y, pdeg) in enumerate(POSES):
            pitch = math.radians(pdeg)
            # tool point such that the (model) fingertips are 3 cm above the plate
            z_low = 0.03 + tip_off * math.sin(pitch)
            q_low = kin.ik(x, y, z_low, pitch)
            q_hi = kin.ik(x, y, z_low + 0.06, pitch)
            if q_low is None or q_hi is None:
                print('pose %d (%.2f, %.2f, %d deg) unreachable - skipped' % (i + 1, x, y, pdeg))
                continue
            robot.move_joints(obs, 0.6)
            robot.follow(joint_move(robot.joints(), q_hi, robot.limits, 0.6))
            robot.follow(cartesian_path(kin, [(x, y, z_low + 0.06), (x, y, z_low)], pitch, robot.limits,
                                        vmax=0.05, q_seed=q_hi))
            tele.sync_target()
            tele.step_i = 0  # 2 mm
            print('\npose %d/%d: gripper at %d deg.  f (2 mm) until the fingertips just touch, then ENTER '
                  '(z = skip, ESC = quit)' % (i + 1, len(POSES), pdeg))
            while rclpy.ok():
                k = kr.get(0.1)
                if k is None:
                    continue
                if k in ('\n', '\r'):
                    q = [float(v) for v in robot.joints()]
                    qs.append(q)
                    print('   recorded: model says fingertips at %+.1f mm' % (tip_z(geom, q, [0, 0, 0, tip0, 0]) * 1e3))
                    break
                if k == 'z':
                    break
                if k == 'ESC':
                    robot.move_joints(obs, 0.6)
                    robot.shutdown()
                    return
                tele.handle(k)
            # lift straight up before moving on
            q = robot.joints()
            x0, y0, z0, p0 = kin.fk(q)
            try:
                robot.follow(cartesian_path(kin, [(x0, y0, z0), (x0, y0, z0 + 0.06)], p0, robot.limits, q_seed=q))
            except ValueError:
                pass
        robot.move_joints(obs, 0.6)
        if len(qs) < 6:
            print('only %d touches - need at least 6. Nothing saved.' % len(qs))
            robot.shutdown()
            return
        before = np.array([tip_z(geom, q, [0, 0, 0, tip0, 0]) for q in qs])
        p, res = fit(geom, qs, tip0)
        d2, d3, d4, tip, h0 = p
        print('\nfingertip height errors with the current model (mm): %s  RMS %.1f'
              % (np.round(before * 1e3, 1).tolist(), math.sqrt(np.mean(before ** 2)) * 1e3))
        print('after calibration (mm): %s  RMS %.1f' % (np.round(res * 1e3, 1).tolist(), math.sqrt(np.mean(res ** 2)) * 1e3))
        print('joint offsets: joint2 %+.2f deg, joint3 %+.2f deg, joint4 %+.2f deg; fingertip length %.1f mm; '
              'plate at %+.1f mm' % (math.degrees(d2), math.degrees(d3), math.degrees(d4), tip * 1e3, h0 * 1e3))
        if max(abs(d2), abs(d3), abs(d4)) > math.radians(15) or not 0.09 < tip < 0.20:
            print('implausible result - not saving (check that the fingertips really touched each time)')
            robot.shutdown()
            return
        print("\nSave to robot.yaml?  y = yes, anything else = no")
        k = None
        while k is None:
            k = kr.get(0.2)
        if k != 'y':
            print('not saved')
            robot.shutdown()
            return
    old = [float(v) for v in cfg['dynamixel']['joint_offsets_rad']]
    # reported angle = sign*(tick-2048)*k - offset ; the true angle is reported + d -> new offset = old - d
    new = [old[0], old[1] - d2, old[2] - d3, old[3] - d4]
    # plain Python floats: yaml cannot write numpy scalars
    path = update_yaml('robot.yaml', {
        'dynamixel': {'joint_offsets_rad': [round(float(v), 5) for v in new]},
        'kinematics': {'tool_length': round(float(tip - tip_off), 4), 'base_z_offset': round(float(-h0), 4)}})
    print('saved %s (%s).  RESTART the driver, vision and teleop now.' % (path, time.strftime('%H:%M')))
    robot.shutdown()


if __name__ == '__main__':
    main()

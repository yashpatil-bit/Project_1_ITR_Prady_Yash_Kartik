"""Refine the camera -> robot calibration by grasping the pegs (driver + vision running).

    ros2 run peg_sorting calibrate_handeye

For every detected peg:
  1. the arm opens the gripper and moves (safely, from above) until the fingertips are
     ~2.5 cm above the peg, where the camera says it is
  2. YOU lower it with 'f' (press '-' first for 2 mm steps) until the open fingers are
     around the peg, about 3 cm below its top; centre it front/back with 'w'/'s' by eye
  3. 'c' closes the gripper: the jaws centre the peg between them
  4. ENTER records "camera position" vs "robot position"; the gripper opens and the arm
     goes straight up and back to the observe pose
After the last peg: 'n' = another round (move the pegs to new spots first, spread over the
workspace - more spread = better), 'k' = compute and save, ESC = quit without saving.

Only the horizontal part is corrected (rotation about the vertical axis + x/y shift): the
camera's tilt and height come from the ArUco + table-plane calibration and are already
checked by the measured peg heights (~100 mm).
"""
import math
import time

import numpy as np
import rclpy

from ..control.robot_interface import RobotInterface
from ..nodes.teleop_keyboard import Teleop
from ..nodes.vision_node import load_extrinsics
from ..utils.config import load_yaml, update_yaml
from ..utils.keyboard import KeyReader
from ..utils.transforms import T_to_dict, fit_planar, make_T, rot_z
from .check_calibration import safe_approach, safe_retreat


def main(args=None):
    rclpy.init(args=args)
    robot = RobotInterface('calibrate_handeye')
    robot.wait_ready()
    calib = load_yaml('camera_calib.yaml')
    T_used = load_extrinsics(calib)            # what the vision node currently uses
    tip = float(robot.robot_cfg['kinematics'].get('fingertip_offset', 0.015))
    tele = Teleop(robot, use_vision=True)
    vis_pts, rob_pts = [], []
    print(__doc__)
    with KeyReader() as kr:
        while rclpy.ok():
            robot.gripper(1.0)
            robot.move_joints(robot.robot_cfg['poses']['observe'], 0.7)
            res = robot.get_scene(10, False)
            if res is None or not res.pegs:
                print('no pegs detected')
                break
            for m in res.pegs:
                target = np.array([m.top_center.x, m.top_center.y, m.top_center.z + 0.025 + tip])
                above = target + np.array([0.0, 0.0, 0.06])
                q_hi, pitch = robot.kin.ik_search_pitch(*above)
                if q_hi is None or robot.kin.ik(*target, pitch) is None:
                    print('%s: out of reach, skipped' % m.color)
                    continue
                print('\n%s %s: lower with f (2 mm steps: press - first) until the fingers are around the '
                      'peg ~3 cm below its top, centre with w/s, then c, then ENTER.  (z = skip this peg)'
                      % (m.color, m.shape))
                safe_approach(robot, q_hi, above, target, pitch)
                tele.sync_target()
                tele.step_i = 1  # 5 mm
                while rclpy.ok():
                    k = kr.get(0.1)
                    if k is None:
                        continue
                    if k in ('\n', '\r'):
                        x, y, _, _ = robot.kin.fk(robot.joints())
                        vis_pts.append([m.top_center.x, m.top_center.y])
                        rob_pts.append([x, y])
                        print('   %s: camera (%.1f, %.1f) mm  robot (%.1f, %.1f) mm  -> diff (%+.1f, %+.1f) mm'
                              % (m.color, m.top_center.x * 1e3, m.top_center.y * 1e3, x * 1e3, y * 1e3,
                                 (x - m.top_center.x) * 1e3, (y - m.top_center.y) * 1e3))
                        break
                    if k == 'z':
                        print('   skipped')
                        break
                    if k == 'ESC':
                        robot.gripper(1.0)
                        safe_retreat(robot)
                        robot.shutdown()
                        return
                    tele.handle(k)
                robot.gripper(1.0)
                time.sleep(0.3)
                safe_retreat(robot)
            print('\n%d points so far.  n = another round (move the pegs to new spots first), '
                  'k = compute and save, ESC = quit' % len(vis_pts))
            k = None
            while k not in ('n', 'k', 'ESC'):
                k = kr.get(0.2)
            if k == 'ESC':
                robot.shutdown()
                return
            if k == 'k':
                break
    if len(vis_pts) < 3:
        print('need at least 3 points (have %d) - nothing saved' % len(vis_pts))
        robot.shutdown()
        return
    theta, t, res = fit_planar(vis_pts, rob_pts)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    print('\nresiduals after the fit (mm):')
    for r in res:
        print('   (%+.1f, %+.1f)' % tuple(r * 1e3))
    print('rotation %+.2f deg, shift (%+.1f, %+.1f) mm, RMS %.1f mm' % (math.degrees(theta), t[0] * 1e3, t[1] * 1e3,
                                                                     rms * 1e3))
    if rms > 0.008:
        print('RMS > 8 mm: the points disagree - not saving.  Re-check (pegs moved? fingers not around the peg?)')
        robot.shutdown()
        return
    # new extrinsics: apply the planar correction in the base frame; the correction offset is folded in
    T_new = make_T(rot_z(theta), [t[0], t[1], 0.0]) @ T_used
    p = update_yaml('camera_calib.yaml', {
        'calibrated': True, 'method': 'aruco + grasp refinement (%d points, rms %.1f mm)' % (len(vis_pts), rms * 1e3),
        'date': time.strftime('%Y-%m-%d %H:%M'), 'T_base_cam': T_to_dict(T_new),
        'xyz_correction_m': [0.0, 0.0, 0.0]})
    print('saved %s - restart the vision node to use it.' % p)
    robot.move_joints(robot.robot_cfg['poses']['observe'], 0.7)
    robot.shutdown()


if __name__ == '__main__':
    main()

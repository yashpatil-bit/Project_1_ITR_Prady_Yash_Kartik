"""Accurate camera -> base calibration from touch points (driver + vision running).

    ros2 run peg_sorting calibrate_handeye

For N >= 4 different spots (spread over the whole workspace, include the board area):
  1. put ONE peg (e.g. the green cylinder) on the spot, arm in the observe pose -> ENTER
     (the vision node measures the peg's top-centre in camera coordinates)
  2. jog the gripper onto the peg with the teleop keys (w/s/a/d/r/f, t/g), close it
     with 'c' so the jaws centre it, check the fingers sit grasp_depth below the top
     -> ENTER (the robot's forward kinematics gives the same point in base coordinates)
  3. 'o' to open, the arm goes back to observe; move the peg; repeat.
Press 'k' when you have enough points to compute and save the calibration.
It uses the same tool point as the grasping, so tool-length errors are absorbed.
"""
import time

import numpy as np
import rclpy

from ..control.robot_interface import RobotInterface
from ..nodes.teleop_keyboard import Teleop
from ..utils.config import load_yaml, update_yaml
from ..utils.keyboard import KeyReader
from ..utils.transforms import T_to_dict, invert_T, kabsch, transform_points
from ..nodes.vision_node import load_extrinsics


def main(args=None):
    rclpy.init(args=args)
    robot = RobotInterface('calibrate_handeye')
    robot.wait_ready()
    calib = load_yaml('camera_calib.yaml')
    T_used = load_extrinsics(calib)         # what the vision node currently uses
    T_cam_base_used = invert_T(T_used)
    task = load_yaml('task.yaml')
    grasp_depth = float(task['grasp']['grasp_depth'])
    tele = Teleop(robot, use_vision=True)
    tele.sync_target()
    cam_pts, base_pts = [], []
    pending = None
    print(__doc__)
    robot.move_joints(robot.robot_cfg['poses']['observe'], 0.7)
    tele.sync_target()
    print('\n>> Place the peg, then ENTER to measure it with the camera.')
    with KeyReader() as kr:
        while rclpy.ok():
            k = kr.get(0.1)
            if k is None:
                continue
            if k in ('\n', '\r'):
                if pending is None:
                    res = robot.get_scene(10, False)
                    if res is None or len(res.pegs) != 1:
                        print('   need exactly ONE peg visible (got %s)' % (None if res is None else len(res.pegs)))
                        continue
                    m = res.pegs[0]
                    p_base_vis = np.array([m.top_center.x, m.top_center.y, m.top_center.z])
                    pending = transform_points(T_cam_base_used, p_base_vis[None])[0]
                    print('   camera: peg top at %s (base, current calib: %s). Now jog onto it, c, ENTER.' % (
                        np.round(pending, 4), np.round(p_base_vis, 4)))
                else:
                    x, y, z, _ = robot.kin.fk(robot.joints())
                    p_robot = np.array([x, y, z + grasp_depth])
                    cam_pts.append(pending)
                    base_pts.append(p_robot)
                    pending = None
                    print('   point %d recorded: robot %s' % (len(cam_pts), np.round(p_robot, 4)))
                    robot.gripper(1.0)
                    tele.jog(dz=0.06)
                    time.sleep(1.0)
                    robot.move_joints(robot.robot_cfg['poses']['observe'], 0.7)
                    tele.sync_target()
                    print('>> Move the peg to a new spot and press ENTER (or k to finish).')
            elif k == 'k':
                if len(cam_pts) < 4:
                    print('   need at least 4 points (have %d)' % len(cam_pts))
                    continue
                T = kabsch(np.array(cam_pts), np.array(base_pts))
                res = transform_points(T, np.array(cam_pts)) - np.array(base_pts)
                rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
                print('   residuals (mm):\n%s\n   RMS %.1f mm' % (np.round(res * 1e3, 1), rms * 1e3))
                if rms > 0.01:
                    print('   RMS > 10 mm - check the points; saving anyway')
                p = update_yaml('camera_calib.yaml', {
                    'calibrated': True, 'method': 'handeye (%d points, rms %.1f mm)' % (len(cam_pts), rms * 1e3),
                    'date': time.strftime('%Y-%m-%d %H:%M'), 'T_base_cam': T_to_dict(T),
                    'xyz_correction_m': [0.0, 0.0, 0.0]})
                print('   saved %s - restart the vision node to use it.' % p)
                break
            elif k == 'ESC':
                break
            else:
                tele.handle(k)
    robot.shutdown()


if __name__ == '__main__':
    main()

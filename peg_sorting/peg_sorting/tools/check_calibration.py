"""Check (and fine-tune) the camera calibration with the real arm (driver + vision running).

    ros2 run peg_sorting check_calibration

The gripper opens and hovers with its FINGERTIPS ~2.5 cm above the top-centre of every
detected peg.  Look from the side and the front: the open fingers should be centred on the peg.  If it is consistently
off, jog it (teleop keys) until it is centred and press ENTER; the average
correction over all pegs is written to camera_calib.yaml -> xyz_correction_m.
ENTER without jogging = "it was fine".  ESC = quit without saving.
"""
import math

import numpy as np
import rclpy

from ..control.robot_interface import RobotInterface
from ..nodes.teleop_keyboard import Teleop
from ..utils.config import load_yaml, update_yaml
from ..utils.keyboard import KeyReader


def main(args=None):
    rclpy.init(args=args)
    robot = RobotInterface('check_calibration')
    robot.wait_ready()
    tele = Teleop(robot, use_vision=True)
    robot.move_joints(robot.robot_cfg['poses']['observe'], 0.7)
    res = robot.get_scene(10, False)
    if res is None or not res.pegs:
        print('no pegs detected')
        robot.shutdown()
        return
    corrections = []
    print(__doc__)
    tip = float(robot.robot_cfg['kinematics'].get('fingertip_offset', 0.015))  # tool point -> fingertip ends
    hover = 0.025
    robot.gripper(1.0)
    with KeyReader() as kr:
        for m in res.pegs:
            # tool point high enough that the fingertips stay `hover` above the top, whatever the pitch
            target = np.array([m.top_center.x, m.top_center.y, m.top_center.z + hover + tip])
            q, pitch = robot.kin.ik_search_pitch(*target)
            if q is None:
                print('%s peg out of reach, skipped' % m.color)
                continue
            print('\n%s %s (top at %.0f mm): fingertips ~%.0f mm above it, pitch %.0f deg. '
                  'Centre it (w/a/s/d, never f), then ENTER.' % (
                      m.color, m.shape, m.top_center.z * 1e3, hover * 1e3, math.degrees(pitch)))
            robot.move_joints(q, 0.6)
            tele.sync_target()
            done = False
            while rclpy.ok() and not done:
                k = kr.get(0.1)
                if k is None:
                    continue
                if k in ('\n', '\r'):
                    x, y, z, _ = robot.kin.fk(robot.joints())
                    corr = np.array([x, y, z]) - target
                    corr[2] = 0.0   # only horizontal offsets are judged here
                    corrections.append(corr)
                    print('   correction %s mm' % np.round(corr * 1e3, 1))
                    tele.jog(dz=0.05)
                    done = True
                elif k == 'ESC':
                    robot.shutdown()
                    return
                else:
                    tele.handle(k)
    if corrections:
        c = np.mean(corrections, axis=0)
        old = np.array(load_yaml('camera_calib.yaml').get('xyz_correction_m') or [0, 0, 0], dtype=float)
        new = old + c
        print('\nmean correction %s mm -> xyz_correction_m = %s' % (np.round(c * 1e3, 1), np.round(new, 4).tolist()))
        p = update_yaml('camera_calib.yaml', {'xyz_correction_m': [float(v) for v in new]})
        print('saved %s - restart the vision node to use it.' % p)
    robot.move_joints(robot.robot_cfg['poses']['observe'], 0.7)
    robot.shutdown()


if __name__ == '__main__':
    main()

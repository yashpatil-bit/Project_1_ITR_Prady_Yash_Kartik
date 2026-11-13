"""Check (and fine-tune) the camera calibration with the real arm (driver + vision running).

    ros2 run peg_sorting check_calibration

The gripper opens and hovers with its FINGERTIPS ~2.5 cm above the top-centre of every
detected peg.  Look from the side and the front: the open fingers should be centred on the peg.  If it is consistently
off, jog it (teleop keys) until it is centred and press ENTER; the average
correction over all pegs is written to camera_calib.yaml -> xyz_correction_m.
ENTER without jogging = "it was fine".  ESC = quit without saving.

    ros2 run peg_sorting check_calibration --per-color             (all pegs)
    ros2 run peg_sorting check_calibration --per-color --color blue
    ros2 run peg_sorting check_calibration --reset-colors          (delete them)

--per-color: every peg's correction is saved for THAT colour only (camera_calib.yaml ->
color_offset_m).  Use it when one peg is off and the others are fine: the camera sees
each colour/shape a bit differently (e.g. the blue square was 20 mm off while the
workspace calibration was done with the green cylinder).  Do it with the pegs standing
roughly where they will be during the run.
"""
import argparse
import math
import sys

import numpy as np
import rclpy

from ..control.robot_interface import RobotInterface
from ..planning.trajectory import cartesian_path, joint_move
from ..nodes.teleop_keyboard import Teleop
from ..utils.config import load_yaml, update_yaml
from ..utils.keyboard import KeyReader


def safe_retreat(robot, dz=0.06):
    """Straight up from wherever the tool is (and WAIT), then back to the observe pose."""
    q = robot.joints()
    x, y, z, pitch = robot.kin.fk(q)
    try:
        robot.follow(cartesian_path(robot.kin, [(x, y, z), (x, y, z + dz)], pitch, robot.limits, q_seed=q))
    except ValueError:
        pass  # cannot go straight up from here; the observe move below starts upward anyway
    robot.move_joints(robot.robot_cfg['poses']['observe'], 0.6)


def safe_approach(robot, q_hi, above, target, pitch):
    """Joint move to a point well above the peg (only if that swing stays high), then straight down."""
    traj = joint_move(robot.joints(), q_hi, robot.limits, 0.6)
    if min(robot.kin.fk(q)[2] for q in traj.q) < target[2] + 0.02:
        # the swing would come low: rise first at the current pose, then go
        safe_retreat(robot, dz=0.04)
        traj = joint_move(robot.joints(), q_hi, robot.limits, 0.6)
    robot.follow(traj)
    robot.follow(cartesian_path(robot.kin, [tuple(above), tuple(target)], pitch, robot.limits,
                                vmax=0.05, q_seed=q_hi))


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-color', action='store_true', help='save a separate correction for each colour')
    ap.add_argument('--color', default=None, help='only this peg colour (e.g. blue)')
    ap.add_argument('--reset-colors', action='store_true', help='delete all per-colour offsets and exit')
    opt, _ = ap.parse_known_args(sys.argv[1:])
    if opt.reset_colors:
        print('per-colour offsets deleted: %s - restart the vision node.'
              % update_yaml('camera_calib.yaml', {'color_offset_m': None}))
        return
    rclpy.init(args=args)
    robot = RobotInterface('check_calibration')
    robot.wait_ready()
    tele = Teleop(robot, use_vision=True)
    robot.move_joints(robot.robot_cfg['poses']['observe'], 0.7)
    res = robot.get_scene(10, False)
    pegs = [m for m in (res.pegs if res else []) if opt.color in (None, m.color)]
    if not pegs:
        print('no %speg detected' % ('%s ' % opt.color if opt.color else ''))
        robot.shutdown()
        return
    corrections = []
    per_color = {}
    print(__doc__)
    tip = float(robot.robot_cfg['kinematics'].get('fingertip_offset', 0.015))  # tool point -> fingertip ends
    hover = 0.025
    robot.gripper(1.0)
    with KeyReader() as kr:
        for m in pegs:
            # tool point high enough that the fingertips stay `hover` above the top, whatever the pitch
            target = np.array([m.top_center.x, m.top_center.y, m.top_center.z + hover + tip])
            above = target + np.array([0.0, 0.0, 0.06])
            q_hi, pitch = robot.kin.ik_search_pitch(*above)
            if q_hi is None or robot.kin.ik(*target, pitch) is None:
                print('%s peg out of reach, skipped' % m.color)
                continue
            print('\n%s %s (top at %.0f mm): fingertips ~%.0f mm above it, pitch %.0f deg. '
                  'Centre it (w/a/s/d, never f), then ENTER.' % (
                      m.color, m.shape, m.top_center.z * 1e3, hover * 1e3, math.degrees(pitch)))
            safe_approach(robot, q_hi, above, target, pitch)
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
                    per_color[m.color] = corr
                    print('   correction %s mm' % np.round(corr * 1e3, 1))
                    safe_retreat(robot)
                    done = True
                elif k == 'ESC':
                    safe_retreat(robot)
                    robot.shutdown()
                    return
                else:
                    tele.handle(k)
    if per_color and opt.per_color:
        old = load_yaml('camera_calib.yaml').get('color_offset_m') or {}
        new = {}
        for color, corr in per_color.items():
            v = np.array(old.get(color) or [0.0, 0.0], dtype=float) + corr[:2]
            new[color] = [round(float(x), 4) for x in v]
            print('%s: correction %s mm -> color_offset_m.%s = %s mm' % (
                color, np.round(corr[:2] * 1e3, 1).tolist(), color, np.round(v * 1e3, 1).tolist()))
        p = update_yaml('camera_calib.yaml', {'color_offset_m': new})
        print('saved %s - restart the vision node to use it.' % p)
    elif corrections:
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

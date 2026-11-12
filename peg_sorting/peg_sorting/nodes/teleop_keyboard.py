"""Keyboard teleoperation of the Manipulator X (driver must be running; vision optional).

    ros2 run peg_sorting teleop_keyboard
"""
import math
import time

import rclpy

from ..control.robot_interface import EmergencyStop, RobotInterface
from ..planning.pick_place_planner import PegInfo, PickPlacePlanner, SocketInfo
from ..planning.trajectory import joint_move
from ..utils.config import load_yaml, update_yaml
from ..utils.keyboard import KeyReader
from ..utils.run_logger import RunLogger

HELP = r"""
 ------------------------- Manipulator X keyboard teleop -------------------------
  MOVE TOOL (robot base frame)          GRIPPER / POSES
   w / s  : forward / back  (+x / -x)     o : open           c : close
   a / d  : left / right    (+y / -y)     h : home pose      b : observe pose
   r / f  : up / down       (+z / -z)     P : save current joints as the 'observe' pose
   t / g  : tool pitch steeper / flatter  (keep pitch constant while carrying a peg!)
   T      : point the gripper straight down (pitch 90 deg)
   q / e  : rotate base left / right
   + / -  : step size            i : print joints + tool pose
  VISION ASSIST (vision node running)
   v      : detect pegs & sockets, list them with numbers
   1..4   : go above peg n (pre-grasp height)
   5..8   : go above the socket matching peg n-4 (insertion start height, same pitch)
  TIMING LOG
   ENTER  : start the run timer   y : last peg placed OK   n : last peg failed
  SAFETY
   SPACE  : EMERGENCY STOP (hold)  x : release emergency stop      ESC : quit
 ---------------------------------------------------------------------------------
"""

STEPS = [0.002, 0.005, 0.01, 0.02, 0.04]


class Teleop:
    def __init__(self, robot: RobotInterface, use_vision):
        self.r = robot
        self.kin = robot.kin
        self.cfg = robot.robot_cfg
        self.task = load_yaml('task.yaml')
        self.use_vision = use_vision
        self.planner = PickPlacePlanner(self.kin, self.task, robot.limits,
                                        table_z=float(load_yaml('vision.yaml').get('table_z', 0.0)))
        self.step_i = 2
        self.pegs, self.sockets = [], {}
        self.last_peg = None
        self.logger = RunLogger(self.task['execution'].get('log_dir', '~/peg_sorting_logs'), 'teleop')
        self.t_peg_start = None
        self.target = None  # (x, y, z, pitch)

    def sync_target(self):
        self.target = list(self.kin.fk(self.r.joints()))

    def go_cartesian(self, x, y, z, pitch, quick=True):
        q = self.kin.ik(x, y, z, pitch)
        if q is None:
            print('  unreachable: (%.3f, %.3f, %.3f) pitch %.0f deg%s' % (
                x, y, z, math.degrees(pitch),
                '' if self.kin.within_limits(self.r.joints()) else
                '  <- the arm is outside its normal range: press h (home) first'))
            return False
        self.target = [x, y, z, pitch]
        traj = joint_move(self.r.joints(), q, self.r.limits, 1.0 if quick else 0.7)
        res = self.r.follow(traj, wait=not quick)
        if not quick and not res:
            print('  move failed: %s' % res.message)
        return bool(res) if not quick else True

    def jog(self, dx=0.0, dy=0.0, dz=0.0, dpitch=0.0, dyaw=0.0):
        if self.target is None:
            self.sync_target()
        x, y, z, p = self.target
        if dyaw:
            r, th = math.hypot(x, y), math.atan2(y, x) + dyaw
            x, y = r * math.cos(th), r * math.sin(th)
        self.go_cartesian(x + dx, y + dy, z + dz, p + dpitch)

    def detect(self):
        if not self.use_vision:
            print('  vision not available')
            return
        res = self.r.get_scene(5, True)
        if res is None or not res.success:
            print('  detection failed')
            return
        self.pegs = [PegInfo(m.color, m.shape, m.top_center.x, m.top_center.y, m.top_center.z, m.height, m.yaw,
                             m.width) for m in res.pegs]
        for m in res.sockets:
            self.sockets[m.shape] = SocketInfo(m.shape, m.center.x, m.center.y, m.center.z, m.yaw, m.width)
        for i, p in enumerate(self.pegs):
            print('  [%d] %-7s %-8s at (%.3f, %.3f) top %.3f' % (i + 1, p.color, p.shape, p.x, p.y, p.z_top))
        for s in self.sockets.values():
            print('      socket %-8s at (%.3f, %.3f)' % (s.shape, s.x, s.y))

    def above_peg(self, i):
        if i >= len(self.pegs):
            print('  no peg %d (press v first)' % (i + 1))
            return
        p = self.pegs[i]
        self.last_peg = p
        self.t_peg_start = self.t_peg_start or time.time()
        z = p.z_top + float(self.task['grasp']['approach_clearance'])
        pitch = None
        s = self.sockets.get(p.shape)
        if s is not None:
            # one pitch that works at the peg AND at its socket, so the peg stays vertical
            h = self.planner.heights(p, s)
            pitch = self.kin.max_feasible_pitch(
                [(p.x, p.y, z), (p.x, p.y, h['grasp']), (p.x, p.y, h['carry']), (s.x, s.y, h['carry']),
                 (s.x, s.y, h['insert_start']), (s.x, s.y, h['place'])],
                lowest=math.radians(float(self.task['pitch']['min_deg'])))
        if pitch is None:
            q, pitch = self.kin.ik_search_pitch(p.x, p.y, z)
            if q is None:
                print('  peg out of reach')
                return
        print('  above %s %s, pitch %.0f deg - lower with f, close with c' % (p.color, p.shape, math.degrees(pitch)))
        self.go_cartesian(p.x, p.y, z, pitch, quick=False)

    def above_socket(self, i):
        if i >= len(self.pegs):
            print('  no peg %d' % (i + 1))
            return
        p = self.pegs[i]
        s = self.sockets.get(p.shape)
        if s is None:
            print('  no socket for %s' % p.shape)
            return
        h = self.planner.heights(p, s)
        if self.target is None:
            self.sync_target()
        print('  above the %s socket at insertion height - lower slowly with f, then o' % s.shape)
        self.go_cartesian(s.x, s.y, h['insert_start'], self.target[3], quick=False)

    def run(self):
        self.r.wait_ready(vision=False)
        self.use_vision = self.use_vision and self.r.scene_cli.wait_for_service(timeout_sec=1.0)
        print(HELP)
        print('vision assist: %s' % ('ON' if self.use_vision else 'OFF (vision node not running)'))
        self.sync_target()
        with KeyReader() as kr:
            if not kr.available:
                print('stdin is not a terminal - run teleop in its own terminal')
                return
            while rclpy.ok():
                k = kr.get(0.1)
                if k is None:
                    continue
                try:
                    if self.handle(k) is False:
                        break
                except EmergencyStop:
                    print('  EMERGENCY STOP active - press x to release')
                except Exception as e:  # noqa: BLE001
                    print('  error: %s' % e)
        s = self.logger.finish(len(self.pegs) or 4)
        print('teleop log: %s  (%d placed, %.1fs)' % (self.logger.csv_path, s['pegs_placed'], s['total_time_s']))

    def handle(self, k):
        st = STEPS[self.step_i]
        da = math.radians(3 if st < 0.01 else 6)
        moves = {'w': dict(dx=st), 's': dict(dx=-st), 'a': dict(dy=st), 'd': dict(dy=-st),
                 'r': dict(dz=st), 'f': dict(dz=-st), 't': dict(dpitch=da), 'g': dict(dpitch=-da),
                 'q': dict(dyaw=da), 'e': dict(dyaw=-da),
                 'UP': dict(dx=st), 'DOWN': dict(dx=-st), 'LEFT': dict(dy=st), 'RIGHT': dict(dy=-st)}
        if k == ' ':
            self.r.emergency_stop()
            print('  !!! EMERGENCY STOP !!!  (x to release)')
        elif k == 'x':
            self.r.release_estop()
            time.sleep(0.2)
            self.sync_target()
            print('  emergency stop released')
        elif k in moves:
            self.jog(**moves[k])
        elif k == 'T':
            if self.target is None:
                self.sync_target()
            x, y, z, _ = self.target
            if self.go_cartesian(x, y, z, math.pi / 2, quick=False):
                print('  gripper pointing straight down')
        elif k in ('+', '='):
            self.step_i = min(self.step_i + 1, len(STEPS) - 1)
            print('  step %.0f mm' % (STEPS[self.step_i] * 1e3))
        elif k == '-':
            self.step_i = max(self.step_i - 1, 0)
            print('  step %.0f mm' % (STEPS[self.step_i] * 1e3))
        elif k == 'o':
            self.r.gripper(1.0, wait=False)
        elif k == 'c':
            res = self.r.gripper(0.0)
            if res is not None:
                print('  gripper %.2f %s' % (res.position, '(holding something)' if res.stalled else ''))
        elif k in ('h', 'b'):
            name = 'home' if k == 'h' else 'observe'
            print('  moving to %s pose ...' % name)
            res = self.r.move_joints(self.cfg['poses'][name], 0.5)
            print('  %s' % ('done' if res else 'FAILED: %s (see the driver terminal)' % res.message))
            self.sync_target()
        elif k == 'P':
            q = [round(float(v), 3) for v in self.r.joints()]
            path = update_yaml('robot.yaml', {'poses': {'observe': q}})
            self.cfg['poses']['observe'] = q
            print('  saved observe pose %s to %s' % (q, path))
        elif k == 'i':
            q = self.r.joints()
            x, y, z, p = self.kin.fk(q)
            print('  joints %s rad | tool (%.3f, %.3f, %.3f) pitch %.1f deg | gripper %s' % (
                [round(float(v), 3) for v in q], x, y, z, math.degrees(p), self.r.gripper_position()))
        elif k == 'v':
            self.detect()
        elif k in '1234':
            self.above_peg(int(k) - 1)
        elif k in '5678':
            self.above_socket(int(k) - 5)
        elif k in ('\n', '\r'):
            self.logger.start_run()
            self.t_peg_start = time.time()
            print('  run timer started')
        elif k in ('y', 'n'):
            p = self.last_peg
            self.logger.attempt(peg_color=p.color if p else '?', shape=p.shape if p else '?', attempt=1,
                                success=(k == 'y'), verified='operator', reason='' if k == 'y' else 'operator: failed',
                                t_start=self.t_peg_start or time.time(), t_end=time.time())
            print('  logged %s for %s' % ('SUCCESS' if k == 'y' else 'FAILURE', p.color if p else '?'))
            self.t_peg_start = time.time()
        elif k == 'ESC':
            return False
        return True


def main(args=None):
    rclpy.init(args=args)
    robot = RobotInterface('teleop_keyboard', with_vision=True)
    try:
        Teleop(robot, use_vision=True).run()
    except KeyboardInterrupt:
        pass
    finally:
        robot.shutdown()


if __name__ == '__main__':
    main()

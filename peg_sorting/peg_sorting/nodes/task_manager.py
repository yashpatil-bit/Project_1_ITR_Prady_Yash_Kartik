"""Fully autonomous peg sorting.

    ros2 run peg_sorting task_manager            (driver + vision must be running)
    ros2 run peg_sorting task_manager --ros-args -p confirm:=false -p speed_scale:=1.3

Keys while running:  SPACE = EMERGENCY STOP   q = abort after the current step
Loop:  observe -> detect pegs + sockets -> match by shape -> for each pair:
       plan -> pick (verify grasp) -> transport -> guarded insertion (retry
       with small offsets if blocked) -> release -> observe & verify -> retry
"""
import math
import sys
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory

from ..control.robot_interface import EmergencyStop, RobotInterface
from ..planning.pick_place_planner import (PegInfo, PickPlacePlanner, PlanningError, SocketInfo, jaw_blockers,
                                           open_jaw_clearance, peg_rotation_hint, predicted_misalignment)
from ..utils.config import load_yaml
from ..utils.keyboard import KeyReader
from ..utils.run_logger import RunLogger

GOAL_TOLERANCE_VIOLATED = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED


def peg_from_msg(m):
    return PegInfo(m.color, m.shape, m.top_center.x, m.top_center.y, m.top_center.z, m.height, m.yaw, m.width)


def socket_from_msg(m):
    return SocketInfo(m.shape, m.center.x, m.center.y, m.center.z, m.yaw, m.width)


class TaskManager:
    def __init__(self, robot: RobotInterface, params):
        self.r = robot
        self.p = params
        self.task = load_yaml('task.yaml')
        self.robot_cfg = robot.robot_cfg
        self.vision_cfg = load_yaml('vision.yaml')
        self.planner = PickPlacePlanner(robot.kin, self.task, robot.limits,
                                        table_z=float(self.vision_cfg.get('table_z', 0.0)))
        ex = self.task['execution']
        self.speed = float(params.get('speed_scale') or ex.get('speed_scale', 1.0))
        self.max_retries = int(ex.get('max_retries', 2))
        self.n_frames = int(ex.get('observe_num_frames', 5))
        self.verify = bool(ex.get('verify', True))
        self.verify_tol = float(ex.get('verify_tolerance', 0.025))
        self.offsets = [tuple(o) for o in ex.get('search_offsets_m', [[0, 0]])]
        self.gr = self.robot_cfg['gripper']
        self.release_fraction = float(self.task['place'].get('release_fraction',
                                                             self.gr.get('release_fraction', 1.0)))
        self.logger = RunLogger(ex.get('log_dir', '~/peg_sorting_logs'), 'autonomy')
        self.abort = False
        self.sockets = {}      # shape -> SocketInfo (cached; holes disappear once filled)
        self.placed = {}       # color -> SocketInfo
        self.true_height = {}  # color -> peg height measured standing on the table
        self.table_z = float(self.vision_cfg.get('table_z', 0.0))

    def yaw_tol(self, shape):
        """Allowed predicted yaw error (deg).  The square hole is the tightest: in the lab it
        went in at 7.6 deg once but sat on the hole at 6 deg + 6 mm, so it gets its own value."""
        al = self.task['alignment']
        return float(al.get('tolerance_deg_by_shape', {}).get(shape, al['tolerance_deg']))

    def log(self, msg):
        self.r.log.info(msg)

    # ------------------------------------------------------------ helpers
    def observe(self, with_sockets=True):
        q_obs = self.robot_cfg['poses']['observe']
        self.r.gripper(1.0, wait=False)
        self.r.move_joints(q_obs, self.speed)
        for attempt in range(3):
            res = self.r.get_scene(self.n_frames, with_sockets)
            if res is not None and res.success:
                pegs = [peg_from_msg(m) for m in res.pegs]
                socks = [socket_from_msg(m) for m in res.sockets]
                if with_sockets:
                    for s in socks:  # keep the most confident detection per shape
                        old = self.sockets.get(s.shape)
                        if old is None or math.hypot(old.x - s.x, old.y - s.y) < 0.03:
                            self.sockets[s.shape] = s
                return pegs
            self.r.log.warn('scene detection failed (%s), retrying' % (None if res is None else res.message))
        return []

    def run_steps(self, steps):
        """Execute plan steps.  Returns (ok, failed_step, info)."""
        for st in steps:
            if self.abort:
                return False, st, 'aborted by user'
            if st.kind == 'traj':
                res = self.r.follow(st.traj, guarded=st.guarded)
                if not res:
                    if res.code == GOAL_TOLERANCE_VIOLATED:  # ended slightly off target (gravity sag)
                        self.r.log.warn('%s: %s (continuing)' % (st.label, res.message))
                        continue
                    return False, st, res.message
            elif st.kind == 'wait':
                time.sleep(st.seconds)
            elif st.kind == 'gripper':
                if st.gripper == 'open':
                    self.r.gripper(1.0)
                elif st.gripper == 'release':
                    self.r.gripper(self.release_fraction)
                elif st.gripper == 'close':
                    res = self.r.gripper(0.0)
                    if not self.grasp_ok(res):
                        return False, st, 'grasp failed (gripper closed on nothing)'
                    self.t_grasp = time.time()
        return True, None, ''

    def grasp_ok(self, res):
        if res is None:
            return False
        frac = float(self.gr.get('grasp_detect_fraction', 0.08))
        return res.stalled or res.position > frac

    # ------------------------------------------------------------ main
    def run(self):
        r = self.r
        r.wait_ready()
        if r.estopped:
            self.log('emergency stop is active - release it (publish false on /emergency_stop) and restart')
            return
        self.log('moving to observe pose ...')
        pegs = self.observe(with_sockets=True)
        self.true_height = {p.color: p.height for p in pegs}
        self.print_setup(pegs)
        if not pegs or not self.sockets:
            self.log('nothing to do: %d pegs, %d sockets detected' % (len(pegs), len(self.sockets)))
            return
        if self.p.get('confirm', True):
            print('\n>>> Press ENTER to start the timed run (SPACE = emergency stop, q = abort) <<<')
            sys.stdin.readline()
        expected = len({p.color for p in pegs if p.shape in self.sockets})
        self.logger.start_run()
        t_run = time.time()

        failures = {}
        while not self.abort:
            todo = [p for p in pegs if p.color not in self.placed and p.shape in self.sockets
                    and failures.get(p.color, 0) <= self.max_retries]
            if not todo:
                break
            pairs = self.planner.assign(todo, [self.sockets[p.shape] for p in todo], r.joints())
            if not pairs:
                break
            peg, sock = pairs[0]
            policy = self.task['alignment'].get('policy', 'warn')
            mis = math.degrees(predicted_misalignment(peg, sock, self.planner.grasp_model))
            if policy == 'skip' and abs(mis) > self.yaw_tol(peg.shape):
                self.log('skipping %s: predicted misalignment %.0f deg' % (peg.color, mis))
                failures[peg.color] = self.max_retries + 1
                continue
            attempt = failures.get(peg.color, 0) + 1
            ok, reason, plan = self.pick_and_place(peg, sock, pegs)
            verified = ''
            if ok and self.verify:
                pegs_now = self.observe(with_sockets=False)
                ok, verified, pegs = self.check_placed(peg, sock, pegs_now, pegs)
                if not ok:
                    reason = 'verification: %s' % verified
            elif not self.abort:
                pegs = self.observe(with_sockets=False) or pegs
            t_end = time.time()
            self.logger.attempt(peg_color=peg.color, shape=peg.shape, attempt=attempt, success=ok,
                                verified=verified, reason=reason, t_start=self.t_start,
                                t_grasp=getattr(self, 't_grasp', ''), t_place=getattr(self, 't_place', ''),
                                t_end=t_end, misalignment_deg=round(mis, 1),
                                pitch_deg=round(math.degrees(plan.pitch), 1) if plan else '',
                                route=plan.route if plan else '')
            if ok:
                self.placed[peg.color] = sock
                self.log('PLACED %s %s (%.1fs since start)' % (peg.color, peg.shape, t_end - t_run))
            else:
                failures[peg.color] = failures.get(peg.color, 0) + 1
                self.r.log.warn('FAILED %s: %s (attempt %d)' % (peg.color, reason, attempt))
                if 'resting ON the board' in reason or 'leaning' in reason:
                    # A peg sitting on / leaning in its socket must not be grabbed again: the gripper
                    # comes down on a tilted peg next to the others and knocks it over (lab video).
                    why = ('predicted yaw error %.0f deg cannot be fixed by this arm - turn the board' % mis
                           if abs(mis) > self.yaw_tol(peg.shape) else
                           'it is leaning in the hole' if 'leaning' in reason else 'it landed a few mm off the hole')
                    self.r.log.warn('%s: resting on its socket (%s) - not retrying it, moving on.' % (peg.color, why))
                    failures[peg.color] = self.max_retries + 1

        summary = self.logger.finish(expected, {'speed_scale': self.speed})
        try:
            r.move_joints(self.robot_cfg['poses']['home'], self.speed)
        except EmergencyStop:
            pass
        self.log('RUN FINISHED: %d/%d pegs placed in %.1fs, %d failed attempts. Log: %s' % (
            summary['pegs_placed'], summary['pegs_expected'], summary['total_time_s'], summary['failures'],
            self.logger.csv_path))

    def pick_and_place(self, peg, sock, all_pegs):
        self.t_start = time.time()
        self.t_grasp = ''
        self.t_place = ''
        if peg.color in self.true_height:  # e.g. re-picking a peg that is resting on the board
            peg.height = self.true_height[peg.color]
        others = [p for p in all_pegs if p.color != peg.color and p.color not in self.placed]
        # pegs already in sockets are obstacles too
        for color, s in self.placed.items():
            src = next((p for p in all_pegs if p.color == color), None)
            w = src.width if src else 0.05
            others.append(PegInfo(color, s.shape, s.x, s.y, 0.1, 0.1, 0.0, w))
        try:
            plan = self.planner.plan(peg, sock, self.r.joints(), others, self.speed)
        except PlanningError as e:
            return False, 'planning: %s' % e, None
        self.log('%s %s -> socket (%.3f, %.3f): pitch %.0f deg, %s, %.1fs planned, predicted yaw error %.0f deg' % (
            peg.color, peg.shape, sock.x, sock.y, math.degrees(plan.pitch), plan.route, plan.duration,
            plan.misalignment_deg))
        ok, failed, info = self.run_steps(plan.steps)
        if ok:
            self.t_place = time.time()
            return True, '', plan
        if failed is not None and failed.label.startswith('insert'):
            # blocked during insertion: back off and try small offsets
            for off in self.offsets[1:]:
                self.r.log.warn('insertion blocked, retrying with offset %s' % (off,))
                steps = self.planner.retry_insertion(peg, sock, self.r.joints(), plan.pitch, off, self.speed)
                ok, failed, info = self.run_steps(steps)
                if ok:
                    self.t_place = time.time()
                    return True, 'inserted with offset %s' % (off,), plan
                if failed is None or not failed.label.startswith('insert'):
                    break
            # give up: release the peg on top of the socket and back off
            self.r.gripper(1.0)
        if failed is not None and failed.kind == 'gripper':
            self.r.gripper(1.0)
        try:
            self.r.follow(self.planner.retreat_up(self.r.joints()))
        except Exception as e:  # noqa: BLE001
            self.r.log.warn('retreat failed: %s' % e)
        return False, '%s: %s' % (failed.label if failed else '?', info), plan

    def check_placed(self, peg, sock, pegs_now, pegs_before):
        if not pegs_now:
            return True, 'unverified', pegs_before
        m = next((p for p in pegs_now if p.color == peg.color), None)
        if m is None:
            return True, 'unverified (peg not visible)', pegs_now
        d = math.hypot(m.x - sock.x, m.y - sock.y)
        # A peg that went into its hole stands on the table again (same top height as
        # before the pick).  One resting ON the board is a board-thickness higher.
        thickness = float(self.vision_cfg['board']['thickness'])
        expected_top = self.table_z + self.true_height.get(peg.color, peg.height)
        on_board = m.z_top > expected_top + 0.5 * thickness
        # Pegs that went in straight measured 0-4 mm from the socket centre.  A peg whose
        # bottom is in the hole but leans over has its top well off the centre.
        lean_tol = float(self.task['execution'].get('lean_tolerance', 0.015))
        leaning = not on_board and lean_tol <= d < self.verify_tol
        ok = d < self.verify_tol and not on_board and not leaning
        info = 'dist %.0fmm, top %+.1fmm' % (d * 1e3, (m.z_top - expected_top) * 1e3)
        if on_board:
            info += ' -> resting ON the board (not inserted; yaw misaligned?)'
        elif leaning:
            info += ' -> leaning in its socket (top > %.0f mm off the centre)' % (lean_tol * 1e3)
        return ok, info, pegs_now

    def print_setup(self, pegs):
        print('\n================ SCENE ================')
        for p in pegs:
            print('  peg    %-7s %-8s at (%6.3f, %6.3f)  top %.3f  yaw %5.1f deg' % (
                p.color, p.shape, p.x, p.y, p.z_top, math.degrees(p.yaw)))
        for s in self.sockets.values():
            print('  socket %-8s at (%6.3f, %6.3f)  yaw %5.1f deg' % (s.shape, s.x, s.y, math.degrees(s.yaw)))
        for p in pegs:
            s = self.sockets.get(p.shape)
            if s is None:
                print('  !! no socket detected for the %s %s' % (p.color, p.shape))
                continue
            mis = math.degrees(predicted_misalignment(p, s, self.planner.grasp_model))
            hint = peg_rotation_hint(p, s, self.planner.grasp_model)
            tol = self.yaw_tol(s.shape)
            if abs(mis) <= tol:
                flag = 'OK'
            elif hint is not None:
                flag = 'WARNING: turn this peg by %+.0f deg (CCW seen from above) before starting' % math.degrees(hint)
            else:
                flag = 'WARNING: 4-DOF arm cannot fix this - move/rotate the board (suggest_board_pose)'
            print('  %-7s -> %-8s predicted yaw error %+5.1f deg (max %.0f)  %s'
                  % (p.color, s.shape, mis, tol, flag))
        half_open = 0.5 * float(self.gr.get('open_width_m', 0.06))
        min_gap = float(self.task['grasp'].get('min_jaw_gap', 0.008))
        for p in pegs:
            gap, turn = open_jaw_clearance(p, half_open)
            if gap < min_gap:
                d = math.degrees(turn)
                face = {'square': 'a flat side faces the robot',
                        'hexagon': 'a corner points at the robot (flat sides left and right)'}.get(p.shape, 'it fits')
                print('  !! %s %s: a corner points at a jaw - only %.0f mm between the open jaw and the peg, '
                      'the jaw will land on it.\n  !!    Turn the %s peg %.0f deg %s (seen from above) so %s.'
                      % (p.color, p.shape, max(gap, 0.0) * 1e3, p.color, abs(d),
                         'anticlockwise' if d > 0 else 'clockwise', face))
        crowded = []
        for p in pegs:
            for o, gap in jaw_blockers(p, [q for q in pegs if q.color != p.color]):
                crowded.append('  !! %s: the open jaw comes down where %s stands (only %.0f mm gap sideways). '
                               'The jaw will push %s over.' % (p.color, o.color, max(gap, 0.0) * 1e3, p.color))
        if crowded:
            print('\n'.join(crowded))
            print('  !! FIX: seen from the robot, keep >= 3 cm free on BOTH sides of every peg '
                  '(centres >= 8 cm apart\n'
                  '  !!      side by side), or put the neighbour >= 4 cm in front of / behind it.  Then restart.')
        print('=======================================\n')


def main(args=None):
    rclpy.init(args=args)
    robot = RobotInterface('task_manager')
    n = robot.node
    params = {}
    for name, default in (('confirm', True), ('speed_scale', 0.0)):
        n.declare_parameter(name, default)
        params[name] = n.get_parameter(name).value
    tm = TaskManager(robot, params)

    stop = threading.Event()

    def keys():
        with KeyReader() as kr:
            if not kr.available:
                return
            while not stop.is_set():
                k = kr.get(0.05)
                if k == ' ':
                    robot.emergency_stop()
                    tm.abort = True
                    print('\n!!! EMERGENCY STOP !!!  (release: ros2 topic pub --once /emergency_stop std_msgs/Bool "data: false")')
                elif k in ('q', 'Q'):
                    tm.abort = True
                    print('\n-- aborting after the current step --')
                elif k in ('\n', '\r'):
                    pass

    kt = None
    if not params['confirm']:
        kt = threading.Thread(target=keys, daemon=True)
        kt.start()
    else:
        # the ENTER prompt needs line input; start the key thread after it
        orig = tm.logger.start_run

        def start_run_and_keys():
            nonlocal kt
            orig()
            kt = threading.Thread(target=keys, daemon=True)
            kt.start()
        tm.logger.start_run = start_run_and_keys
    try:
        tm.run()
    except EmergencyStop:
        robot.log.error('stopped by emergency stop')
    except KeyboardInterrupt:
        robot.emergency_stop()
        robot.log.error('Ctrl-C: emergency stop sent')
    finally:
        stop.set()
        robot.shutdown()


if __name__ == '__main__':
    main()

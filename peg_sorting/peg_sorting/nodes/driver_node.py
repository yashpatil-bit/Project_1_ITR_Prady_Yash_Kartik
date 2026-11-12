"""ROS 2 driver for the Manipulator X (Dynamixel XM430 via U2D2).

Interfaces (standard ROS 2 types, same names as the ROBOTIS ros2_control setup):
  action  /arm_controller/follow_joint_trajectory   control_msgs/FollowJointTrajectory
  action  /gripper_controller/gripper_cmd           control_msgs/GripperCommand
              position: 0.0 = closed ... 1.0 = open (normalised, see robot.yaml gripper ticks)
  topic   /joint_states            sensor_msgs/JointState  (joint1..4 [rad], gripper [0..1])
  topic   /emergency_stop          std_msgs/Bool   true = stop & hold position, false = release
  service /driver/set_torque       std_srvs/SetBool
Parameters:
  mock (bool)          simulate the motors (no hardware)
  port, baudrate       override robot.yaml
"""
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
from std_srvs.srv import SetBool

from ..control.dynamixel_bus import DynamixelBus, MockBus, rad_to_tick, tick_to_rad
from ..utils.config import load_yaml


class DriverNode(Node):
    def __init__(self):
        super().__init__('manipulator_driver')
        self.declare_parameter('mock', False)
        self.declare_parameter('port', '')
        self.declare_parameter('baudrate', 0)
        cfg = load_yaml('robot.yaml')
        self.cfg = cfg
        dx = cfg['dynamixel']
        gr = cfg['gripper']
        self.mock = self.get_parameter('mock').value
        port = self.get_parameter('port').value or dx['port']
        baud = self.get_parameter('baudrate').value or int(dx['baudrate'])
        self.names = list(dx['joint_names'])
        self.signs = [int(s) for s in dx['joint_signs']]
        self.offsets = [float(o) for o in dx['joint_offsets_rad']]
        self.lower = [float(v) for v in cfg['joint_limits_rad']['lower']]
        self.upper = [float(v) for v in cfg['joint_limits_rad']['upper']]
        self.open_ticks = int(gr['open_ticks'])
        self.closed_ticks = int(gr['closed_ticks'])
        self.gr = gr
        self.rate = float(dx.get('control_rate_hz', 100))
        drv = cfg.get('driver', {})
        self.goal_tol = float(drv.get('goal_tolerance_rad', 0.04))
        self.goal_time_tol = float(drv.get('goal_time_tolerance_s', 1.0))

        bus_cls = MockBus if self.mock else DynamixelBus
        init = None
        if self.mock:
            sleep = cfg['poses'].get('sleep', [0, -1.7, 1.25, 0.6])
            init = [rad_to_tick(q, s, o) for q, s, o in zip(sleep, self.signs, self.offsets)] + [self.open_ticks]
        kwargs = {'initial_ticks': init} if self.mock else {}
        self.bus = bus_cls(port, baud, dx['joint_ids'], dx.get('gripper_id'), float(dx.get('protocol', 2.0)), **kwargs)
        if self.mock:
            self.bus.gripper_closed_tick = self.closed_ticks
        self.bus.configure(gripper_current_limit=int(gr.get('current_limit', 300)),
                           gripper_profile_velocity=int(gr.get('profile_velocity', 0)))
        self.bus.set_torque(True)
        self.get_logger().info('%s Manipulator X on %s @ %d (ids %s + gripper %s)' % (
            'MOCK' if self.mock else 'Connected to', port, baud, dx['joint_ids'], dx.get('gripper_id')))

        self.state_lock = threading.Lock()
        self.q = np.zeros(4)
        self.gripper_tick = self.open_ticks
        self.state_time = 0.0
        self._read_state()
        self.cmd_q = self.q.copy()
        self.estop = False
        self._sim_blocked_stamp = 0.0
        self.active_token = 0
        self.token_lock = threading.Lock()

        cb = ReentrantCallbackGroup()
        self.js_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_subscription(Bool, 'emergency_stop', self._on_estop, 10, callback_group=cb)
        self.create_service(SetBool, 'driver/set_torque', self._on_torque, callback_group=cb)
        if self.mock:
            self.create_subscription(Float64, 'sim/gripper_blocked', self._on_sim_blocked, 10, callback_group=cb)
        self.create_timer(1.0 / float(dx.get('state_rate_hz', 50)), self._publish_state, callback_group=cb)
        self.arm_server = ActionServer(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory',
                                       execute_callback=self._exec_traj, goal_callback=self._goal_cb,
                                       cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=cb)
        self.grip_server = ActionServer(self, GripperCommand, 'gripper_controller/gripper_cmd',
                                        execute_callback=self._exec_gripper, goal_callback=self._goal_cb,
                                        cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=cb)

    # ------------------------------------------------------------- state
    def _read_state(self):
        ticks = self.bus.read_positions()
        q = np.array([tick_to_rad(t, s, o) for t, s, o in zip(ticks[:4], self.signs, self.offsets)])
        with self.state_lock:
            self.q = q
            if len(ticks) > 4:
                self.gripper_tick = ticks[4]
            self.state_time = time.time()
        return q

    def gripper_norm(self, tick=None):
        tick = self.gripper_tick if tick is None else tick
        return (tick - self.closed_ticks) / float(self.open_ticks - self.closed_ticks)

    def _publish_state(self):
        try:
            if time.time() - self.state_time > 0.5 / float(self.cfg['dynamixel'].get('state_rate_hz', 50)):
                self._read_state()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('state read failed: %s' % e, throttle_duration_sec=2.0)
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self.state_lock:
            msg.name = self.names + ['gripper']
            msg.position = [float(v) for v in self.q] + [float(self.gripper_norm())]
        self.js_pub.publish(msg)

    def _write_joints(self, q):
        ids = self.bus.joint_ids
        self.bus.write_goal_ticks({i: rad_to_tick(v, s, o) for i, v, s, o in zip(ids, q, self.signs, self.offsets)})
        self.cmd_q = np.array(q, dtype=float)

    def _hold(self):
        try:
            q = self._read_state()
            self._write_joints(q)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('hold failed: %s' % e)

    # ------------------------------------------------------------- safety
    def _on_estop(self, msg):
        if msg.data and not self.estop:
            self.get_logger().error('EMERGENCY STOP - holding position. Publish false on /emergency_stop to release.')
            self.estop = True
            with self.token_lock:
                self.active_token += 1
            self._hold()
        elif not msg.data and self.estop:
            self.get_logger().warn('Emergency stop released.')
            self.estop = False

    def _on_torque(self, req, res):
        try:
            if req.data:
                self._hold()
            self.bus.set_torque(req.data)
            if req.data:
                self._hold()
            res.success = True
            res.message = 'torque %s' % ('on' if req.data else 'off')
        except Exception as e:  # noqa: BLE001
            res.success = False
            res.message = str(e)
        return res

    def _on_sim_blocked(self, msg):
        self._sim_blocked_stamp = time.time()
        if msg.data < 0:
            self.bus.gripper_stall_tick = None
        else:
            self.bus.gripper_stall_tick = int(self.closed_ticks + msg.data * (self.open_ticks - self.closed_ticks))

    def _goal_cb(self, goal):
        if self.estop:
            self.get_logger().warn('goal rejected: emergency stop active')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    # ------------------------------------------------------------- arm
    def _exec_traj(self, gh):
        res = FollowJointTrajectory.Result()
        traj = gh.request.trajectory
        try:
            idx = [list(traj.joint_names).index(n) for n in self.names]
        except ValueError:
            res.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            res.error_string = 'trajectory must contain joints %s' % self.names
            gh.abort()
            return res
        if not traj.points:
            gh.succeed()
            return res
        T = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points])
        Q = np.array([[p.positions[i] for i in idx] for p in traj.points])
        with self.state_lock:
            q_now = self.q.copy()
        # A joint that is currently outside its limit (e.g. the arm resting folded back after
        # power-up) may move anywhere between where it is now and the limit, so that moves back
        # into the valid range (like 'go home') are allowed.
        lo = np.minimum(np.array(self.lower), q_now) - 0.01
        hi = np.maximum(np.array(self.upper), q_now) + 0.01
        if np.any(Q < lo) or np.any(Q > hi):
            bad = sorted({self.names[j] for j in range(4) if np.any(Q[:, j] < lo[j]) or np.any(Q[:, j] > hi[j])})
            res.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            res.error_string = 'trajectory violates joint limits (%s)' % ', '.join(bad)
            self.get_logger().error(res.error_string)
            gh.abort()
            return res
        with self.token_lock:
            self.active_token += 1
            token = self.active_token
        with self.state_lock:
            q_meas0 = self.q.copy()
        # continue from the last command (smooth preemption) unless it is stale
        start_q = self.cmd_q.copy() if np.max(np.abs(self.cmd_q - q_meas0)) < 0.1 else q_meas0
        if T[0] > 1e-6:
            T = np.concatenate([[0.0], T])
            Q = np.vstack([start_q, Q])
        elif min(np.max(np.abs(Q[0] - start_q)), np.max(np.abs(Q[0] - q_meas0))) > 0.15:
            res.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            res.error_string = 'trajectory starts %.2f rad away from the current position' % np.max(np.abs(Q[0] - start_q))
            self.get_logger().error(res.error_string)
            gh.abort()
            return res
        path_tol = None
        if gh.request.path_tolerance:
            tol = {t.name: t.position for t in gh.request.path_tolerance if t.position > 0}
            if tol:
                path_tol = np.array([tol.get(n, np.inf) for n in self.names])

        period = 1.0 / self.rate
        t0 = time.time()
        fb = FollowJointTrajectory.Feedback()
        fb.joint_names = self.names
        n_fb = 0
        while True:
            t = time.time() - t0
            if token != self.active_token or self.estop:
                if self.estop:
                    self._hold()  # (a preempting goal takes over the motors itself)
                res.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                res.error_string = 'emergency stop' if self.estop else 'preempted'
                gh.abort()
                return res
            if gh.is_cancel_requested:
                self._hold()
                gh.canceled()
                res.error_string = 'canceled'
                return res
            q_des = np.array([np.interp(min(t, T[-1]), T, Q[:, j]) for j in range(4)])
            try:
                self._write_joints(q_des)
                q_meas = self._read_state()
            except Exception as e:  # noqa: BLE001
                self.get_logger().error('bus error: %s' % e)
                res.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                res.error_string = 'bus error: %s' % e
                gh.abort()
                return res
            err = q_des - q_meas
            if path_tol is not None and np.any(np.abs(err) > path_tol):
                self.get_logger().warn('blocked: tracking error %s rad -> stopping' % np.round(err, 3))
                self._hold()
                res.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                res.error_string = 'path tolerance violated (arm blocked?)'
                gh.abort()
                return res
            n_fb += 1
            if n_fb % 5 == 0:
                fb.desired.positions = [float(v) for v in q_des]
                fb.actual.positions = [float(v) for v in q_meas]
                fb.error.positions = [float(v) for v in err]
                gh.publish_feedback(fb)
            if t >= T[-1]:
                if np.max(np.abs(Q[-1] - q_meas)) < self.goal_tol:
                    break
                if t > T[-1] + self.goal_time_tol:
                    res.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                    res.error_string = 'final error %.3f rad' % np.max(np.abs(Q[-1] - q_meas))
                    self.get_logger().warn(res.error_string)
                    gh.abort()
                    return res
            sleep = period - ((time.time() - t0) - t)
            if sleep > 0:
                time.sleep(sleep)
        res.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        gh.succeed()
        return res

    # ------------------------------------------------------------- gripper
    def _exec_gripper(self, gh):
        res = GripperCommand.Result()
        pos = float(min(max(gh.request.command.position, 0.0), 1.0))
        goal_tick = int(round(self.closed_ticks + pos * (self.open_ticks - self.closed_ticks)))
        gid = self.bus.gripper_id
        if self.mock:
            # Simulation only: wait until the simulator has told us (after this command) whether a
            # peg is between the fingers, otherwise a slow PC closes the fingers "through" the peg.
            t_cmd = time.time()
            while self._sim_blocked_stamp < t_cmd and time.time() - t_cmd < 1.0:
                time.sleep(0.01)
        try:
            self.bus.write_goal_ticks({gid: goal_tick})
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('gripper write failed: %s' % e)
            gh.abort()
            return res
        t0 = time.time()
        timeout = float(self.gr.get('move_timeout', 1.5))
        last_tick, last_move = None, time.time()
        while True:
            if self.estop or gh.is_cancel_requested:
                break
            self._read_state()
            tick = self.gripper_tick
            if abs(tick - goal_tick) <= 15:
                res.reached_goal = True
                break
            if last_tick is None or abs(tick - last_tick) > 3:
                last_tick, last_move = tick, time.time()
            elif time.time() - last_move > 0.15 and time.time() - t0 > 0.2:
                res.stalled = True
                break
            if time.time() - t0 > timeout:
                break
            time.sleep(0.01)
        res.position = float(self.gripper_norm())
        res.effort = float(self.bus.read_gripper_current()) * 2.69  # mA
        if gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.succeed()
        return res

    def destroy_node(self):
        try:
            self.bus.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DriverNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('driver shutting down (motors keep holding torque)')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

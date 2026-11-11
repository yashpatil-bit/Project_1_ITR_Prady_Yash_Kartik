"""Client-side helper used by the task manager, teleop and calibration tools.

Spins its own ROS node in a background thread so that the calling code can be
written as plain sequential Python (move, wait, grip, ...).
"""
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from control_msgs.msg import JointTolerance
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ..planning.kinematics import ArmGeometry, ManipulatorXKinematics
from ..planning.trajectory import Limits, joint_move
from ..utils.config import load_yaml


class MoveResult:
    def __init__(self, ok, code=0, message=''):
        self.ok, self.code, self.message = ok, code, message

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return 'MoveResult(ok=%s, code=%s, %s)' % (self.ok, self.code, self.message)


class EmergencyStop(RuntimeError):
    pass


class RobotInterface:
    def __init__(self, node_name='peg_sorting_client', with_vision=True):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name)
        self.robot_cfg = load_yaml('robot.yaml')
        self.kin = ManipulatorXKinematics(ArmGeometry.from_config(self.robot_cfg))
        self.limits = Limits.from_config(self.robot_cfg.get('limits'))
        self.joint_names = list(self.robot_cfg['dynamixel']['joint_names'])
        self.path_tol = float(self.robot_cfg.get('driver', {}).get('path_tolerance_rad', 0.2))
        self._q = None
        self._gripper = None
        self._lock = threading.Lock()
        self.estopped = False
        self.node.create_subscription(JointState, 'joint_states', self._on_js, 10)
        self.node.create_subscription(Bool, 'emergency_stop', self._on_estop, 10)
        self.estop_pub = self.node.create_publisher(Bool, 'emergency_stop', 10)
        self.arm = ActionClient(self.node, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        self.grip = ActionClient(self.node, GripperCommand, 'gripper_controller/gripper_cmd')
        self.torque_cli = self.node.create_client(SetBool, 'driver/set_torque')
        self.scene_cli = None
        if with_vision:
            from peg_sorting_interfaces.srv import GetScene
            self.GetScene = GetScene
            self.scene_cli = self.node.create_client(GetScene, 'vision/get_scene')
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self._spin = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin.start()
        self._active_goal = None

    @property
    def log(self):
        return self.node.get_logger()

    # ------------------------------------------------------------ state
    def _on_js(self, msg):
        try:
            idx = [list(msg.name).index(n) for n in self.joint_names]
        except ValueError:
            return
        with self._lock:
            self._q = np.array([msg.position[i] for i in idx])
            if 'gripper' in msg.name:
                self._gripper = msg.position[list(msg.name).index('gripper')]

    def _on_estop(self, msg):
        self.estopped = bool(msg.data)

    def joints(self, timeout=5.0):
        t0 = time.time()
        while self._q is None:
            if time.time() - t0 > timeout:
                raise RuntimeError('no /joint_states - is the driver running?')
            time.sleep(0.02)
        with self._lock:
            return self._q.copy()

    def gripper_position(self):
        return self._gripper

    def tool_pose(self):
        return self.kin.fk(self.joints())

    def wait_ready(self, timeout=20.0, vision=True):
        t0 = time.time()
        ok = self.arm.wait_for_server(timeout_sec=timeout) and self.grip.wait_for_server(
            timeout_sec=max(0.1, timeout - (time.time() - t0)))
        if not ok:
            raise RuntimeError('driver action servers not available - start the driver first')
        self.joints(timeout=timeout)
        if vision and self.scene_cli is not None:
            if not self.scene_cli.wait_for_service(timeout_sec=max(0.1, timeout - (time.time() - t0))):
                raise RuntimeError('vision service not available - start the vision node first')

    # ------------------------------------------------------------ safety
    def emergency_stop(self):
        self.estop_pub.publish(Bool(data=True))
        self.estopped = True
        self.cancel()

    def release_estop(self):
        self.estop_pub.publish(Bool(data=False))
        self.estopped = False

    def cancel(self):
        g = self._active_goal
        if g is not None:
            try:
                g.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass

    def set_torque(self, on):
        if not self.torque_cli.wait_for_service(timeout_sec=2.0):
            return False
        fut = self.torque_cli.call_async(SetBool.Request(data=on))
        return self._wait(fut, 5.0) is not None

    # ------------------------------------------------------------ motion
    @staticmethod
    def _wait(fut, timeout):
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.005)
        return fut.result()

    def follow(self, traj, guarded=False, wait=True):
        """Send a planning.trajectory.Trajectory to the driver."""
        if self.estopped:
            raise EmergencyStop('emergency stop active')
        jt = JointTrajectory()
        jt.joint_names = self.joint_names
        for t, q in zip(traj.t, traj.q):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in q]
            sec = int(t)
            p.time_from_start = Duration(sec=sec, nanosec=int(round((t - sec) * 1e9)))
            jt.points.append(p)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        if guarded:
            goal.path_tolerance = [JointTolerance(name=n, position=self.path_tol) for n in self.joint_names]
        send = self.arm.send_goal_async(goal)
        gh = self._wait(send, 5.0)
        if gh is None or not gh.accepted:
            if self.estopped:
                raise EmergencyStop('emergency stop active')
            return MoveResult(False, -100, 'goal rejected')
        self._active_goal = gh
        if not wait:
            return gh
        res = self._wait(gh.get_result_async(), traj.duration + 10.0)
        self._active_goal = None
        if self.estopped:
            raise EmergencyStop('emergency stop active')
        if res is None:
            return MoveResult(False, -101, 'timeout')
        r = res.result
        ok = r.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        return MoveResult(ok, r.error_code, r.error_string)

    def move_joints(self, q_goal, speed_scale=1.0, duration=None, wait=True):
        traj = joint_move(self.joints(), q_goal, self.limits, speed_scale, duration)
        return self.follow(traj, wait=wait)

    def gripper(self, position, wait=True, timeout=5.0):
        """position: 0 closed ... 1 open.  Returns the GripperCommand result (or None)."""
        if self.estopped:
            raise EmergencyStop('emergency stop active')
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 0.0
        gh = self._wait(self.grip.send_goal_async(goal), 5.0)
        if gh is None or not gh.accepted:
            return None
        if not wait:
            return gh
        res = self._wait(gh.get_result_async(), timeout)
        return None if res is None else res.result

    # ------------------------------------------------------------ vision
    def get_scene(self, num_frames=0, detect_sockets=True, timeout=30.0):
        req = self.GetScene.Request(num_frames=int(num_frames), detect_sockets=bool(detect_sockets))
        return self._wait(self.scene_cli.call_async(req), timeout)

    def shutdown(self):
        try:
            self.executor.shutdown(timeout_sec=1.0)
        except Exception:  # noqa: BLE001
            pass
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

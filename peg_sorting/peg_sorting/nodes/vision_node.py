"""Vision node: owns the RealSense, answers /vision/get_scene, publishes a debug image.

  service /vision/get_scene     peg_sorting_interfaces/GetScene
  topic   /vision/debug_image   sensor_msgs/Image (bgr8)
Parameters:
  mock (bool)          use the synthetic scene (config/sim_scene.yaml) instead of the camera
  show_window (bool)   open an OpenCV window with the live detections (needs a display)
  debug_rate (float)   Hz of the live detection / debug image (0 = off)
"""
import collections
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64

from peg_sorting_interfaces.msg import Peg, Socket
from peg_sorting_interfaces.srv import GetScene

from ..planning.kinematics import ArmGeometry, ManipulatorXKinematics
from ..planning.pick_place_planner import PegInfo, SocketInfo, predicted_misalignment
from ..utils.config import load_yaml
from ..utils.transforms import T_from_dict, planar_correction_from_config
from ..vision.camera import open_camera
from ..vision.scene import SceneDetector, resolve_duplicate_shapes


def load_extrinsics(calib):
    T = T_from_dict(calib['T_base_cam'])
    corr = calib.get('xyz_correction_m') or [0, 0, 0]
    T = T.copy()
    T[:3, 3] += np.asarray(corr, dtype=float)
    return T


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision')
        self.declare_parameter('mock', False)
        self.declare_parameter('show_window', False)
        self.declare_parameter('debug_rate', 3.0)
        self.declare_parameter('sim_scene', 'sim_scene.yaml')
        self.mock = self.get_parameter('mock').value
        self.show_window = self.get_parameter('show_window').value
        self.vcfg = load_yaml('vision.yaml')
        self.colors_cfg = load_yaml('colors.yaml')
        calib = load_yaml('camera_calib.yaml')
        # camera -> robot x-y correction measured by calibrate_workspace (absorbs arm/camera errors)
        self.corr_A, self.corr_b = planar_correction_from_config(calib)
        if not np.allclose(self.corr_A, np.eye(2)) or np.any(self.corr_b):
            self.get_logger().info('using planar correction A=%s b=%s mm' % (
                np.round(self.corr_A, 4).tolist(), np.round(self.corr_b * 1e3, 1).tolist()))
        self.table_z = float(self.vcfg.get('table_z', 0.0))
        self.grasp_model = (load_yaml('task.yaml').get('alignment') or {}).get('grasp_model', 'squeeze')

        self.sim = None
        if self.mock:
            from ..sim.synthetic_scene import SimWorld
            self.sim = SimWorld(load_yaml(self.get_parameter('sim_scene').value))
            self.sim.grasp_model = self.grasp_model
            T = self.sim.T_base_cam
            self.get_logger().info('MOCK camera: synthetic scene from %s' % self.get_parameter('sim_scene').value)
        else:
            T = load_extrinsics(calib)
            if not calib.get('calibrated', False):
                self.get_logger().warn('camera_calib.yaml is NOT calibrated yet - positions will be wrong. '
                                       'Run `ros2 run peg_sorting calibrate_aruco` first.')
        self.camera = open_camera(self.vcfg, mock=self.mock, sim_world=self.sim)
        self.get_logger().info('camera: %s (%s)' % (self.camera.name, self.camera.serial))

        yolo = None
        if self.vcfg.get('shape_backend', 'geometry') == 'yolo':
            from ..vision.yolo_classifier import YoloShapeClassifier
            yolo = YoloShapeClassifier(self.vcfg['yolo'])
        self.detector = SceneDetector(self.colors_cfg, self.vcfg, T, self.table_z, yolo)
        self.kin = ManipulatorXKinematics(ArmGeometry.from_config(load_yaml('robot.yaml')))

        self.frames = collections.deque(maxlen=10)
        self.frame_cv = threading.Condition()
        self.frame_count = 0
        self.running = True
        self.cam_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.cam_thread.start()

        self.debug_pub = self.create_publisher(Image, 'vision/debug_image', 2)
        self.create_service(GetScene, 'vision/get_scene', self._on_get_scene,
                            callback_group=MutuallyExclusiveCallbackGroup())
        rate = float(self.get_parameter('debug_rate').value)
        self.latest_debug = None
        if rate > 0:
            self.create_timer(1.0 / rate, self._debug_tick, callback_group=MutuallyExclusiveCallbackGroup())
        if self.mock:
            self.blocked_pub = self.create_publisher(Float64, 'sim/gripper_blocked', 10)
            self.create_subscription(JointState, 'joint_states', self._on_joint_states, 10,
                                     callback_group=ReentrantCallbackGroup())
        self.get_logger().info('vision ready: service /vision/get_scene')

    # ------------------------------------------------------------ camera
    def _capture_loop(self):
        while self.running:
            try:
                fr = self.camera.get_frame()
            except Exception as e:  # noqa: BLE001
                self.get_logger().error('camera error: %s' % e, throttle_duration_sec=2.0)
                time.sleep(0.2)
                continue
            if fr is None:
                continue
            with self.frame_cv:
                self.frames.append(fr)
                self.frame_count += 1
                self.frame_cv.notify_all()
            if self.mock:
                time.sleep(0.05)

    def _fresh_frames(self, n, timeout=5.0):
        with self.frame_cv:
            start = self.frame_count
            ok = self.frame_cv.wait_for(lambda: self.frame_count >= start + n, timeout=timeout)
            frames = list(self.frames)[-n:]
        if not ok:
            self.get_logger().warn('only got %d fresh frames' % (self.frame_count - start))
        return frames

    # ------------------------------------------------------------ sim
    def _on_joint_states(self, msg):
        if self.sim is None or len(msg.position) < 5:
            return
        q = list(msg.position[:4])
        g = msg.position[4]
        x, y, z, _ = self.kin.fk(q)
        tool = (x, y, z)
        held = any(p.held for p in self.sim.pegs)
        between = held or any(
            math.hypot(p.x - x, p.y - y) < 0.012 and p.z_bottom + p.height - 0.06 < z < p.z_bottom + p.height
            for p in self.sim.pegs)
        self.blocked_pub.publish(Float64(data=0.35 if between else -1.0))
        held_pegs = [p for p in self.sim.pegs if p.held]
        # the fingers must open wider than the peg to let go of it (real gripper opens ~6 cm)
        opening = g * float(self.sim.cfg.get('gripper_max_opening', 0.06))
        if g < 0.45 and between:
            self.sim.update_gripper(tool, q[0], closed=True)
        elif g > 0.5 and held and all(opening > 0.8 * p.size for p in held_pegs):
            self.sim.update_gripper(tool, q[0], closed=False)
        else:
            self.sim.update_gripper(tool, q[0], closed=held)

    # ------------------------------------------------------------ service
    def _on_get_scene(self, req, res):
        n = req.num_frames if req.num_frames > 0 else int(self.vcfg.get('default_num_frames', 5))
        t0 = time.time()
        frames = self._fresh_frames(n)
        if not frames:
            res.success = False
            res.message = 'no camera frames'
            return res
        pegs, board, sockets, _ = self.detector.detect_averaged(frames, with_sockets=req.detect_sockets)
        pegs = resolve_duplicate_shapes(pegs, self.colors_cfg)
        # debug image in camera coordinates (before the robot-frame correction below)
        self.latest_debug = self.detector.draw(frames[-1], pegs, board, sockets, self._misalign(pegs, sockets))
        self._publish_debug(self.latest_debug)
        if self.mock:
            A, b = np.eye(2), np.zeros(2)
        else:
            A, b = self.corr_A, self.corr_b

        def fix(x, y):
            v = A @ np.array([x, y]) + b
            return float(v[0]), float(v[1])
        for p in pegs:
            p.x, p.y = fix(p.x, p.y)
        for so in sockets:
            so.x, so.y = fix(so.x, so.y)
        if board.found:
            board.x, board.y = fix(board.x, board.y)
        for p in pegs:
            m = Peg()
            m.color, m.shape, m.shape_source = p.color, p.shape, p.shape_source
            m.shape_confidence = float(p.shape_confidence)
            m.top_center = Point(x=p.x, y=p.y, z=p.z_top)
            m.height, m.yaw, m.width, m.pixel_area = float(p.height), float(p.yaw), float(p.width), int(p.pixel_area)
            res.pegs.append(m)
        for s in sockets:
            m = Socket()
            m.shape, m.shape_confidence = s.shape, float(s.shape_confidence)
            m.center = Point(x=s.x, y=s.y, z=s.z)
            m.yaw, m.width = float(s.yaw), float(s.width)
            res.sockets.append(m)
        res.board_found = board.found
        res.board_center = Point(x=board.x, y=board.y, z=board.z)
        res.board_yaw = float(board.yaw)
        res.success = True
        res.message = '%d pegs, %d sockets from %d frames in %.2fs' % (len(pegs), len(sockets), len(frames),
                                                                      time.time() - t0)
        self.get_logger().info(res.message)
        return res

    def _misalign(self, pegs, sockets):
        out = {}
        for s in sockets:
            for p in pegs:
                if p.shape == s.shape:
                    out[s.shape] = math.degrees(predicted_misalignment(
                        PegInfo(p.color, p.shape, p.x, p.y, p.z_top, p.height, p.yaw, p.width),
                        SocketInfo(s.shape, s.x, s.y, s.z, s.yaw, s.width), self.grasp_model))
        return out

    # ------------------------------------------------------------ debug
    def _debug_tick(self):
        if not self.frames:
            return
        if self.debug_pub.get_subscription_count() == 0 and not self.show_window:
            return
        fr = self.frames[-1]
        pegs, board, sockets, _ = self.detector.detect(fr)
        self.latest_debug = self.detector.draw(fr, pegs, board, sockets, self._misalign(pegs, sockets))
        self._publish_debug(self.latest_debug)

    def _publish_debug(self, img):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = img.shape[1] * 3
        msg.data = np.ascontiguousarray(img).tobytes()
        self.debug_pub.publish(msg)

    def destroy_node(self):
        self.running = False
        self.cam_thread.join(timeout=1.0)
        self.camera.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        if node.show_window and os.environ.get('DISPLAY'):
            import cv2
            cv2.namedWindow('peg_sorting vision', cv2.WINDOW_NORMAL)
            while rclpy.ok():
                if node.latest_debug is not None:
                    cv2.imshow('peg_sorting vision', node.latest_debug)
                k = cv2.waitKey(50) & 0xFF
                if k == ord('s') and node.latest_debug is not None:
                    path = os.path.expanduser('~/peg_sorting_snapshot_%d.png' % int(time.time()))
                    cv2.imwrite(path, node.latest_debug)
                    node.get_logger().info('saved %s' % path)
        else:
            while rclpy.ok():
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

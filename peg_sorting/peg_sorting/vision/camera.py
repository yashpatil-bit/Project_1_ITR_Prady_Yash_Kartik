"""Intel RealSense RGB-D camera wrapper (pyrealsense2).

Depth is aligned to the colour image, so pixel (u, v) of ``Frame.color`` and
``Frame.depth`` refer to the same point and the colour intrinsics ``K`` apply
to both.
"""
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    color: np.ndarray      # (H, W, 3) uint8 BGR
    depth: np.ndarray      # (H, W) float32 metres, 0 = invalid
    K: np.ndarray          # (3, 3) colour intrinsics
    dist: np.ndarray       # distortion coefficients (RealSense colour is ~ 0)
    stamp: float


class RealSenseCamera:
    def __init__(self, cfg):
        import pyrealsense2 as rs  # imported lazily so the rest works without it
        self.rs = rs
        c = cfg.get('camera', cfg)
        self.width = int(c.get('width', 848))
        self.height = int(c.get('height', 480))
        self.fps = int(c.get('fps', 30))
        self.pipeline = rs.pipeline()
        serial = str(c.get('serial', '') or '')
        # Requested mode first; then modes that also work on a USB 2 connection.  A mode only
        # counts if frames actually arrive (on USB 2 a mode can start and then deliver nothing).
        modes = [(self.width, self.height, self.fps), (640, 480, 15), (640, 480, 6), (424, 240, 15),
                 (424, 240, 6)]
        profile, errors = None, []
        for (w, h, fps) in modes:
            config = rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            try:
                profile = self.pipeline.start(config)
            except RuntimeError as e:
                errors.append('%dx%d@%d: %s' % (w, h, fps, e))
                continue
            try:
                for _ in range(3):
                    self.pipeline.wait_for_frames(10000)
                self.width, self.height, self.fps = w, h, fps
                break
            except RuntimeError as e:
                errors.append('%dx%d@%d: started but %s' % (w, h, fps, e))
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    pass
                profile = None
                time.sleep(1.0)
        if profile is None:
            raise RuntimeError('RealSense could not stream in any mode (USB 2 cable/port? try another '
                               'cable or the USB-C port):\n  ' + '\n  '.join(errors))
        if (self.width, self.height, self.fps) != modes[0]:
            print('WARNING: RealSense running at %dx%d@%dfps (requested %dx%d@%d). Probably a USB 2 '
                  'connection - use a USB 3 port + cable for best depth.' % (
                      self.width, self.height, self.fps, *modes[0]))
        dev = profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        preset = c.get('visual_preset', 'high_accuracy')
        try:
            presets = {'default': 1, 'hand': 2, 'high_accuracy': 3, 'high_density': 4, 'medium_density': 5}
            if preset in presets and depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, presets[preset])
        except Exception:  # noqa: BLE001 - not all models support every preset
            pass
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        color_sensor = None
        for s in dev.query_sensors():
            if s.get_info(rs.camera_info.name).lower().startswith('rgb'):
                color_sensor = s
        if color_sensor is not None:
            if 'exposure' in c and c['exposure']:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, float(c['exposure']))
            if 'white_balance' in c and c['white_balance']:
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                color_sensor.set_option(rs.option.white_balance, float(c['white_balance']))
        self.align = rs.align(rs.stream.color)
        self.filters = []
        if c.get('spatial_filter', True):
            self.filters.append(rs.spatial_filter())
        if c.get('temporal_filter', True):
            self.filters.append(rs.temporal_filter())
        if c.get('hole_filling', False):
            self.filters.append(rs.hole_filling_filter())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=float)
        self.dist = np.array(intr.coeffs[:5], dtype=float)
        self.serial = dev.get_info(rs.camera_info.serial_number)
        self.name = dev.get_info(rs.camera_info.name)
        try:
            self.usb = dev.get_info(rs.camera_info.usb_type_descriptor)
            if self.usb.startswith('2'):
                print('WARNING: RealSense connected over USB %s - use a USB 3 port/cable.' % self.usb)
        except Exception:  # noqa: BLE001
            self.usb = '?'
        # let auto exposure settle
        for _ in range(int(c.get('warmup_frames', 30))):
            self.pipeline.wait_for_frames(10000)

    def get_frame(self, timeout_ms=2000):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return None
        for f in self.filters:
            depth = f.process(depth)
        depth_np = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        color_np = np.asanyarray(color.get_data()).copy()
        return Frame(color_np, depth_np, self.K.copy(), self.dist.copy(), time.time())

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass


def open_camera(vision_cfg, mock=False, sim_world=None):
    """Factory: real RealSense or the synthetic camera used for --mock runs/tests."""
    if mock:
        from ..sim.synthetic_scene import SyntheticCamera
        return SyntheticCamera(sim_world)
    return RealSenseCamera(vision_cfg)

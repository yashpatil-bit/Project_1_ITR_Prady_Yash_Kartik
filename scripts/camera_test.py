#!/usr/bin/env python3
"""Which RealSense stream modes actually deliver frames on this USB connection?

    python3 scripts/camera_test.py
"""
import time

import pyrealsense2 as rs

ctx = rs.context()
for d in ctx.query_devices():
    print('%s  serial %s  firmware %s  USB %s' % (
        d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number),
        d.get_info(rs.camera_info.firmware_version), d.get_info(rs.camera_info.usb_type_descriptor)))
tests = [('depth+color', 848, 480, 30), ('depth+color', 640, 480, 15), ('depth+color', 640, 480, 6),
         ('depth+color', 424, 240, 15), ('depth only', 640, 480, 15), ('color only', 640, 480, 15)]
for name, w, h, fps in tests:
    p = rs.pipeline()
    c = rs.config()
    if name != 'color only':
        c.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    if name != 'depth only':
        c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    try:
        p.start(c)
    except RuntimeError as e:
        print('%-12s %dx%d@%-2d  cannot start (%s)' % (name, w, h, fps, e))
        continue
    n, t0 = 0, time.time()
    try:
        while time.time() - t0 < 4.0:
            p.wait_for_frames(5000)
            n += 1
        print('%-12s %dx%d@%-2d  OK  %.1f fps' % (name, w, h, fps, n / (time.time() - t0)))
    except RuntimeError as e:
        print('%-12s %dx%d@%-2d  started, NO FRAMES (%s)' % (name, w, h, fps, e))
    finally:
        try:
            p.stop()
        except RuntimeError:
            pass
        time.sleep(1.0)

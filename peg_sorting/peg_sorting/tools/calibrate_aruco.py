"""Camera -> robot-base calibration with the ArUco marker on the plate (no ROS needed).

    ros2 run peg_sorting calibrate_aruco            (stop the vision node first - it owns the camera)
    ros2 run peg_sorting calibrate_aruco --mock     (try it on the synthetic scene)

Before running, measure and enter in config/camera_calib.yaml -> aruco:
  marker_length_m       side of the black square (m)
  marker_in_base        x, y of the marker CENTRE from the robot base origin
                        (on the table, directly below joint 1), yaw_deg of the marker
(or pass --length --x --y --yaw).  See docs/calibration.md for pictures.

The marker pose is estimated from the colour corners AND the depth image (table
plane fitted with RANSAC, corner rays intersected with it), which is several
times more accurate than solvePnP alone.  30 frames are averaged.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

from ..utils.config import load_yaml, update_yaml
from ..utils.transforms import T_to_dict, average_rotations, invert_T, make_T, matrix_to_rpy
from ..vision.aruco import detect_markers, marker_in_base_T, marker_pose, marker_pose_on_plane
from ..vision.camera import open_camera
from ..vision.geometry import points_in_base, project


def calibrate(frames, aruco_cfg, use_depth=True, log=print):
    dict_name = aruco_cfg.get('dictionary', 'auto')
    want_id = int(aruco_cfg.get('marker_id', -1))
    L = float(aruco_cfg['marker_length_m'])
    T_bm = marker_in_base_T(aruco_cfg)
    Rs, ts = [], []
    used_dict, used_id, last_corners = None, None, None
    for fr in frames:
        name, found = detect_markers(fr.color, dict_name if used_dict is None else used_dict)
        if not found:
            continue
        used_dict = name
        cand = [f for f in found if want_id < 0 or f[0] == want_id]
        if not cand:
            continue
        mid, corners = cand[0]
        used_id = mid
        last_corners = corners
        T_cm = marker_pose_on_plane(corners, fr.depth, fr.K) if use_depth else None
        if T_cm is None:
            T_cm = marker_pose(corners, L, fr.K, fr.dist)
        if T_cm is None:
            continue
        T = T_bm @ invert_T(T_cm)
        Rs.append(T[:3, :3])
        ts.append(T[:3, 3])
    if not ts:
        return None, {'error': 'marker not found'}
    ts = np.array(ts)
    med = np.median(ts, axis=0)
    keep = np.linalg.norm(ts - med, axis=1) < 0.01
    if keep.sum() < 3:
        keep[:] = True
    R = average_rotations([R for R, k in zip(Rs, keep) if k])
    t = ts[keep].mean(axis=0)
    info = {'dictionary': used_dict, 'marker_id': used_id, 'frames_used': int(keep.sum()),
            'translation_std_mm': (ts[keep].std(axis=0) * 1e3).round(2).tolist(), 'corners': last_corners}
    return make_T(R, t), info


def table_check(frame, T, vcfg):
    pts = points_in_base(frame.depth, frame.K, T)
    ws = vcfg['workspace']
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    with np.errstate(invalid='ignore'):
        m = (x > ws['x'][0]) & (x < ws['x'][1]) & (y > ws['y'][0]) & (y < ws['y'][1]) & np.isfinite(z)
    zz = z[m]
    if zz.size == 0:
        return None
    # the table is the most common height
    hist, edges = np.histogram(zz, bins=np.arange(-0.05, 0.05, 0.002))
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k + 1])


def draw_check(frame, T, vcfg, corners=None):
    img = frame.color.copy()
    if corners is not None:
        c = corners.mean(axis=0)
        d = (corners[1] - corners[0]) + (corners[2] - corners[3])
        tip = c + 1.2 * d / 2
        cv2.polylines(img, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.arrowedLine(img, tuple(c.astype(int)), tuple(tip.astype(int)), (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(img, 'marker x-axis', tuple((tip + 5).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    ws = vcfg['workspace']
    corners = np.array([[ws['x'][0], ws['y'][0], 0], [ws['x'][1], ws['y'][0], 0],
                        [ws['x'][1], ws['y'][1], 0], [ws['x'][0], ws['y'][1], 0]])
    uv = project(corners, frame.K, T).astype(np.int32)
    cv2.polylines(img, [uv], True, (0, 255, 255), 1)
    o = project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]), frame.K, T).astype(int)
    for k, c in zip(range(1, 4), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(img, tuple(o[0]), tuple(o[k]), c, 2)
    cv2.putText(img, 'robot base (x red, y green, z blue)', (int(o[0][0]) + 5, int(o[0][1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for gx in np.arange(0.1, ws['x'][1] + 1e-6, 0.1):
        for gy in np.arange(-0.3, 0.31, 0.1):
            p = project(np.array([[gx, gy, 0]]), frame.K, T)[0].astype(int)
            cv2.circle(img, tuple(p), 3, (255, 0, 255), -1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--length', type=float, help='marker side length (m)')
    ap.add_argument('--x', type=float)
    ap.add_argument('--y', type=float)
    ap.add_argument('--yaw', type=float, help='deg')
    ap.add_argument('--no-depth', action='store_true', help='use solvePnP only')
    ap.add_argument('--dry-run', action='store_true', help='do not write camera_calib.yaml')
    a, _ = ap.parse_known_args()
    vcfg = load_yaml('vision.yaml')
    calib = load_yaml('camera_calib.yaml')
    acfg = calib['aruco']
    if a.length:
        acfg['marker_length_m'] = a.length
    for k in ('x', 'y'):
        if getattr(a, k) is not None:
            acfg['marker_in_base'][k] = getattr(a, k)
    if a.yaw is not None:
        acfg['marker_in_base']['yaw_deg'] = a.yaw
    sim = None
    if a.mock:
        from ..sim.synthetic_scene import SimWorld
        sc = load_yaml('sim_scene.yaml')
        sim = SimWorld(sc)
        acfg['marker_length_m'] = sc['aruco']['length']
        acfg['marker_in_base'] = {'x': sc['aruco']['x'], 'y': sc['aruco']['y'], 'z': 0.0,
                                  'yaw_deg': sc['aruco'].get('yaw_deg', 0)}
    cam = open_camera(vcfg, mock=a.mock, sim_world=sim)
    print('collecting %d frames ...' % a.frames)
    frames = []
    for _ in range(a.frames):
        fr = cam.get_frame()
        if fr is not None:
            frames.append(fr)
    cam.stop()
    T, info = calibrate(frames, acfg, use_depth=not a.no_depth)
    if T is None:
        print('FAILED: %s.  Is the marker visible and flat?  Not covered by the arm?' % info['error'])
        return 1
    roll, pitch, yaw = matrix_to_rpy(T[:3, :3])
    print('marker %s id %s, %d frames, translation std %s mm' % (info['dictionary'], info['marker_id'],
                                                                info['frames_used'], info['translation_std_mm']))
    print('camera position in base frame: x=%.3f y=%.3f z=%.3f m' % tuple(T[:3, 3]))
    print('camera rpy: %.1f %.1f %.1f deg' % tuple(math.degrees(v) for v in (roll, pitch, yaw)))
    tz = table_check(frames[-1], T, vcfg)
    if tz is not None:
        print('table height seen through this calibration: %+.1f mm (should be ~0)' % (tz * 1e3))
    if sim is not None:
        err = (T[:3, 3] - sim.T_base_cam[:3, 3]) * 1e3
        R = T[:3, :3].T @ sim.T_base_cam[:3, :3]
        print('[mock] error vs ground truth: %s mm, %.2f deg' % (
            np.round(err, 1), math.degrees(math.acos(min(1.0, (np.trace(R) - 1) / 2)))))
    img = draw_check(frames[-1], T, vcfg, info.get('corners'))
    out = os.path.expanduser('~/peg_sorting_logs')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'calibration_check_%d.png' % int(time.time()))
    cv2.imwrite(path, img)
    print('check image: %s' % path)
    print('  - red arrow on the marker = marker x-axis.  marker_in_base.yaw_deg must be the angle of that arrow')
    print('    from the robot\'s forward (x) direction, counter-clockwise seen from above (left = +90).')
    print('  - the drawn robot base axes must sit under joint 1, the yellow outline on the work area.')
    print('    If they are rotated/shifted, fix yaw_deg / x / y and run again.')
    if not a.dry_run and not a.mock:
        d = T_to_dict(T)
        p = update_yaml('camera_calib.yaml', {
            'calibrated': True, 'method': 'aruco', 'date': time.strftime('%Y-%m-%d %H:%M'),
            'T_base_cam': d,
            # fine corrections belong to the OLD camera pose - reset them (redo calibrate_workspace)
            'xyz_correction_m': [0.0, 0.0, 0.0],
            'color_offset_m': None,  # (an empty dict would be merged, not reset)
            'planar_correction': {'A': [[1.0, 0.0], [0.0, 1.0]], 'b': [0.0, 0.0],
                                  'note': 'reset by calibrate_aruco - run calibrate_workspace'},
            'aruco': {'dictionary': info['dictionary'], 'marker_id': info['marker_id'],
                                       'marker_length_m': acfg['marker_length_m'],
                                       'marker_in_base': acfg['marker_in_base']}},
            header='Camera -> robot base calibration (written by calibrate_aruco). See docs/calibration.md.')
        print('saved to %s' % p)
        print('The fine corrections from before were reset: now run calibrate_workspace (vision restarted).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

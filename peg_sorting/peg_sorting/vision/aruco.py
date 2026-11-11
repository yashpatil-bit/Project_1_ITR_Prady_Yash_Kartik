"""ArUco marker detection + pose, compatible with old (<4.7) and new OpenCV APIs."""
import math

import cv2
import numpy as np

from ..utils.transforms import make_T, rot_z

COMMON_DICTS = ['DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
                'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_6X6_50', 'DICT_6X6_250',
                'DICT_7X7_50', 'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_36h11', 'DICT_APRILTAG_25h9']


def _detect_with(gray, dict_name):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    if hasattr(aruco, 'ArucoDetector'):
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    else:  # OpenCV < 4.7
        params = aruco.DetectorParameters_create()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)]


def detect_markers(color, dictionary='auto'):
    """Returns (dict_name, [(id, corners(4,2))]).  dictionary='auto' tries COMMON_DICTS."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color
    names = COMMON_DICTS if dictionary in (None, '', 'auto') else [dictionary]
    for name in names:
        if not hasattr(cv2.aruco, name):
            continue
        found = _detect_with(gray, name)
        if found:
            return name, found
    return None, []


def marker_pose(corners, marker_length, K, dist):
    """T_cam_marker (4x4).  Marker frame: origin at centre, x right, y up, z out of the marker."""
    h = marker_length / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE') else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), K, dist, flags=flag)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec.reshape(3))


def marker_in_base_T(cfg):
    """T_base_marker from the measured marker pose in camera_calib.yaml (marker lying flat, face up)."""
    m = cfg['marker_in_base']
    return make_T(rot_z(math.radians(float(m.get('yaw_deg', 0.0)))),
                  [float(m['x']), float(m['y']), float(m.get('z', 0.0))])


def fit_plane(points, iterations=300, inlier_m=0.003, seed=0):
    """RANSAC plane through (N,3) points, refined by least squares on the inliers.
    Returns (normal (unit, the camera origin on its positive side), d) with n.p + d = 0."""
    P = points[np.isfinite(points).all(axis=1)]
    if len(P) < 50:
        return None
    rng = np.random.default_rng(seed)
    best_inl = None
    for _ in range(iterations):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        inl = np.abs((P - a) @ n) < inlier_m
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    Q = P[best_inl]
    c = Q.mean(axis=0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = Vt[2]
    d = -n @ c
    if d < 0:  # make the camera origin lie on the positive side
        n, d = -n, -d
    return n, d


def marker_pose_on_plane(corners, depth, K, radius_px=80):
    """Accurate T_cam_marker for a marker lying on a flat surface: the surface plane is
    fitted to the depth around the marker and the corner rays are intersected with it.
    Returns T or None."""
    from .geometry import deproject
    c = corners.mean(axis=0)
    h, w = depth.shape
    x0, x1 = int(max(0, c[0] - radius_px)), int(min(w, c[0] + radius_px))
    y0, y1 = int(max(0, c[1] - radius_px)), int(min(h, c[1] + radius_px))
    pts = deproject(depth[y0:y1, x0:x1], np.array([[K[0, 0], 0, K[0, 2] - x0], [0, K[1, 1], K[1, 2] - y0],
                                                   [0, 0, 1]]))
    plane = fit_plane(pts.reshape(-1, 3))
    if plane is None:
        return None
    n, d = plane
    rays = np.column_stack([(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(4)])
    t = -d / (rays @ n)
    P = rays * t[:, None]          # 3-D corners: TL, TR, BR, BL (marker frame: x right, y up)
    origin = P.mean(axis=0)
    x = (P[1] - P[0]) + (P[2] - P[3])
    z = n                           # marker z: out of the surface, towards the camera side
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = make_T(np.column_stack([x, y, z]), origin)
    return T

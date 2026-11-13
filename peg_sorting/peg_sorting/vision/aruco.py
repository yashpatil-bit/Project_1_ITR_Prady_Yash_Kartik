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


def _image_variants(gray):
    """The plain image first, then versions that help in dim / low-contrast light:
    local contrast boost (CLAHE), and the same enlarged 2x (small, blurry markers).
    Yields (image, scale) - corners found must be divided by scale."""
    yield gray, 1.0
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield clahe, 1.0
    big = cv2.resize(clahe, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    yield cv2.GaussianBlur(big, (3, 3), 0), 2.0


def detect_markers(color, dictionary='auto'):
    """Returns (dict_name, [(id, corners(4,2))]).  dictionary='auto' tries COMMON_DICTS.
    Retries on contrast-enhanced / enlarged images when the plain one fails (dim light)."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color
    names = COMMON_DICTS if dictionary in (None, '', 'auto') else [dictionary]
    for img, scale in _image_variants(gray):
        for name in names:
            if not hasattr(cv2.aruco, name):
                continue
            found = _detect_with(img, name)
            if found:
                return name, [(i, c / scale) for i, c in found]
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


def marker_pose_on_plane(corners, depth, K, radius_px=None):
    """Accurate T_cam_marker for a marker lying on a flat surface.

    * the surface ORIENTATION comes from a RANSAC plane fitted to the depth of the
      whole visible table (a small patch is not enough: shiny metal gives no depth),
    * the surface HEIGHT comes from the depth right around the marker (paper gives
      good depth), so the plane passes through the marker even if the board / plate
      / table are at slightly different heights,
    * the four corner rays are intersected with that plane.
    Returns T or None."""
    from .geometry import deproject
    P = deproject(depth, K)
    h, w = depth.shape
    c = corners.mean(axis=0)
    side = float(np.mean(np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)))
    # 1) dominant plane over a large region around the marker (or the whole image)
    r = radius_px or max(h, w)
    vv, uu = np.mgrid[0:h, 0:w]
    near = (uu - c[0]) ** 2 + (vv - c[1]) ** 2 < r * r
    pts = P[near & (depth > 0)]
    if len(pts) > 40000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 40000, replace=False)]
    plane = fit_plane(pts, iterations=400, inlier_m=0.004)
    if plane is None:
        return None
    n, d = plane
    # 2) height: depth of the marker paper itself (quad grown by 20 %)
    quad = ((corners - c) * 1.2 + c).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad, 1)
    mp = P[(mask > 0) & (depth > 0)]
    if len(mp) < 30:  # paper without depth: use a ring around it instead
        ring = np.zeros((h, w), np.uint8)
        cv2.circle(ring, tuple(int(v) for v in c), int(2.0 * side), 1, -1)
        mp = P[(ring > 0) & (mask == 0) & (depth > 0)]
    if len(mp) >= 30:
        d_local = -float(np.median(mp @ n))
        if abs(d_local - d) < 0.03:   # sanity: the marker is on (or near) the dominant plane
            d = d_local
    rays = np.column_stack([(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(4)])
    t = -d / (rays @ n)
    Pc = rays * t[:, None]          # 3-D corners: TL, TR, BR, BL (marker frame: x right, y up)
    origin = Pc.mean(axis=0)
    x = (Pc[1] - Pc[0]) + (Pc[2] - Pc[3])
    z = n                           # marker z: out of the surface, towards the camera side
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), origin)

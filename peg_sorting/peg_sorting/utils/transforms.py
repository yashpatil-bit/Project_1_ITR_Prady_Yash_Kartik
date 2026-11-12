"""Small rigid-body transform helpers (numpy only)."""
import math
import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_matrix(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T, pts):
    """Apply 4x4 transform to (N,3) points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def average_rotations(Rs):
    """Chordal L2 mean of rotation matrices (SVD projection)."""
    M = np.sum(np.asarray(Rs), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def kabsch(src, dst):
    """Best rigid transform T (4x4) with dst ~= T * src.  src, dst: (N,3)."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cd - R @ cs
    return make_T(R, t)


def wrap_angle(a, period=2 * math.pi):
    """Wrap angle into [-period/2, period/2)."""
    return (a + period / 2.0) % period - period / 2.0


def T_to_dict(T):
    roll, pitch, yaw = matrix_to_rpy(T[:3, :3])
    return {
        'translation_m': [float(x) for x in T[:3, 3]],
        'rpy_rad': [float(roll), float(pitch), float(yaw)],
        'matrix': [[float(x) for x in row] for row in T],
    }


def T_from_dict(d):
    if 'matrix' in d and d['matrix'] is not None:
        return np.array(d['matrix'], dtype=float)
    R = rpy_to_matrix(*d['rpy_rad'])
    return make_T(R, d['translation_m'])


def fit_planar(src_xy, dst_xy):
    """Rotation about z + 2-D shift with dst ~= R src + t.  Returns (theta, t, residuals)."""
    s, d = np.asarray(src_xy, float), np.asarray(dst_xy, float)
    cs, cd = s.mean(axis=0), d.mean(axis=0)
    H = (s - cs).T @ (d - cd)
    theta = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    t = cd - R @ cs
    res = (s @ R.T + t) - d
    return theta, t, res


def fit_affine_2d(src_xy, dst_xy):
    """Least-squares 2-D affine map dst ~= A @ src + b.  Returns (A (2x2), b (2,), residuals)."""
    s = np.asarray(src_xy, float)
    d = np.asarray(dst_xy, float)
    X = np.column_stack([s, np.ones(len(s))])
    M, _, _, _ = np.linalg.lstsq(X, d, rcond=None)
    A, b = M[:2].T, M[2]
    return A, b, (s @ A.T + b) - d


def fit_similarity_2d(src_xy, dst_xy):
    """dst ~= scale * R(theta) @ src + t.  Returns (scale, theta, t, residuals)."""
    s = np.asarray(src_xy, float)
    d = np.asarray(dst_xy, float)
    cs, cd = s.mean(axis=0), d.mean(axis=0)
    a, c = s - cs, d - cd
    H = a.T @ c
    theta = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    scale = float(np.sum((a @ R.T) * c) / max(np.sum(a * a), 1e-12))
    t = cd - scale * R @ cs
    return scale, theta, t, (scale * s @ R.T + t) - d


def planar_correction_from_config(calib):
    """(A, b) of the camera -> robot x-y correction stored by calibrate_workspace (identity if none)."""
    pc = (calib or {}).get('planar_correction') or {}
    A = np.array(pc.get('A', [[1.0, 0.0], [0.0, 1.0]]), dtype=float)
    b = np.array(pc.get('b', [0.0, 0.0]), dtype=float)
    return A, b

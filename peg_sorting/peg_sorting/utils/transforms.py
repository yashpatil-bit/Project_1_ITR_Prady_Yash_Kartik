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

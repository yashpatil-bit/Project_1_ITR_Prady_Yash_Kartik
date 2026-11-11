"""Pixel <-> 3-D geometry: deprojection, base-frame point clouds, bird's-eye view."""
import numpy as np

from ..utils.transforms import invert_T


def deproject(depth, K):
    """(H,W) depth in metres -> (H,W,3) points in the camera optical frame
    (x right, y down, z forward).  Invalid depth -> NaN."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=-1)
    pts[z <= 0] = np.nan
    return pts


def points_in_base(depth, K, T_base_cam):
    pts = deproject(depth, K)
    R = T_base_cam[:3, :3].astype(np.float32)
    t = T_base_cam[:3, 3].astype(np.float32)
    return pts @ R.T + t


def project(points_base, K, T_base_cam):
    """(N,3) base-frame points -> (N,2) pixel coordinates."""
    T_cam_base = invert_T(T_base_cam)
    p = np.asarray(points_base, dtype=float) @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    uv = p[:, :2] / p[:, 2:3]
    return np.column_stack([K[0, 0] * uv[:, 0] + K[0, 2], K[1, 1] * uv[:, 1] + K[1, 2]])


class BirdsEyeView:
    """Top-down orthographic view of the plane z = z0 (base frame).

    BEV pixel (u, v): u grows towards -y (robot's right), v grows towards -x
    (towards the robot).  So the image is the scene seen from above with the
    robot at the bottom, looking "up" the image.  Use ``to_base`` /
    ``to_pixel`` to convert.
    """

    def __init__(self, x_range, y_range, resolution):
        self.x_min, self.x_max = float(x_range[0]), float(x_range[1])
        self.y_min, self.y_max = float(y_range[0]), float(y_range[1])
        self.res = float(resolution)
        self.width = int(round((self.y_max - self.y_min) / self.res))
        self.height = int(round((self.x_max - self.x_min) / self.res))
        # (u, v, 1) -> (X, Y, 1)
        self.A = np.array([[0.0, -self.res, self.x_max],
                           [-self.res, 0.0, self.y_max],
                           [0.0, 0.0, 1.0]])

    def to_base(self, uv):
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        X = self.x_max - uv[:, 1] * self.res
        Y = self.y_max - uv[:, 0] * self.res
        return np.column_stack([X, Y])

    def to_pixel(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        u = (self.y_max - xy[:, 1]) / self.res
        v = (self.x_max - xy[:, 0]) / self.res
        return np.column_stack([u, v])

    def homography_img_from_bev(self, K, T_base_cam, z0):
        T_cam_base = invert_T(T_base_cam)
        R = T_cam_base[:3, :3]
        t = T_cam_base[:3, 3]
        M = np.column_stack([R[:, 0], R[:, 1], R[:, 2] * z0 + t])
        return K @ M @ self.A

    def warp(self, image, K, T_base_cam, z0, nearest=False):
        import cv2
        H = self.homography_img_from_bev(K, T_base_cam, z0)
        flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
        return cv2.warpPerspective(image, H, (self.width, self.height), flags=flags,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

"""Trajectory generation: minimum-jerk joint-space moves and Cartesian
(straight line / cylindrical) moves sampled through the IK.

A trajectory is a ``Trajectory`` (times [s], positions [N x 4] rad); the
driver interpolates linearly between the (dense, 50 Hz) samples.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Limits:
    joint_vel: float = 1.5        # rad/s  (per joint, peak)
    joint_acc: float = 4.0        # rad/s^2
    cart_vel: float = 0.20        # m/s    (peak, free space)
    cart_vel_slow: float = 0.05   # m/s    (final approach / insertion)
    cart_acc: float = 0.8         # m/s^2
    dt: float = 0.02              # sample period (s)
    min_duration: float = 0.15    # s

    @classmethod
    def from_config(cls, cfg):
        lim = cls()
        for k, v in (cfg or {}).items():
            if hasattr(lim, k):
                setattr(lim, k, float(v))
        return lim


@dataclass
class Trajectory:
    t: np.ndarray        # (N,)
    q: np.ndarray        # (N, 4)

    @property
    def duration(self):
        return float(self.t[-1]) if len(self.t) else 0.0

    def __add__(self, other):
        if len(self.t) == 0:
            return other
        if len(other.t) == 0:
            return self
        t2 = other.t + self.t[-1]
        # drop duplicated first sample of `other`
        return Trajectory(np.concatenate([self.t, t2[1:]]), np.vstack([self.q, other.q[1:]]))


def min_jerk_s(tau):
    """Minimum-jerk time scaling s(tau), tau in [0,1]."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


# peak velocity / accel of min-jerk relative to (distance / T) and (distance / T^2)
_MJ_PEAK_VEL = 1.875
_MJ_PEAK_ACC = 5.7735


def min_jerk_duration(distance, vmax, amax, min_duration=0.0):
    distance = abs(distance)
    if distance < 1e-9:
        return min_duration
    T_v = _MJ_PEAK_VEL * distance / vmax
    T_a = math.sqrt(_MJ_PEAK_ACC * distance / amax)
    return max(T_v, T_a, min_duration)


def joint_move(q_start, q_goal, limits, speed_scale=1.0, duration=None):
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    dq = q_goal - q_start
    if duration is None:
        duration = min_jerk_duration(np.max(np.abs(dq)), limits.joint_vel * speed_scale,
                                     limits.joint_acc * speed_scale, limits.min_duration)
    n = max(2, int(math.ceil(duration / limits.dt)) + 1)
    t = np.linspace(0.0, duration, n)
    s = min_jerk_s(t / duration if duration > 0 else np.ones_like(t))
    q = q_start[None, :] + s[:, None] * dq[None, :]
    return Trajectory(t, q)


def cartesian_path(kin, waypoints, pitch, limits, vmax=None, q_seed=None, speed_scale=1.0):
    """Move the tool through Cartesian waypoints [(x,y,z), ...] at a constant
    tool pitch, with a single minimum-jerk time scaling over the whole path.

    Interpolation is done in *cylindrical* coordinates (r, theta, z) around
    the base so that sweeps around the robot stay at constant radius/height
    (no corner cutting through the robot) while straight vertical/radial moves
    are still straight lines.

    Returns a Trajectory or raises ValueError if any sample is unreachable."""
    vmax = (vmax or limits.cart_vel) * speed_scale
    wp = np.asarray(waypoints, dtype=float)
    cyl = np.column_stack([np.hypot(wp[:, 0], wp[:, 1]), np.unwrap(np.arctan2(wp[:, 1], wp[:, 0])), wp[:, 2]])
    # metric length of each segment (arc length for the theta part)
    seg_len = []
    for a, b in zip(cyl[:-1], cyl[1:]):
        r_mean = 0.5 * (a[0] + b[0])
        seg_len.append(math.sqrt((b[0] - a[0]) ** 2 + (r_mean * (b[1] - a[1])) ** 2 + (b[2] - a[2]) ** 2))
    seg_len = np.array(seg_len)
    total = float(np.sum(seg_len))
    if total < 1e-6:
        q = kin.ik(*wp[0], pitch)
        if q is None:
            raise ValueError('unreachable waypoint %s' % (wp[0],))
        return Trajectory(np.array([0.0, limits.min_duration]), np.array([q, q]))
    duration = min_jerk_duration(total, vmax, limits.cart_acc * speed_scale, limits.min_duration)
    n = max(2, int(math.ceil(duration / limits.dt)) + 1)
    t = np.linspace(0.0, duration, n)
    s = min_jerk_s(t / duration) * total
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    qs = []
    prev = q_seed
    for si in s:
        k = int(np.clip(np.searchsorted(cum, si, side='right') - 1, 0, len(seg_len) - 1))
        frac = 0.0 if seg_len[k] < 1e-9 else (si - cum[k]) / seg_len[k]
        frac = min(max(frac, 0.0), 1.0)
        r, th, z = cyl[k] + frac * (cyl[k + 1] - cyl[k])
        q = kin.ik(r * math.cos(th), r * math.sin(th), z, pitch)
        if q is None:
            raise ValueError('unreachable point r=%.3f th=%.1fdeg z=%.3f pitch=%.1fdeg'
                             % (r, math.degrees(th), z, math.degrees(pitch)))
        if prev is not None:
            # keep joint1 continuous
            q[0] = prev[0] + ((q[0] - prev[0] + math.pi) % (2 * math.pi) - math.pi)
        qs.append(q)
        prev = q
    qs = np.array(qs)
    # Respect joint velocity limit: stretch time if any joint is too fast.
    if len(t) > 1:
        vel = np.max(np.abs(np.diff(qs, axis=0)), axis=0) / np.diff(t).min()
        ratio = float(np.max(vel) / (limits.joint_vel * speed_scale))
        if ratio > 1.0:
            t = t * ratio
    return Trajectory(t, qs)


def chaikin(points, iterations=2):
    """Corner-cutting smoothing of a polyline (keeps end points)."""
    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            new.append(0.75 * a + 0.25 * b)
            new.append(0.25 * a + 0.75 * b)
        new.append(pts[-1])
        pts = np.array(new)
    return pts

"""2-D obstacle-avoiding transport path around the robot base.

While carrying a peg the arm moves at constant height.  The other pegs are
~10 cm tall and stick up higher than the carried peg's bottom, so they are
treated as circular obstacles in the x-y plane.  Paths are built in polar
coordinates (r, theta) around joint 1 because that is how the arm naturally
moves:

* direct  : interpolate r and theta together (a spiral arc)
* inner   : pull in to a smaller radius, rotate, extend out
* outer   : extend to a larger radius, rotate, pull in

The shortest collision-free candidate is returned.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CircleObstacle:
    x: float
    y: float
    radius: float
    label: str = ''


def _polar(p):
    return math.hypot(p[0], p[1]), math.atan2(p[1], p[0])


def _sample_polar_polyline(polar_pts, step=0.005):
    """Densely sample a polyline defined in (r, theta) with linear interpolation in r, theta."""
    out = []
    for (r0, t0), (r1, t1) in zip(polar_pts[:-1], polar_pts[1:]):
        dt = (t1 - t0 + math.pi) % (2 * math.pi) - math.pi
        length = math.hypot(r1 - r0, 0.5 * (r0 + r1) * dt)
        n = max(2, int(length / step) + 1)
        for s in np.linspace(0, 1, n):
            r = r0 + s * (r1 - r0)
            t = t0 + s * dt
            out.append((r * math.cos(t), r * math.sin(t)))
    return np.array(out)


def path_length(xy):
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0


def min_clearance(xy, obstacles):
    if not obstacles:
        return float('inf')
    best = float('inf')
    for o in obstacles:
        d = np.hypot(xy[:, 0] - o.x, xy[:, 1] - o.y) - o.radius
        best = min(best, float(np.min(d)))
    return best


def plan_transport(start_xy, goal_xy, obstacles, r_min=0.10, r_max=0.30, n_radii=9):
    """Return (waypoints_xy list, info dict).  Waypoints include start and goal."""
    r0, t0 = _polar(start_xy)
    r1, t1 = _polar(goal_xy)
    # Ignore obstacles that contain the start or goal (e.g. the peg we are carrying
    # or a peg sitting right next to the target) - nothing we can do about those.
    obs = [o for o in obstacles
           if math.hypot(o.x - start_xy[0], o.y - start_xy[1]) > o.radius
           and math.hypot(o.x - goal_xy[0], o.y - goal_xy[1]) > o.radius]

    candidates = [('direct', [(r0, t0), (r1, t1)])]
    for r in np.linspace(r_min, r_max, n_radii):
        if r < min(r0, r1):
            candidates.append(('inner r=%.2f' % r, [(r0, t0), (r, t0), (r, t1), (r1, t1)]))
        elif r > max(r0, r1):
            candidates.append(('outer r=%.2f' % r, [(r0, t0), (r, t0), (r, t1), (r1, t1)]))

    best = None
    fallback = None
    for name, pol in candidates:
        xy = _sample_polar_polyline(pol)
        clr = min_clearance(xy, obs)
        length = path_length(xy)
        if fallback is None or clr > fallback[2]:
            fallback = (name, pol, clr, length)
        if clr >= 0.0 and (best is None or length < best[3]):
            best = (name, pol, clr, length)
    chosen = best or fallback
    name, pol, clr, length = chosen
    wps = [(r * math.cos(t), r * math.sin(t)) for r, t in pol]
    wps[0] = tuple(start_xy)
    wps[-1] = tuple(goal_xy)
    return wps, {'route': name, 'clearance': clr, 'length': length, 'collision_free': best is not None}

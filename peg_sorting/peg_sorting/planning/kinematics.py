"""Forward / inverse kinematics of the ROBOTIS OpenMANIPULATOR-X (Manipulator X).

Frames and conventions
----------------------
* Robot base frame: origin on the table surface directly below the joint-1
  (base yaw) axis, x forward (where the arm points when joint1 = 0), y to the
  left, z up.  NOTE: ROBOTIS' own "world" frame is 12 mm behind joint 1; we
  put the origin on the joint-1 axis because that is much easier to measure.
* Joint angles are the ROBOTIS convention (0 rad = Dynamixel tick 2048):
  joint1 positive = turn left (CCW seen from above); joints 2-4 positive =
  pitch the arm forward / down.
* "pitch" of the tool = q2 + q3 + q4, measured *downwards* from horizontal:
  pitch = 0 -> gripper points forward horizontally,
  pitch = +pi/2 -> gripper points straight down (top grasp).

Geometry (ROBOTIS URDF, metres)
-------------------------------
joint2 is ``d1`` above the base origin (0.017 + 0.0595 = 0.0765),
joint2 -> joint3 is (0.024 forward, 0.128 up) in the zero pose,
joint3 -> joint4 is 0.124 forward, joint4 -> tool point is ``tool_length``
(0.126 for the stock gripper; measure it for custom fingers - see
docs/calibration.md).

Because joints 2-4 are all parallel, the gripper's jaw-closing axis is always
horizontal and tangential (perpendicular to the arm plane).  The arm has no
wrist-roll, so the gripper yaw is always equal to joint1.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArmGeometry:
    d1: float = 0.0765            # base origin -> joint2 height
    base_z_offset: float = 0.0    # robot base mounted higher/lower than the table (m)
    l1_forward: float = 0.024     # joint2 -> joint3, forward component
    l1_up: float = 0.128          # joint2 -> joint3, upward component
    l2: float = 0.124             # joint3 -> joint4
    tool_length: float = 0.126    # joint4 -> tool point (grasp point between fingers)
    lower: list = field(default_factory=lambda: [-2.827, -1.791, -0.942, -1.791])
    upper: list = field(default_factory=lambda: [2.827, 1.571, 1.382, 2.042])

    @classmethod
    def from_config(cls, cfg):
        k = cfg.get('kinematics', cfg)
        g = cls()
        for name in ('d1', 'base_z_offset', 'l1_forward', 'l1_up', 'l2', 'tool_length'):
            if name in k:
                setattr(g, name, float(k[name]))
        lim = cfg.get('joint_limits_rad')
        if lim:
            g.lower = [float(v) for v in lim['lower']]
            g.upper = [float(v) for v in lim['upper']]
        return g

    @property
    def l1(self):
        return math.hypot(self.l1_forward, self.l1_up)

    @property
    def alpha0(self):
        """Angle of the joint2->joint3 link above horizontal in the zero pose."""
        return math.atan2(self.l1_up, self.l1_forward)

    @property
    def shoulder_z(self):
        return self.d1 + self.base_z_offset


class ManipulatorXKinematics:
    def __init__(self, geometry=None):
        self.g = geometry or ArmGeometry()

    # ------------------------------------------------------------------ FK
    def fk(self, q):
        """q = [q1..q4] -> (x, y, z, pitch) of the tool point."""
        q1, q2, q3, q4 = [float(v) for v in q[:4]]
        g = self.g
        a = g.alpha0 - q2
        r = g.l1 * math.cos(a) + g.l2 * math.cos(q2 + q3) + g.tool_length * math.cos(q2 + q3 + q4)
        z = (g.shoulder_z + g.l1 * math.sin(a) - g.l2 * math.sin(q2 + q3)
             - g.tool_length * math.sin(q2 + q3 + q4))
        return (r * math.cos(q1), r * math.sin(q1), z, q2 + q3 + q4)

    def fk_points(self, q):
        """Positions of base, joint2, joint3, joint4, tool (for plotting/collision checks)."""
        q1, q2, q3, q4 = [float(v) for v in q[:4]]
        g = self.g
        c, s = math.cos(q1), math.sin(q1)
        pts_rz = [(0.0, 0.0), (0.0, g.shoulder_z)]
        a = g.alpha0 - q2
        r3 = g.l1 * math.cos(a)
        z3 = g.shoulder_z + g.l1 * math.sin(a)
        pts_rz.append((r3, z3))
        r4 = r3 + g.l2 * math.cos(q2 + q3)
        z4 = z3 - g.l2 * math.sin(q2 + q3)
        pts_rz.append((r4, z4))
        pts_rz.append((r4 + g.tool_length * math.cos(q2 + q3 + q4),
                       z4 - g.tool_length * math.sin(q2 + q3 + q4)))
        return np.array([[r * c, r * s, z] for r, z in pts_rz])

    # ------------------------------------------------------------------ IK
    def within_limits(self, q, margin=0.0):
        return all(lo + margin <= v <= hi - margin
                   for v, lo, hi in zip(q, self.g.lower, self.g.upper))

    def ik(self, x, y, z, pitch, q1_hint=None):
        """Closed-form IK for a tool position and pitch.  Returns [q1..q4] or None.

        Uses the 'elbow up' solution (the only useful one on this arm)."""
        g = self.g
        r = math.hypot(x, y)
        if r < 1e-6:
            q1 = 0.0 if q1_hint is None else q1_hint
        else:
            q1 = math.atan2(y, x)
        # wrist (joint4) position in the arm plane relative to joint2
        rw = r - g.tool_length * math.cos(pitch)
        zw = z - g.shoulder_z + g.tool_length * math.sin(pitch)
        L1, L2 = g.l1, g.l2
        D = (rw * rw + zw * zw - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        if D > 1.0 or D < -1.0:
            return None
        gamma = -math.acos(D)  # link3 bends down relative to link2 (elbow up)
        theta_a = math.atan2(zw, rw) - math.atan2(L2 * math.sin(gamma), L1 + L2 * math.cos(gamma))
        q2 = g.alpha0 - theta_a
        q3 = -(theta_a + gamma) - q2
        q4 = pitch - q2 - q3
        q = [q1, q2, q3, q4]
        q = [_wrap(v) for v in q]
        if not self.within_limits(q):
            return None
        return q

    def ik_search_pitch(self, x, y, z, preferred=math.pi / 2, lowest=math.radians(20),
                        step=math.radians(2.5)):
        """IK trying the preferred pitch first, then progressively shallower pitches.
        Returns (q, pitch) or (None, None)."""
        p = preferred
        while p >= lowest - 1e-9:
            q = self.ik(x, y, z, p)
            if q is not None:
                return q, p
            p -= step
        return None, None

    def max_feasible_pitch(self, points, preferred=math.pi / 2, lowest=math.radians(20),
                           step=math.radians(2.5)):
        """Largest pitch (<= preferred) for which *all* points (x,y,z) are reachable.
        Used to keep the same tool pitch during a whole pick-and-place so the peg
        stays vertical.  Returns pitch or None."""
        p = preferred
        while p >= lowest - 1e-9:
            if all(self.ik(x, y, z, p) is not None for (x, y, z) in points):
                return p
            p -= step
        return None


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

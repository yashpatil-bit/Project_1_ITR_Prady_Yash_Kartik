"""A tiny ray-casting renderer of the peg-sorting scene (RGB + depth).

Used for ``mock:=true`` runs (whole stack without hardware) and for the unit
tests.  It renders the table/plate, the MDF board with its holes, an ArUco
marker and the pegs as vertical prisms / cylinders, seen from a pinhole
camera.  A peg can be attached to the simulated gripper and released, and it
drops into a socket if it is centred and aligned well enough.
"""
import math
import time
from dataclasses import dataclass

import numpy as np

from ..utils.transforms import invert_T, make_T

N_SIDES = {'triangle': 3, 'square': 4, 'hexagon': 6, 'circle': 0}


def look_at(eye, target, up=(0, 0, 1)):
    """T_base_cam for a camera at `eye` looking at `target` (optical frame: z fwd, x right, y down)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), eye)


def inside_shape(px, py, shape, R, yaw):
    """Vectorised point-in-regular-polygon/circle test (local coords relative to centre)."""
    n = N_SIDES[shape]
    if n == 0:
        return px * px + py * py <= R * R
    ap = R * math.cos(math.pi / n)
    ok = np.ones(np.shape(px), bool)
    for k in range(n):
        a = yaw + math.pi / n + 2 * math.pi * k / n
        ok &= (px * math.cos(a) + py * math.sin(a)) <= ap
    return ok


@dataclass
class SimPeg:
    color: str
    shape: str
    x: float
    y: float
    yaw: float
    size: float      # circumscribed diameter
    height: float
    bgr: tuple
    z_bottom: float = 0.0
    held: bool = False
    grip_offset: float = 0.0   # tool z - peg top when attached
    rel_yaw: float = 0.0       # peg yaw - joint1 while held


@dataclass
class SimSocket:
    shape: str
    x: float
    y: float
    yaw: float
    size: float


class SimWorld:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg['camera']
        self.width, self.height = int(c['width']), int(c['height'])
        self.K = np.array([[c['fx'], 0, self.width / 2.0], [0, c['fy'], self.height / 2.0], [0, 0, 1]])
        self.T_base_cam = look_at(c['eye'], c['target'])
        self.depth_noise = float(c.get('depth_noise_m', 0.0))
        self.color_noise = float(c.get('color_noise', 0.0))
        b = cfg['board']
        self.board_c = np.array(b['center'], float)
        self.board_yaw = math.radians(b['yaw_deg'])
        self.board_len, self.board_w, self.board_t = float(b['length']), float(b['width']), float(b['thickness'])
        self.clearance = float(b.get('clearance', 0.004))
        ax = np.array([math.cos(self.board_yaw), math.sin(self.board_yaw)])
        self.sockets = []
        for s in b['sockets']:
            p = self.board_c + ax * float(s['offset'])
            self.sockets.append(SimSocket(s['shape'], float(p[0]), float(p[1]),
                                          self.board_yaw + math.radians(s.get('yaw_deg', 0)),
                                          float(s['size']) + 2 * self.clearance))
        self.pegs = [SimPeg(p['color'], p['shape'], float(p['x']), float(p['y']), math.radians(p['yaw_deg']),
                            float(p['size']), float(p['height']), tuple(p['bgr'])) for p in cfg['pegs']]
        self._aruco_img = None
        a = cfg.get('aruco')
        if a:
            try:
                import cv2
                d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a['dictionary']))
                gen = cv2.aruco.generateImageMarker if hasattr(cv2.aruco, 'generateImageMarker') \
                    else cv2.aruco.drawMarker
                self._aruco_img = gen(d, int(a['id']), 120)
            except Exception:  # noqa: BLE001
                self._aruco_img = None
        self.rng = np.random.default_rng(1)
        self.grasp_model = 'squeeze'
        self.insert_tolerance = math.radians(float(b.get('insert_tolerance_deg', 8.0)))

    # ------------------------------------------------------------ physics-ish
    def update_gripper(self, tool_xyz, joint1, closed):
        """Attach the nearest peg when the gripper closes around it; release when it opens."""
        held = [p for p in self.pegs if p.held]
        if closed and not held:
            for p in self.pegs:
                top = p.z_bottom + p.height
                if math.hypot(p.x - tool_xyz[0], p.y - tool_xyz[1]) < 0.012 and top - 0.06 < tool_xyz[2] < top:
                    p.held = True
                    p.grip_offset = tool_xyz[2] - top
                    # parallel-jaw squeeze aligns the narrowest width with the jaws
                    p.rel_yaw = _squeeze_rel_yaw(p.shape, p.yaw - joint1) if self.grasp_model == 'squeeze' \
                        else p.yaw - joint1
                    break
        elif not closed and held:
            for p in held:
                p.held = False
                self._drop(p)
        for p in self.pegs:
            if p.held:
                p.x, p.y = float(tool_xyz[0]), float(tool_xyz[1])
                p.z_bottom = tool_xyz[2] - p.grip_offset - p.height
                p.yaw = joint1 + p.rel_yaw

    def _drop(self, p):
        for s in self.sockets:
            if s.shape != p.shape:
                continue
            d = math.hypot(p.x - s.x, p.y - s.y)
            n = N_SIDES[p.shape]
            mis = 0.0
            if n:
                per = 2 * math.pi / n
                mis = abs((p.yaw - s.yaw + per / 2) % per - per / 2)
            if d < 0.006 and mis < self.insert_tolerance:
                p.x, p.y, p.z_bottom = s.x, s.y, 0.0
                return
        # resting on the board or on the table
        p.z_bottom = self.board_t if self._on_board(p.x, p.y) else 0.0

    def _on_board(self, x, y):
        c, s = math.cos(-self.board_yaw), math.sin(-self.board_yaw)
        dx, dy = x - self.board_c[0], y - self.board_c[1]
        u, v = c * dx - s * dy, s * dx + c * dy
        return abs(u) <= self.board_len / 2 and abs(v) <= self.board_w / 2

    def peg_in_socket(self, color):
        for p in self.pegs:
            if p.color == color:
                return any(s.shape == p.shape and math.hypot(p.x - s.x, p.y - s.y) < 0.006 and p.z_bottom < 1e-4
                           for s in self.sockets)
        return False

    # ------------------------------------------------------------ rendering
    def render(self):
        H, W = self.height, self.width
        T = self.T_base_cam
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d_cam = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1)
        d = d_cam @ T[:3, :3].T
        o = T[:3, 3]
        dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
        t_best = np.full((H, W), np.inf)
        col = np.zeros((H, W, 3), np.float32)

        tc = self.cfg['table']
        # table plane z = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            t0 = -o[2] / dz
        hit = (t0 > 0) & np.isfinite(t0)
        X, Y = o[0] + t0 * dx, o[1] + t0 * dy
        on_plate = (X >= tc['plate_x'][0]) & (X <= tc['plate_x'][1]) & (Y >= tc['plate_y'][0]) & (Y <= tc['plate_y'][1])
        t_best[hit] = t0[hit]
        col[hit] = np.array(tc['table_bgr'], np.float32)
        col[hit & on_plate] = np.array(tc['plate_bgr'], np.float32)
        self._draw_aruco(X, Y, hit, col)

        # board top plane
        bt = self.board_t
        with np.errstate(divide='ignore', invalid='ignore'):
            tb = (bt - o[2]) / dz
        Xb, Yb = o[0] + tb * dx, o[1] + tb * dy
        c, s = math.cos(-self.board_yaw), math.sin(-self.board_yaw)
        bu = c * (Xb - self.board_c[0]) - s * (Yb - self.board_c[1])
        bv = s * (Xb - self.board_c[0]) + c * (Yb - self.board_c[1])
        on_board = (np.abs(bu) <= self.board_len / 2) & (np.abs(bv) <= self.board_w / 2) & (tb > 0)
        in_hole = np.zeros_like(on_board)
        near_hole = np.zeros_like(on_board)
        for sk in self.sockets:
            R = sk.size / 2
            lx, ly = Xb - sk.x, Yb - sk.y
            in_hole |= inside_shape(lx, ly, sk.shape, R, sk.yaw)
            near_hole |= inside_shape(lx, ly, sk.shape, R + 0.004, sk.yaw)
        board_hit = on_board & ~in_hole & (tb < t_best)
        t_best[board_hit] = tb[board_hit]
        col[board_hit] = np.array(self.cfg['board']['bgr'], np.float32)
        edge = board_hit & near_hole
        col[edge] = np.array(self.cfg['board']['edge_bgr'], np.float32)

        # pegs
        light = np.array([0.4, 0.6, 0.7])
        for p in self.pegs:
            self._render_peg(p, o, d, t_best, col, light)

        depth = t_best.copy()  # d_cam has z = 1, so the ray parameter is the depth
        depth[~np.isfinite(depth)] = 0.0
        if self.depth_noise > 0:
            depth += self.rng.normal(0, self.depth_noise, depth.shape) * (depth / 0.6) ** 2
            depth[depth < 0] = 0
        if self.color_noise > 0:
            col += self.rng.normal(0, self.color_noise, col.shape)
        color = np.clip(col, 0, 255).astype(np.uint8)
        return color, depth.astype(np.float32)

    def _draw_aruco(self, X, Y, hit, col):
        a = self.cfg.get('aruco')
        if a is None or self._aruco_img is None:
            return
        L = float(a['length'])
        yaw = math.radians(a.get('yaw_deg', 0))
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx = c * (X - a['x']) - s * (Y - a['y'])
        ly = s * (X - a['x']) + c * (Y - a['y'])
        # white quiet zone
        quiet = hit & (np.abs(lx) <= L * 0.75) & (np.abs(ly) <= L * 0.75)
        col[quiet] = 255
        inside = hit & (np.abs(lx) <= L / 2) & (np.abs(ly) <= L / 2)
        n = self._aruco_img.shape[0]
        # marker image: columns along marker x (+lx), rows along marker -y (top row at +ly)
        ci = np.clip(((lx + L / 2) / L * n).astype(int), 0, n - 1)
        ri = np.clip(((L / 2 - ly) / L * n).astype(int), 0, n - 1)
        val = self._aruco_img[ri[inside], ci[inside]].astype(np.float32)
        col[inside] = np.stack([val, val, val], -1) * 0.85 + 10

    def _render_peg(self, p, o, d, t_best, col, light):
        R = p.size / 2
        z0, z1 = p.z_bottom, p.z_bottom + p.height
        base = np.array(p.bgr, np.float32)
        dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
        # top cap
        with np.errstate(divide='ignore', invalid='ignore'):
            tt = (z1 - o[2]) / dz
        hit = (tt > 0) & inside_shape(o[0] + tt * dx - p.x, o[1] + tt * dy - p.y, p.shape, R, p.yaw) & (tt < t_best)
        t_best[hit] = tt[hit]
        col[hit] = base
        n = N_SIDES[p.shape]
        if n == 0:
            # cylinder: |o_xy + t d_xy - c|^2 = R^2
            ox, oy = o[0] - p.x, o[1] - p.y
            A = dx * dx + dy * dy
            B = 2 * (ox * dx + oy * dy)
            C = ox * ox + oy * oy - R * R
            disc = B * B - 4 * A * C
            ok = disc >= 0
            with np.errstate(invalid='ignore', divide='ignore'):
                t = (-B - np.sqrt(np.where(ok, disc, 0))) / (2 * A)
            z = o[2] + t * dz
            hit = ok & (t > 0) & (z >= z0) & (z <= z1) & (t < t_best)
            nx = (ox + t * dx) / R
            ny = (oy + t * dy) / R
            shade = 0.55 + 0.4 * np.abs(nx * light[0] + ny * light[1])
            t_best[hit] = t[hit]
            col[hit] = base * shade[hit][:, None]
            return
        verts = [(p.x + R * math.cos(p.yaw + 2 * math.pi * k / n), p.y + R * math.sin(p.yaw + 2 * math.pi * k / n))
                 for k in range(n)]
        for k in range(n):
            a, b = np.array(verts[k]), np.array(verts[(k + 1) % n])
            e = b - a
            nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)  # outward for CCW vertices
            denom = nrm[0] * dx + nrm[1] * dy
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (nrm[0] * (a[0] - o[0]) + nrm[1] * (a[1] - o[1])) / denom
            px, py, pz = o[0] + t * dx, o[1] + t * dy, o[2] + t * dz
            s_ = ((px - a[0]) * e[0] + (py - a[1]) * e[1]) / (e @ e)
            hit = (t > 0) & (s_ >= 0) & (s_ <= 1) & (pz >= z0) & (pz <= z1) & (t < t_best)
            shade = 0.55 + 0.4 * abs(nrm[0] * light[0] + nrm[1] * light[1])
            t_best[hit] = t[hit]
            col[hit] = base * shade


def _squeeze_rel_yaw(shape, rel):
    stable = {'square': [45.0], 'hexagon': [0.0], 'triangle': [30.0, 90.0]}
    n = N_SIDES[shape]
    if n == 0:
        return rel
    per = 2 * math.pi / n
    cands = [math.radians(s) for s in stable[shape]]
    best = min(cands, key=lambda c: abs((rel - c + per / 2) % per - per / 2))
    return best


class SyntheticCamera:
    """Drop-in replacement for RealSenseCamera."""

    def __init__(self, world: SimWorld):
        self.world = world
        self.K = world.K
        self.dist = np.zeros(5)
        self.serial = 'SIM'
        self.name = 'Synthetic camera'

    def get_frame(self, timeout_ms=2000):
        from ..vision.camera import Frame
        color, depth = self.world.render()
        return Frame(color, depth, self.K.copy(), self.dist.copy(), time.time())

    def stop(self):
        pass


def T_cam_base(world):
    return invert_T(world.T_base_cam)

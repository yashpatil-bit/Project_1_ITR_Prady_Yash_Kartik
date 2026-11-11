"""Where to put the socket board so the pegs arrive aligned with their sockets.

Because the Manipulator X cannot turn the gripper about the vertical axis, a
peg's yaw on arrival is fixed by the socket's bearing from the robot (see
pick_place_planner).  For the 'squeeze' grasp model the error therefore only
depends on the board pose -> search for a reachable board pose (x, y, yaw)
whose sockets all line up.
"""
import math

import numpy as np

from .pick_place_planner import SQUEEZE_STABLE_DEG, SYMMETRY, PegInfo, SocketInfo


def board_model(board_x, board_y, board_yaw, sockets):
    """Socket layout in the board frame: [(shape, u, v, rel_yaw)]."""
    c, s = math.cos(-board_yaw), math.sin(-board_yaw)
    out = []
    for k in sockets:
        dx, dy = k.x - board_x, k.y - board_y
        out.append((k.shape, c * dx - s * dy, s * dx + c * dy, k.yaw - board_yaw))
    return out


def socket_error_deg(shape, socket_yaw, bearing):
    n = SYMMETRY.get(shape)
    if not n:
        return 0.0
    P = 360.0 / n
    return min(abs((math.degrees(socket_yaw) - (math.degrees(bearing) + st) + P / 2) % P - P / 2)
               for st in SQUEEZE_STABLE_DEG[shape])


def place_model(model, bx, by, byaw):
    c, s = math.cos(byaw), math.sin(byaw)
    return [SocketInfo(shape, bx + c * u - s * v, by + s * u + c * v, 0.0, byaw + ry) for shape, u, v, ry in model]


def evaluate(model, bx, by, byaw):
    socks = place_model(model, bx, by, byaw)
    errs = {k.shape: socket_error_deg(k.shape, k.yaw, math.atan2(k.y, k.x)) for k in socks}
    return max(errs.values()) if errs else 0.0, errs, socks


def search(model, planner, current=None, x_range=(0.10, 0.32), y_range=(-0.28, 0.28), step=0.01,
           yaw_step_deg=1.0, peg_height=0.10, board_z=0.005, tolerance_deg=3.0, top=5):
    """Return up to `top` candidate board poses [(worst_err_deg, x, y, yaw, move_cost)], best first.
    A candidate must have every socket reachable by the planner's constant-pitch insertion."""
    kin = planner.kin
    dummy_peg = PegInfo('x', 'circle', 0.15, 0.0, peg_height + planner.table_z, peg_height)
    reach_cache = {}

    def reachable(x, y):
        key = (round(x, 3), round(y, 3))
        if key not in reach_cache:
            h = planner.heights(dummy_peg, SocketInfo('circle', x, y, board_z))
            r = math.hypot(x, y)
            reach_cache[key] = r > 0.11 and kin.max_feasible_pitch(
                [(x, y, h['carry']), (x, y, h['insert_start']), (x, y, h['place'])],
                lowest=math.radians(float(planner.cfg['pitch']['min_deg']))) is not None
        return reach_cache[key]

    cands = []
    for bx in np.arange(x_range[0], x_range[1] + 1e-9, step):
        for by in np.arange(y_range[0], y_range[1] + 1e-9, step):
            for yd in np.arange(0.0, 360.0, yaw_step_deg):
                byaw = math.radians(yd)
                worst, _, socks = evaluate(model, bx, by, byaw)
                if worst > tolerance_deg:
                    continue
                if not all(reachable(k.x, k.y) for k in socks):
                    continue
                cost = 0.0
                if current is not None:
                    dyaw = abs((yd - math.degrees(current[2]) + 180) % 360 - 180)
                    cost = math.hypot(bx - current[0], by - current[1]) + 0.002 * dyaw
                cands.append((worst, float(bx), float(by), byaw, cost))
    cands.sort(key=lambda c: (c[4] + 0.01 * c[0]) if current is not None else c[0])
    # de-duplicate near-identical poses
    out = []
    for c in cands:
        if all(math.hypot(c[1] - o[1], c[2] - o[2]) > 0.03 or abs(c[3] - o[3]) > math.radians(10) for o in out):
            out.append(c)
        if len(out) >= top:
            break
    return out

"""Open-jaw clearance check used by the task manager's scene report."""
import math

from peg_sorting.planning.pick_place_planner import jaw_blockers
from peg_sorting.planning.pick_place_planner import PegInfo


def peg(c, x, y):
    return PegInfo(c, 'circle', x, y, 0.1, 0.1, 0.0, 0.04)


def test_side_by_side_row_is_crowded():
    # lab layout: pegs in a row across the robot's front, centres 5.8 cm apart
    blue, yellow = peg('blue', 0.14, 0.03), peg('yellow', 0.14, -0.028)
    hits = jaw_blockers(yellow, [blue])
    assert len(hits) == 1 and abs(hits[0][1] - 0.018) < 0.004


def test_spread_or_staggered_is_fine():
    a = peg('a', 0.14, 0.0)
    assert not jaw_blockers(a, [peg('b', 0.14, 0.085)])           # 8.5 cm side by side
    assert not jaw_blockers(a, [peg('b', 0.20, 0.0)])             # 6 cm behind (same bearing)
    b = math.atan2(0.0, 0.14)
    assert b == 0.0


def test_triangle_grasp_and_place_offsets():
    from peg_sorting.planning.pick_place_planner import SocketInfo, triangle_grasp_offsets
    R = 0.04 / math.sqrt(3)
    # peg straight ahead, one vertex pointing along +y (towards a jaw), the opposite face towards the other jaw
    tri = PegInfo('yellow', 'triangle', 0.15, 0.0, 0.1, 0.1, math.radians(90), 0.04)
    sock = SocketInfo('triangle', 0.20, 0.0, 0.005, math.radians(90))
    (px, py), (sx, sy) = triangle_grasp_offsets(tri, sock)
    assert abs(px) < 1e-9 and abs(py - R / 4) < 1e-6      # aim midway between face and vertex
    assert abs(sx) < 1e-9 and abs(sy - R / 4) < 1e-6      # the peg centre then lands on the socket centre
    # vertex angle where the jaws see a symmetric outline -> no pick offset
    tri.yaw = math.radians(60)
    (px, py), _ = triangle_grasp_offsets(tri, sock)
    assert math.hypot(px, py) < 1e-9
    # other shapes untouched
    sq = PegInfo('blue', 'square', 0.15, 0.0, 0.1, 0.1, 0.3, 0.04)
    assert triangle_grasp_offsets(sq, SocketInfo('square', 0.2, 0.0, 0.0, 0.0)) == ((0.0, 0.0), (0.0, 0.0))

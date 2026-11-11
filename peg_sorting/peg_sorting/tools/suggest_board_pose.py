"""Suggest where to put the socket board so every peg arrives aligned (vision running).

    ros2 run peg_sorting suggest_board_pose            (driver not needed)
    ros2 run peg_sorting suggest_board_pose --mock     (no ROS at all, synthetic scene)

The vision node measures the board and its sockets; this tool searches for a
reachable board pose where the predicted yaw error of every socket is below
--tol degrees, preferring poses close to where the board is now.  Then move the
board there and watch the live vision window: the numbers next to each socket
are the predicted yaw errors - nudge the board until they are all ~0.
"""
import argparse
import math
import sys

from ..planning.board_placement import board_model, evaluate, search
from ..planning.kinematics import ArmGeometry, ManipulatorXKinematics
from ..planning.pick_place_planner import PickPlacePlanner, SocketInfo
from ..planning.trajectory import Limits
from ..utils.config import load_yaml


def get_scene_ros():
    import rclpy
    from rclpy.node import Node
    from peg_sorting_interfaces.srv import GetScene
    rclpy.init()
    node = Node('suggest_board_pose')
    cli = node.create_client(GetScene, 'vision/get_scene')
    if not cli.wait_for_service(timeout_sec=10.0):
        print('vision node not running')
        sys.exit(1)
    fut = cli.call_async(GetScene.Request(num_frames=10, detect_sockets=True))
    rclpy.spin_until_future_complete(node, fut, timeout_sec=30.0)
    res = fut.result()
    node.destroy_node()
    rclpy.shutdown()
    socks = [SocketInfo(m.shape, m.center.x, m.center.y, m.center.z, m.yaw, m.width) for m in res.sockets]
    return res.board_found, (res.board_center.x, res.board_center.y, res.board_yaw), socks


def get_scene_mock():
    from ..sim.synthetic_scene import SimWorld, SyntheticCamera
    from ..vision.scene import SceneDetector
    w = SimWorld(load_yaml('sim_scene.yaml'))
    det = SceneDetector(load_yaml('colors.yaml'), load_yaml('vision.yaml'), w.T_base_cam)
    _, board, sockets, _ = det.detect_averaged([SyntheticCamera(w).get_frame() for _ in range(2)])
    socks = [SocketInfo(s.shape, s.x, s.y, s.z, s.yaw, s.width) for s in sockets]
    return board.found, (board.x, board.y, board.yaw), socks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--tol', type=float, default=2.0, help='max yaw error (deg)')
    a, _ = ap.parse_known_args()
    found, cur, socks = get_scene_mock() if a.mock else get_scene_ros()
    if not found or len(socks) < 2:
        print('board / sockets not detected (found=%s, %d sockets)' % (found, len(socks)))
        return 1
    robot = load_yaml('robot.yaml')
    task = load_yaml('task.yaml')
    if task['alignment'].get('grasp_model', 'squeeze') != 'squeeze':
        print('grasp_model is "%s": the yaw error depends on how the pegs are placed, not only on the board.'
              % task['alignment']['grasp_model'])
    planner = PickPlacePlanner(ManipulatorXKinematics(ArmGeometry.from_config(robot)), task,
                               Limits.from_config(robot['limits']),
                               table_z=float(load_yaml('vision.yaml').get('table_z', 0.0)))
    model = board_model(cur[0], cur[1], cur[2], socks)
    worst, errs, _ = evaluate(model, *cur)
    print('board now at x=%.3f y=%.3f yaw=%.1f deg -> predicted yaw errors: %s' % (
        cur[0], cur[1], math.degrees(cur[2]), ', '.join('%s %.1f' % kv for kv in errs.items())))
    print('searching ...')
    cands = search(model, planner, current=cur, tolerance_deg=a.tol,
                   board_z=max(s.z for s in socks))
    if not cands:
        print('no reachable board pose with all errors < %.1f deg; try --tol 4' % a.tol)
        return 1
    print('\nSuggested board poses (centre in the robot base frame; yaw = board long axis from robot x, CCW):')
    for worst, x, y, yaw, cost in cands:
        print('  x=%.3f m  y=%+.3f m  yaw=%6.1f deg   worst error %.1f deg   (move %.0f mm, turn %+.0f deg)' % (
            x, y, (math.degrees(yaw) + 180) % 360 - 180, worst, 1e3 * math.hypot(x - cur[0], y - cur[1]),
            (math.degrees(yaw - cur[2]) + 180) % 360 - 180))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

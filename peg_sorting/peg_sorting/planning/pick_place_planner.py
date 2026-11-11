"""Pick-and-place planning for one peg -> socket pair.

The plan is a list of steps that the task manager executes in order:

    PlanStep('gripper', ...)   open / close the gripper
    PlanStep('traj', ...)      follow a joint trajectory
    PlanStep('wait', ...)      settle

Key ideas
---------
* Top grasp with a vertical approach ("approach along the board normal").
  The tool pitch is kept *constant* for the whole pick-and-place, so a peg that
  is grasped vertical stays vertical while carried and when inserted.  The
  largest pitch (closest to straight down) that reaches every waypoint is used.
* Transport happens at the lowest height that lifts the peg's bottom clear of
  the board, along an obstacle-free path around the other pegs.
* Orientation: Manipulator X has no wrist roll, so the gripper yaw is always
  joint1 = atan2(y, x).  A parallel-jaw squeeze turns a polygonal peg so that
  its narrowest width lies between the jaws.  So the peg's yaw when it arrives
  at the socket is fixed by the socket's bearing from the robot.
  ``predicted_misalignment`` computes the error so it can be shown to the
  operator before the run (see README, "Orientation").
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

from .kinematics import ManipulatorXKinematics
from .path_planner import CircleObstacle, plan_transport
from .trajectory import Limits, Trajectory, cartesian_path, joint_move

SYMMETRY = {'triangle': 3, 'square': 4, 'hexagon': 6}

# Stable "vertex angle minus jaw-closing-direction-normal" after a parallel-jaw
# squeeze, in degrees, for each shape (see module docstring).  The jaw-closing
# axis is tangential = joint1 + 90 deg.  Values are relative to joint1.
SQUEEZE_STABLE_DEG = {
    'square': [45.0],          # flats between the jaws
    'hexagon': [0.0],          # flats between the jaws
    'triangle': [30.0, 90.0],  # one face on a jaw, opposite vertex on the other
}


@dataclass
class PegInfo:
    color: str
    shape: str
    x: float
    y: float
    z_top: float
    height: float
    yaw: float = 0.0
    width: float = 0.05


@dataclass
class SocketInfo:
    shape: str
    x: float
    y: float
    z: float           # board top surface height
    yaw: float = 0.0
    width: float = 0.05


@dataclass
class PlanStep:
    kind: str                      # 'traj' | 'gripper' | 'wait'
    label: str
    traj: Optional[Trajectory] = None
    gripper: Optional[str] = None  # 'open' | 'close' | 'release'
    seconds: float = 0.0
    guarded: bool = False          # abort if the arm is blocked (tracking error)


@dataclass
class PickPlacePlan:
    peg: PegInfo
    socket: SocketInfo
    pitch: float
    steps: List[PlanStep] = field(default_factory=list)
    route: str = ''
    misalignment_deg: float = 0.0
    q_end: Optional[list] = None

    @property
    def duration(self):
        return sum(s.traj.duration for s in self.steps if s.traj is not None) + \
            sum(s.seconds for s in self.steps if s.kind == 'wait')


class PlanningError(RuntimeError):
    pass


def _ang_dist(a, b, period):
    d = (a - b + period / 2.0) % period - period / 2.0
    return d


def predicted_peg_yaw_at(shape, peg_yaw, pick_bearing, place_bearing, grasp_model='squeeze'):
    """Predict the peg's vertex angle after grasping at pick_bearing and carrying
    to place_bearing (all radians).  Returns None for circles.

    grasp_model 'squeeze': the parallel jaws turn the peg to its narrowest width
                           (rigid 3-D-printed pegs, hard fingers - the usual case)
    grasp_model 'hold'   : the peg does not turn in the gripper (soft/grippy fingers)
    Either way the peg turns with joint 1 while it is carried."""
    n = SYMMETRY.get(shape)
    if n is None:
        return None
    period = 2 * math.pi / n
    rel = peg_yaw - pick_bearing
    if grasp_model == 'squeeze':
        stable = [math.radians(s) for s in SQUEEZE_STABLE_DEG[shape]]
        rel = min(stable, key=lambda s: abs(_ang_dist(rel, s, period)))
    return (place_bearing + rel) % period


def predicted_misalignment(peg: PegInfo, socket: SocketInfo, grasp_model='squeeze'):
    """Signed yaw error (rad) between the peg and the socket at insertion."""
    n = SYMMETRY.get(socket.shape)
    if n is None:
        return 0.0
    period = 2 * math.pi / n
    yaw_final = predicted_peg_yaw_at(socket.shape, peg.yaw, math.atan2(peg.y, peg.x),
                                     math.atan2(socket.y, socket.x), grasp_model)
    return _ang_dist(socket.yaw, yaw_final, period)


class PickPlacePlanner:
    def __init__(self, kin: ManipulatorXKinematics, task_cfg: dict, limits: Limits, table_z=0.0):
        self.kin = kin
        self.cfg = task_cfg
        self.limits = limits
        self.table_z = table_z
        self.grasp_model = (task_cfg.get('alignment') or {}).get('grasp_model', 'squeeze')

    # ----------------------------------------------------------- matching
    def assign(self, pegs: List[PegInfo], sockets: List[SocketInfo], q_now=None):
        """Match pegs to sockets by shape.  Returns list of (peg, socket), ordered
        so that the closest peg to the current tool position goes first."""
        by_shape = {}
        for s in sockets:
            by_shape.setdefault(s.shape, []).append(s)
        pairs = []
        for p in pegs:
            cands = by_shape.get(p.shape, [])
            if not cands:
                continue
            s = min(cands, key=lambda s: math.hypot(s.x - p.x, s.y - p.y))
            pairs.append((p, s))
        if q_now is not None:
            tx, ty, _, _ = self.kin.fk(q_now)
        else:
            tx, ty = 0.15, 0.0
        ordered = []
        remaining = list(pairs)
        while remaining:
            nxt = min(remaining, key=lambda ps: math.hypot(ps[0].x - tx, ps[0].y - ty))
            ordered.append(nxt)
            remaining.remove(nxt)
            tx, ty = nxt[1].x, nxt[1].y
        return ordered

    # ------------------------------------------------------------ planning
    def heights(self, peg: PegInfo, socket: SocketInfo):
        """All tool-point heights used for one pick-and-place."""
        g, pl, tr = self.cfg['grasp'], self.cfg['place'], self.cfg['transport']
        grasp_depth = min(float(g['grasp_depth']), 0.8 * peg.height)
        below = peg.height - grasp_depth            # peg bottom is this far below the tool point
        board_top = max(socket.z, self.table_z)
        h = {'grasp_depth': grasp_depth, 'peg_bottom_below_tool': below}
        h['grasp'] = peg.z_top - grasp_depth
        h['pre_grasp'] = peg.z_top + float(g['approach_clearance'])
        h['carry'] = max(max(board_top, peg.z_top - peg.height) + float(tr['clearance_above_board']) + below,
                         h['grasp'] + float(tr.get('min_lift', 0.02)))
        h['insert_start'] = board_top + float(pl['insertion_start_above_board']) + below
        h['place'] = self.table_z + float(pl['bottom_clearance']) + below
        h['retreat'] = max(h['place'] + grasp_depth + float(pl['retreat_height']), h['insert_start'])
        return h

    def plan(self, peg: PegInfo, socket: SocketInfo, q_start, other_pegs: List[PegInfo],
             speed_scale=1.0, place_offset=(0.0, 0.0)) -> PickPlacePlan:
        pc = self.cfg['pitch']
        h = self.heights(peg, socket)
        sx = socket.x + place_offset[0]
        sy = socket.y + place_offset[1]
        key_points = [
            (peg.x, peg.y, h['pre_grasp']), (peg.x, peg.y, h['grasp']), (peg.x, peg.y, h['carry']),
            (sx, sy, h['carry']), (sx, sy, h['insert_start']), (sx, sy, h['place']), (sx, sy, h['retreat']),
        ]
        lowest = math.radians(float(pc['min_deg']))
        pitch = self.kin.max_feasible_pitch(key_points, preferred=math.radians(float(pc['preferred_deg'])),
                                            lowest=lowest)
        if pitch is None:
            raise PlanningError('peg %s at (%.3f, %.3f) or socket %s at (%.3f, %.3f) is out of reach '
                                'for a constant-pitch top grasp' % (peg.color, peg.x, peg.y,
                                                                    socket.shape, sx, sy))
        margin = float(self.cfg['transport']['obstacle_margin'])
        obstacles = [CircleObstacle(o.x, o.y, 0.5 * o.width + 0.5 * peg.width + margin, o.color)
                     for o in other_pegs]
        last_err = None
        p = pitch
        while p >= lowest - 1e-9:
            try:
                return self._build(peg, socket, q_start, p, obstacles, (sx, sy), h, speed_scale)
            except ValueError as e:  # an intermediate sample is unreachable -> shallower pitch
                last_err = e
                p -= math.radians(2.5)
        raise PlanningError('no feasible trajectory: %s' % last_err)

    def _build(self, peg, socket, q_start, pitch, obstacles, sxy, h, speed_scale):
        lim = self.limits
        g = self.cfg['grasp']
        kin = self.kin
        sx, sy = sxy
        steps = []

        q_pre = kin.ik(peg.x, peg.y, h['pre_grasp'], pitch)
        if q_pre is None:
            raise ValueError('pre-grasp unreachable')
        steps.append(PlanStep('gripper', 'open gripper', gripper='open'))
        t_pre = joint_move(q_start, q_pre, lim, speed_scale)
        if min(kin.fk(q)[2] for q in t_pre.q) < h['pre_grasp'] - 0.005:
            # the joint-space swing would dip below the approach height (could hit pegs):
            # go to a point higher above the peg first, then straight down
            z_hi = h['pre_grasp'] + 0.06
            q_hi = kin.ik(peg.x, peg.y, z_hi, pitch)
            if q_hi is None:
                raise ValueError('safe approach point unreachable')
            t_pre = joint_move(q_start, q_hi, lim, speed_scale) + cartesian_path(
                kin, [(peg.x, peg.y, z_hi), (peg.x, peg.y, h['pre_grasp'])], pitch, lim, q_seed=q_hi,
                speed_scale=speed_scale)
        steps.append(PlanStep('traj', 'move to pre-grasp', traj=t_pre))
        t_desc = cartesian_path(kin, [(peg.x, peg.y, h['pre_grasp']), (peg.x, peg.y, h['grasp'])], pitch, lim,
                                vmax=float(g.get('approach_speed', lim.cart_vel)), q_seed=q_pre,
                                speed_scale=speed_scale)
        steps.append(PlanStep('traj', 'descend to grasp', traj=t_desc))
        steps.append(PlanStep('wait', 'settle before close', seconds=float(g['settle_time'])))
        steps.append(PlanStep('gripper', 'close gripper', gripper='close'))

        # lift + transport + descend to the insertion start as ONE smooth path
        tr = self.cfg['transport']
        route_xy, info = plan_transport((peg.x, peg.y), (sx, sy), obstacles,
                                        r_min=float(tr.get('r_min', 0.10)), r_max=float(tr.get('r_max', 0.30)))
        wps = [(peg.x, peg.y, h['grasp']), (peg.x, peg.y, h['carry'])]
        wps += [(x, y, h['carry']) for (x, y) in route_xy[1:]]
        wps += [(sx, sy, h['insert_start'])]
        t_move = cartesian_path(kin, wps, pitch, lim, q_seed=t_desc.q[-1], speed_scale=speed_scale)
        steps.append(PlanStep('traj', 'lift and transport (%s)' % info['route'], traj=t_move))
        steps += self._insert_and_release(sx, sy, pitch, h, t_move.q[-1], speed_scale)
        return PickPlacePlan(peg=peg, socket=socket, pitch=pitch, steps=steps, route=info['route'],
                             misalignment_deg=math.degrees(predicted_misalignment(peg, socket, self.grasp_model)),
                             q_end=list(steps[-1].traj.q[-1]))

    def _insert_and_release(self, sx, sy, pitch, h, q_seed, speed_scale):
        lim, pl, kin = self.limits, self.cfg['place'], self.kin
        t_ins = cartesian_path(kin, [(sx, sy, h['insert_start']), (sx, sy, h['place'])], pitch, lim,
                               vmax=lim.cart_vel_slow, q_seed=q_seed)
        t_up = cartesian_path(kin, [(sx, sy, h['place']), (sx, sy, h['retreat'])], pitch, lim,
                              q_seed=t_ins.q[-1], speed_scale=speed_scale)
        return [PlanStep('traj', 'insert (slow)', traj=t_ins, guarded=True),
                PlanStep('wait', 'settle before release', seconds=float(pl['release_settle'])),
                PlanStep('gripper', 'release', gripper='release'),
                PlanStep('traj', 'retreat', traj=t_up)]

    def retry_insertion(self, peg: PegInfo, socket: SocketInfo, q_now, pitch, offset, speed_scale=1.0):
        """After a blocked insertion: back up to the insertion start, shift by `offset`
        (dx, dy) and try again.  Returns a list of PlanSteps."""
        h = self.heights(peg, socket)
        x, y, z, _ = self.kin.fk(q_now)
        sx, sy = socket.x + offset[0], socket.y + offset[1]
        t_back = cartesian_path(self.kin, [(x, y, z), (x, y, h['insert_start']), (sx, sy, h['insert_start'])],
                                pitch, self.limits, vmax=self.limits.cart_vel_slow * 2, q_seed=q_now)
        return [PlanStep('traj', 'back off and shift (%.0f, %.0f) mm' % (offset[0] * 1e3, offset[1] * 1e3),
                         traj=t_back)] + self._insert_and_release(sx, sy, pitch, h, t_back.q[-1], speed_scale)

    def retreat_up(self, q_now, dz=0.05):
        """Straight up from wherever the tool is (used after failures)."""
        x, y, z, pitch = self.kin.fk(q_now)
        return cartesian_path(self.kin, [(x, y, z), (x, y, z + dz)], pitch, self.limits, q_seed=q_now)

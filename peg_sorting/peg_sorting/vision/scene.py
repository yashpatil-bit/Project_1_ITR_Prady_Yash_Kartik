"""Full perception pipeline: frame(s) -> pegs + sockets in the robot base frame."""
import math
from collections import Counter, defaultdict

import cv2
import numpy as np

from .color_segmentation import ColorConfig, to_hsv
from .geometry import points_in_base, project
from .peg_detection import PegDetection, detect_pegs
from .socket_detection import BoardDetection, SocketDetection, detect_board_and_sockets


class SceneDetector:
    def __init__(self, colors_cfg: dict, vision_cfg: dict, T_base_cam: np.ndarray, table_z=0.0, yolo=None):
        self.colors = ColorConfig.from_dict(colors_cfg)
        self.vcfg = vision_cfg
        self.T = np.asarray(T_base_cam, dtype=float)
        self.table_z = float(table_z)
        self.yolo = yolo

    # ----------------------------------------------------------- single frame
    def detect(self, frame, with_sockets=True):
        pts = points_in_base(frame.depth, frame.K, self.T)
        hsv = to_hsv(frame.color, int(self.vcfg['pegs'].get('blur_px', 3)))
        pegs = detect_pegs(hsv, pts, self.colors, self.vcfg, self.table_z, self.yolo, frame.color)
        board, sockets, dbg = BoardDetection(False), [], {}
        if with_sockets:
            board, sockets, dbg = detect_board_and_sockets(frame.color, pts[..., 2], frame.K, self.T,
                                                           self.colors, self.vcfg, self.table_z)
        return pegs, board, sockets, dbg

    # ----------------------------------------------------------- multi frame
    def detect_averaged(self, frames, with_sockets=True):
        all_pegs = defaultdict(list)
        all_sockets = []
        boards = []
        dbg = {}
        for fr in frames:
            pegs, board, sockets, dbg = self.detect(fr, with_sockets)
            for p in pegs:
                all_pegs[p.color].append(p)
            all_sockets.extend(sockets)
            if board.found:
                boards.append(board)
        min_hits = max(1, int(math.ceil(0.5 * len(frames))))
        pegs = [_merge_pegs(v) for v in all_pegs.values() if len(v) >= min_hits]
        sockets = _merge_sockets(all_sockets, min_hits, float(self.vcfg['board'].get('merge_dist', 0.02)))
        board = _merge_boards(boards) if boards else BoardDetection(False)
        return pegs, board, sockets, dbg

    # ----------------------------------------------------------- debug image
    def draw(self, frame, pegs, board, sockets, misalignments=None):
        img = frame.color.copy()
        K, T = frame.K, self.T
        if board.found and board.outline_xy is not None:
            pts3 = np.column_stack([board.outline_xy, np.full(len(board.outline_xy), board.z)])
            uv = project(pts3, K, T).astype(np.int32)
            cv2.polylines(img, [uv], True, (0, 200, 255), 2)
        for s in sockets:
            if s.outline_xy is not None:
                pts3 = np.column_stack([s.outline_xy, np.full(len(s.outline_xy), s.z)])
                uv = project(pts3, K, T).astype(np.int32)
                cv2.polylines(img, [uv], True, (255, 0, 255), 2)
            c = project([[s.x, s.y, s.z]], K, T)[0].astype(int)
            txt = '%s' % s.shape
            if misalignments and s.shape in misalignments:
                txt += ' %+.0fdeg' % misalignments[s.shape]
            cv2.putText(img, txt, (int(c[0]) - 30, int(c[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 2)
        for p in pegs:
            spec = self.colors.by_name(p.color)
            col = tuple(int(v) for v in spec.display_bgr) if spec else (255, 255, 255)
            x, y, w, h = p.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            if p.outline_xy is not None:
                pts3 = np.column_stack([p.outline_xy, np.full(len(p.outline_xy), p.z_top)])
                uv = project(pts3, K, T).astype(np.int32)
                cv2.polylines(img, [uv], True, (255, 255, 255), 1)
            c = project([[p.x, p.y, p.z_top]], K, T)[0].astype(int)
            cv2.circle(img, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)
            cv2.putText(img, '%s %s' % (p.color, p.shape), (x, max(12, y - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.putText(img, '(%.0f,%.0f,%.0f)mm' % (p.x * 1e3, p.y * 1e3, p.z_top * 1e3),
                        (x, max(24, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        return img


def _circmean(angles, period):
    a = np.asarray(angles) * (2 * math.pi / period)
    return (math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a))) * period / (2 * math.pi)) % period


def _period(shape):
    return {'triangle': 2 * math.pi / 3, 'square': math.pi / 2, 'hexagon': math.pi / 3}.get(shape, 2 * math.pi)


def _merge_pegs(dets):
    shape, _ = Counter(d.shape for d in dets).most_common(1)[0]
    same = [d for d in dets if d.shape == shape]
    best = max(same, key=lambda d: d.pixel_area)
    m = PegDetection(
        color=best.color, shape=shape,
        shape_source=Counter(d.shape_source for d in same).most_common(1)[0][0],
        shape_confidence=float(np.mean([d.shape_confidence for d in same])) * len(same) / len(dets),
        x=float(np.median([d.x for d in dets])), y=float(np.median([d.y for d in dets])),
        z_top=float(np.median([d.z_top for d in dets])), height=float(np.median([d.height for d in dets])),
        yaw=_circmean([d.yaw for d in same], _period(shape)),
        width=float(np.median([d.width for d in dets])), pixel_area=best.pixel_area, bbox=best.bbox,
        outline_xy=best.outline_xy, geometry_shape=best.geometry_shape,
        geometry_confidence=best.geometry_confidence)
    return m


def _merge_sockets(dets, min_hits, merge_dist):
    clusters = []
    for d in dets:
        for c in clusters:
            if math.hypot(c[0].x - d.x, c[0].y - d.y) < merge_dist:
                c.append(d)
                break
        else:
            clusters.append([d])
    out = []
    for c in clusters:
        if len(c) < min_hits:
            continue
        shape, _ = Counter(d.shape for d in c).most_common(1)[0]
        same = [d for d in c if d.shape == shape]
        out.append(SocketDetection(
            shape, float(np.mean([d.shape_confidence for d in same])) * len(same) / len(c),
            float(np.median([d.x for d in c])), float(np.median([d.y for d in c])),
            float(np.median([d.z for d in c])), _circmean([d.yaw for d in same], _period(shape)),
            float(np.median([d.width for d in c])), same[0].outline_xy))
    return out


def _merge_boards(boards):
    b = boards[len(boards) // 2]
    return BoardDetection(True, float(np.median([x.x for x in boards])), float(np.median([x.y for x in boards])),
                          b.z, b.yaw, b.length, b.width, b.outline_xy)


def resolve_duplicate_shapes(pegs, colors_cfg):
    """If two pegs claim the same shape, give the less confident one its colour prior."""
    cc = ColorConfig.from_dict(colors_cfg)
    by_shape = defaultdict(list)
    for p in pegs:
        by_shape[p.shape].append(p)
    for shape, lst in by_shape.items():
        if len(lst) <= 1:
            continue
        lst.sort(key=lambda p: -p.shape_confidence)
        for p in lst[1:]:
            spec = cc.by_name(p.color)
            if spec and spec.expected_shape:
                p.shape, p.shape_source = spec.expected_shape, 'color_prior'
    return pegs

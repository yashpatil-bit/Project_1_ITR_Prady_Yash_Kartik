"""Board and socket (hole) detection in a bird's-eye view.

The board is flat, so the colour image can be warped onto the board plane
(z = board top) with a homography computed from the camera calibration.  In
that metric top-down image (1 px = ``bev.resolution`` m):

1. board pixels   = MDF-coloured (incl. the burnt hole edges)  [colors.yaml: board]
2. anything taller than the board (pegs, the arm) is treated as "board" so it
   cannot create fake holes
3. the board = largest blob; its convex hull is the board area
4. holes = board area minus board pixels; each hole outline is classified
   with the same shape classifier as the pegs -> socket shape, centre, yaw.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .color_segmentation import clean_mask, hsv_mask
from .geometry import BirdsEyeView
from .shape_detection import classify_outline, outline_yaw


@dataclass
class SocketDetection:
    shape: str
    shape_confidence: float
    x: float
    y: float
    z: float
    yaw: float
    width: float
    outline_xy: np.ndarray = field(default=None, repr=False)


@dataclass
class BoardDetection:
    found: bool
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    length: float = 0.0
    width: float = 0.0
    outline_xy: np.ndarray = field(default=None, repr=False)


def detect_board_and_sockets(color, height_map, K, T_base_cam, color_cfg, vcfg, table_z=0.0):
    """Returns (BoardDetection, [SocketDetection], debug dict)."""
    bc = vcfg['board']
    ws = vcfg['workspace']
    bev = BirdsEyeView(ws['x'], ws['y'], float(bc['bev_resolution']))
    board_top = table_z + float(bc['thickness'])
    px = 1.0 / bev.res  # pixels per metre

    bev_color = bev.warp(color, K, T_base_cam, board_top)
    h = np.nan_to_num(height_map.astype(np.float32), nan=-1.0)
    bev_height = bev.warp(h, K, T_base_cam, board_top, nearest=True)
    valid_view = bev.warp(np.full(color.shape[:2], 255, np.uint8), K, T_base_cam, board_top, nearest=True) > 0

    hsv = cv2.cvtColor(cv2.GaussianBlur(bev_color, (5, 5), 0), cv2.COLOR_BGR2HSV)
    board_mask = hsv_mask(hsv, color_cfg.board)
    tall = (bev_height > board_top + float(bc['occluder_height'])).astype(np.uint8) * 255
    # grow the tall mask a little: a peg's edges (and its shadow-side depth holes) must
    # never count as board, otherwise a wood-coloured peg next to the board merges into it
    grow = int(max(3, round(0.006 / bev.res))) | 1
    tall = cv2.dilate(tall, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    board_mask = cv2.bitwise_and(board_mask, cv2.bitwise_not(tall))
    board_mask = clean_mask(board_mask, 3, 3)
    board_or_tall = cv2.bitwise_or(board_mask, tall)

    debug = {'bev_color': bev_color, 'board_mask': board_mask, 'bev': bev}
    close = int(max(3, round(float(bc['close_m']) * px))) | 1
    solid = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)))
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BoardDetection(False), [], debug
    board_c = max(contours, key=cv2.contourArea)
    area_m2 = cv2.contourArea(board_c) * bev.res ** 2
    if area_m2 < float(bc['min_area_m2']):
        return BoardDetection(False), [], debug

    hull = cv2.convexHull(board_c)
    (cu, cv), (wu, hv), ang = cv2.minAreaRect(hull)
    center_xy = bev.to_base([[cu, cv]])[0]
    # long axis direction in base frame
    a = math.radians(ang)
    du, dv = (math.cos(a), math.sin(a)) if wu >= hv else (-math.sin(a), math.cos(a))
    p0 = bev.to_base([[cu, cv]])[0]
    p1 = bev.to_base([[cu + du, cv + dv]])[0]
    board_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    board_yaw = (board_yaw + math.pi / 2) % math.pi - math.pi / 2
    board = BoardDetection(True, float(center_xy[0]), float(center_xy[1]), board_top, board_yaw,
                           max(wu, hv) * bev.res, min(wu, hv) * bev.res,
                           bev.to_base(hull.reshape(-1, 2)))

    region = np.zeros_like(board_mask)
    cv2.drawContours(region, [hull], -1, 255, -1)
    shrink = int(max(1, round(float(bc['edge_margin_m']) * px)))
    region = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * shrink + 1,) * 2))
    region[~valid_view] = 0
    holes = cv2.bitwise_and(region, cv2.bitwise_not(board_or_tall))
    holes = clean_mask(holes, int(max(3, round(0.004 * px))) | 1, 3)
    debug['holes_mask'] = holes

    sockets = []
    hcs, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    a_min = float(bc['hole_area_min_m2'])
    a_max = float(bc['hole_area_max_m2'])
    for hc in hcs:
        area = cv2.contourArea(hc) * bev.res ** 2
        if area < a_min or area > a_max:
            continue
        hull_h = cv2.convexHull(hc)
        solidity = cv2.contourArea(hc) / max(cv2.contourArea(hull_h), 1e-9)
        if solidity < float(bc['min_solidity']):
            continue
        xy = bev.to_base(hc.reshape(-1, 2).astype(float))
        shape, conf, f = classify_outline(xy)
        c = f['centroid']
        sockets.append(SocketDetection(shape, conf, float(c[0]), float(c[1]), board_top,
                                       outline_yaw(f, shape), 2.0 * float(f['enclosing_radius']), xy))
    return board, sockets, debug

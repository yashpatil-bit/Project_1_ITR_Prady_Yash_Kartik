"""Peg detection: colour blob + depth -> 3-D position, height, shape, yaw.

1. Height map: every pixel's depth is deprojected and transformed into the
   robot base frame; z is the height above the table.
2. Colour mask AND "taller than min_peg_height" AND "inside the workspace"
   (this removes the orange-ish burnt socket edges, the ruler, things off the
   plate, ...).
3. Largest connected blob per colour = the peg.
4. Its 3-D points within ``top_band`` of the highest point are the peg's top
   face.  Their x-y positions, rasterised top-down, give a metric outline:
   centroid -> peg position, classifier -> shape, harmonic phase -> yaw.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .color_segmentation import segment_color
from .shape_detection import classify_outline, outline_yaw, raster_outline


@dataclass
class PegDetection:
    color: str
    shape: str
    shape_source: str
    shape_confidence: float
    x: float
    y: float
    z_top: float
    height: float
    yaw: float
    width: float
    pixel_area: int
    bbox: tuple = (0, 0, 0, 0)            # x, y, w, h in the colour image
    outline_xy: np.ndarray = field(default=None, repr=False)
    geometry_shape: str = ''
    geometry_confidence: float = 0.0
    outline_source: str = ''       # top_face | blob | depth
    height_source: str = ''        # depth | nominal
    _feats: dict = field(default=None, repr=False)


def workspace_mask(pts_base, ws):
    x, y = pts_base[..., 0], pts_base[..., 1]
    with np.errstate(invalid='ignore'):
        return ((x >= ws['x'][0]) & (x <= ws['x'][1]) & (y >= ws['y'][0]) & (y <= ws['y'][1]))


def pixels_to_plane(uv, K, T_base_cam, z0):
    """Intersect the camera rays of pixels (N,2) with the horizontal plane z = z0 (base frame).
    Returns (N,2) x-y.  Needs no depth."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    rays = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(uv))]) @ T_base_cam[:3, :3].T
    o = T_base_cam[:3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (z0 - o[2]) / rays[:, 2]
    return o[:2] + rays[:, :2] * t[:, None]


def ground_workspace_mask(shape, K, T_base_cam, ws, z0):
    """Pixels whose ray hits the plane z0 inside the workspace (no depth needed)."""
    h, w = shape
    vv, uu = np.mgrid[0:h, 0:w]
    xy = pixels_to_plane(np.column_stack([uu.ravel(), vv.ravel()]), K, T_base_cam, z0).reshape(h, w, 2)
    with np.errstate(invalid='ignore'):
        return ((xy[..., 0] >= ws['x'][0]) & (xy[..., 0] <= ws['x'][1]) &
                (xy[..., 1] >= ws['y'][0]) & (xy[..., 1] <= ws['y'][1]))


def _largest_cc(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def _outline_px(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(float) if len(c) >= 8 else None


def silhouette_footprint(blob, K, T, z_top, table_z, res=0.001, levels=(0.35, 0.55, 0.75, 0.9, 1.0)):
    """Footprint of a vertical prism from its colour silhouette: a ground point belongs to the
    peg if the vertical segment above it (here sampled at several heights) projects inside the
    blob.  Exact for any camera pose, independent of lighting; needs the blob to cover the sides
    (the lowest ~30 % is skipped: shadows / contact with the table)."""
    vv, uu = np.nonzero(blob)
    if len(uu) < 30:
        return None
    h = z_top - table_z
    c = pixels_to_plane(np.array([[uu.mean(), vv.mean()]]), K, T, table_z + 0.6 * h)[0]
    if not np.all(np.isfinite(c)):
        return None
    half = 0.07
    from .geometry import BirdsEyeView
    bev = BirdsEyeView((c[0] - half, c[0] + half), (c[1] - half, c[1] + half), res)
    m8 = blob.astype(np.uint8) * 255
    acc = None
    for f in levels:
        w = bev.warp(m8, K, T, table_z + f * h, nearest=True) > 0
        acc = w if acc is None else (acc & w)
    acc = cv2.morphologyEx(acc.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    acc = _largest_cc(acc)
    if acc is None:
        return None
    px = _outline_px(acc)
    if px is None:
        return None
    return bev.to_base(px)


def top_face_candidates(blob, hsv):
    """Image regions that may be the peg's top face.
    1. the brighter/more saturated part of the blob (Otsu on S*V): with light from above the
       top face is clearly brighter than the sides
    2. the whole blob (when the colour thresholds only caught the top face anyway)"""
    out = []
    sv = hsv[..., 1].astype(np.float32) * hsv[..., 2].astype(np.float32) / 255.0
    vals = np.clip(sv[blob], 0, 255).astype(np.uint8).reshape(-1, 1)
    if len(vals) > 50:
        thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        top = blob & (sv > thr)
        top = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        top = _largest_cc(top)
        if top is not None:
            top = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            if top.sum() > 0.15 * blob.sum():
                out.append(('top_face', top))
    out.append(('blob', blob))
    return out


def detect_pegs(hsv, pts_base, color_cfg, vcfg, table_z=0.0, yolo=None, color_image=None, K=None,
                T_base_cam=None):
    """Find one peg per colour.  The outline comes from the colour image (clean edges) projected
    onto the plane of the peg top; depth is used for the height and to reject flat things."""
    pc = vcfg['pegs']
    z = pts_base[..., 2]
    ws = vcfg['workspace']
    min_h = float(pc['min_height'])
    nominal = float(pc.get('nominal_height', 0.10))
    max_width = float(pc.get('max_width', 0.09))
    in_ws = workspace_mask(pts_base, ws)
    if K is not None and T_base_cam is not None:
        # also without depth: the pixel's ray passes over the workspace at table or peg-top level
        in_ws = in_ws | ground_workspace_mask(z.shape, K, T_base_cam, ws, table_z) \
            | ground_workspace_mask(z.shape, K, T_base_cam, ws, table_z + nominal)
    yolo_dets = yolo.detect(color_image) if (yolo is not None and color_image is not None) else []

    results = []
    for spec in color_cfg.pegs:
        cmask = (segment_color(hsv, spec, color_cfg) > 0) & in_ws
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cmask.astype(np.uint8), connectivity=8)
        order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]) if n > 1 else []
        det = None
        for idx in order[:4]:
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < spec.min_area_px:
                break
            blob = labels == idx
            zz = z[blob]
            zz = zz[np.isfinite(zz)]
            tall = zz[(zz > table_z + min_h) & (zz < table_z + float(pc['max_height']))]
            if len(zz) >= 40 and len(tall) < 0.15 * len(zz) and len(tall) < 30:
                continue  # plenty of depth and it is flat -> not a peg (burnt board edge, tape, ...)
            if len(tall) >= int(pc.get('min_depth_points', 60)):
                z_top = float(np.percentile(tall, 95))
                h_src = 'depth'
            else:
                z_top = table_z + nominal
                h_src = 'nominal'
            det = _measure(spec, blob, hsv, pts_base, z_top, h_src, table_z, area,
                           tuple(int(v) for v in stats[idx, :4]), pc, K, T_base_cam, max_width)
            if det is not None:
                break
        if det is None:
            continue
        # optional YOLO classification (matched to this blob by IoU)
        if yolo_dets:
            best = max(yolo_dets, key=lambda d: _iou(d['bbox'], det.bbox))
            if _iou(best['bbox'], det.bbox) > 0.3 and best['confidence'] >= float(pc.get('yolo_min_conf', 0.5)):
                if det.shape_source != 'geometry' or best['confidence'] > det.shape_confidence:
                    det.shape, det.shape_confidence, det.shape_source = best['shape'], best['confidence'], 'yolo'
        # fall back on the colour -> shape prior from colors.yaml
        min_conf = float(pc.get('min_shape_confidence', 0.6))
        if spec.expected_shape and (det.shape == 'unknown' or det.shape_confidence < min_conf):
            if pc.get('fallback_to_color_prior', True):
                det.shape, det.shape_source = spec.expected_shape, 'color_prior'
                det.shape_confidence = max(det.shape_confidence, 0.5)
        if det._feats is not None and det.shape in ('triangle', 'square', 'hexagon'):
            det.yaw = outline_yaw(det._feats, det.shape)
        results.append(det)
    return results


def _measure(spec, blob, hsv, pts_base, z_top, h_src, table_z, area, bbox, pc, K, T, max_width):
    """Outline, centre, shape and size of one peg blob.  Candidates: the colour top face /
    whole blob projected onto the plane of the peg top (clean edges), and the top face from
    the depth points, and the colour silhouette; they vote on the shape."""
    # depth-based top face (needs dense depth on the top)
    pts = pts_base[blob]
    pts = pts[np.isfinite(pts).all(axis=1)]
    top = pts[np.abs(pts[:, 2] - z_top) < float(pc['top_band'])] if len(pts) else pts
    depth_cand = None
    if len(top) >= int(pc['min_top_points']):
        xy = raster_outline(top[:, :2], resolution=float(pc['raster_resolution']),
                            close_px=int(pc['raster_close_px']))
        if xy is not None and len(xy) >= 8:
            shape, conf, f = classify_outline(xy)
            if 2.0 * float(f['enclosing_radius']) <= max_width:
                depth_cand = ('depth', xy, shape, conf, f)
    cands = []   # (name, outline_xy, shape, conf, features)

    if K is not None and T is not None:
        outlines = []
        faces = top_face_candidates(blob, hsv)
        top_px = next((int(r.sum()) for n, r in faces if n == 'top_face'), int(blob.sum()))
        # the silhouette only means something if the colour blob includes the peg's sides
        sides_in_blob = blob.sum() >= float(pc.get('silhouette_min_ratio', 1.4)) * max(top_px, 1)
        if sides_in_blob:
            sil = silhouette_footprint(blob, K, T, z_top, table_z)
            if sil is not None and len(sil) >= 8:
                outlines.append(('silhouette', sil))
        for name, region in faces:
            if name == 'blob' and sides_in_blob:
                continue  # whole blob incl. sides projected at the top height is smeared
            px = _outline_px(region)
            if px is not None:
                xy = pixels_to_plane(px, K, T, z_top)
                if np.all(np.isfinite(xy)):
                    outlines.append((name, xy))
        for name, xy in outlines:
            shape, conf, f = classify_outline(xy)
            width = 2.0 * float(f['enclosing_radius'])
            if 0.015 <= width <= max_width:
                cands.append((name, xy, shape, conf, f))
    if depth_cand is not None:
        cands.append(depth_cand)
    if not cands:
        if len(top) < 3:
            return None
        cx, cy = float(np.median(top[:, 0])), float(np.median(top[:, 1]))
        det = PegDetection(spec.name, 'unknown', '', 0.0, cx, cy, z_top, z_top - table_z, 0.0,
                           float(pc.get('default_width', 0.05)), area, bbox)
        det._feats = None
        return det
    # vote: each method adds its confidence (weighted by how reliable it is) to its shape
    # (depth outlines are ragged with sparse depth -> weak vote, never a veto; the silhouette is
    #  exact for polygons but rounds a cylinder into a 'hexagon' -> slightly lower weight)
    weight = {'top_face': 1.0, 'silhouette': 0.8, 'blob': 0.5, 'depth': 0.4}
    votes = {}
    for name, _, shp, cf, _ in cands:
        votes[shp] = votes.get(shp, 0.0) + weight.get(name, 0.5) * cf
    shape = max(votes, key=votes.get)
    agreement = votes[shape] / max(sum(votes.values()), 1e-9)
    name, xy, _, conf, f = max((c for c in cands if c[2] == shape),
                               key=lambda c: weight.get(c[0], 0.5) * c[3])
    conf = agreement            # share of the (weighted) votes that went to the winning shape
    c = f['centroid']
    det = PegDetection(spec.name, shape, 'geometry', conf, float(c[0]), float(c[1]), z_top, z_top - table_z,
                       outline_yaw(f, shape), 2.0 * float(f['enclosing_radius']), area, bbox, outline_xy=xy,
                       geometry_shape=shape, geometry_confidence=conf)
    det.outline_source = name
    det.height_source = h_src
    det.votes = [(n, sh, round(float(cf), 2)) for n, _, sh, cf, _ in cands]
    det._feats = f
    return det


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def peg_direction_text(det):
    return '%s %s @ (%.3f, %.3f, top %.3f) yaw %.0f deg [%s %.2f]' % (
        det.color, det.shape, det.x, det.y, det.z_top, math.degrees(det.yaw),
        det.shape_source, det.shape_confidence)

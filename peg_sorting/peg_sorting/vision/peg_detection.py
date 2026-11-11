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


def workspace_mask(pts_base, ws):
    x, y = pts_base[..., 0], pts_base[..., 1]
    with np.errstate(invalid='ignore'):
        return ((x >= ws['x'][0]) & (x <= ws['x'][1]) & (y >= ws['y'][0]) & (y <= ws['y'][1]))


def detect_pegs(hsv, pts_base, color_cfg, vcfg, table_z=0.0, yolo=None, color_image=None):
    pc = vcfg['pegs']
    z = pts_base[..., 2]
    with np.errstate(invalid='ignore'):
        tall = (z > table_z + float(pc['min_height'])) & (z < table_z + float(pc['max_height']))
    valid = tall & workspace_mask(pts_base, vcfg['workspace'])
    valid_u8 = valid.astype(np.uint8) * 255
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(pc.get('edge_erode_px', 3)),) * 2)
    yolo_dets = yolo.detect(color_image) if (yolo is not None and color_image is not None) else []

    results = []
    for spec in color_cfg.pegs:
        cmask = segment_color(hsv, spec, color_cfg)
        m = cv2.bitwise_and(cmask, valid_u8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < spec.min_area_px:
            continue
        blob = (labels == idx).astype(np.uint8) * 255
        bbox = tuple(int(v) for v in stats[idx, :4])
        core = cv2.erode(blob, erode_k)  # drop "flying pixels" at depth edges for the height
        pts_core = pts_base[core > 0]
        pts_core = pts_core[np.isfinite(pts_core).all(axis=1)]
        pts = pts_base[blob > 0]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 20:
            continue
        z_top = float(np.percentile((pts_core if len(pts_core) >= 20 else pts)[:, 2], 95))
        # top face: full blob (the eroded one would lose the far edge of the top face)
        top = pts[np.abs(pts[:, 2] - z_top) < float(pc['top_band'])]
        if len(top) < 3:
            top = pts[pts[:, 2] > z_top - float(pc['top_band'])]
        height = z_top - table_z

        det = PegDetection(color=spec.name, shape='unknown', shape_source='', shape_confidence=0.0,
                           x=float(np.median(top[:, 0])), y=float(np.median(top[:, 1])),
                           z_top=z_top, height=height, yaw=0.0, width=float(pc.get('default_width', 0.05)),
                           pixel_area=area, bbox=bbox)
        outline = None
        outline_feats = None
        if len(top) >= int(pc['min_top_points']):
            outline = raster_outline(top[:, :2], resolution=float(pc['raster_resolution']),
                                     close_px=int(pc['raster_close_px']))
        if outline is not None and len(outline) >= 8:
            shape, conf, f = classify_outline(outline)
            outline_feats = f
            c = f['centroid']
            det.x, det.y = float(c[0]), float(c[1])
            det.width = 2.0 * float(f['enclosing_radius'])
            det.geometry_shape, det.geometry_confidence = shape, conf
            det.outline_xy = outline
            det.shape, det.shape_confidence, det.shape_source = shape, conf, 'geometry'
            det.yaw = outline_yaw(f, shape)

        # optional YOLO classification (matched to this blob by IoU)
        if yolo_dets:
            best = max(yolo_dets, key=lambda d: _iou(d['bbox'], bbox))
            if _iou(best['bbox'], bbox) > 0.3 and best['confidence'] >= float(pc.get('yolo_min_conf', 0.5)):
                if det.shape_source != 'geometry' or best['confidence'] > det.shape_confidence:
                    det.shape, det.shape_confidence, det.shape_source = best['shape'], best['confidence'], 'yolo'

        # fall back on the colour -> shape prior from colors.yaml
        min_conf = float(pc.get('min_shape_confidence', 0.6))
        if spec.expected_shape and (det.shape == 'unknown' or det.shape_confidence < min_conf):
            if pc.get('fallback_to_color_prior', True):
                det.shape, det.shape_source = spec.expected_shape, 'color_prior'
                det.shape_confidence = max(det.shape_confidence, 0.5)
        # yaw for the final shape decision (outline phase of the matching harmonic)
        if outline_feats is not None and det.shape in ('triangle', 'square', 'hexagon'):
            det.yaw = outline_yaw(outline_feats, det.shape)
        results.append(det)
    return results


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

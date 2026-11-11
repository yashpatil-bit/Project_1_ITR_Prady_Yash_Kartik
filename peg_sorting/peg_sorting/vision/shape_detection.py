"""Shape classification (triangle / square / hexagon / circle) of a 2-D
outline and estimation of its orientation.

Works on a *metric, top-down* outline: for pegs this is the peg's top face
re-projected from the depth data onto the x-y plane, for sockets it is the hole
outline in the bird's-eye view of the board.  Because the outline is metric and
top-down, perspective from the oblique side camera does not distort it.

Features (all scale invariant):
* fill ratio  = area / area of minimum enclosing circle
                (circle 1.00, hexagon 0.83, square 0.64, triangle 0.41)
* h3, h4, h6  = relative 3rd/4th/6th Fourier harmonics of the radius-vs-angle
                profile r(phi) around the centroid (triangle -> h3,
                square -> h4, hexagon -> h6, circle -> none)
* vertex count of cv2.approxPolyDP on the convex hull (tie-breaker)

The class whose ideal feature vector is closest wins.  The phase of the
n-th harmonic gives the polygon's vertex angle ("yaw", modulo 360/n deg).
"""
import math

import cv2
import numpy as np

SHAPES = ('triangle', 'square', 'hexagon', 'circle')
N_SIDES = {'triangle': 3, 'square': 4, 'hexagon': 6, 'circle': 0}


def _regular_polygon(n, R=100.0, yaw=0.0, samples=720):
    ang = yaw + 2 * np.pi * np.arange(n) / n
    verts = np.column_stack([R * np.cos(ang), R * np.sin(ang)])
    pts = []
    for a, b in zip(verts, np.roll(verts, -1, axis=0)):
        for s in np.linspace(0, 1, samples // n, endpoint=False):
            pts.append(a + s * (b - a))
    return np.array(pts)


def radial_harmonics(contour_xy, n_list=(3, 4, 6), n_samples=360):
    """Relative harmonics of r(phi).  Returns (dict n->amplitude, dict n->phase, centroid, mean_r)."""
    c = np.asarray(contour_xy, dtype=float).reshape(-1, 2)
    m = cv2.moments(c.astype(np.float32))
    if abs(m['m00']) < 1e-9:
        centroid = c.mean(axis=0)
    else:
        centroid = np.array([m['m10'] / m['m00'], m['m01'] / m['m00']])
    d = c - centroid
    phi = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])
    order = np.argsort(phi)
    phi, r = phi[order], r[order]
    # resample r on a uniform angle grid (outline is star-shaped for convex shapes)
    grid = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    phi_ext = np.concatenate([phi - 2 * np.pi, phi, phi + 2 * np.pi])
    r_ext = np.concatenate([r, r, r])
    rg = np.interp(grid, phi_ext, r_ext)
    mean_r = float(np.mean(rg))
    amps, phases = {}, {}
    for n in n_list:
        z = np.mean(rg * np.exp(1j * n * grid))
        amps[n] = float(2 * abs(z) / max(mean_r, 1e-9))
        phases[n] = float(np.angle(z))
    return amps, phases, centroid, mean_r


def resample_closed(contour_xy, n=360):
    """Resample a closed polyline uniformly by arc length."""
    c = np.asarray(contour_xy, dtype=float).reshape(-1, 2)
    closed = np.vstack([c, c[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return c
    s = np.linspace(0, cum[-1], n, endpoint=False)
    return np.column_stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])])


def shape_features(contour_xy):
    c = np.asarray(contour_xy, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(c)
    area = float(cv2.contourArea(hull))
    (_, _), rc = cv2.minEnclosingCircle(hull)
    fill = area / (math.pi * rc * rc) if rc > 0 else 0.0
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
    amps, phases, centroid, mean_r = radial_harmonics(resample_closed(hull.reshape(-1, 2)))
    return {
        'fill': fill, 'h3': amps[3], 'h4': amps[4], 'h6': amps[6],
        'phase3': phases[3], 'phase4': phases[4], 'phase6': phases[6],
        'vertices': int(len(approx)), 'area': area, 'enclosing_radius': float(rc),
        'centroid': centroid, 'mean_radius': mean_r,
    }


def _ideal_features():
    ideal = {}
    for shape in SHAPES:
        n = N_SIDES[shape]
        if n == 0:
            ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            pts = np.column_stack([100 * np.cos(ang), 100 * np.sin(ang)])
        else:
            pts = _regular_polygon(n)
        f = shape_features(pts)
        ideal[shape] = np.array([f['fill'], f['h3'], f['h4'], f['h6']])
    return ideal


_IDEAL = _ideal_features()
# Prototypes measured on noisy, rasterised depth samples of real-size pegs
# (corners get rounded by the rasterisation; see test/test_shape_detection.py).
_ROUNDED = {
    'triangle': np.array([0.544, 0.230, 0.009, 0.070]),
    'square': np.array([0.729, 0.016, 0.104, 0.007]),
    'hexagon': np.array([0.855, 0.011, 0.011, 0.035]),
    'circle': np.array([0.906, 0.012, 0.012, 0.008]),
}
# Expected noise (std. dev.) of each feature
_SIGMA = np.array([0.035, 0.015, 0.012, 0.008])
_EXPECTED_VERTICES = {'triangle': 3, 'square': 4, 'hexagon': 6, 'circle': 8}


def classify_outline(contour_xy):
    """Classify a closed outline.  Returns (shape, confidence 0..1, features)."""
    f = shape_features(contour_xy)
    v = np.array([f['fill'], f['h3'], f['h4'], f['h6']])
    scores = {}
    for shape in SHAPES:
        d2 = min(float(np.sum(((v - proto) / _SIGMA) ** 2))
                 for proto in (_IDEAL[shape], _ROUNDED[shape]))
        # small vertex-count prior
        ev = _EXPECTED_VERTICES[shape]
        if shape == 'circle':
            d2 += 0.0 if f['vertices'] >= 7 else 1.0
        else:
            d2 += 0.5 * min(abs(f['vertices'] - ev), 3)
        scores[shape] = d2
    best = min(scores, key=scores.get)
    # softmax-like confidence
    vals = np.array([scores[s] for s in SHAPES])
    probs = np.exp(-0.5 * (vals - vals.min()))
    probs /= probs.sum()
    conf = float(probs[SHAPES.index(best)])
    f['scores'] = scores
    return best, conf, f


def outline_yaw(features, shape):
    """Vertex angle (rad) of a regular polygon outline, defined modulo 2*pi/n.
    For the circle returns 0."""
    n = N_SIDES[shape]
    if n == 0:
        return 0.0
    phase = features['phase%d' % n]
    period = 2 * math.pi / n
    return (phase / n) % period


def raster_outline(xy, resolution=0.0015, close_px=5, pad=6):
    """Rasterise scattered x-y points (metres) into a mask and return the largest
    outline as (N,2) metric points, or None."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 10:
        return None
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo) / resolution).astype(int) + pad
    w, h = ij[:, 0].max() + pad + 1, ij[:, 1].max() + pad + 1
    if w * h > 4_000_000:
        return None
    mask = np.zeros((h, w), np.uint8)
    mask[ij[:, 1], ij[:, 0]] = 255
    if close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    # pixel centres back to metres
    return (c - pad + 0.5) * resolution + lo

import math

import numpy as np

from peg_sorting.vision.shape_detection import N_SIDES, SHAPES, classify_outline, outline_yaw, raster_outline


def sample_shape(rng, shape, size, yaw, noise, n):
    R = size / 2
    k = N_SIDES[shape]
    p = rng.uniform(-R, R, (n * 4, 2))
    if k == 0:
        ok = np.hypot(p[:, 0], p[:, 1]) <= R
    else:
        ap = R * math.cos(math.pi / k)
        ok = np.ones(len(p), bool)
        for j in range(k):
            a = yaw + math.pi / k + 2 * math.pi * j / k
            ok &= p @ np.array([math.cos(a), math.sin(a)]) <= ap
    p = p[ok][:n]
    return p + rng.normal(0, noise, p.shape)


def test_classify_noisy_top_faces():
    rng = np.random.default_rng(42)
    total = correct = 0
    yaw_err = []
    for shape in SHAPES:
        for _ in range(40):
            size = rng.uniform(0.04, 0.065)
            yaw = rng.uniform(0, 2 * math.pi)
            pts = sample_shape(rng, shape, size, yaw, rng.choice([0.0005, 0.001, 0.002]), int(rng.uniform(300, 2000)))
            s, conf, f = classify_outline(raster_outline(pts))
            total += 1
            correct += s == shape
            if s == shape and N_SIDES[s]:
                per = 2 * math.pi / N_SIDES[s]
                yaw_err.append(abs((outline_yaw(f, s) - yaw + per / 2) % per - per / 2))
    assert correct / total >= 0.97
    assert math.degrees(np.mean(yaw_err)) < 3.0


def test_clean_polygons():
    for shape in SHAPES:
        n = N_SIDES[shape]
        if n == 0:
            a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
            c = np.column_stack([np.cos(a), np.sin(a)]) * 0.03
        else:
            a = 0.3 + 2 * np.pi * np.arange(n) / n
            v = np.column_stack([np.cos(a), np.sin(a)]) * 0.03
            c = np.vstack([np.linspace(v[i], v[(i + 1) % n], 30, endpoint=False) for i in range(n)])
        s, conf, f = classify_outline(c)
        assert s == shape, (shape, s)
        if n:
            per = 2 * math.pi / n
            assert abs((outline_yaw(f, s) - 0.3 + per / 2) % per - per / 2) < math.radians(1.5)

"""Capture images for a YOLOv8 peg-shape dataset (no ROS; stop the vision node first).

    ros2 run peg_sorting capture_dataset --out ~/peg_dataset

SPACE = save the frame (+ auto-generated YOLO labels from the colour/depth
pipeline, if the camera calibration and colours are set up), q = quit.
Move/rotate the pegs between shots, vary lighting, ~150-300 images.  Check
and fix the auto labels with a labelling tool (e.g. labelImg, CVAT,
Roboflow) before training.  Classes: 0 circle, 1 hexagon, 2 square, 3 triangle.
"""
import argparse
import os
import time

import cv2

from ..utils.config import load_yaml
from ..vision.camera import open_camera
from ..vision.scene import SceneDetector

CLASSES = ['circle', 'hexagon', 'square', 'triangle']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/peg_dataset'))
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--no-auto-label', action='store_true')
    a, _ = ap.parse_known_args()
    img_dir = os.path.join(a.out, 'images')
    lbl_dir = os.path.join(a.out, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    vcfg = load_yaml('vision.yaml')
    sim = None
    if a.mock:
        from ..sim.synthetic_scene import SimWorld
        sim = SimWorld(load_yaml('sim_scene.yaml'))
        T = sim.T_base_cam
    else:
        from ..nodes.vision_node import load_extrinsics
        T = load_extrinsics(load_yaml('camera_calib.yaml'))
    det = SceneDetector(load_yaml('colors.yaml'), vcfg, T, float(vcfg.get('table_z', 0.0)))
    cam = open_camera(vcfg, mock=a.mock, sim_world=sim)
    with open(os.path.join(a.out, 'data.yaml'), 'w') as f:
        f.write('path: %s\ntrain: images\nval: images\nnames:\n' % a.out)
        for i, c in enumerate(CLASSES):
            f.write('  %d: %s\n' % (i, c))
    print(__doc__)
    n = 0
    while True:
        fr = cam.get_frame()
        if fr is None:
            continue
        cv2.imshow('capture_dataset', fr.color)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        if k == ord(' '):
            name = 'peg_%d' % int(time.time() * 1000)
            cv2.imwrite(os.path.join(img_dir, name + '.jpg'), fr.color)
            lines = []
            if not a.no_auto_label:
                pegs, _, _, _ = det.detect(fr, with_sockets=False)
                h, w = fr.color.shape[:2]
                for p in pegs:
                    if p.shape not in CLASSES:
                        continue
                    x, y, bw, bh = p.bbox
                    lines.append('%d %.6f %.6f %.6f %.6f' % (CLASSES.index(p.shape), (x + bw / 2) / w,
                                                             (y + bh / 2) / h, bw / w, bh / h))
            with open(os.path.join(lbl_dir, name + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            n += 1
            print('saved %s (%d labels) - total %d' % (name, len(lines), n))
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

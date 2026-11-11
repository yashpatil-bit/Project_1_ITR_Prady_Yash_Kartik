"""Live HSV threshold tuning for each peg colour and the board (no ROS needed).

    ros2 run peg_sorting hsv_tuner            (stop the vision node first - it owns the camera)
    ros2 run peg_sorting hsv_tuner --mock

Keys: n = next colour, s = save to colors.yaml, q = quit.  Click on the image
to print the HSV value under the mouse (handy to set the ranges).
"""
import argparse

import cv2
import numpy as np

from ..utils.config import load_yaml, save_yaml
from ..vision.camera import open_camera
from ..vision.color_segmentation import clean_mask, hsv_mask, to_hsv

WIN = 'hsv_tuner'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mock', action='store_true')
    a, _ = ap.parse_known_args()
    vcfg = load_yaml('vision.yaml')
    colors = load_yaml('colors.yaml')
    sim = None
    if a.mock:
        from ..sim.synthetic_scene import SimWorld
        sim = SimWorld(load_yaml('sim_scene.yaml'))
    cam = open_camera(vcfg, mock=a.mock, sim_world=sim)
    names = list(colors['pegs'].keys()) + ['board']
    idx = [0]
    state = {}

    def ranges_of(name):
        return colors['board']['hsv_ranges'] if name == 'board' else colors['pegs'][name]['hsv_ranges']

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for t, mx in (('H lo', 180), ('S lo', 255), ('V lo', 255), ('H hi', 180), ('S hi', 255), ('V hi', 255)):
        cv2.createTrackbar(t, WIN, 0, mx, lambda v: None)

    def load_bars():
        lo, hi = ranges_of(names[idx[0]])[0]
        for t, v in zip(('H lo', 'S lo', 'V lo', 'H hi', 'S hi', 'V hi'), list(lo) + list(hi)):
            cv2.setTrackbarPos(t, WIN, int(v))

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and 'hsv' in state:
            h, w = state['hsv'].shape[:2]
            if x < w and y < h:
                print('HSV at (%d,%d): %s' % (x, y, state['hsv'][y, x].tolist()))
    cv2.setMouseCallback(WIN, on_mouse)
    load_bars()
    print(__doc__)
    while True:
        fr = cam.get_frame()
        if fr is None:
            continue
        hsv = to_hsv(fr.color)
        state['hsv'] = hsv
        v = [cv2.getTrackbarPos(t, WIN) for t in ('H lo', 'S lo', 'V lo', 'H hi', 'S hi', 'V hi')]
        name = names[idx[0]]
        rngs = [((v[0], v[1], v[2]), (v[3], v[4], v[5]))] + [tuple(map(tuple, r)) for r in ranges_of(name)[1:]]
        mask = clean_mask(hsv_mask(hsv, rngs), 3, 7)
        vis = fr.color.copy()
        vis[mask == 0] = (vis[mask == 0] * 0.25).astype(np.uint8)
        cv2.putText(vis, 'colour: %s   (n next, s save, q quit)' % name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        if k == ord('n'):
            ranges_of(name)[0] = [[v[0], v[1], v[2]], [v[3], v[4], v[5]]]
            idx[0] = (idx[0] + 1) % len(names)
            load_bars()
        if k == ord('s'):
            ranges_of(name)[0] = [[v[0], v[1], v[2]], [v[3], v[4], v[5]]]
            path = save_yaml('colors.yaml', colors, header='HSV thresholds - saved by hsv_tuner')
            print('saved %s' % path)
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

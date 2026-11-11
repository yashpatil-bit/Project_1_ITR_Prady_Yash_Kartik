"""HSV colour segmentation for the pegs and the wooden board.

Thresholds come from ``config/colors.yaml`` (tune them live with
``ros2 run peg_sorting hsv_tuner``).  OpenCV HSV ranges: H 0-180, S 0-255, V 0-255.
A colour may have several [low, high] ranges (e.g. red wraps around H = 0/180).
"""
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


@dataclass
class ColorSpec:
    name: str
    ranges: List[tuple]
    expected_shape: str = ''
    display_bgr: tuple = (255, 255, 255)
    min_area_px: int = 300


@dataclass
class ColorConfig:
    pegs: List[ColorSpec] = field(default_factory=list)
    board: List[tuple] = field(default_factory=list)
    morph_open_px: int = 3
    morph_close_px: int = 7

    @classmethod
    def from_dict(cls, d):
        cc = cls()
        for name, spec in (d.get('pegs') or {}).items():
            cc.pegs.append(ColorSpec(
                name=name,
                ranges=[(tuple(r[0]), tuple(r[1])) for r in spec['hsv_ranges']],
                expected_shape=spec.get('expected_shape', ''),
                display_bgr=tuple(spec.get('display_bgr', (255, 255, 255))),
                min_area_px=int(spec.get('min_area_px', 300)),
            ))
        b = d.get('board') or {}
        cc.board = [(tuple(r[0]), tuple(r[1])) for r in b.get('hsv_ranges', [])]
        m = d.get('morphology') or {}
        cc.morph_open_px = int(m.get('open_px', cc.morph_open_px))
        cc.morph_close_px = int(m.get('close_px', cc.morph_close_px))
        return cc

    def by_name(self, name):
        for p in self.pegs:
            if p.name == name:
                return p
        return None


def hsv_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def clean_mask(mask, open_px=3, close_px=7):
    if open_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    if close_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    return mask


def to_hsv(bgr, blur=3):
    if blur and blur > 1:
        bgr = cv2.GaussianBlur(bgr, (blur, blur), 0)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def segment_color(hsv, spec: ColorSpec, cfg: ColorConfig):
    return clean_mask(hsv_mask(hsv, spec.ranges), cfg.morph_open_px, cfg.morph_close_px)

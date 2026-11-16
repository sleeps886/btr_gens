# -*- coding: utf-8 -*-
"""validate_level.py —— 用解析真值校准"等阈亮带中点"的电平与合并参数。

真值: 平台 A 静止于 (30.0000°N,122.0000°E), 航向 -30°, 拖曳阵 -180 m;
      T1 (30.0100,122.1000)->(30.0000,122.1000) 19 kn, 走完航点 t=113.8 s;
      T2 (30.0100,121.9000)->(30.0000,121.9000) 23 kn, 走完航点 t=94.0 s。
"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B
from extract_latest import RECT, R_T0, R_T600, T_MAX, image_to_db_fixed, MIN_W

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "btr_out", "20261105_152635")
HEAD = -30.0
ll2xy, xy2ll = B.make_geo(30.0, 122.0)
hx0, hy0 = ll2xy(30.0, 122.0)
psi = math.radians(HEAD)
tx0 = hx0 + (-180.0) * math.sin(psi)
ty0 = hy0 + (-180.0) * math.cos(psi)

TG = {"T1": ((30.0100, 122.1000), (30.0000, 122.1000), 19, 113.8, (0, 600)),
      "T2": ((30.0100, 121.9000), (30.0000, 121.9000), 23, 94.0, (60, 560))}


def truth_bearings(t):
    out = {}
    for name, (s, e, kn, tend, win) in TG.items():
        if not (win[0] <= t <= win[1]):
            continue
        x0, y0 = ll2xy(*s)
        x1, y1 = ll2xy(*e)
        f = min(1.0, t / tend)
        px, py = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        for tag, ax, ay in (("hull", hx0, hy0), ("tow", tx0, ty0)):
            gb = math.degrees(math.atan2(px - ax, py - ay)) % 360.0
            r = (gb - HEAD) % 360.0
            out[(name, tag)] = r if r <= 180 else 360 - r
    return out


def bands(prof, level, merge, min_w=6):
    idx = np.where(prof >= level)[0]
    if len(idx) == 0:
        return []
    segs, s = [], idx[0]
    for a, b_ in zip(idx[:-1], idx[1:]):
        if b_ - a > merge:
            segs.append((s, a))
            s = b_
    segs.append((s, idx[-1]))
    out = []
    for a, b_ in segs:
        if b_ - a + 1 < min_w:
            continue
        # 用 >=level 的首尾像素中点 (亚像素: 线性插值到 level)
        left = ((a - 1) + (level - prof[a - 1]) / max(prof[a] - prof[a - 1], 1e-9)
                if a - 1 >= 0 and prof[a - 1] < level else float(a))
        right = (b_ + (prof[b_] - level) / max(prof[b_] - prof[b_ + 1], 1e-9)
                 if b_ + 1 < len(prof) and prof[b_ + 1] < level else float(b_))
        out.append((0.5 * (left + right), right - left, float(prof[a:b_ + 1].max())))
    return out


lut = B.build_lut(B.CMAP_STOPS["gui"])
db, span = {}, {}
for tag in ("hull", "tow"):
    img = np.asarray(Image.open(os.path.join(SRC, f"btr_A_{tag}.png")).convert("RGB"))
    db[tag] = image_to_db_fixed(img, RECT[tag], lut, -5.0, 16.0)
    span[tag] = RECT[tag][3] - RECT[tag][2] + 1

CASES = [(3.0, "T1"), (20.0, "T1"), (40.0, "T1"), (60.0, "T1"), (77.0, "T1"),
         (113.8, "T1"), (200.0, "T1"), (300.0, "T1"), (500.0, "T1"), (597.0, "T1"),
         (61.0, "T2"), (80.0, "T2"), (94.0, "T2"), (150.0, "T2"), (300.0, "T2"),
         (450.0, "T2"), (559.0, "T2")]

print(f"{'t':>6} {'目标':<4}{'阵':<5}{'真值':>8}" + "".join(f"{f'L{l:g}'.rjust(11)}" for l in (12, 13.5, 14.5, 15.2, 15.6)))
best = {l: [] for l in (12, 13.5, 14.5, 15.2, 15.6)}
for t, name in CASES:
    tb = truth_bearings(t)
    for tag in ("hull", "tow"):
        tv = tb.get((name, tag))
        if tv is None:
            continue
        r = int(round(R_T0 - t / T_MAX * (R_T0 - R_T600))) - RECT[tag][0]
        prof = db[tag][max(r - 1, 0):r + 2].mean(axis=0)
        row = f"{t:6.1f} {name:<4}{tag:<5}{tv:8.2f}"
        for lev in (12, 13.5, 14.5, 15.2, 15.6):
            bs = [(c / span[tag] * 180, w, a) for c, w, a in bands(prof, lev, 20)]
            if not bs:
                row += f"{'--':>11}"
                continue
            m = min(bs, key=lambda b: abs(b[0] - tv))
            err = m[0] - tv
            row += f"{err:+7.2f}/{m[1]:3.0f}"
            if abs(tv - (120 if name == "T1" else 60)) < 60:
                best[lev].append(err)
        print(row)
print()
for lev, e in best.items():
    e = np.array(e)
    print(f"  电平 {lev:4.1f} dB: n={len(e):3d}  均值 {e.mean():+.3f}°  "
          f"RMS {np.sqrt((e ** 2).mean()):.3f}°  最大 |误差| {np.abs(e).max():.2f}°")

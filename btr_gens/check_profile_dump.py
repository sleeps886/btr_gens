# -*- coding: utf-8 -*-
"""对照解析真值舷角, dump 两阵在若干时刻的方位维 dB 剖面, 查清检测偏差。"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "btr_out", "20261105_152635")
# 精确标定 (由网格线得出)
C0, C1 = 98.0, 1262.0          # bearing 0 .. 180
R_TOP, R_BOT = 22.4, 738.6     # t=600 .. t=0
RECT = (22, 736, 98, 1262)
lut = B.build_lut(B.CMAP_STOPS["gui"])

# ---- 解析真值 ----
_, xy2ll = B.make_geo(30.0, 122.0)
ll2xy, _ = B.make_geo(30.0, 122.0)
HEAD = -30.0
hx0, hy0 = ll2xy(30.0, 122.0)
tow_x, tow_y = ll2xy(30.0, 122.0)
# 拖曳阵偏移 -180 m (沿船头负向)
psi = math.radians(HEAD)
tow_x = hx0 + (-180.0) * math.sin(psi)
tow_y = hy0 + (-180.0) * math.cos(psi)

TG = {"T1": ((30.0100, 122.1000), (30.0000, 122.1000), 19, 113.8),
      "T2": ((30.0100, 121.9000), (30.0000, 121.9000), 23, 94.0)}


def tgt_pos(name, t):
    (la0, lo0), (la1, lo1), kn, tend = TG[name]
    x0, y0 = ll2xy(la0, lo0)
    x1, y1 = ll2xy(la1, lo1)
    f = min(1.0, t / tend)
    return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f


def rel_bear(ax, ay, px, py):
    gb = math.degrees(math.atan2(px - ax, py - ay)) % 360.0
    r = (gb - HEAD) % 360.0
    return r if r <= 180 else 360 - r


imgs = {}
for tag in ("hull", "tow"):
    img = np.asarray(Image.open(os.path.join(SRC, f"btr_A_{tag}.png")).convert("RGB"))
    imgs[tag] = B.image_to_db(img, RECT, lut, -5.0, 16.0)

for t in (1.0, 60.0, 120.0, 300.0, 597.0):
    print("=" * 80)
    print(f"t = {t:.0f} s")
    row_t = R_BOT - t / 600.0 * (R_BOT - R_TOP)
    for tag, ax_, ay_ in (("hull", hx0, hy0), ("tow", tow_x, tow_y)):
        db = imgs[tag]
        i0 = int(round(row_t)) - 2
        i1 = int(round(row_t)) + 3
        prof = np.nanmean(db[max(i0, 0):i1], axis=0)
        # 真值舷角 -> 列
        line = []
        for name in TG:
            px, py = tgt_pos(name, t)
            rb = rel_bear(ax_, ay_, px, py)
            col = rb / 180.0 * (C1 - C0 + 1)
            line.append((name, rb, col))
        peaks = []
        for i in range(2, len(prof) - 2):
            if prof[i] >= 8 and prof[i] >= prof[i-1] and prof[i] > prof[i+1] \
               and prof[i] >= prof[i-2] and prof[i] > prof[i+2]:
                peaks.append((B_MIN if False else i, prof[i]))
        print(f"  [{tag}] 真值: " + ", ".join(
            f"{n} 舷角 {b:6.2f}° (列 {c:6.1f})" for n, b, c in line))
        ps = sorted(peaks, key=lambda p: -p[1])[:8]
        print(f"        峰(列, 舷角, dB): " + ", ".join(
            f"{i}->{i/(C1-C0+1)*180:.1f}°:{v:.1f}" for i, v in ps))

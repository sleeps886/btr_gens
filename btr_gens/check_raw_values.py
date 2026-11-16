# -*- coding: utf-8 -*-
"""dump 真值列附近的原始 dB 值, 判定亮线是否饱和/平台化。"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "btr_out", "20261105_152635")
C0, C1 = 98.0, 1262.0
R_TOP, R_BOT = 22.4, 738.6
RECT = (22, 736, 98, 1262)
lut = B.build_lut(B.CMAP_STOPS["gui"])
HEAD = -30.0
ll2xy, xy2ll = B.make_geo(30.0, 122.0)
hx0, hy0 = ll2xy(30.0, 122.0)
psi = math.radians(HEAD)
tx0 = hx0 + (-180.0) * math.sin(psi)
ty0 = hy0 + (-180.0) * math.cos(psi)

imgs, rgbs = {}, {}
for tag in ("hull", "tow"):
    a = np.asarray(Image.open(os.path.join(SRC, f"btr_A_{tag}.png")).convert("RGB"))
    rgbs[tag] = a
    imgs[tag] = B.image_to_db(a, RECT, lut, -5.0, 16.0)

t = 300.0
row = int(round(R_BOT - t / 600.0 * (R_BOT - R_TOP)))
print(f"t=300s -> 行 {row}")
for tag, name, truth_b in (("hull", "T1", 120.00), ("hull", "T2", 60.00),
                           ("tow", "T1", 119.06), ("tow", "T2", 59.08)):
    ct = truth_b / 180.0 * (C1 - C0 + 1)
    lo, hi = int(ct) - 32, int(ct) + 44
    prof = imgs[tag][row]
    print(f"\n[{tag} {name}] 真值舷角 {truth_b}° -> 列 {ct:.1f} (局部列 {lo}..{hi})")
    seg = prof[lo:hi]
    print("  dB :", " ".join(f"{v:5.1f}" if not np.isnan(v) else "  nan" for v in seg))
    print("  列 :", " ".join(f"{c:5d}" for c in range(lo, hi)))
    px = rgbs[tag][row, lo:hi]
    print("  RGB:", " ".join(f"({r:3d},{g:3d},{b:3d})" for r, g, b in px[::4]))

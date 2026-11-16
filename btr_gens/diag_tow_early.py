# -*- coding: utf-8 -*-
"""诊断 tow 图早段 (t<80 s) 亮带碎裂原因: dump 原始 dB 剖面。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_latest import load_bands, band_centers, RECT, row_of, LEVEL, image_to_db_fixed
import btr_infer_track as B
from PIL import Image

lut = B.build_lut(B.CMAP_STOPS["gui"])
img = np.asarray(Image.open(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "btr_out", "20261105_152635",
    "btr_A_tow.png")).convert("RGB"))
d = image_to_db_fixed(img, RECT["tow"], lut, -5.0, 16.0)
span = RECT["tow"][3] - RECT["tow"][2] + 1

for t in (3.0, 20.0, 40.0, 60.0, 77.0):
    r = int(round(row_of(t))) - RECT["tow"][0]
    prof = d[max(r - 1, 0):r + 2].mean(axis=0)
    bc = [(c / span * 180, w, a) for c, w, a in band_centers(prof)]
    print(f"\nt={t:5.1f} 行{r}  亮带: " + ", ".join(f"{b:.2f}°(w={w:.0f},a={a:.1f})" for b, w, a in bc))
    lo = int(110.0 / 180 * span); hi = int(128.0 / 180 * span)
    print("  col :", " ".join(f"{c:5d}" for c in range(lo, hi, 3)))
    print("  dB  :", " ".join(f"{prof[c]:5.1f}" for c in range(lo, hi, 3)))
    print("  RGB :", " ".join(
        f"({img[r + RECT['tow'][0], c + RECT['tow'][2], 0]:3d},"
        f"{img[r + RECT['tow'][0], c + RECT['tow'][2], 1]:3d},"
        f"{img[r + RECT['tow'][0], c + RECT['tow'][2], 2]:3d})"
        for c in range(lo, hi, 6)))

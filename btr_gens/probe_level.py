# -*- coding: utf-8 -*-
"""probe_level.py —— 对比不同批次在若干等阈电平下的历元覆盖率与亮带宽度,
为自适应电平提供依据 (不再以 db_max 为锚)。"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B
import btr_detect_agent as A

BATCHES = [
    ("新/静止/180m", r"btr_out\20261105_152635", "btr_A"),
    ("旧/静止/150m", r"btr_out\20261020_145221", "btr_B"),
    ("旧/静止/180m", r"btr_out\20261020_145532", "btr_A"),
    ("geo/航点/150m", r"gen_datas", "btr_geo"),
]
HERE = os.path.dirname(os.path.abspath(__file__))
luts = [B.build_lut(s) for s in B.CMAP_STOPS.values()]

for label, d, base in BATCHES:
    dd = os.path.join(HERE, d)
    if not os.path.exists(os.path.join(dd, base + "_hull.png")):
        print(f"== {label}: 缺文件, 跳过")
        continue
    print(f"\n== {label}  ({d}\\{base}_*) ==")
    for tag in ("hull", "tow"):
        p = os.path.join(dd, f"{base}_{tag}.png")
        img = np.asarray(Image.open(p).convert("RGB"))
        cal = A.calibrate(img, luts, 600.0, 0.0, 180.0, log=lambda *a: None)
        style = A.B.pick_cmap_style(img, cal["rect"], -5.0, 16.0)
        db = A.image_to_db(img, cal["rect"], B.build_lut(B.CMAP_STOPS[style]), -5.0, 16.0)
        v = db[np.isfinite(db)]
        print(f"  [{tag}] 配色 {style}, px/° {1/cal['deg_per_px']:.2f}, "
              f"dB 实际范围 {v.min():.1f}..{v.max():.1f}, "
              f"p99 {np.percentile(v, 99):.1f}, p99.9 {np.percentile(v, 99.9):.1f}")
        for lv in (8.0, 9.0, 10.0, 11.0, 12.0, 13.5, 15.0):
            hit = np.array([(r >= lv).sum() for r in db])
            cov = float((hit >= A.MIN_W_PX).mean())
            sel = hit[hit >= A.MIN_W_PX]
            wmean = float(sel.mean()) if sel.size else 0.0
            print(f"     lv={lv:5.1f}: 覆盖 {cov*100:5.1f}%  "
                  f"均宽 {wmean:6.1f} px ({wmean / cal['nx'] * 100:4.1f}% nx)")

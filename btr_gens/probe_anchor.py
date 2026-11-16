# -*- coding: utf-8 -*-
"""probe_anchor.py —— 查老批次在锚时刻到底有几条亮带、hull/tow 视差多大。"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B
import btr_detect_agent as A

HERE = os.path.dirname(os.path.abspath(__file__))
BATCHES = [(r"btr_out\20261020_145221", "btr_B"), (r"btr_out\20261020_145532", "btr_A"),
           (r"gen_datas", "btr_geo"), (r"btr_out\20261105_152635", "btr_A")]
luts = [B.build_lut(s) for s in B.CMAP_STOPS.values()]

for d, base in BATCHES:
    print(f"\n===== {d}\\{base} =====")
    raw, cals = {}, {}
    for tag in ("hull", "tow"):
        p = os.path.join(HERE, d, f"{base}_{tag}.png")
        img = np.asarray(Image.open(p).convert("RGB"))
        cals[tag] = A.calibrate(img, luts, 600.0, 0.0, 180.0, log=lambda *a: None)
        style = B.pick_cmap_style(img, cals[tag]["rect"], -5.0, 16.0)
        db = A.image_to_db(img, cals[tag]["rect"], B.build_lut(B.CMAP_STOPS[style]), -5.0, 16.0)
        cals[tag]["_db"] = db
        cals[tag]["_style"] = style
    lev = A.pick_level([cals["hull"]["_db"], cals["tow"]["_db"]], None, 16.0)
    for tag in ("hull", "tow"):
        raw[tag] = A.extract_one(cals[tag]["_db"], cals[tag], 0.0, 180.0, 600.0, 2.0, lev,
                                 log=lambda *a: None)
    print(f"  电平 {lev:.1f} dB")
    for frac in (0.5, 0.25, 0.75):
        ta = max(raw["hull"]) * frac
        ka = min(raw["hull"], key=lambda x: abs(x - ta))
        kb = min(raw["tow"], key=lambda x: abs(x - ta))
        hb = sorted(raw["hull"][ka], key=lambda x: -x[2])
        tb = sorted(raw["tow"][kb], key=lambda x: -x[2])
        print(f"  t={ka:5.0f} s  hull 亮带 {[(round(b,2), round(a,1)) for b, w, a in hb]}"
              f"   tow 亮带 {[(round(b,2), round(a,1)) for b, w, a in tb]}")
        # 最近邻视差
        for b_h, w_h, a_h in hb:
            if not tb:
                break
            dmin = min(abs(b_t - b_h) for b_t, w_t, a_t in tb)
            print(f"      hull {b_h:7.2f}° (峰 {a_h:4.1f}) -> 最近 tow 差 {dmin:5.2f}°")

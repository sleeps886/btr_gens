# -*- coding: utf-8 -*-
"""诊断: 各时刻两阵的等阈亮带, 看 T1 尾段 (t>557) 与 T2 首段 (t~60) 为何缺迹。"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B
from extract_latest import (RECT, ROWS, db, band_centers, bearing_of, LEVEL, times, raw)

print("---- T1 分支 (hull, 期望 120 deg) 尾段 ----")
for t in [540, 550, 555, 558, 560, 562, 570, 580, 590, 599]:
    print(f"  t={t:5.0f}  hull bands: "
          + ", ".join(f"{b:.2f}deg(w={w:.0f},a={a:.1f})" for b, w, a in raw["hull"].get(t, [])))
print("---- T2 分支 (期望 60 deg) 首段 ----")
for t in [3, 30, 55, 58, 60, 62, 66, 70, 100, 555, 558, 560, 562, 570]:
    hb = ", ".join(f"{b:.2f}(w={w:.0f})" for b, w, a in raw["hull"].get(t, []))
    tb = ", ".join(f"{b:.2f}(w={w:.0f})" for b, w, a in raw["tow"].get(t, []))
    print(f"  t={t:5.0f}  hull: [{hb}]   tow: [{tb}]")
print("---- 每时刻亮带个数统计 ----")
for tag in ("hull", "tow"):
    cnt = [len(raw[tag].get(t, [])) for t in times]
    print(f"  {tag}: 时刻数 {len(times)}, 有带时刻 {sum(1 for c in cnt if c)}, "
          f"带数分布 {np.bincount(np.array(cnt))[:6]}")

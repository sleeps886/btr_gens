# -*- coding: utf-8 -*-
"""probe_145532.py —— 核对 145532 批次的真值起终点与重建舷角, 判断 41° 残差是
真 bug 还是场景本身 (目标交叉/弱信号) 造成。"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_detect_agent as A
import btr_infer_track as B

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, r"btr_out\20261020_145532")

plat = A.parse_truth_platform(os.path.join(D, "btr_A_track.txt"))
print("平台参数 (track.txt):", plat)
tt = A.parse_truth_targets({"hull": os.path.join(D, "btr_A_hull.txt"),
                            "tow": os.path.join(D, "btr_A_tow.txt")})
geom = B.Geometry(plat["start_ll"], plat["heading"], plat["speed_kn"],
                  plat.get("hull_m", (0, 0)), plat["tow_m"],
                  *(plat.get("ref", (30.0, 122.0))))
ll2xy, _ = B.make_geo(*(plat.get("ref", (30.0, 122.0))))

for name, recs in tt.items():
    h = recs["hull"]
    print(f"\n{name}: hull txt 舷角 {h['bearing'][0]:.1f} -> {h['bearing'][1]:.1f};"
          f" start {h['start']} end {h['end']}; {h['speed_kn']} kn;"
          f" CPA {h['cpa_km']:.2f} km @ {h['cpa_t']:.0f} s")
    x0, y0 = ll2xy(*h["start"])
    x1, y1 = ll2xy(*h["end"])
    dist = math.hypot(x1 - x0, y1 - y0)
    v = h["speed_kn"] * B.KN2MS
    t_walk = dist / v if v > 1e-6 else 1e9
    print(f"   几何推算: 航程 {dist/1000:.3f} km, 走完需 {t_walk:.0f} s")
    for t in (0.0, 300.0, 600.0):
        rel = A.truth_rel_bearings(geom, ll2xy, h, t_walk, t)[0]
        print(f"   t={t:5.0f}: 重建 hull 舷角 {rel:6.2f}° (txt 端点 {h['bearing'][0 if t == 0 else 1]:.2f}°)")

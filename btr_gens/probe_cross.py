# -*- coding: utf-8 -*-
"""probe_cross.py —— 验证旧批次两目标是否在方位上交叉 (解释 41°/144° 的方位残差)。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_detect_agent as A

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(r"btr_out\20261020_145221", "btr_B"), (r"btr_out\20261020_145532", "btr_A")]

for d, base in CASES:
    print(f"\n===== {d}\\{base} =====")
    tt = A.parse_truth_targets({
        "hull": os.path.join(HERE, d, f"{base}_hull.txt"),
        "tow": os.path.join(HERE, d, f"{base}_tow.txt")})
    names = list(tt)
    for n in names:
        h, t = tt[n]["hull"], tt[n]["tow"]
        print(f"  {n}: hull 舷角 {h['bearing'][0]:6.1f} -> {h['bearing'][1]:6.1f}"
              f" | tow {t['bearing'][0]:6.1f} -> {t['bearing'][1]:6.1f}"
              f" | 观测窗 [{h['window'][0]:.0f},{h['window'][1]:.0f}]"
              f" | CPA {h['cpa_km']:.2f} km @ {h['cpa_t']:.0f} s")
    if len(names) == 2:
        print("  两目标 hull 舷角差 (线性内插近似):")
        for t in range(0, 601, 60):
            bs = []
            for n in names:
                b0, b1 = tt[n]["hull"]["bearing"]
                bs.append(b0 + (b1 - b0) * min(1.0, t / 600.0))
            gap = abs(bs[0] - bs[1])
            print(f"    t={t:4.0f}: {bs[0]:6.2f}° vs {bs[1]:6.2f}°,"
                  f" 差 {gap:5.2f}° = {gap / 180.0 * 1165:5.1f} px")

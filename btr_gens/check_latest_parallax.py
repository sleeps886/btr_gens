# -*- coding: utf-8 -*-
"""从最新一批 hull/tow BTR 图像逐时刻提取目标舷角, 并用双阵视差交会定位。

平台静止 + 目标走完航点后方位恒定 -> 批量 CV 拟合距离不可观 (会发散),
但同一时刻 hull/tow 的视差交会对本场景是良态的, 故走这条路。
"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "btr_out", "20261105_152635")
RECTS = {"hull": (22, 736, 98, 1262), "tow": (22, 737, 98, 1261)}
DB_RANGE = (-5.0, 16.0)
T_MAX, B_MIN, B_MAX = 600.0, 0.0, 180.0
THR = 8.0                       # 峰值门限 (dB, 相对噪声底)

geom = B.Geometry((30.0, 122.0), -30.0, 0.0, (0.0, 0.0), (-180.0, 0.0),
                  30.0, 122.0)
lut = B.build_lut(B.CMAP_STOPS["gui"])

db = {}
peaks = {}
for tag in ("hull", "tow"):
    img = np.asarray(Image.open(os.path.join(SRC, f"btr_A_{tag}.png")).convert("RGB"))
    rect = RECTS[tag]
    d = B.image_to_db(img, rect, lut, *DB_RANGE)
    db[tag] = d
    dk, prof = B.detect_per_time(d, rect, T_MAX, B_MIN, B_MAX, db_thr=THR,
                                 t_step=2.0)
    peaks[tag] = dk
    print(f"[{tag}] 绘图区 {rect}, 峰值点迹 {len(dk)} 个, "
          f"强度范围 {min(x[2] for x in dk):.1f}~{max(x[2] for x in dk):.1f} dB")

BRANCH = {"T1": (105.0, 128.0), "T2": (45.0, 68.0)}   # 折叠舷角搜索窗


def series(tag, lo, hi):
    """该阵上落在 [lo,hi] 舷角窗内的点迹 -> {t: (bearing, dB)}"""
    out = {}
    for t, b, a in peaks[tag]:
        if lo <= b <= hi and t not in out:
            out[t] = (b, a)
        elif lo <= b <= hi and a > out[t][1]:
            out[t] = (b, a)
    return out


print("\n" + "=" * 84)
for name, (lo, hi) in BRANCH.items():
    sh_series = series("hull", lo, hi)
    st_series = series("tow", lo, hi)
    ts = sorted(set(sh_series) & set(st_series))
    if not ts:
        print(f"{name}: 两阵无共同时刻");  continue
    rows = []
    for t in ts:
        bh, ah = sh_series[t]
        bt, at = st_series[t]
        for sg in (+1, -1):        # 左右舷两支镜像 (两阵共线于艏艉线 -> 代价相同)
            p = B.instant_fix(geom, t, bh, bt, sg, sg)
            if np.isfinite(p[0]):
                rows.append((t, bh, bt, abs(bh - bt), p, ah, at, sg))
    print(f"\n===== {name} =====  共同观测时刻 {ts[0]:.0f}~{ts[-1]:.0f} s "
          f"({len(ts)} 个时刻两阵同时可见)")
    for sg in (+1, -1):
        sub = [r for r in rows if r[7] == sg]
        if not sub:
            continue
        side = "右舷" if sg > 0 else "左舷"
        first, last = sub[0], sub[-1]
        # 距离/方位的时间序列
        rng = np.array([math.hypot(r[4][0] - geom.hull_xy(r[0])[0],
                                   r[4][1] - geom.hull_xy(r[0])[1]) for r in sub])
        tt = np.array([r[0] for r in sub])
        k = int(np.argmin(rng))
        print(f"  [{side}分支] 视差中位 {np.median([r[3] for r in sub]):.2f}°")
        for lbl, r, rr in (("初始", first, rng[0]), ("结束", last, rng[-1])):
            t, bh, bt, par, p, ah, at, _ = r
            lat, lon = geom.xy2ll(p[0], p[1])
            dist = math.hypot(p[0] - geom.hull_xy(t)[0], p[1] - geom.hull_xy(t)[1])
            print(f"    {lbl} t={t:5.0f}s: ({lat:.4f}°N, {lon:.4f}°E) | "
                  f"距舷侧阵 {dist/1000:6.3f} km | 舷角 hull {bh:6.2f}° / "
                  f"tow {bt:6.2f}° | 视差 {par:.2f}°")
        print(f"    舷角偏转: hull {first[1]:6.2f}° -> {last[1]:6.2f}° "
              f"(Δ={last[1]-first[1]:+.2f}°) | tow {first[2]:6.2f}° -> "
              f"{last[2]:6.2f}° (Δ={last[2]-first[2]:+.2f}°)")
        print(f"    最近点: {rng[k]/1000:6.3f} km @ t={tt[k]:.0f}s | "
              f"距离跨度 {rng.min()/1000:.3f}~{rng.max()/1000:.3f} km | "
              f"距离标准差 {rng.std(ddof=1)/1000:.3f} km (交会散布)")
        # 目标位移
        p0, p1 = np.array(first[4]), np.array(last[4])
        disp = float(np.hypot(*(p1 - p0)))
        if disp > 1:
            crs = math.degrees(math.atan2(*(p1 - p0)[::-1])) % 360.0
            dt = last[0] - first[0]
            print(f"    拟合运动: 位移 {disp/1000:.3f} km, 航向 {crs:.1f}°, "
                  f"速度 {disp/max(dt,1):.2f} m/s ({disp/max(dt,1)/B.KN2MS:.1f} kn)")

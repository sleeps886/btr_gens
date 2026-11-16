# -*- coding: utf-8 -*-
"""对 btr_out 最新一批 hull/tow 图做纯图像反演, 反馈检测到的起止位置/舷角偏转/距离。

find_plot_rect 在大面积纯黑背景的图上会误判 (信号只占少量列),
这里用像素级诊断得到的精确矩形覆盖它。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B

# ---- 精确绘图区 (行 r0..r1, 列 c0..c1), 由像素诊断得出 ----
RECTS = {"hull": (22, 736, 98, 1262), "tow": (22, 737, 98, 1261)}
_orig_find = B.find_plot_rect
_state = {"tag": None}


def _patched(rgb, lut_list, dist_thr=45.0):
    return RECTS[_state["tag"]]


# run_inference 先 hull 后 tow 各调一次
_calls = []


def _patched2(rgb, lut_list, dist_thr=45.0):
    _calls.append(1)
    return RECTS["hull"] if len(_calls) == 1 else RECTS["tow"]


B.find_plot_rect = _patched2

SRC = os.path.join(B.__file__.replace("btr_infer_track.py", ""),
                   "btr_out", "20261105_152635")
HULL = os.path.join(SRC, "btr_A_hull.png")
TOW = os.path.join(SRC, "btr_A_tow.png")

# 平台 A 参数 (来自同批 btr_A_track.txt)
GEOM = dict(start_ll=(30.0, 122.0), heading_deg=-30.0, speed_kn=0.0,
            hull_m=(0.0, 0.0), tow_m=(-180.0, 0.0),
            ref_lat=30.0, ref_lon=122.0)

res_inf = B.run_inference(HULL, TOW, log=lambda m: print(m), **GEOM)
geom = res_inf["geom"]
print("\n" + "=" * 78)
print("反演结果汇总 (纯图像, 不使用仿真内部数据)")
print("=" * 78)


def rng_of(t, tgt_xy, arr_xy):
    A = np.array(arr_xy(t))
    return float(np.hypot(*(np.array(tgt_xy) - A)))


for res in res_inf["results"]:
    best = res["fit"]["best"]
    s = best["state"]
    t0, t1 = res["t0"], res["t1"]
    print(f"\n--- {res['name']}  共同观测窗口 [{t0:.0f}, {t1:.0f}] s "
          f"| 视差中位 {res['pair']['dparallax']:.3f}° | rms {best['rms']:.3f}° "
          f"| sh={best['sh']:+d} st={best['st']:+d} "
          f"{'[左右舷镜像模糊]' if res['fit']['ambiguous'] else '[已分辨]'}")
    # 检测到的舷角 (图像直接读出)
    print(f"    舷侧阵舷角: {res['th'][0]:6.2f}° (t={t0:.0f}s) -> "
          f"{res['th'][-1]:6.2f}° (t={t1:.0f}s)   Δ={res['th'][-1]-res['th'][0]:+.2f}°")
    print(f"    拖曳阵舷角: {res['tt'][0]:6.2f}° (t={t0:.0f}s) -> "
          f"{res['tt'][-1]:6.2f}° (t={t1:.0f}s)   Δ={res['tt'][-1]-res['tt'][0]:+.2f}°")
    spd = math.hypot(s[2], s[3])
    crs = math.degrees(math.atan2(s[2], s[3])) % 360.0
    print(f"    拟合速度  : {spd:6.2f} m/s ({spd/B.KN2MS:5.1f} kn) | 航向 {crs:6.1f}°")
    for lbl, t in (("初始 t=%.0fs" % t0, t0), ("结束 t=%.0fs" % t1, t1)):
        x, y = s[0] + s[2] * t, s[1] + s[3] * t
        lat, lon = geom.xy2ll(x, y)
        rh = rng_of(t, (x, y), geom.hull_xy)
        rt = rng_of(t, (x, y), geom.tow_xy)
        gb = math.degrees(math.atan2(x - geom.hull_xy(t)[0],
                                    y - geom.hull_xy(t)[1])) % 360.0
        print(f"    {lbl}: ({lat:.4f}°N, {lon:.4f}°E) | 距舷侧阵 {rh/1000:6.3f} km, "
              f"距拖曳阵 {rt/1000:6.3f} km | 地理方位 {gb:6.1f}°")
    # CPA
    ts = np.arange(t0, t1 + 1, 1.0)
    hx, hy = geom.hull_xy(ts)
    tx, ty = s[0] + s[2] * ts, s[1] + s[3] * ts
    d = np.hypot(tx - hx, ty - hy)
    k = int(np.argmin(d))
    print(f"    最近点距离(CPA): {d[k]/1000:6.3f} km @ t={ts[k]:.0f}s")
    if res["fit"]["mirror"] is not None:
        ms = res["fit"]["mirror"]["state"]
        x, y = ms[0] + ms[2] * t0, ms[1] + ms[3] * t0
        lat, lon = geom.xy2ll(x, y)
        print(f"    镜像候选初始位置: ({lat:.4f}°N, {lon:.4f}°E)  "
              f"(关于艏艉轴对称, 距离/舷角大小完全相同)")

# -*- coding: utf-8 -*-
"""plot_latest.py —— 把 extract_latest 的检测结果叠加回 hull/tow 原图并画出几何对照图"""
import math
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B
import extract_latest as E

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

raw, shapes = E.load_bands()
ans = E.anchors(raw)
det, ests = {}, {}
for name, bh, bt, ta in ans:
    h_tr = E.track_from(raw["hull"], ta, bh)
    t_tr = E.track_from(raw["tow"], ta, bt)
    det[name] = (h_tr, t_tr)
    ests[name] = E.branch_est(h_tr, t_tr, +1)
    ests[name + "_neg"] = E.branch_est(h_tr, t_tr, -1)

truth = E.parse_truth()
# 把检测锚点对到真值名
match = {}
for tname, d in truth.items():
    h = d["hull"]
    best, bd = None, 1e9
    for aname, (h_tr, _) in det.items():
        k = E.nearest_key(h_tr, 300.0)
        if k is None:
            continue
        dd = abs(h_tr[k][0] - h["b"][1])
        if dd < bd:
            best, bd = aname, dd
    match[tname] = best

COL = {"T1": "#00e5ff", "T2": "#ff9d00"}
fig = plt.figure(figsize=(15, 13), facecolor="black")
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.22, wspace=0.14)


def pix_axes(ax, tag):
    """把绘图区像素坐标装回图像"""
    r0, r1, c0, c1 = E.RECT[tag]
    img = np.asarray(Image.open(os.path.join(E.SRC, f"btr_A_{tag}.png")).convert("RGB"))
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.set_xlabel(f"{tag} 方位历程图 + 检测方位轨迹叠加", color="white", fontsize=11)
    return r0, r1, c0, c1


def b2px(tag, b):
    r0, r1, c0, c1 = E.RECT[tag]
    return c0 + b / 180.0 * (c1 - r0 * 0 - c0 + 1)


def t2px(tag, t):
    return E.R_T0 - t / E.T_MAX * (E.R_T0 - E.R_T600)


for j, tag in enumerate(("hull", "tow")):
    ax = fig.add_subplot(gs[j, :])
    r0, r1, c0, c1 = pix_axes(ax, tag)
    span = c1 - c0 + 1
    for tname, aname in match.items():
        h_tr, t_tr = det[aname]
        tr = h_tr if tag == "hull" else t_tr
        if not tr:
            continue
        ts = np.array(sorted(tr))
        bs = np.array([tr[t][0] for t in ts])
        xs = c0 + bs / 180.0 * span
        ys = t2px(tag, ts)
        ax.plot(xs, ys, "-", color=COL[tname], lw=1.4, alpha=0.95,
                label=f"{tname} 检测方位 ({aname})")
        for lbl, mk in (("初始", "o"), ("结束", "s")):
            ax.plot(xs[0 if lbl == "初始" else -1],
                    ys[0 if lbl == "初始" else -1], mk, color=COL[tname],
                    ms=9, mec="white", mew=1.2)
        # 运动段标注
        e = ests[aname]
        if e is not None and e["run"] is not None:
            i0, i1 = e["run"]
            sel = slice(i0, i1 + 1)
            ax.plot(xs[sel], ys[sel], "-", color="white", lw=3.0, alpha=0.32)
    ax.legend(loc="upper right", fontsize=9, facecolor="#101828", labelcolor="white")
    ax.set_title(f"{'舷侧阵' if tag == 'hull' else '拖曳阵'} (hull/tow) 检测叠加 —— "
                 f"白粗段 = 方位变化的运动段, 圆/方 = 检测初始/结束",
                 color="white", fontsize=12)

# ---- 第 3 行: 几何对照 (左: 检测 vs 真值平面; 右: 距离-时间) ----
ax = fig.add_subplot(gs[2, 0])
ax.set_facecolor("#0a0f1f")
geom = E.geom
tt = np.linspace(0, 600, 400)
px, py = geom.platform_xy(tt)
ll = [geom.xy2ll(x, y) for x, y in zip(px, py)]
ax.plot([p[1] for p in ll], [p[0] for p in ll], "w-", lw=1.6, label="平台 A (静止)")
hx, hy = geom.hull_xy(0.0)
tx, ty = geom.tow_xy(0.0)
lh, lt = geom.xy2ll(hx, hy), geom.xy2ll(tx, ty)
ax.plot(lh[1], lh[0], "s", color="#ffd400", ms=9, label="舷侧阵 t=0")
ax.plot(lt[1], lt[0], "^", color="#00ffcc", ms=9, label="拖曳阵 t=0")

ll2xy, _ = B.make_geo(30.0, 122.0)
for tname, d in truth.items():
    h = d["hull"]
    x0, y0 = ll2xy(*h["s"])
    x1, y1 = ll2xy(*h["e"])
    f = np.minimum(1.0, tt / h["cpa"][1])
    Tx, Ty = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
    lla = [geom.xy2ll(x, y) for x, y in zip(Tx, Ty)]
    ax.plot([p[1] for p in lla], [p[0] for p in lla], "--", color=COL[tname],
            lw=1.6, alpha=0.9, label=f"{tname} 真值轨迹")
    ax.plot(h["s"][1], h["s"][0], "o", color=COL[tname], ms=10, mec="white")
    ax.plot(h["e"][1], h["e"][0], "s", color=COL[tname], ms=10, mec="white")

for tname, aname in match.items():
    for key, style, ms in ((aname, "-", 8), (aname + "_neg", ":", 7)):
        e = ests.get(key)
        if e is None:
            continue
        lla = [geom.xy2ll(x, y) for x, y in zip(e["X"], e["Y"])]
        ax.plot([p[1] for p in lla], [p[0] for p in lla], style, color="white",
                lw=1.5 if style == "-" else 1.2, alpha=0.9,
                label=f"{tname} 检测 {'右舷+' if e['sg'] > 0 else '左舷-'}分支")
        ax.plot(lla[0][1], lla[0][0], "o", color="white", ms=ms, mec=COL[tname], mew=1.6)
        ax.plot(lla[-1][1], lla[-1][0], "s", color="white", ms=ms, mec=COL[tname], mew=1.6)
ax.set_xlabel("经度 (°E)", color="white")
ax.set_ylabel("纬度 (°N)", color="white")
ax.tick_params(colors="white", labelsize=8)
ax.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
ax.set_aspect(1.0 / math.cos(math.radians(30.0)))
ax.legend(loc="best", fontsize=7, facecolor="#101828", labelcolor="white")
ax.set_title("检测轨迹 (白) vs 真值轨迹 (虚线彩) —— 圆=初始, 方=结束", color="white", fontsize=12)

ax = fig.add_subplot(gs[2, 1])
ax.set_facecolor("#0a0f1f")
for tname, aname in match.items():
    e = ests.get(aname)
    if e is None:
        continue
    ax.plot(e["ts"], e["r_raw"] / 1000, ".", color=COL[tname], ms=3, alpha=0.4)
    ax.plot(e["ts"], e["r_rep"] / 1000, "-", color=COL[tname], lw=2.0,
            label=f"{tname} 检测距离 (平滑)")
    h = truth[tname]["hull"]
    x0, y0 = ll2xy(*h["s"])
    x1, y1 = ll2xy(*h["e"])
    f = np.minimum(1.0, e["ts"] / h["cpa"][1])
    Tx, Ty = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
    r_tr = np.hypot(Tx - hx, Ty - hy)
    ax.plot(e["ts"], r_tr / 1000, "--", color="white", lw=1.4, alpha=0.85,
            label=f"{tname} 真值距离")
ax.set_xlabel("时间 (s)", color="white")
ax.set_ylabel("距舷侧阵距离 (km)", color="white")
ax.tick_params(colors="white", labelsize=8)
ax.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
ax.legend(loc="best", fontsize=8, facecolor="#101828", labelcolor="white")
ax.set_title("距离: 点=单历元交会, 实线=检测平滑, 白虚线=真值", color="white", fontsize=12)

for a in fig.axes:
    a.title.set_color("white")
fig.suptitle("BTR 方位历程图目标检测 —— 初始/结束位置、舷角偏转、距离  "
             f"(btr_out/{E.BATCH})", color="white", fontsize=14)
out = os.path.join(E.HERE, f"detect_overlay_{E.BATCH}.png")
fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="black")
plt.close(fig)
print("OK ->", out)

# -*- coding: utf-8 -*-
"""btr_detect_agent.py —— BTR 方位历程图目标检测代理

输入两张图 (舷侧阵 hull + 拖曳阵 tow 的方位历程图 PNG), 快速输出:
  · 每个检出目标的初始/结束位置 (经纬度)
  · 舷角偏转 (hull / tow 各自的起终舷角与 Δ)
  · 距离 (初始/结束/中位)、CPA 与时刻
  · 运动段拟合速度与航向
并自动与生成端真值 (同批 btr_*_hull.txt / btr_*_tow.txt / btr_*_track.txt)
逐项对比, 给出差异值 (方位残差 RMS、位置误差 m、CPA/航速/航向偏差)。

与 btr_infer_track.py 的区别
  btr_infer_track 走"批量匀速直线 Gauss-Newton 拟合"。本代理针对
  平台静止 / 目标走完航点后方位恒定 这类距离不可观的场景, 改用
  逐历元双站视差交会 + 距离中值平滑 + 通道分解速度, 更稳也更快。

流水线
  1 轴标定   自动定位绘图区 (剔色标条), 用轴边当作数据范围端点,
             再用灰色网格线反推刻度做残差校验 (不写死像素)。
  2 方位提取 等阈亮带亚像素中点。亮线饱和时峰值检测会锁到平台右缘
             (可偏 +5°), 亮带两缘对称于真实方位, 削顶不影响中点。
             电平按 dB 上限自适应 (默认 db_max - 2.5)。
  3 跟迹     观测中段两阵强亮带互配为锚点 (无需先验方位), 前后恒速预测跟迹。
  4 定距     同历元 hull/tow 视差交会; 视差 < PAR_MIN 的历元丢弃 (距离发散);
             距离做中值滤波, 边缘截断历元用局部中值代替 (消除截断偏置)。
  5 速度     通道分解: 切向 = r̄·dθ/dt (方位通道精确),
             径向 = dr/dt (仅在中值滤波满窗历元上取, 否则判为不可观置 0)。
  6 左右舷   两阵共线于艏艉线时镜像不可分辨: 两支都报; 有真值时按位置误差判定。

用法
  python btr_detect_agent.py --hull-img a/btr_A_hull.png --tow-img a/btr_A_tow.png
      (平台参数默认从同目录 btr_*_track.txt 自动读取)
  python btr_detect_agent.py --hull-img ... --tow-img ... ^
      --start-ll 30.0 122.0 --heading -30 --speed-kn 0 --tow-m -180 0
  常用开关: --json / --png / --no-truth / --level 13.5 / --dt 2 / --side auto
"""
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B

KN2MS = B.KN2MS

# ------------------ 可调参数 ------------------
MERGE_PX = 20           # 亮带合并缺口 (px): 容忍网格点线/参考线造成的窄缺口
MIN_W_PX = 6            # 亮带最小宽度 (px)
GATE_DEG = 4.0          # 跟迹门限 (°)
MISS_MAX_S = 30.0       # 跟迹允许断档 (s)
PAR_MIN_DEG = 0.35      # 采纳历元的最小双站视差 (°)
RWIN = 41               # 距离中值滤波窗 (历元数)
RLOCAL = 11             # 边缘历元的局部中值窗 (历元数)
DEV_DEG = 0.20          # 运动段判据: 与末端恒定方位的偏差门限 (°)
RADIAL_SIGMA = 2.5      # 径向速度显著性门限 (|v_r|/标准误), 不足则判为不可观置 0
LEVEL_OFF = 2.5         # 等阈电平 = min(两阵 p99.9) - LEVEL_OFF
COV_MIN = 0.85          # 自适应电平的最低历元覆盖率
WIDTH_CAP = 0.30        # 亮带平均宽度上限 (倍绘图区宽度), 超过即已并入饱和晕台
ANCHOR_FRACS = (0.5, 0.35, 0.65, 0.25, 0.75, 0.45, 0.55)   # 锚点时刻候选 (依次试)
LEVEL_LADDER = (0.0, -1.5, -3.0, -4.5)  # 多电平阶梯: 主电平只捞到最亮目标时逐级放宽
ANCHOR_PEAK_OFF = 2.0   # 锚点亮带最低峰值 = db_thr + 该偏移


# ==================== 1. 轴标定 ====================
def _runs(mask_1d, thr, min_gap=12):
    """一维占比序列 -> 连续 >thr 的区段 [(lo, hi)], 间隔 <= min_gap 的合并"""
    idx = np.where(mask_1d > thr)[0]
    if len(idx) == 0:
        return []
    out, s, prev = [], idx[0], idx[0]
    for v in idx[1:]:
        if v - prev > min_gap:
            out.append((s, prev))
            s = v
        prev = v
    out.append((s, prev))
    return out


def find_axes(rgb, lut_list, db_floor=45, col_thr=0.02, row_thr=0.02):
    """自动定位绘图区 (r0, r1, c0, c1), 并排除右侧色标条

    信号只占绘图区少量列, 因此阈值取得很低 (0.02); 色标条是与主区之间
    有明显间隔的窄列段, 按"宽度 < 主区 5% 且贴右边"识别后剔除。
    """
    h, w, _ = rgb.shape
    rgb16 = rgb.astype(np.int16)
    mask = np.zeros((h, w), bool)
    for lut in lut_list:
        lut = lut[lut.astype(np.int32).sum(1) > db_floor]
        _, dmin = B._color_lut_map(rgb16, lut)
        mask |= dmin <= 45.0 ** 2
    R, G, Bl = rgb16[..., 0], rgb16[..., 1], rgb16[..., 2]
    mask |= (R < 70) & (Bl > R + 40) & (Bl >= 60) & (Bl <= 240)   # 暗蓝背景斑点
    cruns = [r for r in _runs(mask.mean(0), col_thr) if r[1] - r[0] > 30]
    if not cruns:
        raise ValueError("未定位到绘图区列范围 (配色不匹配?)")
    widths = [c[1] - c[0] for c in cruns]
    widest = max(widths)
    # 剔除色标条: 明显窄于主区且位于主区右侧
    keep = [c for c, wd in zip(cruns, widths)
            if not (wd < 0.05 * widest and c[0] > max(x[1] for x in cruns) - widest * 0.2
                    and c[1] > 0.85 * w)]
    if not keep:
        keep = [cruns[int(np.argmax(widths))]]
    c0 = min(c[0] for c in keep)
    c1 = max(c[1] for c in keep)
    rruns = [r for r in _runs(mask[:, c0:c1 + 1].mean(1), row_thr) if r[1] - r[0] > 30]
    r0, r1 = min(r[0] for r in rruns), max(r[1] for r in rruns)
    return (r0, r1, c0, c1), mask


def gridlines(rgb, rect, mask, axis, min_prom=0.06):
    """绘图区内的灰色点线网格 -> 刻度像素位置 (绘图区局部坐标)

    只认中性灰 (R≈G≈B, 中等亮度), 避免把亮带边缘/彩色信号误当网格线。
    """
    r0, r1, c0, c1 = rect
    R = rgb[..., 0].astype(np.int16)
    G = rgb[..., 1].astype(np.int16)
    Bl = rgb[..., 2].astype(np.int16)
    gray = (np.abs(R - G) < 14) & (np.abs(G - Bl) < 14) & (R > 22) & (R < 140)
    sub = gray[r0:r1 + 1, c0:c1 + 1]
    frac = sub.mean(0) if axis == "x" else sub.mean(1)
    base = float(np.median(frac))
    out = []
    for i in range(1, len(frac) - 1):
        if frac[i] > base + min_prom and frac[i] >= frac[i - 1] and frac[i] > frac[i + 1]:
            out.append(i)
    grp, cur = [], []
    for v in out:
        if cur and v - cur[-1] > 3:
            grp.append(float(np.mean(cur)))
            cur = []
        cur.append(v)
    if cur:
        grp.append(float(np.mean(cur)))
    return grp


def calibrate(rgb, lut_list, t_max, b_min, b_max, log=print):
    """轴边 -> 数据范围端点 (xlim=[b_min,b_max], ylim=[0,t_max]); 网格线做校验"""
    rect, mask = find_axes(rgb, lut_list)
    r0, r1, c0, c1 = rect
    info = {"rect": rect, "nx": c1 - c0 + 1, "ny": r1 - r0 + 1}
    info["deg_per_px"] = (b_max - b_min) / info["nx"]
    info["s_per_px"] = t_max / info["ny"]
    gx = gridlines(rgb, rect, mask, "x")
    gy = gridlines(rgb, rect, mask, "y")
    info["grid_x"] = gx
    info["grid_y"] = gy

    # 校验 1: x 刻度把 b_min..b_max 七等分 -> 期望像素间距 nx/6
    exp_px = info["nx"] / 6.0
    res = []
    if len(gx) >= 2:
        d = np.diff(sorted(gx))
        res = [abs(v - exp_px) * info["deg_per_px"] for v in d if abs(v - exp_px) < 0.3 * exp_px]
    # 校验 2: 网格线应落在 "0° + k*exp_px" 的格点上 -> 外推 0° 列与绘图区左边的偏差
    origin_shift = None
    if len(gx) >= 1:
        mod = np.mod(np.array(gx), exp_px)
        mod = np.minimum(mod, exp_px - mod)          # 折到 [0, exp_px/2]
        origin_shift = float(np.median(mod)) * info["deg_per_px"]
    info["grid_resid_deg"] = float(np.max(res)) if res else None
    info["grid_origin_shift_deg"] = origin_shift
    log(f"[标定] 绘图区 rows {r0}-{r1}, cols {c0}-{c1} "
        f"({info['nx']}×{info['ny']} px, {1 / info['deg_per_px']:.2f} px/°); "
        f"网格线 x {len(gx)} 条 / y {len(gy)} 条"
        + (f", 间距 vs 30° 残差 ≤{info['grid_resid_deg']:.2f}°"
           if info["grid_resid_deg"] is not None else "")
        + (f", 0° 原点偏移 {origin_shift:.2f}°" if origin_shift is not None else ""))
    return info


# ==================== 2. 像素 -> dB ====================
def image_to_db(rgb, rect, lut, db_min, db_max, dist_thr=60.0):
    """修正版: 黄色参考线判据收紧, 避免把配色自身的黄停靠点当曲线掩掉"""
    r0, r1, c0, c1 = rect
    sub = rgb[r0:r1 + 1, c0:c1 + 1]
    R = sub[..., 0].astype(np.int16)
    G = sub[..., 1].astype(np.int16)
    Bl = sub[..., 2].astype(np.int16)
    lum = 0.2126 * R + 0.7152 * G + 0.0722 * Bl
    mx = np.maximum(R, G)
    whiteish = (lum > 0.72 * 255) & (Bl > 0.55 * mx)
    yellowish = (R > 248) & (G > 248) & (Bl < 45) & (np.abs(R - G) < 10)
    idx, dmin = B._nearest_lut(sub.astype(np.int16), lut)
    ok = dmin <= dist_thr ** 2
    db = db_min + idx.astype(float) / (len(lut) - 1) * (db_max - db_min)
    bad = B._dilate(whiteish | yellowish | ~ok, it=2)
    db[bad] = np.nan
    cols = np.arange(db.shape[1])
    for i in range(db.shape[0]):
        row = db[i]
        good = ~np.isnan(row)
        if good.sum() >= 2 and (~good).any():
            db[i] = np.interp(cols, cols[good], row[good])
    return db


def pick_level(dbs, level_hint=None, db_max=16.0):
    """自适应等阈电平 (dB) —— 由数据自身决定, 不以 db_max 为锚。

    两阵必须共用一个电平: 电平不一致会让其中一阵的亮带中点系统性偏移零点几度,
    而短基线 (150~180 m) 的视差只有 ~0.9°, 相对误差立刻放大成百分之十几的距离误差。

    规则: 取两阵 dB 分布 p99.9 的较小者再减 LEVEL_OFF (饱和批 => 电平落在饱和核心
    之下, 未饱和批 => 电平落在真实峰值之下), 然后向下微调直到两阵的历元覆盖率
    都达到 COV_MIN —— 覆盖率不足说明电平高于目标亮线峰值, 轨迹会大面积断裂;
    亮带过宽 (超过 WIDTH_CAP 倍绘图区宽度) 说明已并入饱和晕台, 中点会漂。
    实测 (probe_level.py): 新批 p99.9=16 -> 13.5 dB (覆盖 99.7%);
    旧批 p99.9=13.0/12.0 -> 10.5/9.5 dB (覆盖 99%+), 而固定 13.5 只有 1~57%。
    """
    if level_hint is not None:
        return float(level_hint)
    if not isinstance(dbs, (list, tuple)):
        dbs = [dbs]

    def cov_width(db, lv):
        hit = np.array([(r >= lv).sum() for r in db])
        sel = hit[hit >= MIN_W_PX]
        return (float((hit >= MIN_W_PX).mean()),
                float(sel.mean()) / db.shape[1] if sel.size else 0.0)

    p999 = min(float(np.percentile(db[np.isfinite(db)], 99.9)) for db in dbs)
    lv = p999 - LEVEL_OFF
    floor = min(float(np.percentile(db[np.isfinite(db)], 50)) for db in dbs) + 1.0
    while lv > floor:
        cw = [cov_width(db, lv) for db in dbs]
        cov = min(c for c, _ in cw)
        wid = max(w for _, w in cw)
        if cov >= COV_MIN and wid <= WIDTH_CAP:
            break
        lv -= 0.5
    return float(lv)


def band_centers(prof, level, min_w=MIN_W_PX, merge=MERGE_PX):
    """一行 dB 剖面 -> [(center_col, width, peak_db)]: 等阈亮带亚像素中点"""
    idx = np.where(prof >= level)[0]
    if len(idx) == 0:
        return []
    segs, s = [], idx[0]
    for a, b_ in zip(idx[:-1], idx[1:]):
        if b_ - a > merge:
            segs.append((s, a))
            s = b_
    segs.append((s, idx[-1]))
    out, n = [], len(prof)
    for a, b_ in segs:
        if b_ - a + 1 < min_w:
            continue
        left = ((a - 1) + (level - prof[a - 1]) / max(prof[a] - prof[a - 1], 1e-9)
                if a - 1 >= 0 and prof[a - 1] < level else float(a))
        right = (b_ + (prof[b_] - level) / max(prof[b_] - prof[b_ + 1], 1e-9)
                 if b_ + 1 < n and prof[b_ + 1] < level else float(b_))
        out.append((0.5 * (left + right), right - left, float(prof[a:b_ + 1].max())))
    return out


# ==================== 3. 提取 + 跟迹 ====================
def extract_one(db, cal, b_min, b_max, t_max, dt, level, log):
    """逐历元提取等阈亮带 -> {t: [(bearing_deg, width_px, peak_db), ...]}"""
    r0, r1, c0, c1 = cal["rect"]
    ny = db.shape[0]
    span_px = c1 - c0 + 1
    raw = {}
    for t in np.arange(dt / 2, t_max, dt):
        # 行号: t=0 -> 绘图区底 (db 行 ny-1), t=t_max -> 顶 (db 行 0)
        rr = (1.0 - t / t_max) * (ny - 1)
        half = dt / t_max * (ny - 1)
        lo, hi = int(math.floor(rr + half)), int(math.ceil(rr - half))
        lo, hi = max(0, min(lo, ny - 1)), max(0, min(hi, ny - 1))
        if hi < lo:
            lo, hi = hi, lo
        prof = db[lo:hi + 1].mean(axis=0)
        raw[float(t)] = [(b_min + c / span_px * (b_max - b_min), w, a)
                         for c, w, a in band_centers(prof, level)]
    cov = sum(1 for v in raw.values() if v)
    log(f"  亮带: {cov}/{len(raw)} 历元有带, 共 {sum(len(v) for v in raw.values())} 条, "
        f"电平 {level:.1f} dB")
    return raw


def nearest_key(d, t, tol):
    k = min(d, key=lambda x: abs(x - t))
    return k if abs(k - t) <= tol else None


def track_dir(raw, keys, t0, b0):
    out, p, v, last = {}, b0, 0.0, t0
    for t in keys:
        cands = raw.get(t, [])
        dtt = t - last
        if not cands:
            if abs(dtt) > MISS_MAX_S:
                break
            continue
        best = min(cands, key=lambda c: abs(c[0] - p))
        if abs(best[0] - p) > GATE_DEG and abs(dtt) > MISS_MAX_S:
            break
        if abs(best[0] - p) > GATE_DEG:
            continue
        out[t] = best
        v = 0.7 * v + 0.3 * (best[0] - p) / max(abs(dtt), 1e-9)
        p = best[0] + v * dtt
        last = t
    return out


def track_from(raw, t_anchor, b_anchor):
    ks = sorted(raw)
    merged = track_dir(raw, [k for k in ks if k < t_anchor][::-1], t_anchor, b_anchor)
    merged.update(track_dir(raw, [k for k in ks if k > t_anchor], t_anchor, b_anchor))
    merged[t_anchor] = min(raw[t_anchor], key=lambda c: abs(c[0] - b_anchor))
    return merged


def _pair_at(raw, ka, kb, db_thr):
    hb = [b for b in raw["hull"][ka] if b[2] >= db_thr]
    tt = [b for b in raw["tow"][kb] if b[2] >= db_thr]
    used, pairs = set(), []
    for hh in sorted(hb, key=lambda x: -x[2]):
        cand = [(i, t) for i, t in enumerate(tt)
                if i not in used and abs(t[0] - hh[0]) <= GATE_DEG]
        if cand:
            i, m = min(cand, key=lambda p: abs(p[1][0] - hh[0]))
            used.add(i)
            pairs.append((hh, m))
    return pairs


def anchors(raw, db_thr, fracs=ANCHOR_FRACS, log=None):
    """两阵强亮带互配 -> 锚点 (不依赖先验方位)

    依次试多个锚时刻, 并在必要时放宽峰值门限 (暗目标批次的亮带峰值可能
    低于 db_max-2), 保证至少能起迹; 全都失败才返回空。
    """
    t_span = max(raw["hull"])
    for frac in fracs:
        ta = t_span * frac
        ka = min(raw["hull"], key=lambda x: abs(x - ta))
        kb = min(raw["tow"], key=lambda x: abs(x - ta))
        for thr in (db_thr, db_thr - 1.5, db_thr - 3.0, 0.0):
            pairs = _pair_at(raw, ka, kb, thr)
            if pairs:
                if log and thr != db_thr:
                    log(f"  [锚点] t={ka:.0f} s 处放宽峰值门限至 {thr:.1f} dB "
                        f"才配到 {len(pairs)} 个目标")
                return [{"bearing_hull": h[0], "bearing_tow": t[0], "t_anchor": ka,
                         "peak_hull": h[2], "peak_tow": t[2], "anchor_frac": frac}
                        for h, t in pairs]
    return []


def anchors_multi(levels, db_thr, log=None):
    """跨电平阶梯累积锚点。

    主电平由最亮目标定, 常常捞不到暗目标 —— 实测旧批次锚时刻 hull 只有 1 条
    亮带而 tow 有 3~4 条。故按 LEVEL_LADDER 逐级放宽, 每级都取锚, 并去掉与
    已取锚点 (同历元、hull 方位相近) 重复者。
    levels: [(level, {"hull":..., "tow":...}), ...] 按电平从高到低。
    """
    out, seen = [], {}
    top = levels[0][0]
    for lv, raw in levels:
        got = anchors(raw, db_thr - (top - lv))
        added = 0
        for a in got:
            ka = a["t_anchor"]
            if any(abs(a["bearing_hull"] - b) < GATE_DEG for b in seen.get(ka, ())):
                continue                      # 与该历元已取锚点重复
            a["level"] = lv
            out.append(a)
            seen.setdefault(ka, []).append(a["bearing_hull"])
            added += 1
        if log:
            log(f"  [锚点] 电平 {lv:5.1f} dB: 新增 {added} 个 (累计 {len(out)})")
    return out


# ==================== 4. 交会 / 距离 / 速度 ====================
def medfilt(x, k):
    x = np.asarray(x, float)
    h = k // 2
    return np.array([np.median(x[max(i - h, 0):min(i + h + 1, len(x))])
                     for i in range(len(x))])


def longest_run(mask):
    best, s = None, None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if best is None or i - s > best[1] - best[0]:
                best = (s, i - 1)
            s = None
    if s is not None and (best is None or len(mask) - s > best[1] - best[0]):
        best = (s, len(mask) - 1)
    return best


def estimate_branch(geom, h_tr, t_tr, sg, par_min=PAR_MIN_DEG, rwin=RWIN):
    ts, bh, bt = [], [], []
    for t in sorted(set(h_tr) & set(t_tr)):
        p = B.instant_fix(geom, t, h_tr[t][0], t_tr[t][0], sg, sg)
        if not np.isfinite(p[0]):
            continue
        ts.append(t)
        bh.append(h_tr[t][0])
        bt.append(t_tr[t][0])
    if len(ts) < 15:
        return None
    ts, bh, bt = np.array(ts), np.array(bh), np.array(bt)
    par = np.abs(bh - bt)
    acc = par >= par_min
    if acc.sum() < 15:
        return None
    ts, bh, bt, par = ts[acc], bh[acc], bt[acc], par[acc]
    hx, hy = geom.hull_xy(ts)
    X, Y = np.array([B.instant_fix(geom, t, b_, c, sg, sg)
                     for t, b_, c in zip(ts, bh, bt)]).T
    r_raw = np.hypot(X - hx, Y - hy)
    r_sm = medfilt(r_raw, rwin)
    val = np.zeros(len(ts), bool)
    half = rwin // 2
    if len(ts) > 2 * half:
        val[half:len(ts) - half] = True
    r_rep = r_sm.copy()
    kl = RLOCAL // 2
    for j in np.where(~val)[0]:
        r_rep[j] = float(np.median(r_raw[max(j - kl, 0):j + kl + 1]))
    az = np.radians(geom.heading + sg * bh)
    Px, Py = hx + r_rep * np.sin(az), hy + r_rep * np.cos(az)
    tail = float(np.median(bh[-max(20, len(ts) // 5):]))
    mv = np.abs(bh - tail) > DEV_DEG
    return {"t": ts.tolist(), "bearing_hull": bh.tolist(), "bearing_tow": bt.tolist(),
            "parallax": par.tolist(), "range_raw_m": r_raw.tolist(),
            "range_rep_m": r_rep.tolist(), "x_m": Px.tolist(), "y_m": Py.tolist(),
            "radial_ok": val.tolist(), "motion_run": longest_run(mv),
            "side_sign": sg, "n_epoch": int(len(ts)),
            "par_median": float(np.median(par))}


def motion_of(est, geom):
    """通道分解速度 -> dict 或 None"""
    run = est["motion_run"]
    if not run or run[1] - run[0] < 8:
        return None
    i0, i1 = run
    idx = np.arange(i0, i1 + 1)
    t = np.array(est["t"])[idx]
    seg_t = t - t[0]
    if seg_t[-1] < 1e-6:
        return None
    A = np.stack([np.ones_like(seg_t), seg_t], 1)
    cb, *_ = np.linalg.lstsq(A, np.radians(np.array(est["bearing_hull"])[idx]), rcond=None)
    r_bar = float(np.mean(np.array(est["range_rep_m"])[idx]))
    v_t = float(cb[1]) * r_bar
    # 径向通道: 用"单历元交会距离" (样本独立) 做回归, 而不是中值平滑后的序列 ——
    # 平滑输出是阶梯状自相关序列, 残差标准误会严重偏小, 把滤噪声当成真信号。
    # 180 m 基线在 10 km 量级上单历元 σ≈1 km, dr/dt 通常不可观 -> 显著性不足就置 0
    # (假径向速度会把航向带偏几十度)。
    rv = np.array(est["range_raw_m"])[idx]
    n = len(rv)
    v_r, sig_r, radial_ok_flag = 0.0, 0.0, False
    if n >= 12:
        cr, *_ = np.linalg.lstsq(A, rv, rcond=None)
        v_r = float(cr[1])
        resid = rv - A @ cr
        sst = float(((seg_t - seg_t.mean()) ** 2).sum())
        se = math.sqrt(float((resid ** 2).sum()) / max(n - 2, 1) / max(sst, 1e-9))
        sig_r = abs(v_r) / se if se > 1e-9 else float("inf")
        if sig_r >= RADIAL_SIGMA:
            radial_ok_flag = True
        else:
            v_r = 0.0
    sg = est["side_sign"]
    a = math.radians(geom.heading + sg * float(np.mean(np.array(est["bearing_hull"])[idx])))
    vx = v_r * math.sin(a) + v_t * sg * math.cos(a)
    vy = v_r * math.cos(a) - v_t * sg * math.sin(a)
    return {"speed_mps": math.hypot(vx, vy),
            "course_deg": math.degrees(math.atan2(vx, vy)) % 360.0,
            "t_start": float(t[0]), "t_end": float(t[-1]),
            "v_tangent_mps": v_t, "v_radial_mps": v_r,
            "radial_observable": radial_ok_flag,
            "radial_significance": sig_r}


def cpa_of(est, geom):
    r = np.array(est["range_rep_m"])
    t = np.array(est["t"])
    k = int(np.argmin(r))
    return float(r[k]), float(t[k])


def summarize(geom, est):
    t = np.array(est["t"])
    lat0, lon0 = geom.xy2ll(est["x_m"][0], est["y_m"][0])
    lat1, lon1 = geom.xy2ll(est["x_m"][-1], est["y_m"][-1])
    cpad, cpat = cpa_of(est, geom)
    mo = motion_of(est, geom)
    hx, hy = geom.hull_xy(t)
    tx, ty = geom.tow_xy(t)
    return {
        "t_start": float(t[0]), "t_end": float(t[-1]), "n_epoch": est["n_epoch"],
        "start": {"lat": lat0, "lon": lon0,
                  "range_hull_km": est["range_rep_m"][0] / 1000,
                  "range_tow_km": float(np.hypot(est["x_m"][0] - tx[0], est["y_m"][0] - ty[0])) / 1000,
                  "bearing_hull": est["bearing_hull"][0], "bearing_tow": est["bearing_tow"][0],
                  "parallax": est["parallax"][0]},
        "end": {"lat": lat1, "lon": lon1,
                "range_hull_km": est["range_rep_m"][-1] / 1000,
                "range_tow_km": float(np.hypot(est["x_m"][-1] - tx[-1], est["y_m"][-1] - ty[-1])) / 1000,
                "bearing_hull": est["bearing_hull"][-1], "bearing_tow": est["bearing_tow"][-1],
                "parallax": est["parallax"][-1]},
        "bearing_sweep_hull": est["bearing_hull"][-1] - est["bearing_hull"][0],
        "bearing_sweep_tow": est["bearing_tow"][-1] - est["bearing_tow"][0],
        "range_change_km": (est["range_rep_m"][-1] - est["range_rep_m"][0]) / 1000,
        "range_median_km": float(np.median(est["range_rep_m"])) / 1000,
        "range_epoch_sigma_km": float(np.std(est["range_raw_m"], ddof=1)) / 1000,
        "cpa_km": cpad / 1000, "cpa_t": cpat,
        "parallax_median": est["par_median"],
        "displacement_km": float(np.hypot(est["x_m"][-1] - est["x_m"][0],
                                         est["y_m"][-1] - est["y_m"][0])) / 1000,
        "motion": mo}


# ==================== 5. 真值解析与差异对比 ====================
# 目标名后可跟描述标签 (如 "T1 远/相向"); 严禁跨行 —— 字符类里必须排除 \n,
# 否则 "====" 分隔行会吞掉下一行的 T1, 导致真值名变成一串等号。
REC_PAT = (r"^([^\S\n]*)(\S+?)(?:[^\S\n]+[^\|\n]*?)?[^\S\n]*\|"
           r"[^\|\n]*?([\d.]+)\s*kn[^\n]*?start=\(([\d.+-]+)°N,([\d.+-]+)°E\)\s*"
           r"end=\(([\d.+-]+)°N,([\d.+-]+)°E\)[^\n]*?舷角\s*([\d.+-]+)°->\s*([\d.+-]+)°[^\n]*?"
           r"CPA=\s*([\d.+-]+)\s*km\s*@\s*t=\s*(-?[\d.]+)s[^\n]*?观测窗\[([\d.]+),([\d.]+)\]s")


def parse_truth_targets(txt_paths):
    out = {}
    for tag, p in txt_paths.items():
        if not p or not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        for m in re.finditer(REC_PAT, txt, re.M):
            g = m.groups()[1:]          # 丢掉 group(1) 的前导缩进
            out.setdefault(g[0], {})[tag] = dict(
                speed_kn=float(g[1]), start=(float(g[2]), float(g[3])),
                end=(float(g[4]), float(g[5])), bearing=(float(g[6]), float(g[7])),
                cpa_km=float(g[8]), cpa_t=float(g[9]), window=(float(g[10]), float(g[11])))
    return out


_NUM = r"[\d.+-]+"
# 数值常被括号包住: "(30.0623°N, 122.0415°E)" / "相对偏移 (-150.0, 0.0) m";
# 键名有两种方言: "初始经纬度" (btr_out 各批) 与 "起点经纬度" (gen_datas/geo)。
_LL = rf"(?:[（(]\s*)?({_NUM})\s*°N\s*,\s*({_NUM})\s*°E(?:\s*[）)])?"
_OFF = rf"(?:[（(]\s*)?({_NUM})\s*,\s*({_NUM})\s*(?:[）)])?"


def parse_truth_platform(track_txt, log=None):
    """从 btr_*_track.txt 读平台/阵列/观测设置 (缺参数时用)

    任一字段解析失败都必须报警告而不是静默用默认值 —— 平台起点错 6 km
    就会污染全部真值舷角/位置/距离对比, 且残差看起来"只是有点大"而已。
    """
    if not track_txt or not os.path.exists(track_txt):
        return {}
    s = open(track_txt, encoding="utf-8").read()

    def f(pat, dflt=None):
        m = re.search(pat, s)
        return cast_float(m.group(1)) if m else dflt

    def cast_float(x):
        return float(x)
    d, missing = {}, []
    m = re.search(rf"(?:初始|起点)经纬度[:：]\s*{_LL}", s)
    if m:
        d["start_ll"] = (float(m.group(1)), float(m.group(2)))
    else:
        missing.append("平台起点经纬度")
    h = f(rf"船头朝向\(航向\)[:：]\s*({_NUM})°")
    if h is not None:
        d["heading"] = h
    else:
        missing.append("航向")
    v = f(rf"航速[:：]\s*({_NUM})\s*kn")
    if v is not None:
        d["speed_kn"] = v
    else:
        missing.append("航速")
    m = re.search(rf"(?:舷侧阵|弦测阵|舰壳)[^:：]*[:：]\s*相对偏移\s*{_OFF}\s*m", s)
    if m:
        d["hull_m"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(rf"拖曳阵[^:：]*[:：]\s*相对偏移\s*{_OFF}\s*m", s)
    if m:
        d["tow_m"] = (float(m.group(1)), float(m.group(2)))
    else:
        missing.append("拖曳阵偏移")
    tm = f(r"观测时长[:：]\s*([\d.]+)\s*s")
    if tm:
        d["t_max"] = tm
    m = re.search(r"方位范围[:：]\s*({_NUM})~({_NUM})\s*deg", s)
    if m:
        d["b_range"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(rf"投影基准[:：]\s*lat=({_NUM})\s*,\s*lon=({_NUM})", s)
    if m:
        d["ref"] = (float(m.group(1)), float(m.group(2)))
    # 平台若是航点折线 (航向随时间变), 本代理的"匀速直线 + 固定航向"模型不适用
    if re.search(r"航点[:：]", s) and "heading" not in d:
        d["platform_waypoints"] = True
    if missing and log:
        log(f"[警告] track.txt 未解析出: {', '.join(missing)} —— "
            f"这些项将退回默认值, 真值对比可能整体失真")
    return d


def truth_traj_fn(rec, ll2xy):
    """真值航点直线: t -> (x, y) 米制; 走完航点后停在终点"""
    x0, y0 = ll2xy(*rec["start"])
    x1, y1 = ll2xy(*rec["end"])
    t_end = rec["cpa_t"] if rec["cpa_t"] > 0 else 1e9
    # 走完航点时刻优先取 hull txt 里的 "走完航点 t=" (与 CPA 时刻不同源时)
    def fn(t):
        f = min(1.0, t / t_end) if t_end > 0 else 1.0
        return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
    return fn


def truth_rel_bearings(geom, ll2xy, rec, t_end_walk, t, tow=False):
    """真值轨迹在时刻 t 的折叠舷角序列 (船首=0°, 正横=90°, 正尾=180°)"""
    fn = truth_traj_fn({**rec, "cpa_t": t_end_walk}, ll2xy)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ax, ay = (geom.tow_xy(t) if tow else geom.hull_xy(t))
    ax, ay = np.atleast_1d(ax), np.atleast_1d(ay)
    out = []
    for i, ti in enumerate(t):
        px, py = fn(ti)
        gb = math.degrees(math.atan2(px - ax[i], py - ay[i]))
        rel = (gb - geom.heading) % 360.0
        out.append(rel if rel <= 180 else 360 - rel)
    return np.array(out)


def diff_vs_truth(geom, ll2xy, est, summ, rec_h, rec_t, t_end_walk):
    """检出结果 vs 真值 -> 差异字典"""
    t = np.array(est["t"])
    fn = truth_traj_fn({**rec_h, "cpa_t": t_end_walk}, ll2xy)
    Tx = np.empty(len(t))
    Ty = np.empty(len(t))
    for i, tt in enumerate(t):
        Tx[i], Ty[i] = fn(tt)
    hx, hy = geom.hull_xy(t)
    r_tr = np.hypot(Tx - hx, Ty - hy)
    bh = np.array(est["bearing_hull"])
    bt = np.array(est["bearing_tow"])
    # 真值折叠舷角序列 (由真值轨迹直接算, 不用 txt 的 t=0/600 端点值)
    rel = truth_rel_bearings(geom, ll2xy, rec_h, t_end_walk, t)
    rel_t = truth_rel_bearings(geom, ll2xy, rec_h, t_end_walk, t, tow=True)
    # 真值航向: 全程位移方向
    crs_t = math.degrees(math.atan2(Tx[-1] - Tx[0], Ty[-1] - Ty[0])) % 360.0
    if abs(Tx[-1] - Tx[0]) < 1 and abs(Ty[-1] - Ty[0]) < 1:
        crs_t = float("nan")
    mo = summ.get("motion")
    # 同窗口真值偏转: 检出窗口 [t[0], t[-1]] 上的真值舷角变化 (与检出偏转可比)
    sweep_tr_hull = float(rel[-1] - rel[0])
    sweep_tr_tow = float(rel_t[-1] - rel_t[0])
    # 运动段同窗口真值速度/航向
    if mo:
        i0 = int(np.argmin(np.abs(t - mo["t_start"])))
        i1 = int(np.argmin(np.abs(t - mo["t_end"])))
        dx, dy = Tx[i1] - Tx[i0], Ty[i1] - Ty[i0]
        dtm = max(t[i1] - t[i0], 1e-9)
        spd_tr_win = math.hypot(dx, dy) / dtm
        crs_tr_win = math.degrees(math.atan2(dx, dy)) % 360.0
    else:
        spd_tr_win = crs_tr_win = float("nan")
    pos_err = np.hypot(np.array(est["x_m"]) - Tx, np.array(est["y_m"]) - Ty)
    d = {
        "truth": {"speed_kn": rec_h["speed_kn"], "course_deg": crs_t,
                  "start": rec_h["start"], "end": rec_h["end"],
                  "cpa_km": rec_h["cpa_km"], "cpa_t": rec_h["cpa_t"],
                  "window": rec_h["window"],
                  "bearing_hull": rec_h["bearing"], "bearing_tow": rec_t["bearing"],
                  "t_end_walk": t_end_walk,
                  "bearing_at_det_start": float(rel[0]),
                  "bearing_at_det_end": float(rel[-1]),
                  "range_at_det_start_km": float(r_tr[0] / 1000),
                  "range_at_det_end_km": float(r_tr[-1] / 1000)},
        "bearing_rms_deg": float(np.sqrt(np.mean((bh - rel) ** 2))),
        "bearing_max_deg": float(np.max(np.abs(bh - rel))),
        "bearing_tow_rms_deg": float(np.sqrt(np.mean((bt - rel_t) ** 2))),
        "sweep_det_hull_deg": float(summ["bearing_sweep_hull"]),
        "sweep_truth_hull_deg": sweep_tr_hull,
        "sweep_err_hull_deg": float(summ["bearing_sweep_hull"] - sweep_tr_hull),
        "sweep_err_tow_deg": float(summ["bearing_sweep_tow"] - sweep_tr_tow),
        "bearing_err_start_deg": float(bh[0] - rel[0]),
        "bearing_err_end_deg": float(bh[-1] - rel[-1]),
        "range_err_mean_km": float(np.mean(np.array(est["range_rep_m"]) - r_tr)) / 1000,
        "range_err_pct": float(np.mean(np.array(est["range_rep_m"]) - r_tr) / np.mean(r_tr)) * 100,
        "cpa_err_km": summ["cpa_km"] - rec_h["cpa_km"],
        "cpa_err_pct": (summ["cpa_km"] - rec_h["cpa_km"]) / rec_h["cpa_km"] * 100,
        "cpa_t_err_s": summ["cpa_t"] - rec_h["cpa_t"],
        "pos_err_start_m": float(pos_err[0]),
        "pos_err_end_m": float(pos_err[-1]),
        "pos_err_mean_m": float(np.mean(pos_err)),
        "pos_err_median_m": float(np.median(pos_err)),
        "parallax_expected_deg": float(np.median(np.abs(rel - rel_t))),
        "truth_speed_window_mps": spd_tr_win,
        "truth_course_window_deg": crs_tr_win,
    }
    if mo:
        d["det_speed_kn"] = mo["speed_mps"] / KN2MS
        d["det_course_deg"] = mo["course_deg"]
        d["truth_window_speed_kn"] = spd_tr_win / KN2MS if spd_tr_win == spd_tr_win else float("nan")
        d["truth_window_course_deg"] = crs_tr_win
        d["speed_err_kn"] = mo["speed_mps"] / KN2MS - spd_tr_win / KN2MS
        d["speed_err_pct"] = ((mo["speed_mps"] - spd_tr_win) / max(spd_tr_win, 1e-9) * 100
                              if spd_tr_win == spd_tr_win else float("nan"))
        d["course_err_deg"] = ((mo["course_deg"] - crs_tr_win + 180) % 360 - 180
                               if crs_tr_win == crs_tr_win else float("nan"))
        d["speed_err_vs_nominal_kn"] = mo["speed_mps"] / KN2MS - rec_h["speed_kn"]
    return d


# ==================== 5b. 质量定级与漏检/交叉诊断 ====================
def quality_grade(diff):
    """按方位残差与位置误差定级, 避免把失真的数字当结论。

    A 可直接用; B 趋势可信, 绝对量宜复核; C 仅相对量 (舷角/偏转) 可信;
    D 不可用 (轨迹合并 / 平台模型不符 / 牵引到错误亮带)。
    """
    b = diff["bearing_rms_deg"]
    p = diff["pos_err_median_m"]
    if b <= 0.5 and p <= 1000:
        return "A"
    if b <= 2.0 and p <= 3000:
        return "B"
    if b <= 5.0:
        return "C"
    return "D"


def detect_notes(targets, truth_map, tt, unmatched, min_cov):
    """给出每个检出目标的失准原因 (若有), 以及真值漏检清单"""
    notes = {}
    for tg in targets:
        tm = truth_map.get(tg["name"])
        if not tm:
            continue
        d = tm["diffs"][tm["chosen"]]
        why = []
        if d["bearing_rms_deg"] > 5.0:
            why.append("方位残差过大: 轨迹很可能被邻近亮带牵引")
        if d["pos_err_median_m"] > 3000 and d["bearing_rms_deg"] <= 5.0:
            why.append("方位尚可但位置误差大: 距离 (双站视差) 条件数差")
        est = (tg["branches"].get(tm["chosen"])
               or next(iter(tg["branches"].values())))
        span = est["t"][-1] - est["t"][0]
        expect = tm["records"]["hull"]["window"][1] - tm["records"]["hull"]["window"][0]
        if expect > 0 and span < 0.6 * expect:
            why.append(f"检出时长 {span:.0f}s 远短于真值观测窗 {expect:.0f}s: 轨迹断裂")
        notes[tg["name"]] = "; ".join(why)
    if unmatched:
        notes["_missed"] = ("真值目标未被检出: " + ", ".join(
            f"{n} (hull 舷角 {tt[n]['hull']['bearing'][0]:.1f}°->"
            f"{tt[n]['hull']['bearing'][1]:.1f}°)" for n in unmatched))
    if min_cov is not None and min_cov < 0.8:
        notes["_dim"] = f"亮带历元覆盖率仅 {min_cov*100:.0f}%: 信号偏弱, 结果宜复核"
    return notes


def crossing_risk(targets, geom, ll2xy, tt, walk_of):
    """两真值目标在 hull 方位上的最小间距 (px) —— 小于亮带宽度就有合并风险"""
    if len(tt) < 2:
        return None
    names = list(tt)
    worst = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ts = np.arange(0.0, geom_t_span(targets) + 1e-9, 10.0)
            bi = truth_rel_bearings(geom, ll2xy, tt[names[i]]["hull"], walk_of(names[i]), ts)
            bj = truth_rel_bearings(geom, ll2xy, tt[names[j]]["hull"], walk_of(names[j]), ts)
            gap_px = float(np.min(np.abs(bi - bj)) / 180.0 * 1165.0)
            if worst is None or gap_px < worst[0]:
                worst = (gap_px, names[i], names[j])
    return worst


def geom_t_span(targets):
    return max((max(tg["tracks"]["hull"]) for tg in targets if tg["tracks"]["hull"]),
               default=600.0)


def summary_table(targets, truth_map, grades, notes):
    L = ["", "=" * 96,
         "汇总 (等级: A 可直接用 / B 趋势可信, 绝对量宜复核 / "
         "C 仅相对量可信 / D 不可用)", "=" * 96]
    L.append(f"  {'检出':<8}{'真值':<7}{'舷角RMS':>9}{'位置中位误差':>12}{'CPA误差':>10}"
             f"{'航速Δ':>9}{'航向Δ':>9}  {'级':<3}备注")
    for tg in targets:
        tm = truth_map.get(tg["name"])
        if not tm:
            L.append(f"  {tg['name']:<8}{'—':<7}{'':>9}{'':>12}{'':>10}{'':>9}{'':>9}"
                     f"  {'-':<3}未匹配到真值目标")
            continue
        d = tm["diffs"][tm["chosen"]]
        sp = f"{d['speed_err_kn']:+.1f}kn" if "speed_err_kn" in d else "—"
        cr = f"{d['course_err_deg']:+.1f}°" if "course_err_deg" in d else "—"
        L.append(f"  {tg['name']:<8}{tm['truth_name']:<7}{d['bearing_rms_deg']:>8.3f}°"
                 f"{d['pos_err_median_m']:>11.0f} m{d['cpa_err_km']:>+9.3f} km"
                 f"{sp:>9}{cr:>9}  {grades.get(tg['name'], '?'):<3}"
                 f"{notes.get(tg['name'], '')}")
    for key in ("_missed", "_dim"):
        if key in notes:
            L.append(f"  ! {notes[key]}")
    return "\n".join(L)


# ==================== 6. 输出 ====================
def fmt_summary(idx, name, side, summ, d):
    s, e = summ["start"], summ["end"]
    L = [f"----- {name} [{'右舷 +' if side > 0 else '左舷 -'}] "
         f"检出历元 {summ['n_epoch']} 个, 覆盖 {summ['t_start']:.0f}~{summ['t_end']:.0f} s, "
         f"双站视差中位 {summ['parallax_median']:.2f}°"]
    L.append(f"  初始 t={summ['t_start']:6.1f} s : "
             f"({s['lat']:.4f}°N, {s['lon']:.4f}°E)")
    L.append(f"       距舷侧阵 {s['range_hull_km']:6.3f} km | 距拖曳阵 {s['range_tow_km']:6.3f} km"
             f" | 舷角 hull {s['bearing_hull']:6.2f}° / tow {s['bearing_tow']:6.2f}°"
             f" | 视差 {s['parallax']:.2f}°")
    L.append(f"  结束 t={summ['t_end']:6.1f} s : ({e['lat']:.4f}°N, {e['lon']:.4f}°E)")
    L.append(f"       距舷侧阵 {e['range_hull_km']:6.3f} km | 距拖曳阵 {e['range_tow_km']:6.3f} km"
             f" | 舷角 hull {e['bearing_hull']:6.2f}° / tow {e['bearing_tow']:6.2f}°"
             f" | 视差 {e['parallax']:.2f}°")
    L.append(f"  舷角偏转 : hull {s['bearing_hull']:.2f}° -> {e['bearing_hull']:.2f}°"
             f" (Δ={summ['bearing_sweep_hull']:+.2f}°)"
             f"  |  tow {s['bearing_tow']:.2f}° -> {e['bearing_tow']:.2f}°"
             f" (Δ={summ['bearing_sweep_tow']:+.2f}°)")
    L.append(f"  距离     : {s['range_hull_km']:.3f} -> {e['range_hull_km']:.3f} km"
             f" (Δ={summ['range_change_km']:+.3f} km) | 中位 {summ['range_median_km']:.3f} km"
             f" | CPA {summ['cpa_km']:.3f} km @ t={summ['cpa_t']:.0f} s"
             f" | 单历元散布 σ {summ['range_epoch_sigma_km']:.3f} km")
    mo = summ["motion"]
    if mo:
        L.append(f"  运动段   : t={mo['t_start']:.0f}~{mo['t_end']:.0f} s"
                 f" | 速度 {mo['speed_mps']:.2f} m/s ({mo['speed_mps'] / KN2MS:.1f} kn)"
                 f" | 航向 {mo['course_deg']:.1f}°"
                 f" | 切向 {mo['v_tangent_mps']:+.2f} m/s"
                 f" | 径向 " + (
                     f"{mo['v_radial_mps']:+.2f} m/s ({mo['radial_significance']:.1f}σ)"
                     if mo["radial_observable"]
                     else (f"不可观, 置 0 ({mo['radial_significance']:.1f}σ < {RADIAL_SIGMA}σ)"
                           if mo["radial_significance"]
                           else "不可观 (历元不足)")))
    else:
        L.append("  运动段   : 方位近似恒定 (静止或已走完航点), 无可用运动段")
    L.append(f"  首末位移 : {summ['displacement_km']:.3f} km")
    if d:
        tr = d["truth"]
        L.append("  ── 与真值差异 ──")
        L.append(f"    方位残差 : RMS {d['bearing_rms_deg']:.3f}°, 最大 {d['bearing_max_deg']:.3f}°"
                 f" (hull) | tow RMS {d['bearing_tow_rms_deg']:.3f}°")
        L.append(f"    舷角对照 : 检出 {summ['start']['bearing_hull']:.2f}° -> "
                 f"{summ['end']['bearing_hull']:.2f}° (Δ={d['sweep_det_hull_deg']:+.2f}°)"
                 f"  |  真值 {tr['bearing_at_det_start']:.2f}° -> "
                 f"{tr['bearing_at_det_end']:.2f}° (Δ={d['sweep_truth_hull_deg']:+.2f}°)"
                 f"  [同为检出窗口]")
        L.append(f"    偏转误差 : hull {d['sweep_err_hull_deg']:+.3f}°,"
                 f" tow {d['sweep_err_tow_deg']:+.3f}°"
                 f"  (端点残差 初始 {d['bearing_err_start_deg']:+.3f}°,"
                 f" 结束 {d['bearing_err_end_deg']:+.3f}°)")
        L.append(f"    距离误差 : 平均 {d['range_err_mean_km']:+.3f} km ({d['range_err_pct']:+.1f}%)"
                 f" | CPA {d['cpa_err_km']:+.3f} km ({d['cpa_err_pct']:+.1f}%)"
                 f" @Δt {d['cpa_t_err_s']:+.0f} s"
                 f"  (真值距离 {tr['range_at_det_start_km']:.3f} -> "
                 f"{tr['range_at_det_end_km']:.3f} km)")
        L.append(f"    位置误差 : 初始 {d['pos_err_start_m']:.0f} m,"
                 f" 结束 {d['pos_err_end_m']:.0f} m,"
                 f" 全程中位 {d['pos_err_median_m']:.0f} m")
        if "speed_err_kn" in d and d["speed_err_pct"] == d["speed_err_pct"]:
            L.append(f"    运动误差 : 航速 检出 {d['det_speed_kn']:.1f} kn"
                     f" / 真值(同窗口) {d['truth_window_speed_kn']:.1f} kn"
                     f" (名义 {tr['speed_kn']:.0f} kn), Δ={d['speed_err_kn']:+.1f} kn"
                     f" ({d['speed_err_pct']:+.1f}%)")
            L.append(f"               航向 检出 {d['det_course_deg']:.1f}°"
                     f" / 真值(同窗口) {d['truth_window_course_deg']:.1f}°"
                     f" (全程 {tr['course_deg']:.0f}°), Δ={d['course_err_deg']:+.1f}°")
        L.append(f"    真值     : {tr['speed_kn']:.0f} kn, 航向 {tr['course_deg']:.0f}°,"
                 f" ({tr['start'][0]:.4f}°N,{tr['start'][1]:.4f}°E) ->"
                 f" ({tr['end'][0]:.4f}°N,{tr['end'][1]:.4f}°E),"
                 f" CPA {tr['cpa_km']:.2f} km @ {tr['cpa_t']:.0f} s,"
                 f" 观测窗 [{tr['window'][0]:.0f},{tr['window'][1]:.0f}] s")
    return "\n".join(L)


def make_png(*_a, **_k):        # 兼容别名: 实际绘图见 plot_png
    raise DeprecationWarning("use plot_png()")


# ==================== 主流程 ====================
def run(hull_png, tow_png, params=None, dt=2.0, level=None, side="auto",
        t_max=None, b_range=None, db_range=(-5.0, 16.0), truth=True,
        png_out=None, log=print, json_out=None):
    t0_wall = datetime.now()
    lut_list = [B.build_lut(s) for s in B.CMAP_STOPS.values()]
    imgs = {}
    for tag, p in (("hull", hull_png), ("tow", tow_png)):
        imgs[tag] = np.asarray(Image.open(p).convert("RGB"))

    # ---- 平台参数: 显式 > 同目录 track.txt 自动读取 > 默认 ----
    d = os.path.dirname(os.path.abspath(hull_png))
    track_txt = params.get("track_txt") if params else None
    if track_txt is None:
        cand = os.path.join(d, os.path.basename(hull_png).replace("_hull.png", "_track.txt"))
        track_txt = cand if os.path.exists(cand) else None
    auto = parse_truth_platform(track_txt, log=log)
    prm = dict(auto)
    for k, v in (params or {}).items():
        if v is not None:
            prm[k] = v
    DEFAULTS = {"start_ll": (30.0, 122.0), "heading": 0.0, "speed_kn": 0.0,
                "hull_m": (0.0, 0.0), "tow_m": (-180.0, 0.0), "ref": (30.0, 122.0)}
    defaulted = [k for k, v in DEFAULTS.items() if k not in prm]
    for k in defaulted:
        prm[k] = DEFAULTS[k]
    if defaulted:
        log(f"[警告] 未取得平台参数 {', '.join(defaulted)} —— 已退回默认值 "
            f"{ {k: DEFAULTS[k] for k in defaulted} }; 若与实际仿真设置不符, "
            f"所有绝对量 (位置/距离/CPA) 与真值对比都会整体失真")
    t_max = float(t_max or prm.get("t_max", 600.0))
    b_range = tuple(b_range or prm.get("b_range", (0.0, 180.0)))
    log(f"[参数] 平台起点 {prm['start_ll'][0]:.4f}°N/{prm['start_ll'][1]:.4f}°E, "
        f"航向 {prm['heading']:.1f}°, 航速 {prm['speed_kn']:.1f} kn, "
        f"舷侧阵 {prm['hull_m']} m, 拖曳阵 {prm['tow_m']} m, "
        f"t_max {t_max:.0f} s, 方位 {b_range[0]:.0f}~{b_range[1]:.0f}°"
        + ("  (由同目录 track.txt 自动读取)" if auto else "  (显式传入)"))
    if prm.pop("platform_waypoints", False):
        log("[警告] 该平台按航点折线运动 (航向随时间变), 本代理的平台模型是"
            "'匀速直线 + 固定航向', 只能给出相对量 (舷角/偏转/视差) 与"
            "近似绝对量; 绝对位置与距离会有系统性偏差。请用 --heading/"
            "--speed-kn 指定平台等效运动, 或仅采信方位类指标。")

    geom = B.Geometry(prm["start_ll"], prm["heading"], prm["speed_kn"],
                      prm["hull_m"], prm["tow_m"], prm["ref"][0], prm["ref"][1])
    ll2xy, _ = B.make_geo(prm["ref"][0], prm["ref"][1])

    # ---- 配色 + 轴标定 ----
    style = prm.get("cmap")
    if style in (None, "auto"):
        cal_tmp = find_axes(imgs["hull"], lut_list)[0]
        style = B.pick_cmap_style(imgs["hull"], cal_tmp, *db_range)
    lut = B.build_lut(B.CMAP_STOPS[style])
    log(f"[配色] 还原为 {style}")
    cals, raw, dbs = {}, {}, {}
    for tag, p in (("hull", hull_png), ("tow", tow_png)):
        log(f"[{tag}] {os.path.basename(p)}")
        cals[tag] = calibrate(imgs[tag], [lut], t_max, b_range[0], b_range[1], log)
        cals[tag]["t_max"] = t_max
        dbs[tag] = image_to_db(imgs[tag], cals[tag]["rect"], lut, *db_range)
    # 两阵共用一个电平 (电平不一致会把距离误差放大到百分之十几);
    # 再按 LEVEL_LADDER 向下放宽几档, 用于捞主电平够不着的暗目标。
    lev = pick_level([dbs["hull"], dbs["tow"]], level, db_max=db_range[1])
    ladder = [lev + off for off in LEVEL_LADDER]
    levels = []
    for lv in ladder:
        st = {}
        for tag in ("hull", "tow"):
            st[tag] = extract_one(dbs[tag], cals[tag], b_range[0], b_range[1],
                                  t_max, dt, lv, log if lv == ladder[0] else lambda *a: None)
        levels.append((lv, st))
    raw = levels[0][1]
    for tag in ("hull", "tow"):
        cals[tag]["level"] = lev
    log(f"[电平] 主 {lev:.1f} dB, 阶梯 {[round(l,1) for l in ladder]}")

    # ---- 锚点 + 跟迹 ----
    # 峰值门限跟着自适应电平走 (暗目标批次的亮带峰值本身就低于 db_max-2)
    db_thr = lev + 0.5
    ans = anchors_multi(levels, db_thr, log=log) if level is None else anchors(raw, db_thr, log=log)
    if not ans:
        log("[错误] 任何锚时刻都配不出 hull/tow 亮带对: 检查图像是否为同一次仿真的"
            "舷侧阵/拖曳阵阵位图, 或用 --level/--db-range 调整")
        return {"report": "", "targets": [], "truth": {}, "cals": cals, "geom": geom}
    log(f"[锚点] 共 {len(ans)} 个目标: "
        + ", ".join(f"t={a['t_anchor']:.0f}s hull {a['bearing_hull']:.2f}°/"
                    f"tow {a['bearing_tow']:.2f}° (视差 {abs(a['bearing_hull'] - a['bearing_tow']):.2f}°,"
                    f" {a.get('level', lev):.1f}dB)" for a in ans))
    targets = []
    by_level = dict(levels)
    for i, a in enumerate(ans):
        name = f"目标{i + 1}"
        raw_i = by_level.get(a.get("level", lev), raw)
        h_tr = track_from(raw_i["hull"], a["t_anchor"], a["bearing_hull"])
        t_tr = track_from(raw_i["tow"], a["t_anchor"], a["bearing_tow"])
        branches = {sg: estimate_branch(geom, h_tr, t_tr, sg) for sg in (+1, -1)}
        branches = {k: v for k, v in branches.items() if v}
        summaries = {k: summarize(geom, v) for k, v in branches.items()}
        targets.append({"name": name, "anchor": a, "branches": branches,
                        "summaries": summaries,
                        "tracks": {"hull": h_tr, "tow": t_tr}})

    # ---- 真值匹配 + 差异 ----
    truth_map, tt_all, walk_all, unmatched = {}, {}, {}, []
    min_cov = min(float(np.mean([1 if raw[tag].get(t) else 0 for t in raw[tag]]))
                  for tag in ("hull", "tow"))
    if truth:
        tp = {"hull": os.path.join(d, os.path.basename(hull_png).replace(".png", ".txt")),
              "tow": os.path.join(d, os.path.basename(tow_png).replace(".png", ".txt"))}
        tt = parse_truth_targets(tp)
        if tt:
            walk = {}
            s = open(tp["hull"], encoding="utf-8").read()
            for m in re.finditer(
                    r"^[^\S\n]*(\S+?)(?:[^\S\n]+[^\|\n]*?)?[^\S\n]*\|[^\n]*?走完航点 t=(-?[\d.]+)s",
                    s, re.M):
                walk[m.group(1)] = float(m.group(2))

            def _walk_t(tname):
                """走完航点时刻: 优先 txt 明示, 否则由几何 (start->end 距离 / 航速) 推算。
                早先直接用 cpa_t 兜底是错的 —— CPA 时刻与走完航点时刻无关,
                会把真值轨迹在时间轴上压扁/拉伸, 造成几十度的虚假方位残差。"""
                rec = tt[tname]["hull"]
                if tname in walk:
                    return walk[tname]
                x0, y0 = ll2xy(*rec["start"])
                x1, y1 = ll2xy(*rec["end"])
                dist = math.hypot(x1 - x0, y1 - y0)
                v = rec["speed_kn"] * KN2MS
                return dist / v if v > 1e-6 else 1e9

            # 代价 = 检出舷角序列 vs 真值舷角序列 的 RMS (比单点比较更有区分度)
            cand = []
            for tg in targets:
                est = tg["branches"].get(1) or next(iter(tg["branches"].values()), None)
                if not est:
                    continue
                t_arr = np.array(est["t"])
                bh = np.array(est["bearing_hull"])
                for tname in tt:
                    tw = _walk_t(tname)
                    rel = truth_rel_bearings(geom, ll2xy, tt[tname]["hull"], tw, t_arr)
                    cost = float(np.sqrt(np.mean((bh - rel) ** 2)))
                    cand.append((cost, tg["name"], tname))
            cand.sort()
            used_t, used_n = set(), set()
            pairs = {}
            for cost, tname_det, tname_true in cand:
                if tname_det in used_t or tname_true in used_n:
                    continue                     # 一对一: 已配对的目标/真值不再复用
                used_t.add(tname_det)
                used_n.add(tname_true)
                pairs[tname_det] = (tname_true, cost)
            tt_all = tt
            walk_all = _walk_t
            unmatched = [n for n in tt if n not in used_n]
            for tg in targets:
                if tg["name"] not in pairs:
                    continue
                best, cost_mid = pairs[tg["name"]]
                recs = tt[best]
                t_end_walk = _walk_t(best)
                chosen, diffs = None, {}
                for sg, estb in tg["branches"].items():
                    dd = diff_vs_truth(geom, ll2xy, estb, tg["summaries"][sg],
                                       recs["hull"], recs["tow"], t_end_walk)
                    diffs[sg] = dd
                if side == "auto" and len(diffs) > 1:
                    chosen = min(diffs, key=lambda k_: diffs[k_]["pos_err_median_m"])
                else:
                    want = 1 if side == "pos" else -1
                    chosen = want if want in diffs else next(iter(diffs))
                truth_map[tg["name"]] = {"truth_name": best, "diffs": diffs,
                                         "chosen": chosen, "records": recs,
                                         "t_end_walk": t_end_walk}
                log(f"[匹配] {tg['name']} <- 真值 {best}: 方位 RMS "
                    f"{diffs[chosen]['bearing_rms_deg']:.3f}°, 位置中位误差 "
                    f"{diffs[chosen]['pos_err_median_m']:.0f} m, CPA 误差 "
                    f"{diffs[chosen]['cpa_err_km']:+.3f} km "
                    f"({'判定 ' + ('右舷+' if chosen > 0 else '左舷-') if side == 'auto' else '指定 ' + side})")
        else:
            log("[真值] 未找到同目录 txt, 跳过差异对比")

    # ---- 报告 ----
    R = ["=" * 96,
         f"BTR 目标检测代理报告  hull={os.path.basename(hull_png)}, "
         f"tow={os.path.basename(tow_png)}",
         f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 96,
         f"平台: 起点 ({prm['start_ll'][0]:.4f}°N, {prm['start_ll'][1]:.4f}°E), "
         f"航向 {prm['heading']:.1f}°, 航速 {prm['speed_kn']:.1f} kn",
         f"阵列: 舷侧阵 {prm['hull_m']} m, 拖曳阵 {prm['tow_m']} m "
         f"(基线 {abs(prm['tow_m'][0]):.0f} m)",
         f"标定: hull cols {cals['hull']['rect'][2]}-{cals['hull']['rect'][3]} = "
         f"{b_range[0]:.0f}~{b_range[1]:.0f}°, rows {cals['hull']['rect'][0]}-"
         f"{cals['hull']['rect'][1]} = {t_max:.0f}~0 s; 配色 {style}",
         f"提取: 等阈亮带亚像素中点 (hull {cals['hull']['level']:.1f} dB, "
         f"tow {cals['tow']['level']:.1f} dB), 步长 {dt:.0f} s, "
         f"视差门限 {PAR_MIN_DEG}°, 距离中值窗 {RWIN} 历元", ""]
    for tg in targets:
        R.append(f"===== {tg['name']} =====")
        tm = truth_map.get(tg["name"])
        for sg in sorted(tg["branches"], reverse=True):
            summ = tg["summaries"][sg]
            d_ = tm["diffs"][sg] if tm else None
            mark = ""
            if tm:
                mark = "  ★选定分支" if sg == tm["chosen"] else "  (镜像候选)"
            R.append(fmt_summary(sg, tg["name"], sg, summ, d_) + mark)
            R.append("")
    # ---- 质量定级 / 漏检与交叉诊断 ----
    grades, notes = {}, {}
    if truth_map:
        for tg in targets:
            tm = truth_map.get(tg["name"])
            if tm:
                grades[tg["name"]] = quality_grade(tm["diffs"][tm["chosen"]])
        notes = detect_notes(targets, truth_map, tt_all, unmatched, min_cov)
        if tt_all:
            x = crossing_risk(targets, geom, ll2xy, tt_all, walk_all)
            if x is not None and x[0] < 2 * 12.0:      # 间距 < 约两条亮带宽度
                notes["_cross"] = (f"交叉风险: 真值 {x[1]}/{x[2]} 的 hull 方位最小间距仅"
                                   f" {x[0]:.1f} px (小于亮带宽度量级), "
                                   f"跟迹会把两者合并成一条轨迹 (漏检一个)")
        R.append("")
        R.append(summary_table(targets, truth_map, grades,
                               {k: v for k, v in notes.items() if not k.startswith("_")}))
        if notes.get("_cross"):
            R.append(f"  ! {notes['_cross']}")
        R.append("")
    R.append(f"亮带历元覆盖率: hull "
             f"{np.mean([1 if raw['hull'].get(t) else 0 for t in raw['hull']])*100:.0f}%, "
             f"tow {np.mean([1 if raw['tow'].get(t) else 0 for t in raw['tow']])*100:.0f}%")
    R.append(f"耗时 {((datetime.now() - t0_wall).total_seconds()):.1f} s")
    report = "\n".join(R)
    print(report)

    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)

        def _tgt_json(tg):
            tm = truth_map.get(tg["name"])
            return {"name": tg["name"],
                    "anchor": tg["anchor"],
                    "branches": {str(k): v for k, v in tg["summaries"].items()},
                    "truth": ({"truth_name": tm["truth_name"],
                               "chosen_side": tm["chosen"],
                               "diffs": {str(k): v for k, v in tm["diffs"].items()}}
                              if tm else None)}
        payload = {"params": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in prm.items()},
                   "t_max": t_max, "b_range": list(b_range), "cmap": style,
                   "calib": {k: {"rect": list(v["rect"]), "level": v["level"],
                                 "grid_x": v.get("grid_x"), "grid_y": v.get("grid_y"),
                                 "grid_resid_deg": v.get("grid_resid_deg"),
                                 "grid_origin_shift_deg": v.get("grid_origin_shift_deg")}
                             for k, v in cals.items()},
                   "targets": [_tgt_json(tg) for tg in targets]}
        with open(json_out, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2, default=float)
        log(f"\n[JSON] {json_out}")
    if png_out:
        os.makedirs(os.path.dirname(os.path.abspath(png_out)), exist_ok=True)
        try:
            plot_png(hull_png, tow_png, cals, targets, truth_map, png_out, prm)
            log(f"[PNG] {png_out}")
        except Exception as exc:                       # 绘图失败不影响检测结果
            log(f"[PNG] 绘制失败: {exc}")
    return {"report": report, "targets": targets, "truth": truth_map,
            "cals": cals, "geom": geom}


def plot_png(hull_png, tow_png, cals, targets, truth_map, out_png, prm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cols = ["#00e5ff", "#ff9d00", "#ff5ec4", "#8cff5e", "#ffffff"]
    fig = plt.figure(figsize=(14, 11), facecolor="black")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.05], hspace=0.22)
    for j, (tag, png) in enumerate((("hull", hull_png), ("tow", tow_png))):
        ax = fig.add_subplot(gs[j, 0])
        img = np.asarray(Image.open(png).convert("RGB"))
        r0, r1, c0, c1 = cals[tag]["rect"]
        span, tmax = c1 - c0 + 1, cals[tag]["t_max"]
        ax.imshow(img)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        for i, tg in enumerate(targets):
            est = tg["branches"].get(1) or next(iter(tg["branches"].values()))
            bs = np.array(est["bearing_hull"] if tag == "hull" else est["bearing_tow"])
            ts = np.array(est["t"])
            xs, ys = c0 + bs / 180.0 * span, r1 - ts / tmax * (r1 - r0)
            c = cols[i % len(cols)]
            ax.plot(xs, ys, "-", color=c, lw=1.3, label=f"{tg['name']} 检出方位")
            run = est["motion_run"]
            if run:
                ax.plot(xs[run[0]:run[1] + 1], ys[run[0]:run[1] + 1], "-",
                        color="white", lw=3.2, alpha=0.30)
            ax.plot(xs[0], ys[0], "o", color=c, ms=9, mec="white", mew=1.2)
            ax.plot(xs[-1], ys[-1], "s", color=c, ms=9, mec="white", mew=1.2)
        ax.legend(loc="upper right", fontsize=8, facecolor="#101828", labelcolor="white")
        ax.set_title(f"{tag} 方位历程图 + 检出轨迹 (○初始 / □结束, 白粗段=运动段)",
                     color="white", fontsize=11)
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor("#0a0f1f")
    geom = B.Geometry(prm["start_ll"], prm["heading"], prm["speed_kn"],
                      prm["hull_m"], prm["tow_m"], prm["ref"][0], prm["ref"][1])
    tt = np.linspace(0, cals["hull"]["t_max"], 300)
    px, py = geom.platform_xy(tt)
    pl = [geom.xy2ll(x, y) for x, y in zip(px, py)]
    ax.plot([p[1] for p in pl], [p[0] for p in pl], "w-", lw=1.6, label="平台 (输入参数)")
    hx, hy = geom.hull_xy(0.0)
    tx, ty = geom.tow_xy(0.0)
    lh, lt = geom.xy2ll(hx, hy), geom.xy2ll(tx, ty)
    ax.plot(lh[1], lh[0], "s", color="#ffd400", ms=9, label="舷侧阵")
    ax.plot(lt[1], lt[0], "^", color="#00ffcc", ms=9, label="拖曳阵")
    ll2xy, _ = B.make_geo(prm["ref"][0], prm["ref"][1])
    for i, tg in enumerate(targets):
        c = cols[i % len(cols)]
        tm = truth_map.get(tg["name"])
        est = tg["branches"].get(tm["chosen"] if tm else 1) or next(iter(tg["branches"].values()))
        lla = [geom.xy2ll(x, y) for x, y in zip(est["x_m"], est["y_m"])]
        ax.plot([p[1] for p in lla], [p[0] for p in lla], "-", color=c, lw=2.0,
                label=f"{tg['name']} 检测 ({'右舷+' if est['side_sign'] > 0 else '左舷-'})")
        ax.plot(lla[0][1], lla[0][0], "o", color=c, ms=8, mec="white", mew=1.3)
        ax.plot(lla[-1][1], lla[-1][0], "s", color=c, ms=8, mec="white", mew=1.3)
        if tm:
            rec = tm["records"]["hull"]
            t_end = tm["t_end_walk"]
            x0, y0 = ll2xy(*rec["start"])
            x1, y1 = ll2xy(*rec["end"])
            f = np.minimum(1.0, tt / t_end) if t_end > 0 else np.ones_like(tt)
            Tx, Ty = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            tl = [geom.xy2ll(x, y) for x, y in zip(Tx, Ty)]
            ax.plot([p[1] for p in tl], [p[0] for p in tl], "--", color=c, lw=1.4,
                    alpha=0.85, label=f"{tm['truth_name']} 真值")
    ax.set_aspect(1.0 / math.cos(math.radians(prm["ref"][0])))
    ax.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("经度 (°E)", color="white")
    ax.set_ylabel("纬度 (°N)", color="white")
    ax.legend(loc="best", fontsize=7, facecolor="#101828", labelcolor="white")
    ax.set_title("检测轨迹 (实线) vs 真值轨迹 (虚线) —— ○初始 / □结束", color="white", fontsize=11)
    fig.suptitle("BTR 目标检测代理 —— 初始/结束位置 · 舷角偏转 · 距离 · 与真值差异",
                 color="white", fontsize=13)
    fig.savefig(out_png, dpi=115, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="BTR 方位历程图目标检测代理 (hull+tow -> 起止位置/舷角偏转/距离 + 真值差异)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--hull-img", required=True)
    ap.add_argument("--tow-img", required=True)
    ap.add_argument("--start-ll", nargs=2, type=float, default=None, metavar=("LAT", "LON"))
    ap.add_argument("--heading", type=float, default=None)
    ap.add_argument("--speed-kn", type=float, default=None)
    ap.add_argument("--hull-m", nargs=2, type=float, default=None, metavar=("X", "Y"))
    ap.add_argument("--tow-m", nargs=2, type=float, default=None, metavar=("X", "Y"))
    ap.add_argument("--ref", nargs=2, type=float, default=None, metavar=("LAT", "LON"))
    ap.add_argument("--track-txt", default=None,
                    help="平台参数来源 (默认自动找同目录 btr_*_track.txt)")
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--b-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--db-range", nargs=2, type=float, default=[-5.0, 16.0], metavar=("MIN", "MAX"))
    ap.add_argument("--dt", type=float, default=2.0, help="提取时间步长 (s)")
    ap.add_argument("--level", type=float, default=None, help="等阈电平 (dB), 默认自适应")
    ap.add_argument("--cmap", choices=["auto", "gui", "geo"], default="auto")
    ap.add_argument("--side", choices=["auto", "pos", "neg"], default="auto")
    ap.add_argument("--no-truth", action="store_true", help="不与真值对比")
    ap.add_argument("--json", default=None, help="JSON 输出路径")
    ap.add_argument("--png", default=None, help="叠加核验图输出路径")
    ap.add_argument("--report", default=None, help="文本报告输出路径")
    args = ap.parse_args(argv)

    params = dict(start_ll=args.start_ll, heading=args.heading, speed_kn=args.speed_kn,
                  hull_m=args.hull_m, tow_m=args.tow_m, ref=args.ref,
                  cmap=args.cmap, track_txt=args.track_txt)
    res = run(args.hull_img, args.tow_img, params=params, dt=args.dt, level=args.level,
              side=args.side, t_max=args.t_max, b_range=args.b_range,
              db_range=(args.db_range[0], args.db_range[1]), truth=not args.no_truth,
              png_out=args.png, json_out=args.json)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fp:
            fp.write(res["report"] + "\n")
        print(f"[报告] {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

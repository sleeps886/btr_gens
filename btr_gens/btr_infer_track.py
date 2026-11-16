# -*- coding: utf-8 -*-
"""
btr_infer_track.py —— 由 hull / tow 方位历程图 (BTR) 图像反推目标轨迹
=====================================================================
纯图像反演: 只依赖两张 BTR 图 (PNG) 和平台/阵列几何参数, 不使用仿真内部
数据。流程:

  1. 读图: 自动定位坐标区 (剔除外围坐标轴/色标条), 按生成端配色查找表把
     每个像素还原为 dB 强度 (同时剔除叠加的白色/黄色参考曲线)。
  2. 检测: 逐时间片沿方位轴做峰值检测 → 方位点迹; 点迹关联成方位轨迹
     (容忍交叉、短断线), 过滤杂散斑点。
  3. 配对: hull / tow 两组方位轨迹按"视差最小"原则一一配对。
  4. 定位: 由平台运动参数 (起点经纬度 / 航向 / 航速) 与两阵偏移推算任意
     时刻两阵位置, 对每个目标:
       a) 瞬时双阵交会 —— 逐时刻三角定位 (直观, 但远距离病态);
       b) 批量匀速直线 (CV) Gauss-Newton 拟合 —— 利用全程视差与方位变化
          率约束 4 维状态 (x0,y0,vx,vy), 远距离也能稳定收敛 (主要输出)。
     舷角已折叠 0~180° 存在左右舷模糊: 两阵共线于艏艉线时镜像分支代价
     完全相同, 无法从数据分辨, 此时输出两支候选 (或用 --side 指定);
     若阵列存在横向偏移 (oy≠0) 则自动按拟合代价分辨左右舷。
  5. 输出: 反推轨迹图 PNG + 反演信息 TXT (含舷角残差/视差等质量指标)。

用法:
  python btr_infer_track.py ^
      --hull-img btr_out/xxx/btr_A_hull.png --tow-img btr_out/xxx/btr_A_tow.png ^
      --start-ll 30.0623 122.0415 --heading 210 --speed-kn 0 ^
      --hull-m 0 0 --tow-m -150 0 --tag A

依赖: numpy, Pillow, matplotlib (仅读图与绘图, 与生成端代码无关)。

坐标与约定 (与 btr_gui_server.py 完全一致):
  - heading: 地理方位角 (北=0° 顺时针); 平台沿航向匀速直线运动;
  - 阵列偏移 (ox, oy): ox>0 指向船头, oy>0 指向左舷, 随船头航向旋转;
  - 舷角: 船首=0°, 右舷正横=90°, 船尾=180°, 左右舷模糊折叠到 0~180°;
  - 经纬度换算: 等距圆柱投影, 基准 (ref_lat, ref_lon), 地球半径 6371000 m。
"""
import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image

try:                      # matplotlib 仅绘图需要, 缺失时文本输出仍可用
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:       # pragma: no cover
    matplotlib = None
    plt = None

KN2MS = 0.514444
EARTH_R_DEF = 6371000.0

# 生成端两种配色的颜色停靠点 (与 btr_gui_server.py / btr_geo_ll.py 一致)
CMAP_STOPS = {
    "gui": [(0,0,0),(0,0,0.35),(0,0.15,0.5),(0,0.45,0.75),(0.2,0.85,0.85),
            (0.95,0.95,0.25),(1,0.55,0.1),(1,0.2,0),(1,0,0)],
    "geo": [(0,0,0.15),(0,0.2,0.6),(0,0.7,0.9),(1,1,0.2),(1,0.4,0),(1,0,0)],
}


# ----------------------------------------------------------------------
# 基础数学工具
# ----------------------------------------------------------------------
def build_lut(stops, n=256):
    """颜色停靠点 -> (n,3) uint8 查找表 (等距停靠, 与 from_list 一致)"""
    pos = np.linspace(0.0, 1.0, len(stops))
    xs = np.linspace(0.0, 1.0, n)
    lut = np.zeros((n, 3), dtype=np.uint8)
    for c in range(3):
        vals = np.array([s[c] for s in stops], dtype=float)
        lut[:, c] = np.clip(np.round(np.interp(xs, pos, vals) * 255.0), 0, 255)
    return lut


def wrap180(a):
    """角度归一到 (-180, 180]"""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def make_geo(ref_lat, ref_lon, earth_r=EARTH_R_DEF):
    """等距圆柱投影 (与生成端一致的经纬度 <-> 米制)"""
    lat0 = math.radians(ref_lat)
    lon0 = math.radians(ref_lon)

    def ll2xy(lat, lon):
        return (earth_r * math.cos(lat0) * (math.radians(lon) - lon0),
                earth_r * (math.radians(lat) - lat0))

    def xy2ll(x, y):
        return (math.degrees(lat0 + y / earth_r),
                math.degrees(lon0 + x / (earth_r * math.cos(lat0))))

    return ll2xy, xy2ll


class Geometry:
    """平台 + 双阵运动学 (与 btr_gui_server.build_platform 相同的模型)"""

    def __init__(self, start_ll, heading_deg, speed_kn, hull_m, tow_m,
                 ref_lat=30.0, ref_lon=122.0, earth_r=EARTH_R_DEF):
        self.ll2xy, self.xy2ll = make_geo(ref_lat, ref_lon, earth_r)
        self.start_ll = tuple(float(v) for v in start_ll)
        self.x0, self.y0 = self.ll2xy(*self.start_ll)
        self.psi = math.radians(float(heading_deg))     # 地理方位角
        self.heading = float(heading_deg)
        self.v = float(speed_kn) * KN2MS
        self.hull_m = tuple(float(v) for v in hull_m)
        self.tow_m = tuple(float(v) for v in tow_m)
        self.ref_lat = float(ref_lat)
        self.ref_lon = float(ref_lon)

    def platform_xy(self, t):
        t = np.asarray(t, dtype=float)
        return (self.x0 + self.v * math.sin(self.psi) * t,
                self.y0 + self.v * math.cos(self.psi) * t)

    def array_xy(self, t, off):
        ox, oy = off
        px, py = self.platform_xy(t)
        sh, ch = math.sin(self.psi), math.cos(self.psi)
        return (px + ox * sh - oy * ch, py + ox * ch + oy * sh)

    def hull_xy(self, t):
        return self.array_xy(t, self.hull_m)

    def tow_xy(self, t):
        return self.array_xy(t, self.tow_m)


# ----------------------------------------------------------------------
# 图像 -> dB 图
# ----------------------------------------------------------------------
def _longest_run(mask):
    """一维 bool 序列中最长连续 True 段 (start, end) 含端点; 无 True 返回 None"""
    best = None
    s = None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if best is None or (i - 1 - s) > (best[1] - best[0]):
                best = (s, i - 1)
            s = None
    if s is not None:
        i = len(mask)
        if best is None or (i - 1 - s) > (best[1] - best[0]):
            best = (s, i - 1)
    return best


def _color_lut_map(rgb, lut):
    """每个像素 -> 最近 LUT 序号与精确距离平方。

    BTR 图是 flat 四边形渲染, 唯一颜色数远小于像素数: 先对唯一颜色
    求 LUT 近邻, 再按像素映射回去, 复杂度从 O(像素xLUT) 降到
    O(唯一颜色xLUT + 像素)。
    """
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3).astype(np.int32)
    lut32 = lut.astype(np.int32)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    d = ((uniq[:, None, :] - lut32[None, :, :]) ** 2).sum(-1)
    idx = d.argmin(1)[inv].reshape(h, w)
    dmin = d.min(1)[inv].reshape(h, w)
    return idx, dmin


def _nearest_lut(rgb16, lut, chunk=48):
    """逐块求每个像素的最近 LUT 序号与距离平方; rgb16: (H,W,3) int16"""
    return _color_lut_map(rgb16, lut)


def find_plot_rect(rgb, lut_list, dist_thr=45.0):
    """自动定位 pcolormesh 绘图区 (剔除轴外文字/色标条)

    背景中低于色标下限的像素被渲染为纯黑, 因此除 LUT 颜色外, 还需把
    "暗蓝色背景斑点" 计入绘图区特征 (B 通道显著高于 R, 中等亮度)。
    """
    h, w, _ = rgb.shape
    rgb16 = rgb.astype(np.int16)
    thr2 = dist_thr ** 2
    # 剔除 LUT 中的纯黑端 (背景也是黑色, 会把整幅图误判为绘图区)
    luts = [lut[lut.astype(np.int32).sum(1) > 45] for lut in lut_list]
    mask = np.zeros((h, w), dtype=bool)
    for lut in luts:
        _, dmin = _color_lut_map(rgb16, lut)
        mask |= dmin <= thr2
    R = rgb16[..., 0]
    G = rgb16[..., 1]
    B = rgb16[..., 2]
    bluish = (R < 70) & (B > R + 40) & (B >= 60) & (B <= 240)   # 暗蓝背景斑点
    mask |= bluish
    colfrac = mask.mean(0)
    crun = _longest_run(colfrac > 0.15)
    if crun is None or crun[1] - crun[0] < 100:
        raise ValueError("未能在图像中定位 BTR 绘图区 (配色不匹配?)")
    rowfrac = mask[:, crun[0]:crun[1] + 1].mean(1)
    rrun = _longest_run(rowfrac > 0.15)
    if rrun is None or rrun[1] - rrun[0] < 60:
        raise ValueError("未能定位绘图区行范围")
    return rrun[0], rrun[1], crun[0], crun[1]      # r0, r1, c0, c1


def _dilate(m, it=3):
    """3x3 二值膨胀 it 次 (纯 numpy)"""
    out = m
    for _ in range(it):
        out = (out | np.roll(out, 1, 0) | np.roll(out, -1, 0)
               | np.roll(out, 1, 1) | np.roll(out, -1, 1))
    return out


def image_to_db(rgb, rect, lut, db_min=-5.0, db_max=16.0, dist_thr=60.0):
    """绘图区像素 -> dB 图 (叠加参考曲线/网格处为 NaN 并沿方位轴修补)"""
    r0, r1, c0, c1 = rect
    sub = rgb[r0:r1 + 1, c0:c1 + 1]
    R = sub[..., 0].astype(np.int16)
    G = sub[..., 1].astype(np.int16)
    B = sub[..., 2].astype(np.int16)
    lum = 0.2126 * R + 0.7152 * G + 0.0722 * B
    mx = np.maximum(R, G)
    whiteish = (lum > 0.72 * 255) & (B > 0.55 * mx)          # 白色参考曲线
    yellowish = (R > 240) & (G > 240) & (B < 100)            # 黄色虚线参考曲线

    idx, dmin = _nearest_lut(sub.astype(np.int16), lut)
    ok = dmin <= dist_thr ** 2
    db = db_min + idx.astype(float) / (len(lut) - 1) * (db_max - db_min)
    bad = whiteish | yellowish | ~ok
    # 关键: 参考曲线的抗锯齿混色像素即使 dist<60, 其 dB 也已失真 (偏向白/黄),
    # 会把亮线核心"拉低"。将坏点带膨胀数像素后再修补, 让插值采样远离混色区。
    bad = _dilate(bad, it=2)
    db[bad] = np.nan

    # 沿方位轴 (每行) 线性修补 NaN —— 高斯亮线对称, 修补后峰值位置不变
    Hh, Ww = db.shape
    cols = np.arange(Ww)
    for i in range(Hh):
        row = db[i]
        good = ~np.isnan(row)
        if good.sum() >= 2 and (~good).any():
            db[i] = np.interp(cols, cols[good], row[good])
    return db


def pick_cmap_style(rgb, rect, db_min, db_max):
    """用绘图区子采样对比两种配色的还原残差, 自动选择生成端配色"""
    r0, r1, c0, c1 = rect
    sub = rgb[r0:r1 + 1:8, c0:c1 + 1:8].astype(np.int16)
    best, bestmd = None, None
    for name, stops in CMAP_STOPS.items():
        _, dmin = _nearest_lut(sub, build_lut(stops))
        md = float(np.median(dmin))
        if best is None or md < bestmd:
            best, bestmd = name, md
    return best


# ----------------------------------------------------------------------
# 峰值检测与点迹关联
# ----------------------------------------------------------------------
def detect_per_time(db, rect, t_max, b_min, b_max, db_thr=8.0,
                    t_step=1.0, sup_px=6, smooth=3):
    """逐时间片峰值检测。

    返回 (dets, profiles):
      dets     [(t, bearing, peak_db), ...] 按时间排序 (用于轨迹起始);
      profiles {t: np.ndarray} 每个时间片平滑后的方位维 dB 剖面 (用于质心跟踪)。
    """
    r0, r1, c0, c1 = rect
    Hh, Ww = db.shape
    span = b_max - b_min

    def rr(t):                      # 时间 -> 绘图区内行号 (t=0 在底部)
        return (r1 - t / t_max * (r1 - r0)) - r0

    kern = np.ones(smooth) / smooth
    dets = []
    profiles = {}
    n_t = int(round(t_max / t_step))
    for k in range(n_t):
        t = k * t_step
        lo = int(math.floor(rr(t + t_step)))
        hi = int(math.ceil(rr(t)))
        lo = max(0, min(lo, Hh - 1))
        hi = max(0, min(hi, Hh - 1))
        if hi < lo:
            lo, hi = hi, lo
        prof = np.nanmean(db[lo:hi + 1], axis=0)
        if np.all(np.isnan(prof)):
            continue
        prof_s = np.convolve(np.nan_to_num(prof, nan=-99.0), kern, mode="same")
        profiles[t + 0.5 * t_step] = prof_s
        for i in range(2, len(prof_s) - 2):
            v = prof_s[i]
            if v < db_thr:
                continue
            if v >= prof_s[i - 1] and v > prof_s[i + 1] and v >= prof_s[i - 2] and v > prof_s[i + 2]:
                l, c, r = prof_s[i - 1], prof_s[i], prof_s[i + 1]
                den = l - 2 * c + r
                off = 0.5 * (l - r) / den if abs(den) > 1e-9 else 0.0
                off = float(np.clip(off, -1.0, 1.0))
                b = b_min + (i + off + 0.5) / Ww * span
                dets.append((t + 0.5 * t_step, float(b), float(v)))
    # 同一时间片近邻峰去重 (保留强者)
    dets.sort(key=lambda d: d[0])
    out = []
    for d in dets:
        if out and abs(d[0] - out[-1][0]) < 1e-6 and abs(d[1] - out[-1][1]) < sup_px * span / Ww:
            if d[2] > out[-1][2]:
                out[-1] = d
            continue
        out.append(d)
    return out, profiles


def link_tracks(dets, profiles=None, db_thr=8.0, gate0=0.9, gate_vel=2.0,
                gate_max=5.0, max_gap=5.0, min_len=40.0, merge_gap=50.0,
                min_amp=None, b_min=0.0, b_max=180.0, halfwin_deg=2.5,
                miss_max=6, min_coverage=0.45):
    """点迹/剖面 -> 方位轨迹。

    主策略: 质心跟踪 (track-before-detect 简化版) —— 对每条已有轨迹按
    速度预测位置, 在预测窗内对 dB 剖面做加权质心作为该时刻方位测量;
    峰值点迹仅用于轨迹起始。这样宽线/修补平台/断续峰都不会撕裂轨迹。
    折叠方位在 0°/180° 处镜像反射 (目标穿越艏艉线)。
    """
    span = b_max - b_min
    npx = None
    if profiles:
        p0 = next(iter(profiles.values()))
        npx = len(p0)
    if npx is None:
        npx = max(1, int(round(span / 0.25)))

    def reflect(b):
        """把预测角以三角形波方式折回 [b_min, b_max]"""
        period = 2 * span
        x = (b - b_min) % period
        if x > span:
            x = period - x
        return b_min + x

    def col2b(c):
        return b_min + (c + 0.5) / npx * span

    def b2col(b):
        return (b - b_min) / span * npx - 0.5

    def slope(pts):
        """中位两两斜率 (稳健全于噪声/短反射); 向量化, 供逐点调用"""
        n = len(pts)
        if n < 3:
            return 0.0
        arr = np.asarray(pts, dtype=float)
        t, b = arr[:, 0], arr[:, 1]
        iu = np.triu_indices(n, 1)
        dt = t[iu[1]] - t[iu[0]]
        m = dt >= 1.0
        if not m.any():
            return 0.0
        return float(np.median((b[iu[1]][m] - b[iu[0]][m]) / dt[m]))

    seeds = {}
    for t, b, amp in dets:
        seeds.setdefault(round(t, 3), []).append((b, amp))

    tracks = []      # dict: pts [(t,b,amp)], miss, done
    finished = []

    def grow(tr, t, prof):
        dt = t - tr["pts"][-1][0]
        pred = reflect(tr["pts"][-1][1] + tr["v"] * dt)
        half = min(max(halfwin_deg, gate0 + gate_vel * abs(tr["v"]) * dt),
                   gate_max)
        c0 = int(round(b2col(pred) - half / span * npx))
        c1 = int(round(b2col(pred) + half / span * npx))
        idx = np.arange(max(0, c0), min(npx - 1, c1) + 1)
        vals = prof[idx]
        # 成熟轨迹允许 1.5 dB 宽限, 避免 T4 这类弱线因逐秒起伏断链
        thr_eff = db_thr if len(tr["pts"]) < 30 else db_thr - 1.5
        good = ~np.isnan(vals) & (vals >= thr_eff - 3.0)
        if good.sum() < 3:
            return None
        if float(np.nanmax(vals)) < thr_eff:
            return None                      # 窗口内没有真实峰 -> 不增长
        w = np.clip(vals - (thr_eff - 3.0), 0.0, 8.0) * good
        cent = float((idx * w).sum() / w.sum())
        # 亮带边缘中点修正: 掩膜/参考线削顶会让质心偏向残存一侧,
        # 而高斯线形的两侧过阈位置对称于真实中心, 取包含峰值的连续亮带
        # 首尾像素中点作为无偏中心, 与质心各取一半。
        imax = int(np.argmax(np.where(good, vals, -99.0)))
        a = imax
        while a - 1 >= 0 and good[a - 1]:
            a -= 1
        b_ = imax
        while b_ + 1 < len(good) and good[b_ + 1]:
            b_ += 1
        cent_edge = 0.5 * (idx[a] + idx[b_])
        cent = 0.5 * cent + 0.5 * cent_edge
        b_new = col2b(cent)
        res = float(wrap180(b_new - pred))
        if abs(res) > min(gate0 + gate_vel * abs(tr["v"]) * dt + 1.0, gate_max):
            return None
        return b_new, float(np.nanmax(vals))

    times = sorted(profiles.keys()) if profiles else sorted(seeds.keys())
    active = []                      # 仅未终止轨迹, 避免逐秒全量扫描
    for t in times:
        prof = profiles.get(t) if profiles else None
        # 1) 增长已有轨迹
        still = []
        for tr in active:
            if t - tr["pts"][-1][0] > max_gap:
                tr["done"] = True
                continue
            if prof is None:
                tr["miss"] += 1
                if tr["miss"] > miss_max:
                    tr["done"] = True
                else:
                    still.append(tr)
                continue
            got = grow(tr, t, prof)
            if got is None:
                tr["miss"] += 1
                if tr["miss"] > miss_max:
                    tr["done"] = True
                else:
                    still.append(tr)
            else:
                # 霸主检查: 更长的活跃轨迹已占据该时刻 ±1.5° 时, 本轨迹
                # 不再收点 (只记 miss)。宽线两翼孵化出的平行轨迹会因此
                # 迅速饿死, 而真实轨迹穿过交叉区 (短于 miss_max) 后自动恢复。
                b_new = got[0]
                sup = False
                for other in active:
                    if other is tr or len(other["pts"]) <= len(tr["pts"]):
                        continue
                    dt2 = t - other["pts"][-1][0]
                    if dt2 < 0 or dt2 > max_gap:
                        continue
                    pred2 = reflect(other["pts"][-1][1] + other["v"] * dt2)
                    if abs(float(wrap180(b_new - pred2))) < 1.5:
                        sup = True
                        break
                if sup:
                    tr["miss"] += 1
                    if tr["miss"] > miss_max:
                        tr["done"] = True
                    else:
                        still.append(tr)
                else:
                    tr["pts"].append((t, got[0], got[1]))
                    tr["v"] = slope(tr["pts"][-20:])
                    tr["miss"] = 0
                    still.append(tr)
        active = still
        # 2) 未被认领的峰点 -> 新轨迹 (起始要求更强的峰, 弱峰只允许并入);
        #    认领门限取半窗宽: 同一宽线的两翼峰与中心峰相距可达 2~4°,
        #    用 gate0+1.0 会反复孵化平行轨迹 (数量爆炸且拖慢后续去重)
        for b, amp in seeds.get(round(t, 3), []):
            if amp < db_thr + 1.0:
                continue
            claimed = False
            for tr in active:
                dt = t - tr["pts"][-1][0]
                if dt <= 0 or dt > max_gap:
                    continue
                pred = reflect(tr["pts"][-1][1] + tr["v"] * dt)
                if abs(float(wrap180(b - pred))) <= halfwin_deg + 1.5:
                    claimed = True
                    break
            if not claimed:
                tr = {"pts": [(t, b, amp)], "v": 0.0, "miss": 0,
                      "done": False}
                tracks.append(tr)
                active.append(tr)

    def as_arr(tr):
        return np.array(tr["pts"], dtype=float)      # (N,3): t, b, amp

    # 0) 预过滤: 明显的杂波短迹 (少于 8 点或短于 10s) 直接丢弃, 降低后续开销
    tracks = [tr for tr in tracks
              if len(tr["pts"]) >= 8
              and tr["pts"][-1][0] - tr["pts"][0][0] >= 10.0]

    # 1) 断段合并: 首尾间隔小且方位预测一致 (先合并, 让长轨迹成形)。
    #    起点时间排序 + 二分索引: 每条轨迹只检查起点落在其尾点之后
    #    merge_gap 内的少数候选, 避免 O(K^2) 全对扫描。
    merged = True
    while merged:
        merged = False
        order = sorted(range(len(tracks)),
                       key=lambda i: tracks[i]["pts"][0][0])
        start_ts = [tracks[j]["pts"][0][0] for j in order]
        import bisect
        for i in range(len(tracks)):
            ti = as_arr(tracks[i])
            tail_t, tail_b = ti[-1, 0], ti[-1, 1]
            lo_pos = bisect.bisect_right(start_ts, tail_t)
            hi_pos = bisect.bisect_right(start_ts, tail_t + merge_gap)
            best_j, best_gap = None, None
            for pos in range(lo_pos, hi_pos):
                j = order[pos]
                if j == i:
                    continue
                tj = tracks[j]["pts"]
                if not tj:
                    continue
                gap = tj[0][0] - tail_t
                if not (0 < gap <= merge_gap):
                    continue
                pred = reflect(tail_b + tracks[i]["v"] * gap)
                if abs(float(wrap180(tj[0][1] - pred))) > 2.0:
                    continue
                if best_j is None or gap < best_gap:
                    best_j, best_gap = j, gap
            if best_j is not None:
                tracks[i]["pts"].extend(tracks[best_j]["pts"])
                tracks[best_j]["pts"] = []
                merged = True
                break
        if merged:
            tracks = [tr for tr in tracks if tr["pts"]]

    # 2) 轨迹去重: 与更长轨迹高度重叠的丢弃; 按时长优先排序, 避免长轨迹
    #    被高幅碎片挤掉。阈值按重叠占比自适应: ≥80% 用 5.0° (同一目标的
    #    变体), ≥60% 用 3.0°, 部分重叠用 1.8° (防误并交叉目标)。
    tracks.sort(key=lambda tr: (-(as_arr(tr)[-1, 0] - as_arr(tr)[0, 0]),
                                -np.mean([p[2] for p in tr["pts"]])))
    kept = []
    for tr in tracks:
        a = as_arr(tr)
        span_a = a[-1, 0] - a[0, 0]
        dup = False
        for ktr in kept:
            ka = ktr["arr"]
            lo = max(a[0, 0], ka[0, 0])
            hi = min(a[-1, 0], ka[-1, 0])
            if hi - lo < 30:
                continue
            frac = (hi - lo) / max(min(span_a, ka[-1, 0] - ka[0, 0]), 1.0)
            thr_d = 5.0 if frac >= 0.8 else (3.0 if frac >= 0.6 else 1.8)
            tg = np.arange(math.ceil(lo), math.floor(hi) + 1, 1.0)
            d = np.abs(wrap180(np.interp(tg, a[:, 0], a[:, 1])
                               - np.interp(tg, ka[:, 0], ka[:, 1])))
            if float(np.median(d)) < thr_d:
                dup = True
                break
        if not dup:
            tr["arr"] = a
            kept.append(tr)
    tracks = kept

    # 3) 过滤: 时长 / 点数 / 覆盖率 / 幅度
    keep = []
    for tr in tracks:
        a = as_arr(tr)
        span_t = a[-1, 0] - a[0, 0]
        if span_t < min_len or len(a) < 20:
            continue
        if min_coverage and len(a) / max(span_t, 1.0) < min_coverage:
            continue
        if min_amp is not None and a[:, 2].mean() < min_amp:
            continue
        tt, bb = a[:, 0], a[:, 1]
        tr["v"] = slope(list(zip(tt, bb)))
        tr["arr"] = a
        keep.append(tr)
    keep.sort(key=lambda tr: -tr["arr"][:, 2].mean())
    for n, tr in enumerate(keep):
        tr["name"] = f"B{n+1}"
    return keep


def track_at(tr, t_grid):
    a = tr["arr"]
    return np.interp(t_grid, a[:, 0], a[:, 1])


def track_segment_split(tr, min_moving_sec=100.0):
    """按滑动方位变化率剔除'静止尾段'并按断口分段, 返回 [(t0,t1), ...]"""
    a = tr["arr"]
    ts, bs = a[:, 0], a[:, 1]
    win = 41.0
    rate = np.zeros_like(ts)
    for i, t in enumerate(ts):
        m = (ts >= t - win / 2) & (ts <= t + win / 2)
        if m.sum() >= 8:
            rate[i] = np.polyfit(ts[m], bs[m], 1)[0]
    med = float(np.median(np.abs(rate))) + 1e-6
    active = np.abs(rate) > max(0.15 * med, 0.004)
    segs = []
    i = 0
    n = len(ts)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and active[j + 1] and ts[j + 1] - ts[j] <= 15.0:
            j += 1
        segs.append((ts[i], ts[j]))
        i = j + 1
    return [(s, e) for s, e in segs if e - s >= min_moving_sec]


def associate(htracks, ttracks, min_overlap=30.0, max_parallax=3.0):
    """hull/tow 方位轨迹配对: 重叠段内平均方位差最小 (贪心)"""
    cand = []
    for i, h in enumerate(htracks):
        for j, k in enumerate(ttracks):
            lo = max(h["arr"][0, 0], k["arr"][0, 0])
            hi = min(h["arr"][-1, 0], k["arr"][-1, 0])
            if hi - lo < min_overlap:
                continue
            tg = np.arange(math.ceil(lo), math.floor(hi) + 1, 1.0)
            d = np.abs(wrap180(track_at(h, tg) - track_at(k, tg)))
            m = float(np.median(d))
            if m > max_parallax:        # 视差过大 → 不可能是同一目标
                continue
            cand.append((m, i, j))
    cand.sort()
    uh, ut, out = set(), set(), []
    for m, i, j in cand:
        if i in uh or j in ut:
            continue
        uh.add(i)
        ut.add(j)
        out.append({"ih": i, "it": j, "dparallax": m})
    return out


# ----------------------------------------------------------------------
# 定位: 瞬时交会 + 批量 CV 拟合
# ----------------------------------------------------------------------
def instant_fix(geom, t, th, tt, sh, st):
    """单时刻双阵交会; 返回 (x, y) 或 (nan, nan)"""
    A = np.array(geom.hull_xy(t))
    B = np.array(geom.tow_xy(t))
    a1 = math.radians(geom.heading + sh * th)
    a2 = math.radians(geom.heading + st * tt)
    u1 = np.array([math.sin(a1), math.cos(a1)])
    u2 = np.array([math.sin(a2), math.cos(a2)])
    M = np.array([[u1[0], -u2[0]], [u1[1], -u2[1]]])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-6:
        return (np.nan, np.nan)
    rhs = B - A
    d1 = (M[1, 1] * rhs[0] - M[0, 1] * rhs[1]) / det
    d2 = (-M[1, 0] * rhs[0] + M[0, 0] * rhs[1]) / det
    if d1 <= 0 or d2 <= 0:
        return (np.nan, np.nan)
    P1 = A + d1 * u1
    P2 = B + d2 * u2
    return tuple(0.5 * (P1 + P2))


def pseudo_linear_init(ts, S, az):
    """纯方位伪线性最小二乘初值 (有偏但稳健)"""
    a = np.radians(np.asarray(az, dtype=float))
    u = np.stack([np.sin(a), np.cos(a)], 1)
    nvec = np.stack([-np.cos(a), np.sin(a)], 1)
    M = np.concatenate([nvec, ts[:, None] * nvec], 1)
    rhs = (nvec * S).sum(1)
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return sol


def gn_fit_cv(ts, S, az, s0, iters=50, trim_deg=2.5, quick=False):
    """批量匀速直线 (CV) Gauss-Newton 拟合: state=(x0,y0,vx,vy) 米制 ENU

    quick=True: 少量迭代、不做离群剔除, 用于多起点粗筛。
    """
    if quick:
        iters = min(iters, 12)
    s = np.array(s0, dtype=float)
    lam = 1.0
    ts = np.asarray(ts, dtype=float)
    S = np.asarray(S, dtype=float)
    az = np.asarray(az, dtype=float)

    def cost_of(st, mask):
        X = np.stack([st[0] + st[2] * ts, st[1] + st[3] * ts], 1)
        D = X - S
        d2 = (D ** 2).sum(1)
        al = np.degrees(np.arctan2(D[:, 0], D[:, 1]))
        r = wrap180(al - az)
        return float((r[mask] ** 2).sum()), r, d2, X

    mask = np.ones(len(ts), dtype=bool)
    c, r, d2, _ = cost_of(s, mask)
    for it in range(iters):
        X = np.stack([s[0] + s[2] * ts, s[1] + s[3] * ts], 1)
        D = X - S
        d2 = np.maximum((D ** 2).sum(1), 1.0)[:, None]
        # d(al)/dx0 = Dy*(180/pi)/d2, d(al)/dy0 = -Dx*(180/pi)/d2
        Jx = np.stack([D[:, 1], -D[:, 0]], 1) * (180.0 / math.pi) / d2
        J = np.concatenate([Jx, ts[:, None] * Jx], 1)
        H = J[mask].T @ J[mask] + lam * np.eye(4)
        g = J[mask].T @ r[mask]
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        s_new = s + step
        c_new, r_new, _, _ = cost_of(s_new, mask)
        if c_new < c:
            s, c, r = s_new, c_new, r_new
            lam = max(lam * 0.5, 1e-6)
        else:
            lam *= 4.0
            if lam > 1e8:
                break
        if np.abs(step).max() < 1e-9:
            break
    # 粗大残差剔除后重拟合 (quick 模式跳过)
    for _ in range(0 if quick else 2):
        bad = np.abs(r) > trim_deg
        if bad.sum() == 0 or (~bad).sum() < 20:
            break
        mask = ~bad
        s = s + 0.0
        c, r, _, _ = cost_of(s, mask)
        for it in range(30):
            X = np.stack([s[0] + s[2] * ts, s[1] + s[3] * ts], 1)
            D = X - S
            d2 = np.maximum((D ** 2).sum(1), 1.0)[:, None]
            Jx = np.stack([D[:, 1], -D[:, 0]], 1) * (180.0 / math.pi) / d2
            J = np.concatenate([Jx, ts[:, None] * Jx], 1)
            H = J[mask].T @ J[mask] + lam * np.eye(4)
            g = J[mask].T @ r[mask]
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break
            s_new = s + step
            c_new, r_new, _, _ = cost_of(s_new, mask)
            if c_new < c:
                s, c, r = s_new, c_new, r_new
                lam = max(lam * 0.5, 1e-6)
            else:
                lam *= 4.0
                if lam > 1e8:
                    break
            if np.abs(step).max() < 1e-9:
                break
    rms = float(np.sqrt(c / max(mask.sum(), 1)))
    return s, rms, mask, r


def _gn_from(ts, S, az, s0, quick=False):
    s, rms, mask, r = gn_fit_cv(ts, S, az, s0, quick=quick)
    # 鲁棒代价: 被剔除的点按 trim 上限计费, 防止"丢弃数据换低 rms"的简并解
    score = float(np.minimum(r ** 2, 2.5 ** 2).sum())
    return {"state": s, "rms": rms, "cost": float((r[mask] ** 2).sum()),
            "score": score, "nuse": int(mask.sum()), "nall": int(len(r)),
            "s0": np.asarray(s0, dtype=float).copy()}


def fit_pair_cv(geom, ts_h, th, ts_t, tt):
    """对配对目标尝试 4 种左右舷符号组合, 返回最优/镜像等信息

    初值策略: 伪线性最小二乘 (病态时限幅缩放) + 距离网格多起点,
    取代价最低者, 避免远距离弱视差几何下发散。
    """
    combos = []
    for sh in (+1, -1):
        for st in (+1, -1):
            azh = geom.heading + sh * np.asarray(th, dtype=float)
            azt = geom.heading + st * np.asarray(tt, dtype=float)
            ts = np.concatenate([ts_h, ts_t])
            Sn = np.stack([geom.hull_xy(ts_h)[0], geom.hull_xy(ts_h)[1]], 1)
            St = np.stack([geom.tow_xy(ts_t)[0], geom.tow_xy(ts_t)[1]], 1)
            S = np.concatenate([Sn, St], 0)
            az = np.concatenate([azh, azt])
            inits = []
            try:
                s0 = pseudo_linear_init(ts, S, az)
                if np.all(np.isfinite(s0)):
                    if np.abs(s0).max() > 1e5:      # 病态: 限幅缩放
                        p = s0[:2]
                        v = s0[2:]
                        pn = np.linalg.norm(p)
                        vn = np.linalg.norm(v)
                        s0 = np.concatenate([
                            p / (pn + 1e-9) * 1.0e4,
                            v / (vn + 1e-9) * 8.0])
                    inits.append(s0)
            except Exception:
                pass
            # 距离网格起点: 沿中位视线方向, 速度取 0 与 ±8 m/s 横向
            amid = float(np.median(th))
            a0 = math.radians(geom.heading + sh * amid)
            u = np.array([math.sin(a0), math.cos(a0)])
            ut = np.array([u[1], -u[0]])        # 视线右侧法向
            Ac = geom.hull_xy(float(np.median(ts_h)))
            for rng in (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0):
                for vv in (0.0, 8.0, -8.0):
                    inits.append(np.concatenate([Ac + u * rng, ut * vv]))
            best = None
            # 两阶段: 全部初值快速粗评 (12 轮), 仅 top3 全量精化
            quicks = []
            for s0 in inits:
                try:
                    cand = _gn_from(ts, S, az, s0, quick=True)
                    if not np.all(np.isfinite(cand["state"])):
                        continue
                    if np.abs(cand["state"]).max() > 1e7:
                        continue
                    quicks.append(cand)
                except Exception:
                    continue
            for cand in quicks:                     # quick 结果作兜底
                if best is None or cand["score"] < best["score"]:
                    best = cand
            for cand in quicks[:3]:                 # top3 全量精化
                try:
                    full = _gn_from(ts, S, az, cand["s0"])
                    if not np.all(np.isfinite(full["state"])):
                        continue
                    if np.abs(full["state"]).max() > 1e7:
                        continue
                    if best is None or full["score"] < best["score"]:
                        best = full
                except Exception:
                    continue
            if best is None:
                continue
            best.update({"sh": sh, "st": st})
            combos.append(best)
    if not combos:
        return None
    combos.sort(key=lambda c: c["score"])
    best = combos[0]
    ambiguous = False
    mirror = None
    if len(combos) > 1:
        c2 = combos[1]
        is_mirror = (c2["sh"] == -best["sh"] and c2["st"] == -best["st"])
        tol = 0.02 * max(best["score"], 1.0)
        if is_mirror and (c2["score"] - best["score"]) <= tol:
            ambiguous = True
            mirror = c2
    return {"best": best, "mirror": mirror, "ambiguous": ambiguous,
            "all": combos}


# ----------------------------------------------------------------------
# 主推理流程 (可编程调用)
# ----------------------------------------------------------------------
def run_inference(hull_img, tow_img, start_ll, heading_deg, speed_kn,
                  hull_m=(0.0, 0.0), tow_m=(-150.0, 0.0),
                  t_max=600.0, b_min=0.0, b_max=180.0,
                  db_min=-5.0, db_max=16.0,
                  ref_lat=30.0, ref_lon=122.0, earth_r=EARTH_R_DEF,
                  db_threshold=7.0, min_track_sec=40.0, min_seg_sec=100.0,
                  max_parallax=3.0,
                  side="auto", cmap_style="auto", log=print, debug_dir=None):
    geom = Geometry(start_ll, heading_deg, speed_kn, hull_m, tow_m,
                    ref_lat, ref_lon, earth_r)
    img_h = np.asarray(Image.open(hull_img).convert("RGB"))
    img_t = np.asarray(Image.open(tow_img).convert("RGB"))
    lut_list = [build_lut(s) for s in CMAP_STOPS.values()]
    rect_h = find_plot_rect(img_h, lut_list)
    rect_t = find_plot_rect(img_t, lut_list)
    if cmap_style == "auto":
        cmap_style = pick_cmap_style(img_h, rect_h, db_min, db_max)
    log(f"[INFO] 配色还原: {cmap_style}; hull 绘图区 rows={rect_h[0]}-{rect_h[1]}, "
        f"cols={rect_h[2]}-{rect_h[3]}; tow rows={rect_t[0]}-{rect_t[1]}, "
        f"cols={rect_t[2]}-{rect_t[3]}")
    lut = build_lut(CMAP_STOPS[cmap_style])

    db_h = image_to_db(img_h, rect_h, lut, db_min, db_max)
    db_t = image_to_db(img_t, rect_t, lut, db_min, db_max)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for tag, d in (("hull", db_h), ("tow", db_t)):
            arr = np.clip((d - db_min) / (db_max - db_min), 0, 1)
            Image.fromarray((np.nan_to_num(arr) * 255).astype(np.uint8)
                            ).save(os.path.join(debug_dir, f"debug_dbmap_{tag}.png"))

    dets_h, prof_h = detect_per_time(db_h, rect_h, t_max, b_min, b_max,
                                     db_threshold)
    dets_t, prof_t = detect_per_time(db_t, rect_t, t_max, b_min, b_max,
                                     db_threshold)
    log(f"[INFO] 峰值检测: hull {len(dets_h)} 点, tow {len(dets_t)} 点")
    htracks = link_tracks(dets_h, profiles=prof_h, db_thr=db_threshold,
                          min_len=min_track_sec,
                          min_amp=db_threshold - 1.5,
                          b_min=b_min, b_max=b_max)
    ttracks = link_tracks(dets_t, profiles=prof_t, db_thr=db_threshold,
                          min_len=min_track_sec,
                          min_amp=db_threshold - 1.5,
                          b_min=b_min, b_max=b_max)
    def _summarize(trs):
        return [(tr["name"], round(float(tr["arr"][0, 0])),
                 round(float(tr["arr"][-1, 0]))) for tr in trs[:10]]

    log(f"[INFO] 方位轨迹: hull {len(htracks)} 条 {_summarize(htracks)}")
    log(f"[INFO] 方位轨迹: tow  {len(ttracks)} 条 {_summarize(ttracks)}")

    pairs = associate(htracks, ttracks, max_parallax=max_parallax)
    log(f"[INFO] hull/tow 配对: {[(p['ih'], p['it'], round(p['dparallax'],3)) for p in pairs]}")

    results = []
    for n, pr in enumerate(pairs):
        h, k = htracks[pr["ih"]], ttracks[pr["it"]]
        segs_h = track_segment_split(h, min_moving_sec=min_seg_sec)
        segs_t = track_segment_split(k, min_moving_sec=min_seg_sec)
        lo0 = max(h["arr"][0, 0], k["arr"][0, 0])
        hi0 = min(h["arr"][-1, 0], k["arr"][-1, 0])
        if hi0 - lo0 < 30:
            log(f"[WARN] 目标#{n+1}: hull/tow 共同窗口过短 ({hi0-lo0:.0f}s), 跳过")
            continue
        # 候选拟合窗口: 全重叠 + 各自运动段与共同窗口的交集。
        # 全程窗口让 CV 拟合用满约束; 若目标有机动/反射 (V 形), 分段择优。
        # 窗口长度下限 = max(min_seg_sec, 一半重叠跨度): 过短的窗口在
        # 近 CPA 处方位曲率趋零, 距离不可观, 会出现"低残差错盆地"简并。
        ov_span = hi0 - lo0
        min_win = max(min_seg_sec, 0.5 * ov_span)
        cands = [(lo0, hi0)]
        for segs in (segs_h, segs_t):
            for s0, s1 in segs:
                lo, hi = max(s0, lo0), min(s1, hi0)
                if hi - lo >= min_win and (lo, hi) not in cands:
                    cands.append((lo, hi))
        # 滑窗候选: 交叉区污染/抑制造成的缺口会改变分段位置, 固定步长
        # 扫描可让 [中段,末段] 这类优质窗口重新入围
        _step, _w = 60.0, 0.7 * ov_span
        _t = lo0
        while _t + min_win <= hi0 + 1e-9:
            _hi = min(_t + _w, hi0)
            if _hi - _t >= min_win and all(abs(_t - a) > 1e-6 or abs(_hi - b) > 1e-6
                                           for a, b in cands):
                cands.append((_t, _hi))
            _t += _step
        best_fit, best_win = None, None
        for lo, hi in cands:
            tg = np.arange(math.ceil(lo), math.floor(hi) + 1, 1.0)
            th = track_at(h, tg)
            tt = track_at(k, tg)
            f = fit_pair_cv(geom, tg, th, tg, tt)
            if f is None:
                continue
            if best_fit is None or (f["best"]["score"] / max(f["best"]["nall"], 1)
                                    < best_fit["best"]["score"]
                                    / max(best_fit["best"]["nall"], 1)):
                best_fit, best_win = f, (lo, hi, tg, th, tt)
        if best_fit is None:
            log(f"[WARN] 目标#{n+1}: CV 拟合失败, 跳过")
            continue
        fit = best_fit
        lo, hi, tg, th, tt = best_win
        best = fit["best"]
        # 瞬时交会 (按选定组合)
        inst = np.array([instant_fix(geom, t, bh, bt, best["sh"], best["st"])
                         for t, bh, bt in zip(tg, th, tt)])
        res = {"name": f"目标{n+1}", "pair": pr, "t0": float(tg[0]), "t1": float(tg[-1]),
               "th": th, "tt": tt, "tgrid": tg, "fit": fit, "instant": inst,
               "side": side, "geom": geom}
        results.append(res)
        amb = "镜像模糊" if fit["ambiguous"] else "已分辨"
        log(f"[INFO] {res['name']}: 窗口[{res['t0']:.0f},{res['t1']:.0f}]s "
            f"组合(sh,st)=({best['sh']:+d},{best['st']:+d}) rms={best['rms']:.3f}° "
            f"视差中位={pr['dparallax']:.3f}° [{amb}]")
    return {"geom": geom, "results": results, "htracks": htracks,
            "ttracks": ttracks, "cmap_style": cmap_style,
            "rects": (rect_h, rect_t)}


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------
def _setup_fonts():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def resolve_states(res):
    """按 --side 决定最终输出状态; 返回 (states, mirror_states, note)"""
    fit = res["fit"]
    best = fit["best"]
    side = res["side"]
    if side == "auto":
        states = [best]
        mirrors = [fit["mirror"]] if fit["mirror"] is not None else []
        note = "镜像模糊, 两支候选均已绘出" if fit["mirror"] else "已由拟合代价分辨"
    else:
        want = +1 if side == "pos" else -1
        same = [c for c in fit["all"] if c["sh"] == want]
        if same:
            states = [min(same, key=lambda c: c["cost"])]
            others = [c for c in fit["all"]
                      if c not in states and (c["sh"], c["st"]) == (-states[0]["sh"], -states[0]["st"])]
            mirrors = others[:1]
        else:
            states = [best]
            mirrors = []
        note = f"按 --side {side} 强制"
    return states, mirrors, note


def draw_figure(res_inf, out_png, tag="A", show_instant=True, dpi=150):
    _setup_fonts()
    geom = res_inf["geom"]
    fig = plt.figure(figsize=(11, 6.5), dpi=dpi, facecolor="black")
    ax = fig.add_subplot(111)
    ttt = np.linspace(0.0, max(res_inf["results"], key=lambda r: r["t1"])["t1"]
                      if res_inf["results"] else 600.0, 800)
    px, py = geom.platform_xy(ttt)
    la0, lo0 = geom.xy2ll(px[0], py[0])
    la1, lo1 = geom.xy2ll(px[-1], py[-1])
    ax.plot([lo0, lo1], [la0, la1], "w-", lw=1.8, label="平台航迹(输入参数推算)")
    hx, hy = geom.hull_xy(0.0)
    tx, ty = geom.tow_xy(0.0)
    lhh = geom.xy2ll(hx, hy)
    ltt = geom.xy2ll(tx, ty)
    ax.plot(lhh[1], lhh[0], "s", color="#ffd400", ms=8, label="舷侧阵(hull) t=0")
    ax.plot(ltt[1], ltt[0], "^", color="#00ffcc", ms=8, label="拖曳阵(tow) t=0")

    colors = ["#ff9d00", "#ff00cc", "#00ff88", "#66aaff", "#ff5555",
              "#dddddd", "#bb88ff", "#7fff00"]
    for i, res in enumerate(res_inf["results"]):
        c = colors[i % len(colors)]
        states, mirrors, _note = resolve_states(res)
        for si, st in enumerate(states):
            s = st["state"]
            X0 = np.array(geom.xy2ll(s[0], s[1]))
            X1 = np.array(geom.xy2ll(s[0] + s[2] * res["t1"], s[1] + s[3] * res["t1"]))
            lbl = f"{res['name']} CV拟合" if si == 0 else None
            ax.plot([X0[1], X1[1]], [X0[0], X1[0]], "-", color=c, lw=2.2, label=lbl)
            ax.plot(X0[1], X0[0], "o", color=c, ms=6)
        for st in mirrors:
            s = st["state"]
            X0 = np.array(geom.xy2ll(s[0], s[1]))
            X1 = np.array(geom.xy2ll(s[0] + s[2] * res["t1"], s[1] + s[3] * res["t1"]))
            ax.plot([X0[1], X1[1]], [X0[0], X1[0]], "--", color=c, lw=1.4,
                    alpha=0.75, label=f"{res['name']} 镜像候选" if si == 0 and st is mirrors[0] else None)
        if show_instant:
            ins = res["instant"]
            ok = np.isfinite(ins[:, 0])
            if ok.any():
                ll = [geom.xy2ll(x, y) for x, y in ins[ok]]
                ax.plot([p[1] for p in ll], [p[0] for p in ll], ".",
                        color=c, ms=3, alpha=0.35)
    ax.set_xlabel("经度 Longitude (°E)", fontsize=10)
    ax.set_ylabel("纬度 Latitude (°N)", fontsize=10)
    ax.set_facecolor("#0a0f1f")
    ax.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
    ax.set_aspect(1.0 / math.cos(math.radians(geom.ref_lat)))
    ax.legend(loc="best", fontsize=8, facecolor="#101828", labelcolor="white")
    ax.set_title(f"由 hull+tow BTR 图像反推目标轨迹 (平台{tag})", fontsize=12,
                 color="white")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def write_txt(res_inf, out_txt, params, tag="A"):
    geom = res_inf["geom"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["=" * 68,
         f"反推结果: btr_infer_{tag}.png (由 hull+tow BTR 图像反演, 平台{tag})",
         f"生成时间: {now}", "=" * 68,
         "[输入参数]",
         f"  hull 图: {params['hull_img']}",
         f"  tow  图: {params['tow_img']}",
         f"  平台起点: ({geom.start_ll[0]:.4f}°N, {geom.start_ll[1]:.4f}°E)",
         f"  航向: {geom.heading:.1f}° (北=0°顺时针) | 航速: {geom.v/KN2MS:.1f} kn",
         f"  舷侧阵偏移: {geom.hull_m} m | 拖曳阵偏移: {geom.tow_m} m",
         f"  观测时长: {params['t_max']:.0f} s | 方位范围: "
         f"{params['b_min']:.0f}~{params['b_max']:.0f}° | "
         f"配色还原: {res_inf['cmap_style']}",
         f"  检测门限: {params['db_threshold']:.0f} dB | 方位轨迹: hull "
         f"{len(res_inf['htracks'])} 条 / tow {len(res_inf['ttracks'])} 条",
         "", "[反推目标]"]
    for res in res_inf["results"]:
        states, mirrors, note = resolve_states(res)
        best = res["fit"]["best"]
        L.append("-" * 68)
        L.append(f"  {res['name']}: 共同观测窗口 [{res['t0']:.0f}, {res['t1']:.0f}]s | "
                 f"双阵视差中位数 {res['pair']['dparallax']:.3f}° | {note}")
        for si, st in enumerate(states + (mirrors or [])):
            s = st["state"]
            spd = math.hypot(s[2], s[3])
            course = math.degrees(math.atan2(s[2], s[3])) % 360.0 if spd > 1e-6 else 0.0
            ll0 = geom.xy2ll(s[0], s[1])
            ll1 = geom.xy2ll(s[0] + s[2] * res["t1"], s[1] + s[3] * res["t1"])
            role = "CV拟合" if si < len(states) else "镜像候选"
            L.append(f"    [{role}] t0=({ll0[0]:.4f}°N,{ll0[1]:.4f}°E) "
                     f"-> t={res['t1']:.0f}s ({ll1[0]:.4f}°N,{ll1[1]:.4f}°E)")
            L.append(f"            速度 {spd:.2f} m/s ({spd/KN2MS:.1f} kn), "
                     f"航向 {course:.1f}° | 舷角残差 RMS {st['rms']:.3f}° "
                     f"(符号组合 sh={st['sh']:+d}, st={st['st']:+d})")
    L.append("-" * 68)
    L.append("说明: CV拟合=批量匀速直线最小二乘(主要结果); 镜像候选=左右舷模糊的")
    L.append("      另一支等价解 (两阵共线于艏艉线时无法由数据分辨, 用 --side 指定)。")
    L.append("      机动目标(变向变速)按匀速直线拟合, 轨迹为时间平均意义上的估计。")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="由 hull/tow BTR 图像反推目标轨迹 (输入: 两张图 + 平台/阵列参数)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--hull-img", required=True, help="舷侧阵 BTR PNG 路径")
    ap.add_argument("--tow-img", required=True, help="拖曳阵 BTR PNG 路径")
    ap.add_argument("--start-ll", nargs=2, type=float, required=True,
                    metavar=("LAT", "LON"), help="平台起点经纬度 (纬度 经度)")
    ap.add_argument("--heading", type=float, required=True,
                    help="平台航向 (地理方位角, 北=0°顺时针, 全程不变)")
    ap.add_argument("--speed-kn", type=float, required=True, help="平台航速 (节)")
    ap.add_argument("--hull-m", nargs=2, type=float, default=[0.0, 0.0],
                    metavar=("X", "Y"), help="舷侧阵偏移 (x>0 船头, y>0 左舷)")
    ap.add_argument("--tow-m", nargs=2, type=float, default=[-150.0, 0.0],
                    metavar=("X", "Y"), help="拖曳阵偏移 (x>0 船头, y>0 左舷)")
    ap.add_argument("--t-max", type=float, default=600.0, help="图像时间轴上限 (s)")
    ap.add_argument("--b-min", type=float, default=0.0, help="图像方位轴下限 (°)")
    ap.add_argument("--b-max", type=float, default=180.0, help="图像方位轴上限 (°)")
    ap.add_argument("--db-min", type=float, default=-5.0, help="色标 dB 下限")
    ap.add_argument("--db-max", type=float, default=16.0, help="色标 dB 上限")
    ap.add_argument("--ref-lat", type=float, default=30.0, help="投影基准纬度")
    ap.add_argument("--ref-lon", type=float, default=122.0, help="投影基准经度")
    ap.add_argument("--db-threshold", type=float, default=7.0,
                    help="峰值检测门限 (dB, 相对噪声底)")
    ap.add_argument("--max-parallax", type=float, default=3.0,
                    help="hull/tow 配对允许的最大视差中位数 (°)")
    ap.add_argument("--min-track-sec", type=float, default=40.0,
                    help="方位轨迹最短时长 (s)")
    ap.add_argument("--min-seg-sec", type=float, default=100.0,
                    help="CV 拟合段最短时长 (s)")
    ap.add_argument("--side", choices=["auto", "pos", "neg"], default="auto",
                    help="左右舷: auto=按代价/输出两支, pos=右舷+, neg=左舷−")
    ap.add_argument("--cmap", choices=["auto", "gui", "geo"], default="auto",
                    help="生成端配色 (gui=GUI服务器, geo=btr_geo_ll)")
    ap.add_argument("--tag", default="A", help="平台名 (输出文件名)")
    ap.add_argument("--no-instant", action="store_true",
                    help="不绘制瞬时交会散点")
    ap.add_argument("--debug", action="store_true", help="保存中间 dB 图")
    ap.add_argument("--out-dir", default=None, help="输出目录 (默认在 hull 图同目录)")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.hull_img)),
        f"infer_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        print(msg)

    params = {"hull_img": args.hull_img, "tow_img": args.tow_img,
              "t_max": args.t_max, "b_min": args.b_min, "b_max": args.b_max,
              "db_threshold": args.db_threshold}
    res_inf = run_inference(
        args.hull_img, args.tow_img, args.start_ll, args.heading, args.speed_kn,
        hull_m=tuple(args.hull_m), tow_m=tuple(args.tow_m),
        t_max=args.t_max, b_min=args.b_min, b_max=args.b_max,
        db_min=args.db_min, db_max=args.db_max,
        ref_lat=args.ref_lat, ref_lon=args.ref_lon,
        db_threshold=args.db_threshold, min_track_sec=args.min_track_sec,
        min_seg_sec=args.min_seg_sec, max_parallax=args.max_parallax,
        side=args.side, cmap_style=args.cmap,
        log=log, debug_dir=out_dir if args.debug else None)

    out_png = os.path.join(out_dir, f"btr_infer_{args.tag}.png")
    out_txt = os.path.join(out_dir, f"btr_infer_{args.tag}.txt")
    draw_figure(res_inf, out_png, tag=args.tag, show_instant=not args.no_instant)
    write_txt(res_inf, out_txt, params, tag=args.tag)
    print(f"OK -> {out_png}")
    print(f"OK -> {out_txt}")
    return res_inf


if __name__ == "__main__":
    main()

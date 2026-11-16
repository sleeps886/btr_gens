# -*- coding: utf-8 -*-
"""extract_latest.py —— 从 btr_out 最新一批 hull/tow 方位历程图提取目标检测结果

流程 (纯图像反演, 不使用仿真内部数据):
  1. 像素<->坐标标定: 由灰色网格线反推 (列 98..1262 = 0..180°, 行 738.6..22.4 = 0..600 s)。
  2. 定方位: 本批亮线饱和 (16 dB 平台宽约 ±5°), 峰值检测会锁到平台右缘 (约 +5° 偏置),
     故改用"等阈亮带亚像素中点" —— 亮带两缘对称于真实方位, 削顶不影响中点。
     实测精度约 0.1° (t=300 s: 检出 hull 119.89° / 真值 120.0°)。
  3. 跟迹: 取观测中段 (t=300 s) 两阵强亮带互配为锚点 (不预设目标方位),
     由锚点向前/向后恒速预测跟迹, 自然覆盖目标进出场时刻。
  4. 定距离: 同一时刻舷侧阵/拖曳阵双站视差交会 (基线 180 m)。
     视差约 0.9° -> 单历元距离误差约 ±11%, 且视差趋 0 时发散;
     故只采纳视差 >= PAR_MIN 的历元, 再对距离做宽窗中值滤波,
     位置 = 阵点 + 平滑距离 × 检出方位单位矢量 (角向精确, 径向稳健)。
  5. 速度: 由极坐标导数 v = ṙ·u + r·θ̇·u⊥ 给出 (对停走型目标比整体直线拟合合理)。
  6. 左右舷: 两阵共线于艏艉线 -> 镜像分支的舷角/距离完全相同, 两支位置都报出。
"""
import math
import os
import re
import sys
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = "20261105_152635"
SRC = os.path.join(HERE, "btr_out", BATCH)

# ---- 像素标定 ----
RECT = {"hull": (22, 736, 98, 1262), "tow": (22, 737, 98, 1261)}
R_T0, R_T600 = 738.6, 22.4
T_MAX = 600.0
# ---- 提取参数 ----
LEVEL = 13.5            # 等阈电平 (dB): 由 validate_level.py 对解析真值标定
                        #   (12 dB 会把亮线的宽晕台并进来, 中点漂移 +4~+6°;
                        #    13.5 dB 只取饱和核心, 误差约 ±0.2°)
MIN_W = 6               # 亮带最小宽度 (px)
MERGE = 20              # 亮带合并缺口 (px): 容忍网格点线/参考线造成的窄缺口
DT = 2.0                # 时间步长 (s)
GATE = 4.0              # 跟迹门限 (deg)
ANCHOR_T = 300.0        # 起迹锚时刻
ANCHOR_PEAK = 14.0      # 锚点亮带最低峰值 (dB)
MISS_MAX = 30.0         # 允许断档 (s)
PAR_MIN = 0.35          # 采纳历元的最小双站视差 (deg)
RWIN = 41               # 距离中值滤波窗 (历元数, 41×2 s = 82 s)
VWIN = 21               # 速度估计窗 (历元数)
DEV_DEG = 0.20          # 运动段判据: 与末端恒定方位的偏差门限 (deg)

GEOM = dict(start_ll=(30.0, 122.0), heading_deg=-30.0, speed_kn=0.0,
            hull_m=(0.0, 0.0), tow_m=(-180.0, 0.0), ref_lat=30.0, ref_lon=122.0)
geom = B.Geometry(**GEOM)


def row_of(t):
    return R_T0 - t / T_MAX * (R_T0 - R_T600)


# ================= 0. 修正版像素->dB 还原 =================
# btr_infer_track.image_to_db 的"黄色参考线"判据 (R>240 & G>240 & B<100) 会把
# 配色自身的黄停靠点 (0.95,0.95,0.25)*255 = (242,242,64) 一并当成参考线掩掉,
# 于是亮线 crest (约 12.8 dB 一带) 被挖空后按行插值成假平台, 早段宽亮带的中点
# 因此漂移。这里收紧为"纯黄 (255,255,0) 且 R,G 均衡", 只掩真正的叠加曲线。
def image_to_db_fixed(rgb, rect, lut, db_min=-5.0, db_max=16.0, dist_thr=60.0):
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
    Hh, Ww = db.shape
    cols = np.arange(Ww)
    for i in range(Hh):
        row = db[i]
        good = ~np.isnan(row)
        if good.sum() >= 2 and (~good).any():
            db[i] = np.interp(cols, cols[good], row[good])
    return db


# ================= 1. 亮带提取 =================
def band_centers(prof, level=LEVEL, min_w=MIN_W, merge=MERGE):
    idx = np.where(prof >= level)[0]
    if len(idx) == 0:
        return []
    segs, s = [], idx[0]
    for a, b_ in zip(idx[:-1], idx[1:]):
        if b_ - a > merge:                    # 容忍参考曲线/网格线窄缺口
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


def load_bands():
    lut = B.build_lut(B.CMAP_STOPS["gui"])
    raw, shapes = {}, {}
    for tag in ("hull", "tow"):
        img = np.asarray(Image.open(os.path.join(SRC, f"btr_A_{tag}.png")).convert("RGB"))
        d = image_to_db_fixed(img, RECT[tag], lut, -5.0, 16.0)
        shapes[tag] = d.shape
        span = RECT[tag][3] - RECT[tag][2] + 1
        raw[tag] = {}
        for t in np.arange(DT / 2, T_MAX, DT):
            r = int(round(row_of(t))) - RECT[tag][0]
            if not (0 <= r < d.shape[0]):
                continue
            prof = d[max(r - 1, 0):r + 2].mean(axis=0)          # 3 行平均
            raw[tag][t] = [(c / span * 180.0, w, a)
                           for c, w, a in band_centers(prof)]
    return raw, shapes


# ================= 2. 锚点 + 跟迹 =================
def nearest_key(d, t):
    k = min(d, key=lambda x: abs(x - t))
    return k if abs(k - t) <= DT else None


def track_dir(raw_tag, keys, t0, b0):
    out, p, v, last = {}, b0, 0.0, t0
    for t in keys:
        cands = raw_tag.get(t, [])
        dt = t - last
        if not cands:
            if abs(dt) > MISS_MAX:
                break
            continue
        best = min(cands, key=lambda c: abs(c[0] - p))
        if abs(best[0] - p) > GATE:
            if abs(dt) > MISS_MAX:
                break
            continue
        out[t] = best
        v = 0.7 * v + 0.3 * (best[0] - p) / max(abs(dt), 1e-9)
        p = best[0] + v * dt
        last = t
    return out


def track_from(raw_tag, t_anchor, b_anchor):
    ks = sorted(raw_tag)
    merged = track_dir(raw_tag, [k for k in ks if k < t_anchor][::-1], t_anchor, b_anchor)
    merged.update(track_dir(raw_tag, [k for k in ks if k > t_anchor], t_anchor, b_anchor))
    merged[t_anchor] = min(raw_tag[t_anchor], key=lambda c: abs(c[0] - b_anchor))
    return merged


def anchors(raw):
    """观测中段两阵强亮带互配 -> 目标锚 (不依赖任何先验方位)"""
    ta, tb = nearest_key(raw["hull"], ANCHOR_T), nearest_key(raw["tow"], ANCHOR_T)
    hb = [b for b in raw["hull"][ta] if b[2] >= ANCHOR_PEAK]
    tt = [b for b in raw["tow"][tb] if b[2] >= ANCHOR_PEAK]
    used, pairs = set(), []
    for h in hb:
        cand = [(i, t) for i, t in enumerate(tt)
                if i not in used and abs(t[0] - h[0]) <= GATE]
        if cand:
            i, m = min(cand, key=lambda p: abs(p[1][0] - h[0]))
            used.add(i)
            pairs.append((h, m))
    pairs.sort(key=lambda p: -p[0][2])
    return [(f"目标{i + 1}", h[0], t[0], ta) for i, (h, t) in enumerate(pairs)]


# ================= 3. 交会 + 距离平滑 =================
def medfilt(x, k):
    x = np.asarray(x, float)
    h = k // 2
    return np.array([np.median(x[max(i - h, 0):min(i + h + 1, len(x))])
                     for i in range(len(x))])


def branch_est(bh_tr, bt_tr, sg):
    """一个左右舷分支的完整估计 -> dict 或 None"""
    ts, bh, bt = [], [], []
    for t in sorted(set(bh_tr) & set(bt_tr)):
        p = B.instant_fix(geom, t, bh_tr[t][0], bt_tr[t][0], sg, sg)
        if not np.isfinite(p[0]):
            continue
        ts.append(t)
        bh.append(bh_tr[t][0])
        bt.append(bt_tr[t][0])
    if len(ts) < 30:
        return None
    ts, bh, bt = np.array(ts), np.array(bh), np.array(bt)
    par = np.abs(bh - bt)
    acc = par >= PAR_MIN
    if acc.sum() < 30:
        return None
    # 单历元距离 (由交会点) -> 宽窗中值平滑
    hx, hy = geom.hull_xy(ts)
    X, Y = np.array([B.instant_fix(geom, t, b_, c, sg, sg) for t, b_, c in
                     zip(ts[acc], bh[acc], bt[acc])]).T
    tt = ts[acc]
    r_raw = np.hypot(X - hx[acc], Y - hy[acc])
    r_sm = medfilt(r_raw, RWIN)
    # 满窗历元 val: 边缘 RWIN//2 个历元的截断中值有偏 -> 径向通道不采用
    val = np.zeros(len(tt), bool)
    h = RWIN // 2
    if len(tt) > 2 * h:
        val[h:len(tt) - h] = True
    # 代表距离 r_rep: 满窗用宽窗中值, 边缘历元用局部中值 (消除截断偏置)
    r_rep = r_sm.copy()
    K = 11
    for j in np.where(~val)[0]:
        r_rep[j] = float(np.median(r_raw[max(j - K // 2, 0):j + K // 2 + 1]))
    # 位置 = 阵点 + 代表距离 × 检出方位单位矢量 (角向精确, 径向稳健)
    az = np.radians(geom.heading + sg * bh[acc])
    Px = hx[acc] + r_rep * np.sin(az)
    Py = hy[acc] + r_rep * np.cos(az)
    # 运动段: 目标"走完航点"后方位恒定 -> 取与末端方位偏差超过 DEV_DEG 的连续段
    tail = float(np.median(bh[acc][-max(20, len(tt) // 5):]))
    mv = np.abs(bh[acc] - tail) > DEV_DEG
    run = longest_run(mv)
    return {"ts": tt, "bh": bh[acc], "bt": bt[acc], "par": par[acc],
            "r_raw": r_raw, "r_sm": r_sm, "r_rep": r_rep, "X": Px, "Y": Py,
            "moving": mv, "run": run, "val": val, "sg": sg,
            "n_all": len(ts), "n_acc": int(acc.sum()),
            "par_med": float(np.median(par)), "par_rng": (float(par.min()), float(par.max()))}


def longest_run(mask):
    """布尔序列中最长连续 True 段的 (起, 止) 下标; 无则 None"""
    best, cur, s = None, 0, None
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


def cpa(est):
    """滑窗内最近点: 用平滑距离序列的最小值"""
    k = int(np.argmin(est["r_rep"]))
    return float(est["r_rep"][k]), float(est["ts"][k])


def fmt_point(est, i, lbl):
    t = est["ts"][i]
    lat, lon = geom.xy2ll(est["X"][i], est["Y"][i])
    dh = est["r_rep"][i]
    dt_ = math.hypot(est["X"][i] - geom.tow_xy(t)[0], est["Y"][i] - geom.tow_xy(t)[1])
    return (f"    {lbl} t={t:6.1f} s : ({lat:.4f}°N, {lon:.4f}°E)\n"
            f"         距舷侧阵 {dh / 1000:6.3f} km | 距拖曳阵 {dt_ / 1000:6.3f} km"
            f" | 舷角 hull {est['bh'][i]:6.2f}° / tow {est['bt'][i]:6.2f}°"
            f" | 视差 {est['par'][i]:.2f}°")


def motion(est):
    """运动段上的速度/航向估计 —— 通道分解:
         切向 v_t = r̄ · dθ/dt   (方位通道精确, 约 ±0.2°)
         径向 v_r = dr/dt        (距离通道; 只用中值滤波满窗历元, 否则不可观)
       返回 (速度 m/s, 航向°, t0, t1, v_r, v_t, radial_ok)。
    """
    run = est["run"]
    if run is None or run[1] - run[0] < 8:
        return None
    i0, i1 = run
    idx = np.arange(i0, i1 + 1)
    seg_t = est["ts"][idx] - est["ts"][i0]
    dt = float(seg_t[-1])
    if dt < 1e-6:
        return None
    A = np.stack([np.ones_like(seg_t), seg_t], 1)
    # 切向: 方位通道 (全段最小二乘斜率)
    cb, *_ = np.linalg.lstsq(A, np.radians(est["bh"][idx]), rcond=None)
    r_bar = float(np.mean(est["r_rep"][idx]))
    v_t = float(cb[1]) * r_bar
    # 径向: 距离通道, 仅限满窗历元; 不足则视为不可观 (v_r = 0)
    ok = est["val"][idx]
    radial_ok = int(ok.sum()) >= 10
    if radial_ok:
        Av = A[ok]
        cr, *_ = np.linalg.lstsq(Av, est["r_sm"][idx][ok], rcond=None)
        v_r = float(cr[1])
    else:
        v_r = 0.0
    a = math.radians(geom.heading + est["sg"] * float(np.mean(est["bh"][idx])))
    # 地理方位 a 的单位矢量 u=(sin a, cos a); 舷角增大方向的切向单位矢量 = sg*(cos a, -sin a)
    vx = v_r * math.sin(a) + v_t * est["sg"] * math.cos(a)
    vy = v_r * math.cos(a) - v_t * est["sg"] * math.sin(a)
    return (math.hypot(vx, vy), math.degrees(math.atan2(vx, vy)) % 360.0,
            float(est["ts"][i0]), float(est["ts"][i1]), v_r, v_t, radial_ok)


def branch_block(name, est):
    if est is None:
        return "    (该分支有效历元不足, 无法交会定距)"
    L = [f"  [{'右舷 +' if est['sg'] > 0 else '左舷 -'}分支]  "
         f"有效交会 {est['ts'][0]:.0f}~{est['ts'][-1]:.0f} s"
         f" ({est['n_acc']}/{est['n_all']} 历元, 视差≥{PAR_MIN}°)"
         f" | 双站视差 {est['par_rng'][0]:.2f}~{est['par_rng'][1]:.2f}"
         f" (中位 {est['par_med']:.2f})°",
         fmt_point(est, 0, "初始"), fmt_point(est, -1, "结束")]
    L.append(f"    舷角偏转 : hull {est['bh'][0]:.2f}° -> {est['bh'][-1]:.2f}°"
             f"  (Δ={est['bh'][-1] - est['bh'][0]:+.2f}°)")
    L.append(f"               tow  {est['bt'][0]:.2f}° -> {est['bt'][-1]:.2f}°"
             f"  (Δ={est['bt'][-1] - est['bt'][0]:+.2f}°)")
    r0, r1 = est["r_rep"][0], est["r_rep"][-1]
    rc, tc = cpa(est)
    L.append(f"    距离变化 : {r0 / 1000:.3f} -> {r1 / 1000:.3f} km"
             f"  (Δ={(r1 - r0) / 1000:+.3f} km)"
             f"   |   CPA {rc / 1000:.3f} km @ t={tc:.0f} s"
             f"   |   距离中位 {np.median(est['r_rep']) / 1000:.3f} km,"
             f" 单历元散布 σ {est['r_raw'].std(ddof=1) / 1000:.3f} km")
    mo = motion(est)
    if mo is not None:
        vm, cm, mt0, mt1, v_r, v_t, rok = mo
        L.append(f"    运动段   : t={mt0:.0f}~{mt1:.0f} s (方位持续变化, Δt={mt1 - mt0:.0f} s)"
                 f" | 拟合速度 {vm:.2f} m/s ({vm / B.KN2MS:.1f} kn)"
                 f" | 拟合航向 {cm:.1f}°")
        L.append(f"               切向 (r·dθ/dt) {v_t:+6.2f} m/s"
                 f" | 径向 (dr/dt) "
                 + (f"{v_r:+6.2f} m/s" if rok else "不可观 (置 0)")
                 + f" | 段内位移 {vm * (mt1 - mt0) / 1000:.3f} km")
    else:
        L.append("    运动段   : 方位近似恒定 (目标静止或已走完航点), 无可拟合运动段")
    disp = math.hypot(est["X"][-1] - est["X"][0], est["Y"][-1] - est["Y"][0])
    L.append(f"    首末位移 : {disp / 1000:.3f} km, 平均地速 "
             f"{disp / max(est['ts'][-1] - est['ts'][0], 1e-9):.2f} m/s"
             f" ({disp / max(est['ts'][-1] - est['ts'][0], 1e-9) / B.KN2MS:.1f} kn)")
    return "\n".join(L)


# ================= 4. 真值校验 =================
PAT = (r"^\s*(\S+)\s*\|\s*([\d.]+)\s*kn.*?start=\(([\d.]+)°N,([\d.]+)°E\)\s*"
       r"end=\(([\d.]+)°N,([\d.]+)°E\).*?舷角\s*([\d.]+)°->\s*([\d.]+)°.*?"
       r"CPA=\s*([\d.]+)\s*km\s*@\s*t=\s*(-?[\d.]+)s.*?观测窗\[([\d.]+),([\d.]+)\]s")


def parse_truth():
    out = {}
    for tag in ("hull", "tow"):
        txt = open(os.path.join(SRC, f"btr_A_{tag}.txt"), encoding="utf-8").read()
        for m in re.finditer(PAT, txt, re.M):
            g = m.groups()
            out.setdefault(g[0], {})[tag] = dict(
                kn=float(g[1]), s=(float(g[2]), float(g[3])), e=(float(g[4]), float(g[5])),
                b=(float(g[6]), float(g[7])), cpa=(float(g[8]), float(g[9])),
                win=(float(g[10]), float(g[11])))
    return out


def truth_block(truth, det, ests):
    L = ["", "-" * 96, "生成端真值 (btr_A_hull.txt / btr_A_tow.txt) 与检测对照:"]
    L.append(f"  {'真值':<6}{'检测':<9}{'舷角@300 检/真':>17}{'CPA 检/真 km':>18}"
             f"{'航速 检/真 kn':>17}{'航向 检/真':>16}")
    for tname, d in truth.items():
        h, t_ = d["hull"], d["tow"]
        best, bd = None, 1e9
        for aname, (h_tr, _) in det.items():
            k = nearest_key(h_tr, 300.0)
            if k is None:
                continue
            dd = abs(h_tr[k][0] - h["b"][1])
            if dd < bd:
                best, bd = (aname, h_tr, k), dd
        if best is None:
            L.append(f"  {tname:<6}{'未配对':<9}")
            continue
        aname, h_tr, k = best
        e = ests.get(aname)
        cs = ss = rs = "n/a"
        if e is not None:
            rc, tc = cpa(e)
            cs = f"{rc / 1000:.2f}"
            mo = motion(e)
            if mo is not None:
                ss = f"{mo[0] / B.KN2MS:.1f}"
                rs = f"{mo[1]:.0f}"
        _, l2xy = B.make_geo(30.0, 122.0)
        x0, y0 = l2xy(*h["s"])
        x1, y1 = l2xy(*h["e"])
        crs_t = math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0
        L.append(f"  {tname:<6}{aname:<9}{h_tr[k][0]:>9.2f}/{h['b'][1]:<7.1f}"
                 f"{cs + ' / ' + format(h['cpa'][0], '.2f'):>18}"
                 f"{ss + ' / ' + format(h['kn'], '.0f'):>17}"
                 f"{rs + ' / ' + format(crs_t, '.0f'):>16}")
        L.append(f"        真值 舷角 hull {h['b'][0]:.1f}°->{h['b'][1]:.1f}°"
                 f" (Δ={h['b'][1] - h['b'][0]:+.1f}°), tow {t_['b'][0]:.1f}°->{t_['b'][1]:.1f}°"
                 f" (Δ={t_['b'][1] - t_['b'][0]:+.1f}°)"
                 f" | 观测窗 [{h['win'][0]:.0f},{h['win'][1]:.0f}] s"
                 f" | 走完航点 t={h['cpa'][1]:.0f} s")
        L.append(f"        真值位置 ({h['s'][0]:.4f}°N,{h['s'][1]:.4f}°E) ->"
                 f" ({h['e'][0]:.4f}°N,{h['e'][1]:.4f}°E)"
                 f" | 真值 CPA hull {h['cpa'][0]:.2f} km @{h['cpa'][1]:.0f} s,"
                 f" tow {t_['cpa'][0]:.2f} km @{t_['cpa'][1]:.0f} s")
    return "\n".join(L)


# ================= 主流程 =================
def main():
    raw, shapes = load_bands()
    for tag in ("hull", "tow"):
        print(f"[{tag}] dB 图 {shapes[tag]}, 有亮带时刻 "
              f"{sum(1 for v in raw[tag].values() if v)}/{len(raw[tag])}, "
              f"亮带总数 {sum(len(v) for v in raw[tag].values())}")
    ans = anchors(raw)
    print(f"自动锚点 (t={ANCHOR_T:.0f} s, 峰值≥{ANCHOR_PEAK:.0f} dB): "
          + ", ".join(f"{n} hull {bh:.2f}°/tow {bt:.2f}° (视差 {abs(bh - bt):.2f}°)"
                      for n, bh, bt, _ in ans))

    rep = ["=" * 96,
           f"hull/tow 方位历程图目标检测报告  (btr_gens/btr_out/{BATCH})",
           f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 96,
           "平台 A: 起点 (30.0000°N, 122.0000°E), 航向 -30°, 航速 0 kn (静止)",
           "阵列: 舷侧阵 (0,0) m, 拖曳阵 (-180,0) m, 双站基线 180 m",
           f"方法: 等阈亮带亚像素中点定方位 (电平 {LEVEL:.1f} dB, 步长 {DT:.0f} s)"
           f" -> 双站视差交会 -> 距离中值滤波 ({RWIN} 历元)",
           "标定: 列 98..1262 = 0..180°, 行 738.6..22.4 = 0..600 s (由网格线反推)", ""]
    det, ests = {}, {}
    for name, bh, bt, ta in ans:
        h_tr = track_from(raw["hull"], ta, bh)
        t_tr = track_from(raw["tow"], ta, bt)
        det[name] = (h_tr, t_tr)
        rep.append(f"===== {name} =====")
        rep.append(f"  方位跟迹: hull {len(h_tr)} 历元 [{min(h_tr):.0f}~{max(h_tr):.0f} s]"
                   f" {h_tr[min(h_tr)][0]:.2f}°->{h_tr[max(h_tr)][0]:.2f}°"
                   f" | tow {len(t_tr)} 历元 [{min(t_tr):.0f}~{max(t_tr):.0f} s]"
                   f" {t_tr[min(t_tr)][0]:.2f}°->{t_tr[max(t_tr)][0]:.2f}°")
        pos_est = branch_est(h_tr, t_tr, +1)
        ests[name] = pos_est
        for sg in (+1, -1):
            rep.append(branch_block(name, branch_est(h_tr, t_tr, sg)))
        rep.append("")
    rep.append(truth_block(parse_truth(), det, ests))
    rep.append("")
    rep.append("注 1: 真值 txt 的舷角起止按 t=0/600 s 计算 (与观测窗无关); 目标只在观测窗内")
    rep.append("        出现在图上, 故检测到的初始舷角对应入场时刻 (T2 约 t=60 s)。")
    rep.append("注 2: 两阵共线于艏艉线 -> 左右舷镜像不可分辨, 两支的舷角/距离完全相同,")
    rep.append("        仅位置关于艏艉轴镜像; 距离由 180 m 基线的双站视差决定, 精度约 ±10%。")
    txt = "\n".join(rep)
    print(txt)
    if "--no-save" not in sys.argv:
        out = os.path.join(HERE, f"detect_report_{BATCH}.txt")
        with open(out, "w", encoding="utf-8") as fp:
            fp.write(txt + "\n")
        print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()

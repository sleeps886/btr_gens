#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTR 方位历程图仿真 —— 参数设置界面 (后端服务器)
================================================
功能:
  - 模式一: 单平台 (舷侧阵 + 拖曳阵) × 多目标
  - 模式二: 多平台 × 多目标
  - 目标路径: ① 航点列表  ② 曲线公式  ③ 前端鼠标划线取点(自动转航点)
  - 每个平台生成三张图 (track / hull / tow) + 三个同名 txt

运行:
    python3 btr_gui_server.py
浏览器打开:  http://127.0.0.1:8765

依赖: numpy, matplotlib (无其他第三方依赖)
"""
import json
import math
import os
import re
import threading
import traceback
from urllib.parse import unquote
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# =========================   仿真核心   ===============================
# =====================================================================
# 与 btr_geo_ll.py / btr_multi_platform.py 相同的运动学/波束形成逻辑,
# 改为函数式: run_simulation(params) 接收前端提交的 JSON 参数。

KN2MS = 0.514444

# 模块级默认值 (每次 run_simulation 时由参数覆盖)
T_MAX = 600.0; DT = 1.0; B_MIN = 0.0; B_MAX = 180.0; DB_STEP = 0.25
EARTH_R = 6371000.0; REF_LAT = 30.0; REF_LON = 122.0
NOISE_STD = 0.030; CLUTTER_LAM = 0.012; DB_RANGE = (-5.0, 16.0)
SEED = 20260911
OUT_DIR = "."


def ll2xy(lat, lon):
    lat0 = np.radians(REF_LAT); lon0 = np.radians(REF_LON)
    x = EARTH_R * np.cos(lat0) * (np.radians(lon) - lon0)
    y = EARTH_R * (np.radians(lat) - lat0)
    return x, y


def xy2ll(x, y):
    lat0 = np.radians(REF_LAT); lon0 = np.radians(REF_LON)
    lat = np.degrees(lat0 + y / EARTH_R)
    lon = np.degrees(lon0 + x / (EARTH_R * np.cos(lat0)))
    return lat, lon


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1; dlam = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlam/2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def gcbearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    y = np.sin(dlam) * np.cos(p2)
    x = (np.cos(p1)*np.sin(p2) - np.sin(p1)*np.cos(p2)*np.cos(dlam))
    return np.degrees(np.arctan2(y, x)) % 360.0


def fold_bearing(th):
    th = th % 360.0
    return 360.0 - th if th > 180.0 else th


class Path:
    """显式路径对象: p(t) -> (lat, lon); p.t_end 走完全程时刻(s)"""

    def __init__(self, fn, t_end=None, mode="waypoints"):
        self._fn = fn
        self.t_end = t_end
        self.mode = mode

    def __call__(self, t):
        return self._fn(t)

    def __repr__(self):
        return f"<Path mode={self.mode}, t_end={self.t_end}>"


def build_path(start_ll, speed_kn, waypoints=None, curve_func=None):
    if curve_func is not None:
        return Path(curve_func, t_end=None, mode="curve")
    if waypoints is None:
        raise ValueError("build_path: waypoints 与 curve_func 至少提供一个")
    pts = [tuple(start_ll)]
    for p in waypoints:
        p = tuple(p)
        if haversine_m(pts[-1][0], pts[-1][1], p[0], p[1]) < 1e-3:
            continue
        pts.append(p)
    if len(pts) < 2:
        def path0(t):
            return float(start_ll[0]), float(start_ll[1])
        return Path(path0, t_end=0.0, mode="waypoints")
    times = [0.0]
    v = speed_kn * KN2MS
    for i in range(1, len(pts)):
        d = haversine_m(pts[i-1][0], pts[i-1][1],
                        pts[i][0],  pts[i][1])
        times.append(times[-1] + d / max(v, 1e-6))
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    t_end = times[-1]

    def path(t):
        return (float(np.interp(t, times, lats)),
                float(np.interp(t, times, lons)))
    return Path(path, t_end=t_end, mode="waypoints")


# ---------- 曲线公式安全解析: 只允许数学表达式 ----------
_ALLOWED_NAMES = {"np", "math", "t", "pi", "e"}


def parse_curve_expr(expr):
    """把用户输入的曲线公式 (返回 (lat, lon) 的表达式) 安全编译为 f(t)"""
    import ast
    expr = expr.strip()
    if not expr:
        raise ValueError("曲线公式为空")
    tree = ast.parse(expr, mode="eval")
    allowed_types = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                     ast.Name, ast.Load, ast.Call, ast.Attribute,
                     ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                     ast.USub, ast.UAdd, ast.Tuple, ast.List, ast.IfExp,
                     ast.Compare, ast.Lt, ast.Gt, ast.Eq, ast.NotEq,
                     ast.LtE, ast.GtE, ast.And, ast.Or, ast.Not)
    for node in ast.walk(tree):
        if not isinstance(node, allowed_types):
            raise ValueError(f"曲线公式包含不允许的语法: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"曲线公式引用了不允许的名字: {node.id}")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or \
                    node.value.id not in ("np", "math"):
                raise ValueError("曲线公式只允许调用 np.* / math.*")
    code = compile(f"def f(t):\n    return ({expr})", "<curve>", "exec")
    env = {"np": np, "math": math, "pi": math.pi, "e": math.e,
           "__builtins__": {}}
    exec(code, env)
    return env["f"]


def check_bounds(pos_fn, bounds, label, t_arr, log):
    if bounds is None:
        return
    lat = np.array([pos_fn(t)[0] for t in t_arr])
    lon = np.array([pos_fn(t)[1] for t in t_arr])
    la_lo, la_hi = bounds["lat"]; lo_lo, lo_hi = bounds["lon"]
    if (lat.min() < la_lo or lat.max() > la_hi
            or lon.min() < lo_lo or lon.max() > lo_hi):
        log(f"[WARN] {label} 轨迹超出经纬度 bounds: "
            f"lat=[{lat.min():.4f},{lat.max():.4f}] "
            f"(限 {la_lo},{la_hi}), "
            f"lon=[{lon.min():.4f},{lon.max():.4f}] "
            f"(限 {lo_lo},{lo_hi})")


def build_platform(pc):
    """构造平台对象: 位置函数 / 航向 / 两阵位置函数"""
    x0, y0 = ll2xy(*pc["start_ll"])
    psi = np.radians(90.0 - pc["heading_deg"])
    v = pc["speed_kn"] * KN2MS

    def plat_xy(t):
        return x0 + v*np.cos(psi)*t, y0 + v*np.sin(psi)*t

    def plat_ll(t):
        return xy2ll(*plat_xy(t))

    def array_xy(t, off):
        px, py = plat_xy(t)
        c, s = np.cos(psi), np.sin(psi)
        ox, oy = off
        return px + ox*c - oy*s, py + ox*s + oy*c

    def hull_ll(t):
        return xy2ll(*array_xy(t, pc["hull_m"]))

    def tow_ll(t):
        return xy2ll(*array_xy(t, pc["tow_m"]))

    return {"name": pc["name"], "start_ll": pc["start_ll"],
            "heading": float(pc["heading_deg"]), "plat_ll": plat_ll,
            "hull_ll": hull_ll, "tow_ll": tow_ll,
            "hull_m": pc["hull_m"], "tow_m": pc["tow_m"],
            "speed_kn": pc["speed_kn"], "bounds": pc.get("bounds"),
            "color": pc.get("color", "#ffd400")}


def rel_bearing(t, tgt_fn, array_fn, heading):
    """目标相对船头舷角 (船首=0°, 正横=90°, 正尾=180°, 折叠 0~180°)"""
    lat_t, lon_t = tgt_fn(t)
    lat_a, lon_a = array_fn(t)
    gb = gcbearing(lat_a, lon_a, lat_t, lon_t)
    rel = (gb - heading) % 360.0
    return fold_bearing(rel)


def build_btr(array_fn, sigma_key, seed_off, heading, tgt_ll, t_arr):
    r = np.random.default_rng(SEED + seed_off)
    br = np.arange(B_MIN, B_MAX + DB_STEP, DB_STEP)
    S = (r.normal(0.0, NOISE_STD, size=(len(br), len(t_arr)))
         + 0.02*np.sin(2*np.pi*t_arr/180.0)[None, :]
         * np.cos(2*np.pi*br/120.0)[:, None]
         + r.exponential(CLUTTER_LAM, size=(len(br), len(t_arr))))
    for g, pf in zip(tgt_ll, [t["fn"] for t in tgt_ll]):
        sig = g[sigma_key]
        for k, tk in enumerate(t_arr):
            if tk < g["t_on"] or tk > g["t_off"]:
                continue
            bk = rel_bearing(tk, pf, array_fn, heading)
            peak = g["amp"] * (1.0 + 0.2*r.standard_normal())
            S[:, k] += peak * np.exp(-0.5*((br - bk)/sig)**2)
    S = np.clip(S, 1e-4, None)
    P0 = np.median(S)
    return br, np.clip(10.0*np.log10(S/P0), *DB_RANGE)


def _cpa(pf, array_fn):
    dmin, tmin = 1e18, 0.0
    for tk in np.linspace(0, T_MAX, 2000):
        d = haversine_m(*array_fn(tk), *pf(tk))
        if d < dmin:
            dmin, tmin = d, tk
    return dmin/1000.0, tmin


# ---------- 主仿真入口 ----------
def run_simulation(p):
    """执行仿真, 返回 {ok, platforms:[{name, imgs, txts}], log}"""
    global T_MAX, DT, B_MIN, B_MAX, DB_STEP, EARTH_R, REF_LAT, REF_LON
    global NOISE_STD, CLUTTER_LAM, DB_RANGE, SEED, OUT_DIR

    logs = []
    def log(msg):
        logs.append(str(msg))
        print(msg)

    # ---- 读取公共参数 ----
    from btr_batch import validate_params
    validate_params(p)
    T_MAX = float(p.get("T_MAX", 600.0))
    DT    = float(p.get("DT", 1.0))
    B_MIN = float(p.get("B_MIN", 0.0))
    B_MAX = float(p.get("B_MAX", 180.0))
    DB_STEP = float(p.get("DB_STEP", 0.25))
    EARTH_R = float(p.get("EARTH_R", 6371000.0))
    REF_LAT = float(p.get("REF_LAT", 30.0))
    REF_LON = float(p.get("REF_LON", 122.0))
    NOISE_STD = float(p.get("NOISE_STD", 0.030))
    CLUTTER_LAM = float(p.get("CLUTTER_LAM", 0.012))
    DB_RANGE = (float(p.get("DB_RANGE", [-5, 16])[0]),
                float(p.get("DB_RANGE", [-5, 16])[1]))
    SEED = int(p.get("SEED", 20260911))

    # 输出目录: 未指定则自动创建带时间戳子目录, 避免覆盖旧结果
    user_out = (p.get("OUT_DIR") or "").strip()
    if user_out:
        OUT_DIR = user_out
        os.makedirs(OUT_DIR, exist_ok=True)
    else:
        OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "btr_out",
                               datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 平台 ----
    platforms_raw = p.get("platforms") or []
    if not platforms_raw:
        raise ValueError("未设置任何平台")
    if len(platforms_raw) > 8:
        raise ValueError("平台数量过多 (最多 8 个)")
    platforms = []
    for i, pc in enumerate(platforms_raw):
        start = [float(pc["start_ll"][0]), float(pc["start_ll"][1])]
        pl = {"name": str(pc["name"]) or f"P{i+1}",
              "start_ll": tuple(start),
              "heading_deg": float(pc["heading_deg"]),
              "speed_kn": float(pc["speed_kn"]),
              "hull_m": (float(pc["hull_m"][0]), float(pc["hull_m"][1])),
              "tow_m": (float(pc["tow_m"][0]), float(pc["tow_m"][1])),
              "bounds": pc.get("bounds") or None,
              "color": pc.get("color", "#ffd400")}
        # bounds 深度转换
        if pl["bounds"]:
            pl["bounds"] = {"lat": tuple(float(x) for x in pl["bounds"]["lat"]),
                            "lon": tuple(float(x) for x in pl["bounds"]["lon"])}
        platforms.append(build_platform(pl))

    # ---- 目标 ----
    targets_raw = p.get("targets") or []
    if not targets_raw:
        raise ValueError("未设置任何目标")
    if len(targets_raw) > 12:
        raise ValueError("目标数量过多 (最多 12 个)")
    t_arr = np.arange(0, T_MAX + 1e-9, DT)
    tgt_ll = []      # [ {**g, "fn": Path} ]
    for j, g in enumerate(targets_raw):
        start = [float(g["start"][0]), float(g["start"][1])]
        speed = float(g["speed_kn"])
        path_type = str(g.get("path_type", "waypoints"))
        curve_func = None
        waypoints = None
        if path_type == "curve":
            expr = str(g.get("curve_func", "")).strip()
            curve_func = parse_curve_expr(expr)
            log(f"[INFO] {g['name']} 使用 curve 模式: "
                f"速度由曲线导数决定, speed_kn 不生效")
        else:
            wps = g.get("waypoints") or []
            waypoints = [tuple(float(x) for x in wp) for wp in wps]
        fn = build_path(start, speed, waypoints, curve_func)
        check_bounds(fn, g.get("bounds"), g["name"], t_arr, log)
        if fn.t_end is not None and fn.t_end > float(g["t_off"]):
            log(f"[INFO] {g['name']} 航点全程需 {fn.t_end:.0f}s, "
                f"大于 t_off={g['t_off']:.0f}s: "
                f"目标将在 t={g['t_off']:.0f}s 停止显示")
        tgt_ll.append({
            "name": str(g["name"]) or f"T{j+1}",
            "fn": fn, "speed": speed,
            "start": tuple(start),
            "amp": float(g["amp"]),
            "sigma_hull": float(g["sigma_hull"]),
            "sigma_tow": float(g["sigma_tow"]),
            "t_on": float(g["t_on"]), "t_off": float(g["t_off"]),
            "color": g.get("color", "#ff9d00"),
            "t_end": fn.t_end,
        })

    # ---- 逐平台渲染 ----
    ttt = np.linspace(0, T_MAX, 1200)
    colors = [(0,0,0),(0,0,0.35),(0,0.15,0.5),(0,0.45,0.75),
              (0.2,0.85,0.85),(0.95,0.95,0.25),(1,0.55,0.1),
              (1,0.2,0),(1,0,0)]
    cmap = LinearSegmentedColormap.from_list("sonar", colors, N=256)

    results = []
    for idx, plat in enumerate(platforms):
        name = plat["name"]
        log("="*84)
        log(f"[平台 {name}] start=({plat['start_ll'][0]:.4f}°N,"
            f"{plat['start_ll'][1]:.4f}°E) 航向={plat['heading']:.0f}°"
            f"(北=0°顺时针) speed={plat['speed_kn']:.1f} kn "
            f"拖曳基线={abs(plat['tow_m'][0]):.0f} m")
        log("="*84)
        check_bounds(plat["plat_ll"], plat["bounds"], f"平台{name}", t_arr, log)

        if p.get("EXPORT_TRUTH", False):
            truth_path = os.path.join(OUT_DIR, f"btr_{name}_truth.jsonl")
            with open(truth_path, "w", encoding="utf-8") as f:
                for tk in t_arr:
                    platform_ll = plat["plat_ll"](float(tk))
                    for g in tgt_ll:
                        if not g["t_on"] <= tk <= g["t_off"]:
                            continue
                        lat, lon = g["fn"](float(tk))
                        hlat, hlon = plat["hull_ll"](float(tk))
                        tlat, tlon = plat["tow_ll"](float(tk))
                        point = {"platform_id": name, "target_id": g["name"],
                                 "time_s": float(tk), "lat": float(lat), "lon": float(lon),
                                 "platform_lat": float(platform_ll[0]),
                                 "platform_lon": float(platform_ll[1]),
                                 "hull_bearing_deg": float(rel_bearing(tk, g["fn"], plat["hull_ll"], plat["heading"])),
                                 "tow_bearing_deg": float(rel_bearing(tk, g["fn"], plat["tow_ll"], plat["heading"])),
                                 "hull_range_m": float(haversine_m(lat, lon, hlat, hlon)),
                                 "tow_range_m": float(haversine_m(lat, lon, tlat, tlon)),
                                 "depth_m": None, "source": "simulation_truth"}
                        f.write(json.dumps(point, ensure_ascii=False, allow_nan=False) + "\n")

        # BTR
        br_h, B_hull = build_btr(plat["hull_ll"], "sigma_hull",
                                 1+idx*10, plat["heading"], tgt_ll, t_arr)
        br_t, B_tow = build_btr(plat["tow_ll"], "sigma_tow",
                                2+idx*10, plat["heading"], tgt_ll, t_arr)

        # 摘要与数据收集
        records = []
        for g in tgt_ll:
            bh0 = rel_bearing(0, g["fn"], plat["hull_ll"], plat["heading"])
            bhe = rel_bearing(T_MAX, g["fn"], plat["hull_ll"], plat["heading"])
            bt0 = rel_bearing(0, g["fn"], plat["tow_ll"], plat["heading"])
            bte = rel_bearing(T_MAX, g["fn"], plat["tow_ll"], plat["heading"])
            cpah, tch = _cpa(g["fn"], plat["hull_ll"])
            cpat, tct = _cpa(g["fn"], plat["tow_ll"])
            rec = {"name": g["name"], "speed": g["speed"],
                   "lat0": g["start"][0], "lon0": g["start"][1],
                   "latE": g["fn"](T_MAX)[0], "lonE": g["fn"](T_MAX)[1],
                   "bh0": bh0, "bhe": bhe, "bt0": bt0, "bte": bte,
                   "cpa_h": cpah, "tc_h": tch, "cpa_t": cpat, "tc_t": tct,
                   "amp": g["amp"], "t_on": g["t_on"], "t_off": g["t_off"],
                   "t_end": g["t_end"]}
            records.append(rec)
            log(f"  {rec['name']:12s} | {rec['speed']:4.0f} kn "
                f"({rec['speed']*KN2MS:4.2f} m/s) | "
                f"弦测舷角 {bh0:5.1f}°->{bhe:5.1f}° | "
                f"拖曳舷角 {bt0:5.1f}°->{bte:5.1f}° | "
                f"CPA弦测={cpah:5.2f} km @ t={tch:4.0f}s")
        bh3 = rel_bearing(300, tgt_ll[min(2, len(tgt_ll)-1)]["fn"],
                          plat["hull_ll"], plat["heading"])
        bt3 = rel_bearing(300, tgt_ll[min(2, len(tgt_ll)-1)]["fn"],
                          plat["tow_ll"], plat["heading"])
        d_parallax = abs(bh3 - bt3)
        log(f"T3 @ t=300s: 弦测阵舷角={bh3:.1f}°, "
            f"拖曳阵舷角={bt3:.1f}°, 双站视差 |Δ|={d_parallax:.1f}°")

        # 理论航迹
        bh_curves = [np.array([rel_bearing(tk, g["fn"], plat["hull_ll"],
                                           plat["heading"]) for tk in ttt])
                     for g in tgt_ll]
        bt_curves = [np.array([rel_bearing(tk, g["fn"], plat["tow_ll"],
                                           plat["heading"]) for tk in ttt])
                     for g in tgt_ll]

        # ---- 图1: 轨迹 ----
        fig1 = plt.figure(figsize=(11, 6.5), dpi=150, facecolor="black")
        ax0 = fig1.add_subplot(111)
        pl_la = [plat["plat_ll"](tk)[0] for tk in ttt]
        pl_lo = [plat["plat_ll"](tk)[1] for tk in ttt]
        tw_la = [plat["tow_ll"](tk)[0] for tk in ttt]
        tw_lo = [plat["tow_ll"](tk)[1] for tk in ttt]
        ax0.plot(pl_lo, pl_la, "w-", lw=1.8)
        ax0.plot(tw_lo, tw_la, "c--", lw=1.0)
        ax0.plot(pl_lo[0], pl_la[0], "s", color=plat["color"], ms=6)
        for g in tgt_ll:
            la = [g["fn"](tk)[0] for tk in ttt]
            lo = [g["fn"](tk)[1] for tk in ttt]
            ax0.plot(lo, la, "-", lw=1.0, alpha=0.9, color=g["color"])
            ax0.plot(lo[0], la[0], "o", ms=5, color=g["color"])
        ax0.set_xlabel("经度 Longitude (°E)", fontsize=10)
        ax0.set_ylabel("纬度 Latitude (°N)", fontsize=10)
        ax0.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
        ax0.set_facecolor("#0a0f1f")
        out1 = os.path.join(OUT_DIR, f"btr_{name}_track.png")
        fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig1)

        # ---- 图2: 舷侧阵(舰壳) BTR ----
        fig2 = plt.figure(figsize=(12, 6.2), dpi=150, facecolor="black")
        ax1 = fig2.add_subplot(111)
        im1 = ax1.pcolormesh(br_h, t_arr, B_hull.T, cmap=cmap, shading="auto",
                             vmin=DB_RANGE[0], vmax=DB_RANGE[1])
        ax1.set_xlim(B_MIN, B_MAX)
        ax1.set_xticks(np.linspace(B_MIN, B_MAX, 7))
        ax1.set_ylim(0, T_MAX)
        ax1.set_yticks(np.arange(0, int(T_MAX)+1, max(1, int(T_MAX)//10)))
        ax1.set_xlabel("方位 Bearing (°)", fontsize=11)
        ax1.set_ylabel("时间 Time (s)", fontsize=10)
        ax1.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
        for g, bh in zip(tgt_ll if p.get("SHOW_TRUTH_OVERLAY", True) else [], bh_curves):
            m = (ttt >= g["t_on"]) & (ttt <= g["t_off"])
            ax1.plot(bh[m], ttt[m], "w-", lw=0.8, alpha=0.7)
        cbar1 = fig2.colorbar(im1, ax=ax1, pad=0.015)
        cbar1.set_label("波束输出强度 (dB, 相对噪声底)", fontsize=10)
        out2 = os.path.join(OUT_DIR, f"btr_{name}_hull.png")
        fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig2)

        # ---- 图3: 拖曳阵 BTR ----
        fig3 = plt.figure(figsize=(12, 6.2), dpi=150, facecolor="black")
        ax2 = fig3.add_subplot(111)
        im2 = ax2.pcolormesh(br_t, t_arr, B_tow.T, cmap=cmap, shading="auto",
                             vmin=DB_RANGE[0], vmax=DB_RANGE[1])
        ax2.set_xlim(B_MIN, B_MAX)
        ax2.set_xticks(np.linspace(B_MIN, B_MAX, 7))
        ax2.set_ylim(0, T_MAX)
        ax2.set_yticks(np.arange(0, int(T_MAX)+1, max(1, int(T_MAX)//10)))
        ax2.set_xlabel("方位 Bearing (°)", fontsize=11)
        ax2.set_ylabel("时间 Time (s)", fontsize=10)
        ax2.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
        for g, bh, bt in zip(tgt_ll if p.get("SHOW_TRUTH_OVERLAY", True) else [], bh_curves, bt_curves):
            m = (ttt >= g["t_on"]) & (ttt <= g["t_off"])
            ax2.plot(bh[m], ttt[m], "w-", lw=0.8, alpha=0.7)
            ax2.plot(bt[m], ttt[m], "y--", lw=0.8, alpha=0.8)
        cbar2 = fig3.colorbar(im2, ax=ax2, pad=0.015)
        cbar2.set_label("波束输出强度 (dB, 相对噪声底)", fontsize=10)
        out3 = os.path.join(OUT_DIR, f"btr_{name}_tow.png")
        fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig3)

        # ---- txt ----
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _fmt(rec, tag):
            if tag == "hull":
                b0, be, cpa, tc = rec["bh0"], rec["bhe"], rec["cpa_h"], rec["tc_h"]
            else:
                b0, be, cpa, tc = rec["bt0"], rec["bte"], rec["cpa_t"], rec["tc_t"]
            extra = (f" | 走完航点 t={rec['t_end']:.0f}s"
                     if rec["t_end"] is not None else "")
            return (f"  {rec['name']:12s} | {rec['speed']:4.0f} kn "
                    f"({rec['speed']*KN2MS:4.2f} m/s) | "
                    f"start=({rec['lat0']:.4f}°N,{rec['lon0']:.4f}°E) "
                    f"end=({rec['latE']:.4f}°N,{rec['lonE']:.4f}°E) | "
                    f"舷角 {b0:5.1f}°->{be:5.1f}° | "
                    f"CPA={cpa:5.2f} km @ t={tc:4.0f}s | "
                    f"amp={rec['amp']:.2f} | "
                    f"观测窗[{rec['t_on']:.0f},{rec['t_off']:.0f}]s{extra}")

        L1 = ["="*68,
              f"图像文件: btr_{name}_track.png (经纬度平面运动轨迹, 平台{name})",
              f"生成时间: {now}", "="*68,
              "[仿真设置]",
              f"  观测时长: {T_MAX:.0f} s, 时间步长: {DT:.1f} s",
              f"  方位范围: {B_MIN:.0f}~{B_MAX:.0f} deg "
              f"(相对船头舷角, 船首=0°, 正横=90°, 正尾=180°)",
              f"  地球半径: {EARTH_R:.0f} m, 投影基准: lat={REF_LAT}, lon={REF_LON}",
              "", "[平台信息]",
              f"  平台名: {name}",
              f"  初始经纬度: ({plat['start_ll'][0]:.4f}°N, "
              f"{plat['start_ll'][1]:.4f}°E)",
              f"  船头朝向(航向): {plat['heading']:.0f}° "
              f"(地理方位角, 北=0°顺时针)",
              f"  航速: {plat['speed_kn']:.1f} kn "
              f"({plat['speed_kn']*KN2MS:.2f} m/s)",
              "", "[阵列配置]",
              f"  舷侧阵(舰壳): 相对偏移 {plat['hull_m'][0]:.1f},"
              f"{plat['hull_m'][1]:.1f} m",
              f"  拖曳阵: 相对偏移 {plat['tow_m'][0]:.1f},"
              f"{plat['tow_m'][1]:.1f} m "
              f"(基线 {abs(plat['tow_m'][0]):.0f} m, 随船头航向旋转)",
              "", "[目标信息]"]
        for rec in records:
            L1.append(f"  {rec['name']:12s} | {rec['speed']:4.0f} kn "
                      f"({rec['speed']*KN2MS:4.2f} m/s) | "
                      f"start=({rec['lat0']:.4f}°N,{rec['lon0']:.4f}°E) "
                      f"end=({rec['latE']:.4f}°N,{rec['lonE']:.4f}°E) | "
                      f"amp={rec['amp']:.2f} | "
                      f"观测窗[{rec['t_on']:.0f},{rec['t_off']:.0f}]s")
        txt1 = os.path.join(OUT_DIR, f"btr_{name}_track.txt")
        with open(txt1, "w", encoding="utf-8") as f:
            f.write("\n".join(L1) + "\n")

        L2 = ["="*68,
              f"图像文件: btr_{name}_hull.png (舷侧阵方位历程图, 平台{name})",
              f"生成时间: {now}", "="*68,
              "[目标相对舷侧阵的舷角与距离信息] "
              "(舷角: 船首=0°, 正横=90°, 正尾=180°)"]
        for rec in records:
            L2.append(_fmt(rec, "hull"))
        txt2 = os.path.join(OUT_DIR, f"btr_{name}_hull.txt")
        with open(txt2, "w", encoding="utf-8") as f:
            f.write("\n".join(L2) + "\n")

        L3 = ["="*68,
              f"图像文件: btr_{name}_tow.png (拖曳线列阵方位历程图, 平台{name})",
              f"生成时间: {now}", "="*68,
              "[目标相对拖曳阵的舷角与距离信息]"]
        for rec in records:
            L3.append(_fmt(rec, "tow"))
        L3 += ["-"*68, "[双站视差对比 (t=300s)]",
               f"  T3 近/高速: 舷侧阵舷角={bh3:.1f}°, "
               f"拖曳阵舷角={bt3:.1f}°, 双站视差 |Δ|={d_parallax:.1f}°",
               f"  说明: 基线 {abs(plat['tow_m'][0]):.0f} m, "
               f"近距离目标 (CPA≈{records[min(2,len(records)-1)]['cpa_h']:.2f} km) "
               f"视差显著, 远距离目标两阵舷角趋于一致"]
        txt3 = os.path.join(OUT_DIR, f"btr_{name}_tow.txt")
        with open(txt3, "w", encoding="utf-8") as f:
            f.write("\n".join(L3) + "\n")

        log(f"OK -> {name}: track/hull/tow 图 + txt")
        results.append({
            "name": name,
            "imgs": {"track": out1, "hull": out2, "tow": out3},
            "txts": {"track": txt1, "hull": txt2, "tow": txt3},
            "parallax": d_parallax,
        })

    log(f"\n输出目录: {OUT_DIR}")
    return {"ok": True, "platforms": results, "out_dir": OUT_DIR,
            "log": "\n".join(logs)}


# =====================================================================
# =========================    HTTP 服务   =============================
# =====================================================================
HOST = "127.0.0.1"
PORT = 8765
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_LOCK = threading.Lock()
from btr_auto import StudioBatchManager
BATCH_ROOT = os.path.join(BASE_DIR, "btr_out")
BATCHES = StudioBatchManager(BATCH_ROOT, run_simulation, RUN_LOCK)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/batch/active":
            with BATCHES.lock:
                active_ids = [j['id'] for j in BATCHES.jobs.values()
                              if j['state'] in ('queued', 'running', 'stopping')]
            active = [BATCHES.status(job_id) for job_id in active_ids]
            self._send(200, json.dumps({"jobs": active}, ensure_ascii=False))
            return
        if path.startswith("/batch/status/"):
            try:
                self._send(200, json.dumps(BATCHES.status(path.rsplit("/", 1)[-1]), ensure_ascii=False))
            except KeyError:
                self._send(404, json.dumps({"ok": False, "error": "批次不存在"}, ensure_ascii=False))
            return
        if path.startswith("/batch/files/"):
            relative = unquote(path[len("/batch/files/"):])
            root = os.path.realpath(BATCH_ROOT)
            fp = os.path.realpath(os.path.join(root, relative))
            if os.path.commonpath([root, fp]) != root or not os.path.isfile(fp):
                self._send(404, "not found", "text/plain")
                return
            ctype = "image/png" if fp.endswith(".png") else "application/json; charset=utf-8" if fp.endswith(".json") else "application/octet-stream"
            with open(fp, "rb") as f:
                self._send(200, f.read(), ctype)
            return
        studio_assets = {
            "/studio/advanced": ("btr_studio.html", "text/html; charset=utf-8"),
            "/quick.css": ("btr_quick.css", "text/css; charset=utf-8"),
            "/quick.js": ("btr_quick.js", "text/javascript; charset=utf-8"),
            "/studio": ("btr_quick.html", "text/html; charset=utf-8"),
            "/studio/": ("btr_quick.html", "text/html; charset=utf-8"),
            "/studio.js": ("btr_studio.js", "text/javascript; charset=utf-8"),
            "/studio.css": ("btr_studio.css", "text/css; charset=utf-8"),
        }
        if path == "/studio/defaults":
            from btr_studio import default_config
            self._send(200, json.dumps(default_config(), ensure_ascii=False))
            return
        if path in studio_assets:
            filename, ctype = studio_assets[path]
            with open(os.path.join(BASE_DIR, filename), "rb") as f:
                self._send(200, f.read(), ctype)
            return
        if path == "/" or path == "/index.html":
            try:
                with open(os.path.join(BASE_DIR, "btr_gui_index.html"),
                          "rb") as f:
                    self._send(200, f.read(),
                               "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, "缺少前端文件 btr_gui_index.html")
            return
        if path.startswith("/files/"):
            rel = path[len("/files/"):]
            fp = os.path.abspath(os.path.join(OUT_DIR, rel))
            root = os.path.realpath(OUT_DIR)
            if os.path.commonpath([root, os.path.realpath(fp)]) == root and os.path.isfile(fp):
                if fp.lower().endswith(".png"):
                    ctype = "image/png"
                elif fp.lower().endswith(".txt"):
                    ctype = "text/plain; charset=utf-8"
                else:
                    ctype = "application/octet-stream"
                with open(fp, "rb") as f:
                    self._send(200, f.read(), ctype)
            else:
                self._send(404, "file not found", "text/plain")
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path == "/studio/preview":
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 200000:
                    raise ValueError("配置请求大小无效")
                from btr_studio import preview_config
                result = preview_config(json.loads(self.rfile.read(length).decode("utf-8")))
                self._send(200, json.dumps(result, ensure_ascii=False))
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                self._send(400, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return
        if self.path in ("/batch/start", "/batch/stop"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000:
                    raise ValueError("请求体大小无效")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("请求体须为对象")
                job = BATCHES.start(payload) if self.path.endswith("start") else BATCHES.stop(payload['id'])
                self._send(200, json.dumps({"ok": True, "job": job}, ensure_ascii=False))
            except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
                self._send(400, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return
        if self.path != "/run":
            self._send(404, "not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            params = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send(400, json.dumps({"ok": False,
                                        "error": f"参数解析失败: {e}"}))
            return
        with RUN_LOCK:
            try:
                res = run_simulation(params)
                # 图/文件 URL 供前端展示
                for pl in res["platforms"]:
                    for key in ("track", "hull", "tow"):
                        from urllib.parse import quote
                        for field in ("imgs", "txts"):
                            filename = os.path.abspath(pl[field][key])
                            if os.path.commonpath([os.path.abspath(BATCH_ROOT), filename]) == os.path.abspath(BATCH_ROOT):
                                relative = os.path.relpath(filename, BATCH_ROOT).replace(os.sep, "/")
                                pl[field][key] = "/batch/files/" + quote(relative)
                            else:
                                pl[field][key] = "/files/" + quote(os.path.basename(filename))
                res["out_dir"] = os.path.basename(res["out_dir"])
                self._send(200, json.dumps(res, ensure_ascii=False))
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                self._send(500, json.dumps({"ok": False,
                                            "error": str(e),
                                            "trace": tb[-1500:]},
                                           ensure_ascii=False))


def main():
    # Windows SO_REUSEADDR can otherwise bind alongside an old server and
    # deliver new HTML with requests still routed to the old backend.
    ThreadingHTTPServer.allow_reuse_address = False
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("="*60)
    print("BTR 仿真参数设置界面已启动")
    print(f"  请用浏览器打开:  http://{HOST}:{PORT}")
    print("  按 Ctrl+C 停止服务")
    print("="*60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()

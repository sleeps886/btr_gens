# -*- coding: utf-8 -*-
"""
多平台 × 多目标 拖曳阵方位历程图 (BTR) —— 经纬度坐标版
============================================================
- 多个平台分别位于不同位置、拥有不同航向(船头朝向)。
  每个平台携带两套阵列:
      弦测阵 (hull): 舰体中心
      拖曳阵 (tow):  舰艉正后方, 基线可设 (默认 150 m, 随船头航向旋转)
- 同一批水下目标在区域内按各自轨迹航行 (支持 waypoints / curve 两种模式)。
- 对【每个平台】独立计算目标相对本平台船头的舷角
      (船首=0°, 正横=90°, 正尾=180°; 左右舷模糊折叠到 0~180°),
  生成三张图 + 三个同名 txt, 分开存储, 文件名明确平台:
      btr_{平台名}_track.png / .txt   经纬度平面运动轨迹
      btr_{平台名}_hull.png  / .txt   弦测阵 BTR (横轴=方位 0~180°, 纵轴=时间)
      btr_{平台名}_tow.png   / .txt   拖曳阵 BTR
- 所有可调参数集中在下方【参数区】, 按需修改即可。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# =======================    参 数 区 (可调)    =======================
# =====================================================================

# ---------- 仿真总览 ----------
T_MAX    = 600.0      # 仿真时长 (s)
DT       = 1.0        # 时间步长 (s)
B_MIN    = 0.0        # 方位范围下限 (度, 折叠后)
B_MAX    = 180.0      # 方位范围上限 (度)
DB_STEP  = 0.25       # 方位分辨率 (度)

# ---------- 地球模型 ----------
EARTH_R   = 6371000.0          # 地球半径 (m)
REF_LAT   = 30.0               # 投影基准纬度 (度)
REF_LON   = 122.0              # 投影基准经度 (度)

# ---------- 平台列表 (每个平台一套阵列, 各自位置 + 航向) ----------
# start_ll:    平台初始经纬度 (lat, lon)
# heading_deg: 平台航向 = 船头朝向, 使用【地理方位角】:
#              北=0° 顺时针, 东=90°, 南=180°, 西=270°
#              (东南≈135°, 西北≈315°; 例 "东南30°方向" ≈ 方位角 30°)
# speed_kn:    平台航速 (节); =0 表示平台静止
#              (仅船头朝向有效, 拖曳阵航向 = 平台航向仍起作用)
# hull_m / tow_m: 两套阵列相对舰体的偏移 (m, 随船头航向旋转)
# bounds:      平台经纬度运动范围 (可选, None 则不检查)
# color:       轨迹图中该平台航迹的颜色
#
# 位置换算示例 (相对参考点 30.0°N,122.0°E, 方位角 az, 距离 d km):
#   lat = 30.0 + d*cos(az)/111.19
#   lon = 122.0 + d*sin(az)/(111.32*cos(30°))
PLATFORMS = [
    {"name": "A",            # 平台名 (用于文件名与 txt 标注)
     "start_ll": (30.0623, 122.0415),   # 参考点东北偏东 8km (方位角 30°)
     "heading_deg": 30.0,               # 船头朝向 方位角 30° (东北偏东)
     "speed_kn": 0.0,                   # 静止 (改为 >0 则沿航向匀速航行)
     "hull_m": (0.0, 0.0),
     "tow_m": (-150.0, 0.0),
     "bounds": {"lat": (29.90, 30.15), "lon": (121.80, 122.30)},
     "color": "#ffd400"},

    {"name": "B",
     "start_ll": (30.0509, 121.9413),   # 参考点西北 8km (方位角 315°)
     "heading_deg": 315.0,              # 船头朝向 方位角 315° (西北)
     "speed_kn": 0.0,
     "hull_m": (0.0, 0.0),
     "tow_m": (-150.0, 0.0),
     "bounds": {"lat": (29.90, 30.15), "lon": (121.80, 122.30)},
     "color": "#00ffcc"},

    {"name": "C",
     "start_ll": (29.9377, 122.0415),   # 参考点东南偏南 8km (方位角 150°)
     "heading_deg": 150.0,              # 船头朝向 方位角 150° (东南偏南)
     "speed_kn": 0.0,
     "hull_m": (0.0, 0.0),
     "tow_m": (-150.0, 0.0),
     "bounds": {"lat": (29.90, 30.15), "lon": (121.80, 122.30)},
     "color": "#ff6b6b"},
]

# ---------- 目标列表 (同一批目标, 区域不同位置, 各自轨迹) ----------
# start:     初始经纬度 (lat, lon)
# speed_kn:  航速 (节)
# path_type: 'waypoints' (按 speed_kn 匀速过航点) / 'curve' (曲线函数)
# waypoints: 航点列表 (lat, lon)
# curve_func: f(t) -> (lat, lon) 曲线函数
# amp:       目标强度 (相对噪声底, 即相对强度)
# sigma_hull / sigma_tow: 两阵波束主瓣宽度 (度)
# t_on / t_off: 目标出现与消失时刻 (s)
TARGETS = [
    {
        "name": "T1 远/相向",
        "start":  (30.0360, 122.1240),   # 参考点东北 12 km / 4 km
        "speed_kn": 19.0,                # ≈ 9.77 m/s, 向西
        "path_type": "waypoints",
        "waypoints": [(30.0360, 122.0630)],
        "curve_func": None,
        "amp": 0.30, "sigma_hull": 3.5, "sigma_tow": 1.4,
        "t_on": 0, "t_off": 600,
        "color": "#ff9d00",
    },
    {
        "name": "T2 中/机动",
        "start":  (30.0225, 121.9690),   # 参考点西北 3 km / 北 2.5 km
        "speed_kn": 23.0,                # ≈ 11.83 m/s, 向东 + 横向正弦
        "path_type": "curve",
        "waypoints": None,
        "curve_func": lambda t: (
            30.0225 + 0.0045*np.sin(2*np.pi*t/400.0),      # 纬度摆动
            121.9690 + 0.0001227*t),                        # 经度东移
        "amp": 0.22, "sigma_hull": 4.0, "sigma_tow": 1.6,
        "t_on": 60, "t_off": 560,
        "color": "#ff00cc",
    },
    {
        "name": "T3 近/高速",
        "start":  (30.0045, 121.9780),   # 参考点西南 2.1 km / 北 0.5 km
        "speed_kn": 29.0,                # ≈ 14.92 m/s, 向东超越
        "path_type": "waypoints",
        "waypoints": [(30.0045, 122.0580)],
        "curve_func": None,
        "amp": 0.18, "sigma_hull": 5.0, "sigma_tow": 2.0,
        "t_on": 150, "t_off": 450,
        "color": "#00ffff",
    },
    {
        "name": "T4 远/弱",
        "start":  (30.0630, 122.0620),   # 参考点北 7 km / 东 6 km
        "speed_kn": 10.0,                # ≈ 5.14 m/s, 向西
        "path_type": "waypoints",
        "waypoints": [(30.0630, 122.0090)],
        "curve_func": None,
        "amp": 0.10, "sigma_hull": 6.0, "sigma_tow": 2.4,
        "t_on": 100, "t_off": 560,
        "color": "#aaff00",
    },
]

# ---------- 噪声与显示 ----------
NOISE_STD   = 0.030      # 高斯噪声标准差 (线性强度)
CLUTTER_LAM = 0.012      # 杂波指数分布参数
DB_RANGE    = (-5, 16)   # BTR 显示动态范围 (dB, 相对噪声底)
SEED        = 20260911   # 随机种子 (可复现)

# ---------- 输出目录 (图像与 txt 保存位置) ----------
OUT_DIR = "./gen_datas/multi_multi"

# =====================================================================
# =======================    内部实现    ==============================
# =====================================================================
KN2MS = 0.514444          # 1 节 = 0.514444 m/s

def ll2xy(lat, lon):
    """经纬度 -> 等距圆柱投影米制坐标 (x 东, y 北)"""
    lat0 = np.radians(REF_LAT); lon0 = np.radians(REF_LON)
    x = EARTH_R * np.cos(lat0) * (np.radians(lon) - lon0)
    y = EARTH_R * (np.radians(lat) - lat0)
    return x, y

def xy2ll(x, y):
    """米制坐标 -> 经纬度"""
    lat0 = np.radians(REF_LAT); lon0 = np.radians(REF_LON)
    lat = np.degrees(lat0 + y / EARTH_R)
    lon = np.degrees(lon0 + x / (EARTH_R * np.cos(lat0)))
    return lat, lon

def haversine_m(lat1, lon1, lat2, lon2):
    """两经纬度点间大圆距离 (m)"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1; dlam = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2
         + np.cos(p1)*np.cos(p2)*np.sin(dlam/2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def gcbearing(lat1, lon1, lat2, lon2):
    """大圆方位 (北=0°, 顺时针 0~360°)"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    y = np.sin(dlam) * np.cos(p2)
    x = (np.cos(p1)*np.sin(p2)
         - np.sin(p1)*np.cos(p2)*np.cos(dlam))
    return np.degrees(np.arctan2(y, x)) % 360.0

def fold_bearing(th):
    """线列阵左右舷模糊: 折叠到 0~180°"""
    th = th % 360.0
    return 360.0 - th if th > 180.0 else th

# ---------- 路径构建 (经纬度) ----------
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
    """
    统一路径生成器 (经纬度版), 返回 Path 对象:
        p(t)  -> (lat, lon)
        p.t_end  # waypoints 模式走完全程的时刻; curve 模式为 None
    - waypoints: 从 start_ll 出发按 speed_kn 匀速过航点, 走完后保持终点。
    - curve:     直接使用 curve_func(t)->(lat, lon), speed_kn 不生效。
    """
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
        times.append(times[-1] + d / v)
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    t_end = times[-1]
    def path(t):
        return float(np.interp(t, times, lats)), float(np.interp(t, times, lons))
    return Path(path, t_end=t_end, mode="waypoints")

def check_bounds(pos_fn, bounds, label, t_arr):
    """检查经纬度轨迹是否越界"""
    if bounds is None:
        return
    lat = np.array([pos_fn(t)[0] for t in t_arr])
    lon = np.array([pos_fn(t)[1] for t in t_arr])
    la_lo, la_hi = bounds["lat"]; lo_lo, lo_hi = bounds["lon"]
    if (lat.min() < la_lo or lat.max() > la_hi
            or lon.min() < lo_lo or lon.max() > lo_hi):
        print(f"[WARN] {label} 轨迹超出经纬度 bounds: "
              f"lat=[{lat.min():.4f},{lat.max():.4f}] "
              f"(限 {la_lo},{la_hi}), "
              f"lon=[{lon.min():.4f},{lon.max():.4f}] "
              f"(限 {lo_lo},{lo_hi})")

# ---------- 目标路径 (同一批, 全局构建一次) ----------
t_arr = np.arange(0, T_MAX + DT, DT)
ttt   = np.linspace(0, T_MAX, 1200)      # 理论航迹采样

tgt_ll = []
for g in TARGETS:
    pf = build_path(g["start"], g["speed_kn"],
                    g["waypoints"], g["curve_func"])
    if g["path_type"] == "curve":
        print(f"[INFO] {g['name']} 使用 curve 模式: speed_kn 不生效")
    check_bounds(pf, g.get("bounds"), g["name"], t_arr)
    if pf.t_end is not None and pf.t_end > g["t_off"]:
        print(f"[INFO] {g['name']} 航点全程需 {pf.t_end:.0f}s, "
              f"大于 t_off={g['t_off']:.0f}s: "
              f"目标将在 t={g['t_off']:.0f}s 停止显示")
    tgt_ll.append(pf)

# ---------- 声呐伪彩色 ----------
colors = [(0,0,0),(0,0,0.35),(0,0.15,0.5),(0,0.45,0.75),
          (0.2,0.85,0.85),(0.95,0.95,0.25),(1,0.55,0.1),
          (1,0.2,0),(1,0,0)]
cmap = LinearSegmentedColormap.from_list("sonar", colors, N=256)

# ---------- 平台构造 ----------
def build_platform(plat):
    """构造单个平台对象 (位置函数/航向/两阵位置函数)"""
    x0, y0 = ll2xy(*plat["start_ll"])
    psi = np.radians(90.0 - plat["heading_deg"])   # 米制平面运动角 (东=0 逆时针)
    v = plat["speed_kn"] * KN2MS
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
        return xy2ll(*array_xy(t, plat["hull_m"]))
    def tow_ll(t):
        return xy2ll(*array_xy(t, plat["tow_m"]))
    return {"name": plat["name"],
            "start_ll": plat["start_ll"],
            "heading_deg": float(plat["heading_deg"]),
            "heading": float(plat["heading_deg"]),
            "plat_ll": plat_ll, "hull_ll": hull_ll, "tow_ll": tow_ll,
            "hull_m": plat["hull_m"], "tow_m": plat["tow_m"],
            "speed_kn": plat["speed_kn"], "bounds": plat.get("bounds"),
            "color": plat["color"]}

def rel_bearing(t, tgt_fn, array_fn, heading):
    """目标相对船头舷角 (船首=0°, 正横=90°, 正尾=180°, 折叠 0~180°)"""
    lat_t, lon_t = tgt_fn(t)
    lat_a, lon_a = array_fn(t)
    gb = gcbearing(lat_a, lon_a, lat_t, lon_t)   # 大圆方位 (北=0 顺时针)
    rel = (gb - heading) % 360.0                 # 减去平台航向
    return fold_bearing(rel)

def build_btr(array_fn, sigma_key, seed_off, heading):
    """常规波束形成 (CBF) 仿真的方位-时间能量图 (dB, 相对噪声底)"""
    r = np.random.default_rng(SEED + seed_off)
    br = np.arange(B_MIN, B_MAX + DB_STEP, DB_STEP)
    S = (r.normal(0.0, NOISE_STD, size=(len(br), len(t_arr)))
         + 0.02*np.sin(2*np.pi*t_arr/180.0)[None, :]
         * np.cos(2*np.pi*br/120.0)[:, None]
         + r.exponential(CLUTTER_LAM, size=(len(br), len(t_arr))))
    for g, pf in zip(TARGETS, tgt_ll):
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
    """目标相对指定阵列的最近距离 (km) 与时刻 (s)"""
    dmin, tmin = 1e18, 0.0
    for tk in np.linspace(0, T_MAX, 2000):
        d = haversine_m(*array_fn(tk), *pf(tk))
        if d < dmin:
            dmin, tmin = d, tk
    return dmin/1000.0, tmin

# ---------- 单平台渲染 (三图 + 三 txt) ----------
def render_platform(plat, idx):
    name = plat["name"]
    print("="*84)
    print(f"[平台 {name}] start=({plat['start_ll'][0]:.4f}°N,"
          f"{plat['start_ll'][1]:.4f}°E) 航向={plat['heading_deg']:.0f}°"
          f"(北=0°顺时针) speed={plat['speed_kn']:.1f} kn "
          f"拖曳基线={abs(plat['tow_m'][0]):.0f} m")
    print("="*84)
    check_bounds(plat["plat_ll"], plat["bounds"], f"平台{name}", t_arr)

    # ---- 两阵 BTR ----
    br_h, B_hull = build_btr(plat["hull_ll"], "sigma_hull", 1+idx*10, plat["heading"])
    br_t, B_tow  = build_btr(plat["tow_ll"],  "sigma_tow",  2+idx*10, plat["heading"])

    # ---- 运动学摘要 + 数据收集 (供 txt) ----
    records = []
    for g, pf in zip(TARGETS, tgt_ll):
        bh0 = rel_bearing(0, pf, plat["hull_ll"], plat["heading"])
        bhe = rel_bearing(T_MAX, pf, plat["hull_ll"], plat["heading"])
        bt0 = rel_bearing(0, pf, plat["tow_ll"], plat["heading"])
        bte = rel_bearing(T_MAX, pf, plat["tow_ll"], plat["heading"])
        cpah, tch = _cpa(pf, plat["hull_ll"])
        cpat, tct = _cpa(pf, plat["tow_ll"])
        rec = {"name": g["name"], "speed": g["speed_kn"],
               "lat0": g["start"][0], "lon0": g["start"][1],
               "latE": pf(T_MAX)[0], "lonE": pf(T_MAX)[1],
               "bh0": bh0, "bhe": bhe, "bt0": bt0, "bte": bte,
               "cpa_h": cpah, "tc_h": tch, "cpa_t": cpat, "tc_t": tct,
               "amp": g["amp"], "t_on": g["t_on"], "t_off": g["t_off"],
               "t_end": pf.t_end}
        records.append(rec)
        print(f"  {rec['name']:12s} | {rec['speed']:4.0f} kn "
              f"({rec['speed']*KN2MS:4.2f} m/s) | "
              f"弦测舷角 {bh0:5.1f}°->{bhe:5.1f}° | "
              f"拖曳舷角 {bt0:5.1f}°->{bte:5.1f}° | "
              f"CPA弦测={cpah:5.2f} km @ t={tch:4.0f}s")
    bh3 = rel_bearing(300, tgt_ll[2], plat["hull_ll"], plat["heading"])
    bt3 = rel_bearing(300, tgt_ll[2], plat["tow_ll"], plat["heading"])
    d_parallax = abs(bh3 - bt3)
    print(f"T3 @ t=300s: 弦测阵舷角={bh3:.1f}°, 拖曳阵舷角={bt3:.1f}°, "
          f"双站视差 |Δ|={d_parallax:.1f}°")

    # ---- 理论航迹曲线 ----
    bh_curves = [np.array([rel_bearing(tk, pf, plat["hull_ll"], plat["heading"])
                           for tk in ttt]) for pf in tgt_ll]
    bt_curves = [np.array([rel_bearing(tk, pf, plat["tow_ll"], plat["heading"])
                           for tk in ttt]) for pf in tgt_ll]

    # ---- 图1: 经纬度平面运动轨迹 (纯图) ----
    fig1 = plt.figure(figsize=(11, 6.5), dpi=150, facecolor="black")
    ax0 = fig1.add_subplot(111)
    pl_la = [plat["plat_ll"](tk)[0] for tk in ttt]
    pl_lo = [plat["plat_ll"](tk)[1] for tk in ttt]
    tw_la = [plat["tow_ll"](tk)[0] for tk in ttt]
    tw_lo = [plat["tow_ll"](tk)[1] for tk in ttt]
    ax0.plot(pl_lo, pl_la, "w-", lw=1.8)
    ax0.plot(tw_lo, tw_la, "c--", lw=1.0)
    ax0.plot(pl_lo[0], pl_la[0], "s", color=plat["color"], ms=6)
    for g, pf in zip(TARGETS, tgt_ll):
        la = [pf(tk)[0] for tk in ttt]
        lo = [pf(tk)[1] for tk in ttt]
        ax0.plot(lo, la, "-", lw=1.0, alpha=0.9, color=g["color"])
        ax0.plot(lo[0], la[0], "o", ms=5, color=g["color"])
    ax0.set_xlabel("经度 Longitude (°E)", fontsize=10)
    ax0.set_ylabel("纬度 Latitude (°N)", fontsize=10)
    ax0.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
    ax0.set_facecolor("#0a0f1f")
    out1 = f"{OUT_DIR}/btr_{name}_track.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="black")
    print("OK ->", out1)
    plt.close(fig1)

    # ---- 图2: 弦测阵 BTR (横轴=方位 0~180° 船头0, 纵轴=时间) ----
    fig2 = plt.figure(figsize=(12, 6.2), dpi=150, facecolor="black")
    ax1 = fig2.add_subplot(111)
    im1 = ax1.pcolormesh(br_h, t_arr, B_hull.T, cmap=cmap, shading="auto",
                         vmin=DB_RANGE[0], vmax=DB_RANGE[1])
    ax1.set_xlim(B_MIN, B_MAX)
    ax1.set_xticks(np.arange(0, 181, 30))
    ax1.set_ylim(0, T_MAX)
    ax1.set_yticks(np.arange(0, 601, 60))
    ax1.set_xlabel("方位 Bearing (°)", fontsize=11)
    ax1.set_ylabel("时间 Time (s)", fontsize=10)
    ax1.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
    for g, bh in zip(TARGETS, bh_curves):
        m = (ttt >= g["t_on"]) & (ttt <= g["t_off"])
        ax1.plot(bh[m], ttt[m], "w-", lw=0.8, alpha=0.7)
    cbar1 = fig2.colorbar(im1, ax=ax1, pad=0.015)
    cbar1.set_label("波束输出强度 (dB, 相对噪声底)", fontsize=10)
    out2 = f"{OUT_DIR}/btr_{name}_hull.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="black")
    print("OK ->", out2)
    plt.close(fig2)

    # ---- 图3: 拖曳阵 BTR ----
    fig3 = plt.figure(figsize=(12, 6.2), dpi=150, facecolor="black")
    ax2 = fig3.add_subplot(111)
    im2 = ax2.pcolormesh(br_t, t_arr, B_tow.T, cmap=cmap, shading="auto",
                         vmin=DB_RANGE[0], vmax=DB_RANGE[1])
    ax2.set_xlim(B_MIN, B_MAX)
    ax2.set_xticks(np.arange(0, 181, 30))
    ax2.set_ylim(0, T_MAX)
    ax2.set_yticks(np.arange(0, 601, 60))
    ax2.set_xlabel("方位 Bearing (°)", fontsize=11)
    ax2.set_ylabel("时间 Time (s)", fontsize=10)
    ax2.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
    for g, bh, bt in zip(TARGETS, bh_curves, bt_curves):
        m = (ttt >= g["t_on"]) & (ttt <= g["t_off"])
        ax2.plot(bh[m], ttt[m], "w-", lw=0.8, alpha=0.7)
        ax2.plot(bt[m], ttt[m], "y--", lw=0.8, alpha=0.8)
    cbar2 = fig3.colorbar(im2, ax=ax2, pad=0.015)
    cbar2.set_label("波束输出强度 (dB, 相对噪声底)", fontsize=10)
    out3 = f"{OUT_DIR}/btr_{name}_tow.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="black")
    print("OK ->", out3)
    plt.close(fig3)

    # ---- txt: 与三图同名, 记录平台与目标信息 ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    heading_txt = (f"船头朝向(航向)={plat['heading_deg']:.0f}° (地理方位角, "
                   f"北=0°顺时针)")
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

    L1 = []
    L1.append("="*68)
    L1.append(f"图像文件: btr_{name}_track.png (经纬度平面运动轨迹, 平台{name})")
    L1.append(f"生成时间: {now}")
    L1.append("="*68)
    L1.append("[仿真设置]")
    L1.append(f"  观测时长: {T_MAX:.0f} s, 时间步长: {DT:.1f} s")
    L1.append(f"  方位范围: {B_MIN:.0f}~{B_MAX:.0f} deg "
              f"(相对船头舷角, 船首=0°, 正横=90°, 正尾=180°)")
    L1.append(f"  地球半径: {EARTH_R:.0f} m, 投影基准: lat={REF_LAT}, lon={REF_LON}")
    L1.append("")
    L1.append("[平台信息]")
    L1.append(f"  平台名: {name}")
    L1.append(f"  初始经纬度: ({plat['start_ll'][0]:.4f}°N, "
              f"{plat['start_ll'][1]:.4f}°E)")
    L1.append(f"  {heading_txt}")
    L1.append(f"  航速: {plat['speed_kn']} kn "
              f"({plat['speed_kn']*KN2MS:.2f} m/s)")
    b = plat["bounds"]
    if b:
        L1.append(f"  运动范围 bounds: lat=[{b['lat'][0]},{b['lat'][1]}], "
                  f"lon=[{b['lon'][0]},{b['lon'][1]}]")
    L1.append("")
    L1.append("[阵列配置]")
    L1.append(f"  弦测阵: 舰体中心, 相对偏移 {plat['hull_m']} m")
    L1.append(f"  拖曳阵: 舰艉正后方, 相对偏移 {plat['tow_m']} m "
              f"(基线 {abs(plat['tow_m'][0]):.0f} m, 随船头航向旋转)")
    L1.append("")
    L1.append("[目标信息]")
    for rec in records:
        L1.append(f"  {rec['name']:12s} | {rec['speed']:4.0f} kn "
                  f"({rec['speed']*KN2MS:4.2f} m/s) | "
                  f"start=({rec['lat0']:.4f}°N,{rec['lon0']:.4f}°E) "
                  f"end=({rec['latE']:.4f}°N,{rec['lonE']:.4f}°E) | "
                  f"amp={rec['amp']:.2f} | "
                  f"观测窗[{rec['t_on']:.0f},{rec['t_off']:.0f}]s")
    with open(f"{OUT_DIR}/btr_{name}_track.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L1) + "\n")

    L2 = []
    L2.append("="*68)
    L2.append(f"图像文件: btr_{name}_hull.png (弦测阵方位历程图, 平台{name})")
    L2.append(f"生成时间: {now}")
    L2.append("="*68)
    L2.append("[目标相对弦测阵的舷角与距离信息] "
              "(舷角: 船首=0°, 正横=90°, 正尾=180°)")
    for rec in records:
        L2.append(_fmt(rec, "hull"))
    with open(f"{OUT_DIR}/btr_{name}_hull.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L2) + "\n")

    L3 = []
    L3.append("="*68)
    L3.append(f"图像文件: btr_{name}_tow.png (拖曳线列阵方位历程图, 平台{name})")
    L3.append(f"生成时间: {now}")
    L3.append("="*68)
    L3.append("[目标相对拖曳阵的舷角与距离信息]")
    for rec in records:
        L3.append(_fmt(rec, "tow"))
    L3.append("-"*68)
    L3.append("[双站视差对比 (t=300s)]")
    L3.append(f"  T3 近/高速: 弦测阵舷角={bh3:.1f}°, "
              f"拖曳阵舷角={bt3:.1f}°, 双站视差 |Δ|={d_parallax:.1f}°")
    L3.append(f"  说明: 基线 {abs(plat['tow_m'][0]):.0f} m, "
              f"近距离目标 (CPA≈{records[2]['cpa_h']:.2f} km) 视差显著, "
              f"远距离目标两阵舷角趋于一致")
    with open(f"{OUT_DIR}/btr_{name}_tow.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L3) + "\n")

    print(f"OK -> btr_{name}_track.txt / btr_{name}_hull.txt / "
          f"btr_{name}_tow.txt")
    return records

# ---------- 主循环: 逐平台渲染 ----------
ALL_RECORDS = {}
for i, plat_cfg in enumerate(PLATFORMS):
    plat = build_platform(plat_cfg)
    ALL_RECORDS[plat["name"]] = render_platform(plat, i)

print("\n全部平台渲染完成: " + ", ".join(p["name"] for p in PLATFORMS))
#（注：内容由AI生成）

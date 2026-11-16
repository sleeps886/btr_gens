# -*- coding: utf-8 -*-
"""
双阵列方位历程图 (BTR) —— 经纬度坐标版
=======================================
所有位置参数均使用【经纬度】(lat, lon)，单位：度；
速度使用【节 kn】(1 kn = 0.514444 m/s)。
内部将经纬度按等距圆柱投影换算为米制平面做运动学计算，
方位角采用【大圆方位】(北为0°, 顺时针) 并按线列阵左右舷模糊
折叠到 0~180°。

路径支持两种模式:
  1) waypoints: 给定经纬度航点列表, 按 speed_kn 匀速线性插值
  2) curve:    给定函数 f(t) -> (lat, lon)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# =======================    参 数 区 (可调)    =======================
# =====================================================================

# ---------- 仿真总览 ----------
T_MAX    = 600.0      # 仿真时长 (s)
DT       = 1.0        # 时间步长 (s)
B_MIN    = 0.0        # 方位范围下限 (度, 左右舷模糊后)
B_MAX    = 180.0      # 方位范围上限 (度)
DB_STEP  = 0.25       # 方位分辨率 (度)

# ---------- 地球模型 ----------
EARTH_R   = 6371000.0          # 地球半径 (m)
REF_LAT   = 30.0               # 投影基准纬度 (度), 用于米制换算
REF_LON   = 122.0              # 投影基准经度 (度)

# ---------- 平台运动 (经纬度) ----------
PLATFORM = {
    "start":  (30.0000, 122.0000),     # 初始经纬度 (lat, lon)
    "speed_kn": 15.0,                  # 航速 (节), 15 kn ≈ 7.72 m/s
    "bounds": {"lat": (29.90, 30.15),  # 纬度运动范围
               "lon": (121.80, 122.30)},  # 经度运动范围
    "path_type": "waypoints",          # 'waypoints' / 'curve'
    "waypoints": [                     # 航点 (lat, lon), 按 speed_kn 匀速
        (30.0000, 122.0410),
        (30.0045, 122.0830),
        (30.0045, 122.1240)],
    "curve_func": None,                # 曲线模式备用: f(t)->(lat, lon)
}

# ---------- 阵列位置 (相对舰体的偏移, 米; 随平台航向旋转) ----------
ARRAYS = {
    "hull": (0.0,    0.0),     # 弦测阵: 舰体中心
    "tow":  (-150.0, 0.0),     # 拖曳阵: 舰艉正后方 150 m
}

# ---------- 目标列表 (经纬度) ----------
# start:    初始经纬度 (lat, lon)
# speed_kn: 航速 (节)
# amp:      目标强度 (相对噪声底)
# sigma_hull / sigma_tow: 两阵主瓣宽度 (度)
TARGETS = [
    {
        "name": "T1 远/相向",
        "start":  (30.0360, 122.1240),   # 平台东北 12 km / 4 km
        "speed_kn": 19.0,                # ≈ 9.77 m/s, 向西
        "bounds": None,
        "path_type": "waypoints",
        "waypoints": [(30.0360, 122.0630)],
        "curve_func": None,
        "amp": 0.30, "sigma_hull": 3.5, "sigma_tow": 1.4,
        "t_on": 0, "t_off": 600,
    },
    {
        "name": "T2 中/机动",
        "start":  (30.0225, 121.9690),   # 平台西北 3 km / 北 2.5 km
        "speed_kn": 23.0,                # ≈ 11.83 m/s, 向东+横向正弦
        "bounds": None,
        "path_type": "curve",
        "waypoints": None,
        "curve_func": lambda t: (
            30.0225 + 0.0045*np.sin(2*np.pi*t/400.0),      # 纬度摆动
            121.9690 + 0.0001227*t),                        # 经度东移
        "amp": 0.22, "sigma_hull": 4.0, "sigma_tow": 1.6,
        "t_on": 60, "t_off": 560,
    },
    {
        "name": "T3 近/高速",
        "start":  (30.0045, 121.9780),   # 平台西南 2.1 km / 北 0.5 km
        "speed_kn": 29.0,                # ≈ 14.92 m/s, 向东超越
        "bounds": None,
        "path_type": "waypoints",
        "waypoints": [(30.0045, 122.0580)],   # 全程 7.7 km, 走完 ~517 s > t_off
        "curve_func": None,
        "amp": 0.18, "sigma_hull": 5.0, "sigma_tow": 2.0,
        "t_on": 150, "t_off": 450,
    },
    {
        "name": "T4 远/弱",
        "start":  (30.0630, 122.0620),   # 平台北 7 km / 东 6 km
        "speed_kn": 10.0,                # ≈ 5.14 m/s, 向西
        "bounds": None,
        "path_type": "waypoints",
        "waypoints": [(30.0630, 122.0090)],
        "curve_func": None,
        "amp": 0.10, "sigma_hull": 6.0, "sigma_tow": 2.4,
        "t_on": 100, "t_off": 560,
    },
]

# ---------- 噪声与显示 ----------
NOISE_STD   = 0.030
CLUTTER_LAM = 0.012
DB_RANGE    = (-5, 16)
SEED        = 20260911

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
    th = np.degrees(np.arctan2(y, x)) % 360.0
    return th

def fold_bearing(th):
    """线列阵左右舷模糊: 折叠到 0~180°"""
    th = th % 360.0
    return 360.0 - th if th > 180.0 else th

# ---------- 路径构建 (经纬度) ----------
class Path:
    """
    显式路径对象 (消除闭包隐式依赖, 输入输出一目了然)。

    用法:
        p = build_path(start_ll, speed_kn, waypoints=[...])
        lat, lon = p(t)     # 输入: t 时间(s); 输出: (lat, lon)
        p.t_end             # 航点模式走完全程的时刻(s); 曲线模式为 None
        p.mode              # 'waypoints' / 'curve'
    """

    def __init__(self, fn, t_end=None, mode="waypoints"):
        self._fn = fn       # 内部位置函数: t -> (lat, lon)
        self.t_end = t_end
        self.mode = mode

    def __call__(self, t):
        """输入时间 t (s), 返回该时刻经纬度 (lat, lon)"""
        return self._fn(t)

    def __repr__(self):
        return f"<Path mode={self.mode}, t_end={self.t_end}>"


def build_path(start_ll, speed_kn, waypoints=None, curve_func=None):
    """
    统一路径生成器 (经纬度版), 返回 Path 对象:
        p(t)  -> (lat, lon)   # t: 时间(s)
        p.t_end               # 航点模式走完全程的时刻(s); 曲线模式为 None

    - waypoints 模式: 从 start_ll 出发, 按 speed_kn 匀速依次经过航点,
      走完全程后保持终点静止 (np.interp 越界钳制到端点)。
    - curve 模式:    直接使用 curve_func(t)->(lat, lon),
      速度由曲线函数导数决定, speed_kn 不生效 (调用处应提示)。
    """
    if curve_func is not None:
        return Path(curve_func, t_end=None, mode="curve")
    if waypoints is None:
        raise ValueError("build_path: waypoints 与 curve_func 至少提供一个")
    # 过滤距离 < 1e-3 m 的重复航点, 保证 times 严格递增 (np.interp 要求)
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
        lat = float(np.interp(t, times, lats))
        lon = float(np.interp(t, times, lons))
        return lat, lon
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

# ---------- 平台/阵列运动 ----------
t_arr = np.arange(0, T_MAX + DT, DT)
plat_ll = build_path(PLATFORM["start"], PLATFORM["speed_kn"],
                     PLATFORM["waypoints"], PLATFORM["curve_func"])
plat_tend = plat_ll.t_end
if PLATFORM["path_type"] == "curve":
    print("[INFO] 平台使用 curve 模式: speed_kn 不生效, 速度由曲线函数决定")
check_bounds(plat_ll, PLATFORM["bounds"], "平台", t_arr)

tgt_ll = []          # Path 对象列表
for g in TARGETS:
    pf = build_path(g["start"], g["speed_kn"],
                    g["waypoints"], g["curve_func"])
    if g["path_type"] == "curve":
        print(f"[INFO] {g['name']} 使用 curve 模式: speed_kn 不生效")
    check_bounds(pf, g["bounds"], g["name"], t_arr)
    if pf.t_end is not None and pf.t_end > g["t_off"]:
        print(f"[INFO] {g['name']} 航点全程需 {pf.t_end:.0f}s, "
              f"大于 t_off={g['t_off']:.0f}s: "
              f"目标将在 t={g['t_off']:.0f}s 停止显示")
    tgt_ll.append(pf)

def platform_xy(t):
    """平台米制坐标"""
    lat, lon = plat_ll(t)
    return ll2xy(lat, lon)

def heading_deg(t):
    """平台瞬时航向 (米制平面, 相对 +x 东向, 度)"""
    h = 1.0
    x1, y1 = platform_xy(t - h)
    x2, y2 = platform_xy(t + h)
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

def array_xy(t, offset_m):
    """阵列米制坐标 = 平台 + 随航向旋转的偏移"""
    px, py = platform_xy(t)
    psi = np.radians(heading_deg(t))
    ox, oy = offset_m
    return (px + ox*np.cos(psi) - oy*np.sin(psi),
            py + ox*np.sin(psi) + oy*np.cos(psi))

def hull_ll(t):
    x, y = array_xy(t, ARRAYS["hull"])
    return xy2ll(x, y)

def tow_ll(t):
    x, y = array_xy(t, ARRAYS["tow"])
    return xy2ll(x, y)

# ---------- 目标方位 (相对船头舷角, 折叠 0~180) ----------
def tgt_bearing_ll(t, tgt_fn, array_fn):
    """目标相对船头的舷角: 船首=0°, 正横=90°, 正尾=180°
    (左右舷模糊后折叠到 0~180°, 不含左右舷区分)"""
    lat_t, lon_t = tgt_fn(t)
    lat_a, lon_a = array_fn(t)
    gb = gcbearing(lat_a, lon_a, lat_t, lon_t)   # 大圆方位 (北=0, 顺时针)
    hd = heading_deg(t)                          # 平台航向 (东=0, 逆时针)
    hd_north = (90.0 - hd) % 360.0               # 统一为 北=0 顺时针
    rel = (gb - hd_north) % 360.0                # 相对船头角 (0=正前)
    return fold_bearing(rel)

# ---------- 生成 BTR ----------
def build_btr(array_fn, sigma_key, seed_off):
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
            bk = tgt_bearing_ll(tk, pf, array_fn)
            peak = g["amp"] * (1.0 + 0.2*r.standard_normal())
            S[:, k] += peak * np.exp(-0.5*((br - bk)/sig)**2)
    S = np.clip(S, 1e-4, None)
    P0 = np.median(S)
    return br, np.clip(10.0*np.log10(S/P0), *DB_RANGE)

br, B_hull = build_btr(hull_ll, "sigma_hull", 1)
_,  B_tow  = build_btr(tow_ll,  "sigma_tow", 2)

# ---------- 运动学摘要 (经纬度) 与 数据收集 (供 txt 文件) ----------
records = []
print("="*84)
print(f"平台: start=({PLATFORM['start'][0]:.4f}°N, "
      f"{PLATFORM['start'][1]:.4f}°E), speed={PLATFORM['speed_kn']} kn "
      f"({PLATFORM['speed_kn']*KN2MS:.2f} m/s)")
print(f"阵列偏移: 弦测阵={ARRAYS['hull']} m, "
      f"拖曳阵={ARRAYS['tow']} m (随航向旋转)")
if plat_tend is not None:
    print(f"平台走完全部航点需 t={plat_tend:.0f}s "
          f"(T_MAX={T_MAX:.0f}s)")
print("="*84)

def _cpa(pf, array_fn):
    """目标相对指定阵列的最近距离 (km) 与时刻 (s)"""
    dmin, tmin = 1e18, 0.0
    for tk in np.linspace(0, T_MAX, 2000):
        lat_t, lon_t = pf(tk); lat_a, lon_a = array_fn(tk)
        d = haversine_m(lat_a, lon_a, lat_t, lon_t)
        if d < dmin: dmin, tmin = d, tk
    return dmin/1000.0, tmin

for g, pf in zip(TARGETS, tgt_ll):
    bh0 = tgt_bearing_ll(0, pf, hull_ll)
    bhe = tgt_bearing_ll(T_MAX, pf, hull_ll)
    bt0 = tgt_bearing_ll(0, pf, tow_ll)
    bte = tgt_bearing_ll(T_MAX, pf, tow_ll)
    cpa_h, tc_h = _cpa(pf, hull_ll)
    cpa_t, tc_t = _cpa(pf, tow_ll)
    lat0, lon0 = pf(0)
    latE, lonE = pf(T_MAX)
    rec = dict(name=g["name"], speed=g["speed_kn"], amp=g["amp"],
               sigma_hull=g["sigma_hull"], sigma_tow=g["sigma_tow"],
               t_on=g["t_on"], t_off=g["t_off"],
               lat0=lat0, lon0=lon0, latE=latE, lonE=lonE,
               t_end=pf.t_end,
               bh0=bh0, bhe=bhe, bt0=bt0, bte=bte,
               cpa_h=cpa_h, tc_h=tc_h, cpa_t=cpa_t, tc_t=tc_t)
    records.append(rec)
    extra = f" | 走完航点 t={pf.t_end:.0f}s" if pf.t_end is not None else ""
    print(f"{g['name']:12s} | {g['speed_kn']:4.0f} kn "
          f"({g['speed_kn']*KN2MS:4.2f} m/s) | "
          f"start=({lat0:.4f}°N,{lon0:.4f}°E) "
          f"end=({latE:.4f}°N,{lonE:.4f}°E) | "
          f"弦测舷角 {bh0:5.1f}°->{bhe:5.1f}° | "
          f"拖曳舷角 {bt0:5.1f}°->{bte:5.1f}° | "
          f"CPA弦测={cpa_h:5.2f} km @ t={tc_h:4.0f}s{extra}")
bh3 = tgt_bearing_ll(300, tgt_ll[2], hull_ll)
bt3 = tgt_bearing_ll(300, tgt_ll[2], tow_ll)
d_parallax = abs(bh3 - bt3)
print("-"*84)
print(f"T3 @ t=300s: 弦测阵舷角={bh3:.1f}°, 拖曳阵舷角={bt3:.1f}°, "
      f"双站视差 |Δ|={d_parallax:.1f}°")
print("="*84)

# ---------- 绘图: 三张独立图分开保存 ----------
colors = [(0,0,0.15),(0,0.2,0.6),(0,0.7,0.9),
          (1,1,0.2),(1,0.4,0),(1,0,0)]
cmap = LinearSegmentedColormap.from_list("sonar", colors, N=256)
ttt = np.linspace(0, T_MAX, 1200)

# 理论航迹曲线 (共用)
bh_curves = [np.array([tgt_bearing_ll(tk, pf, hull_ll) for tk in ttt])
             for pf in tgt_ll]
bt_curves = [np.array([tgt_bearing_ll(tk, pf, tow_ll)  for tk in ttt])
             for pf in tgt_ll]

# ---- 图1: 经纬度平面运动轨迹 ----
fig1 = plt.figure(figsize=(11, 6.5), dpi=150, facecolor="black")
ax0 = fig1.add_subplot(111)
plat_lat = [plat_ll(tk)[0] for tk in ttt]
plat_lon = [plat_ll(tk)[1] for tk in ttt]
tow_la, tow_lo = [], []
for tk in ttt:
    la, lo = tow_ll(tk); tow_la.append(la); tow_lo.append(lo)
ax0.plot(plat_lon, plat_lat, "w-", lw=1.6, label="平台航迹")
ax0.plot(tow_lo, tow_la, "c--", lw=1.0, label="拖曳阵轨迹")
for g, pf in zip(TARGETS, tgt_ll):
    la = [pf(tk)[0] for tk in ttt]; lo = [pf(tk)[1] for tk in ttt]
    ax0.plot(lo, la, "-", lw=1.0, alpha=0.9, label=g["name"])
    s = pf(0)
    ax0.plot(s[1], s[0], "o", ms=5)
ax0.plot(plat_lon[0], plat_lat[0], "s", color="white", ms=6)
ax0.set_xlabel("经度 Longitude (°E)", fontsize=10)
ax0.set_ylabel("纬度 Latitude (°N)", fontsize=10)
ax0.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
ax0.set_facecolor("#0a0f1f")
out1 = "./gen_datas/btr_geo_track.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="black")
print("OK ->", out1)
plt.close(fig1)

# ---- 图2: 弦测阵 BTR (横轴=方位 0~180°, 纵轴=时间) ----
fig2 = plt.figure(figsize=(12, 6.2), dpi=150, facecolor="black")
ax1 = fig2.add_subplot(111)
im1 = ax1.pcolormesh(br, t_arr, B_hull.T, cmap=cmap, shading="auto",
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
out2 = "./gen_datas/btr_geo_hull.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="black")
print("OK ->", out2)
plt.close(fig2)

# ---- 图3: 拖曳阵 BTR (横轴=方位 0~180°, 纵轴=时间) ----
fig3 = plt.figure(figsize=(12, 6.2), dpi=150, facecolor="black")
ax2 = fig3.add_subplot(111)
im2 = ax2.pcolormesh(br, t_arr, B_tow.T, cmap=cmap, shading="auto",
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
out3 = "./gen_datas/btr_geo_tow.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="black")
print("OK ->", out3)
plt.close(fig3)

# ---------- 输出信息 txt (与图像同名, 记录平台与目标信息) ----------
from datetime import datetime

BASE_DIR = "./gen_datas"

def _fmt_rec(rec, array_tag):
    """格式化单个目标的信息行 (hull=弦测阵视角, tow=拖曳阵视角)"""
    if array_tag == "hull":
        b0, be, cpa, tc = rec["bh0"], rec["bhe"], rec["cpa_h"], rec["tc_h"]
    else:
        b0, be, cpa, tc = rec["bt0"], rec["bte"], rec["cpa_t"], rec["tc_t"]
    extra = f" | 走完航点 t={rec['t_end']:.0f}s" if rec["t_end"] is not None else ""
    return (f"  {rec['name']:12s} | {rec['speed']:4.0f} kn "
            f"({rec['speed']*KN2MS:4.2f} m/s) | "
            f"start=({rec['lat0']:.4f}°N,{rec['lon0']:.4f}°E) "
            f"end=({rec['latE']:.4f}°N,{rec['lonE']:.4f}°E) | "
            f"舷角 {b0:5.1f}°->{be:5.1f}° | "  # 相对船头: 船首=0°, 正横=90°, 正尾=180°
            f"CPA={cpa:5.2f} km @ t={tc:4.0f}s | "
            f"amp={rec['amp']:.2f} | 观测窗[{rec['t_on']:.0f},{rec['t_off']:.0f}]s"
            f"{extra}")

def write_info_txt():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- btr_geo_track.txt: 轨迹图信息 ----
    L1 = []
    L1.append("="*64)
    L1.append("图像文件: btr_geo_track.png (经纬度平面运动轨迹)")
    L1.append(f"生成时间: {now}")
    L1.append("="*64)
    L1.append("[仿真设置]")
    L1.append(f"  观测时长: {T_MAX:.0f} s, 时间步长: {DT:.1f} s")
    L1.append(f"  方位范围: {B_MIN:.0f}~{B_MAX:.0f} deg "
              f"(相对船头舷角, 船首=0°, 正横=90°, 正尾=180°)")
    L1.append(f"  地球半径: {EARTH_R:.0f} m, 投影基准: lat={REF_LAT}, lon={REF_LON}")
    L1.append("")
    L1.append("[平台信息]")
    L1.append(f"  起点经纬度: ({PLATFORM['start'][0]:.4f}°N, "
              f"{PLATFORM['start'][1]:.4f}°E)")
    L1.append(f"  航速: {PLATFORM['speed_kn']} kn "
              f"({PLATFORM['speed_kn']*KN2MS:.2f} m/s)")
    b = PLATFORM["bounds"]
    L1.append(f"  运动范围 bounds: lat=[{b['lat'][0]},{b['lat'][1]}], "
              f"lon=[{b['lon'][0]},{b['lon'][1]}]")
    L1.append(f"  路径类型: {PLATFORM['path_type']}")
    if PLATFORM["waypoints"]:
        wp = " -> ".join([f"({p[0]:.4f},{p[1]:.4f})"
                          for p in PLATFORM["waypoints"]])
        L1.append(f"  航点: {wp}")
    if plat_tend is not None:
        L1.append(f"  走完全部航点需 t={plat_tend:.0f} s "
                  f"(T_MAX={T_MAX:.0f} s)")
    L1.append("")
    L1.append("[阵列配置]")
    L1.append(f"  弦测阵: 舰体中心, 相对偏移 {ARRAYS['hull']} m")
    L1.append(f"  拖曳阵: 舰艉正后方, 相对偏移 {ARRAYS['tow']} m "
              f"(基线 {abs(ARRAYS['tow'][0]):.0f} m, 随航向旋转)")
    L1.append("")
    L1.append("[目标信息]")
    for rec in records:
        L1.append(f"  {rec['name']:12s} | {rec['speed']:4.0f} kn "
                  f"({rec['speed']*KN2MS:4.2f} m/s) | "
                  f"start=({rec['lat0']:.4f}°N,{rec['lon0']:.4f}°E) "
                  f"end=({rec['latE']:.4f}°N,{rec['lonE']:.4f}°E) | "
                  f"amp={rec['amp']:.2f} | sigma_hull={rec['sigma_hull']}°, "
                  f"sigma_tow={rec['sigma_tow']}° | "
                  f"观测窗[{rec['t_on']:.0f},{rec['t_off']:.0f}]s")
    with open(f"{BASE_DIR}/btr_geo_track.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L1) + "\n")

    # ---- btr_geo_hull.txt: 弦测阵图信息 ----
    L2 = []
    L2.append("="*64)
    L2.append("图像文件: btr_geo_hull.png (弦测阵方位历程图)")
    L2.append(f"生成时间: {now}")
    L2.append("="*64)
    L2.append("[目标相对弦测阵的方位与距离信息]")
    for rec in records:
        L2.append(_fmt_rec(rec, "hull"))
    with open(f"{BASE_DIR}/btr_geo_hull.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L2) + "\n")

    # ---- btr_geo_tow.txt: 拖曳阵图信息 ----
    L3 = []
    L3.append("="*64)
    L3.append("图像文件: btr_geo_tow.png (拖曳线列阵方位历程图)")
    L3.append(f"生成时间: {now}")
    L3.append("="*64)
    L3.append("[目标相对拖曳阵的方位与距离信息]")
    for rec in records:
        L3.append(_fmt_rec(rec, "tow"))
    L3.append("-"*64)
    L3.append("[双站视差对比 (t=300s)]")
    L3.append(f"  T3 近/高速: 弦测阵舷角={bh3:.1f}°, "
              f"拖曳阵舷角={bt3:.1f}°, 双站视差 |Δ|={d_parallax:.1f}°")
    L3.append(f"  说明: 基线 {abs(ARRAYS['tow'][0]):.0f} m, "
              f"近距离目标 (CPA≈{records[2]['cpa_h']:.2f} km) 视差显著, "
              f"远距离目标两阵方位趋于一致")
    with open(f"{BASE_DIR}/btr_geo_tow.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L3) + "\n")

    print("OK -> btr_geo_track.txt / btr_geo_hull.txt / btr_geo_tow.txt")

write_info_txt()

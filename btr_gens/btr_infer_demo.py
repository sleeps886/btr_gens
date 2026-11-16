# -*- coding: utf-8 -*-
"""演示/验证脚本: 用 btr_infer_track 反推 regen_A 图像并与真值对比

用法 (工作目录 btr_gens):
    python btr_infer_demo.py

真值 = regen_check.py 重跑场景的参数 (与 20261026_141830 图像逐峰一致):
  T1 航路点  (30.0360,122.1240)->(30.0360,122.0630) 19 kn, 全程
  T2 曲线    lat=30.0225+0.0045*sin(2pi t/400), lon=121.9690+0.0001227 t, [60,560]
  T3 航路点  (30.0045,121.9780)->(30.0045,122.0580) 29 kn, [150,450], 静止尾段
  T4 航路点  (30.0630,122.0620)->(30.0630,122.0090) 10 kn, [100,560]
输出: ..\\infer_out\\demo\\btr_infer_demo.png / btr_infer_demo.txt
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as T

HULL_IMG = os.path.join("..", "infer_out", "regen_A", "btr_A_hull.png")
TOW_IMG = os.path.join("..", "infer_out", "regen_A", "btr_A_tow.png")
OUT_DIR = os.path.join("..", "infer_out", "demo")
PLATFORM = (30.0623, 122.0415)
HEADING, SPEED_KN = 210.0, 0.0
HULL_M, TOW_M = (0.0, 0.0), (-150.0, 0.0)
T_MAX = 600.0


def truth_xy(name, t, geom):
    """目标真实位置 (米制 ENU), 与生成端公式一致"""
    t = np.asarray(t, dtype=float)
    if name == "T1":
        p0 = np.array(geom.ll2xy(30.0360, 122.1240))
        p1 = np.array(geom.ll2xy(30.0360, 122.0630))
        f = np.clip(t / 601.0, 0.0, 1.0)
        return p0[None, :] + (p1 - p0)[None, :] * f[:, None]
    if name == "T2":
        lat = 30.0225 + 0.0045 * np.sin(2 * np.pi * t / 400.0)
        lon = 121.9690 + 0.0001227 * t
        return np.stack(geom.ll2xy_vec(lat, lon), 1) \
            if hasattr(geom, "ll2xy_vec") else np.array(
                [geom.ll2xy(a, b) for a, b in zip(lat, lon)])
    if name == "T3":
        p0 = np.array(geom.ll2xy(30.0045, 121.9780))
        p1 = np.array(geom.ll2xy(30.0045, 122.0580))
        f = np.clip(t / 517.0, 0.0, 1.0)
        return p0[None, :] + (p1 - p0)[None, :] * f[:, None]
    p0 = np.array(geom.ll2xy(30.0630, 122.0620))
    p1 = np.array(geom.ll2xy(30.0630, 122.0090))
    f = np.clip(t / 993.0, 0.0, 1.0)
    return p0[None, :] + (p1 - p0)[None, :] * f[:, None]


TRUTH = {
    "T1": {"win": (0.0, 600.0), "kn": 19.0},
    "T2": {"win": (60.0, 560.0), "kn": 23.0},
    "T3": {"win": (150.0, 450.0), "kn": 29.0},
    "T4": {"win": (100.0, 560.0), "kn": 10.0},
}


def fold(b):
    b = np.asarray(b, dtype=float) % 360.0
    return np.where(b > 180.0, 360.0 - b, b)


def rel_bearings(geom, name, t, arr):
    """目标相对某阵的折叠舷角 (真值)"""
    X = truth_xy(name, t, geom)
    ax, ay = arr(np.asarray(t, dtype=float))
    dx = X[:, 0] - ax
    dy = X[:, 1] - ay
    az = np.degrees(np.arctan2(dx, dy)) % 360.0
    return fold((az - geom.heading) % 360.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    res = T.run_inference(
        HULL_IMG, TOW_IMG, PLATFORM, HEADING, SPEED_KN,
        hull_m=HULL_M, tow_m=TOW_M,
        log=log)
    geom = res["geom"]

    # ---- 真值 <-> 推断结果关联: 用 (hull,tow) 舷角曲线的公共段中位差 ----
    rows = []
    for name, info in TRUTH.items():
        best = None
        for r in res["results"]:
            tg = r["tgrid"]
            m = (tg >= info["win"][0]) & (tg <= info["win"][1])
            if m.sum() < 40:
                continue
            tg2 = tg[m]
            bh_t = rel_bearings(geom, name, tg2, geom.hull_xy)
            bt_t = rel_bearings(geom, name, tg2, geom.tow_xy)
            dh = np.abs(T.wrap180(r["th"][m] - bh_t))
            dt = np.abs(T.wrap180(r["tt"][m] - bt_t))
            score = float(np.median(dh) + np.median(dt))
            if best is None or score < best[0]:
                best = (score, r, tg2)
        if best is None:
            rows.append((name, None))
            continue
        _, r, tg2 = best
        # 两支候选各算位置误差, 取小者为"较优支"
        X_true = truth_xy(name, tg2, geom)
        rec = {"name": name, "res": r, "tg": tg2, "X_true": X_true,
               "score": best[0]}
        cand_err = []
        states, mirrors, _ = T.resolve_states(r)
        for si, st in enumerate(states + mirrors):
            s = st["state"]
            X_fit = np.stack([s[0] + s[2] * tg2, s[1] + s[3] * tg2], 1)
            err = np.hypot(X_fit[:, 0] - X_true[:, 0],
                           X_fit[:, 1] - X_true[:, 1]) / 1000.0
            cand_err.append((float(np.sqrt((err ** 2).mean())),
                             float(err.max()), si < len(states), s))
        cand_err.sort()
        rec["cands"] = cand_err
        rows.append(rec)

    # ---- 文本报告 ----
    log("")
    log("=" * 68)
    log("真值对比 (regen_A 场景, 平台A 静止 @ 30.0623N 122.0415E)")
    log("=" * 68)
    for rec in rows:
        if rec is None or isinstance(rec, tuple) and rec[1] is None:
            continue
        if not isinstance(rec, dict):
            continue
        rms, mx, is_main, s = rec["cands"][0]
        kind = "主输出支" if is_main else "镜像候选支"
        log(f"{rec['name']}: 真值 {TRUTH[rec['name']]['kn']} kn, "
            f"关联到 {rec['res']['name']} (舷角差中位 "
            f"{rec['score']:.2f}°), 拟合窗口 "
            f"[{rec['res']['t0']:.0f},{rec['res']['t1']:.0f}]s")
        log(f"   较优支={kind}: 位置误差 RMS={rms:.2f} km, 最大={mx:.2f} km "
            f"| 拟合航速 {math.hypot(s[2], s[3]) / T.KN2MS:.1f} kn")
        for rms2, mx2, main2, _ in rec["cands"][1:]:
            log(f"   另一支: RMS={rms2:.2f} km, 最大={mx2:.2f} km")
    miss = [n for n, r in zip(TRUTH, [x for x in rows])
            if not isinstance(r, dict)]
    if miss:
        log(f"未关联到推断结果的目标: {miss}")
    with open(os.path.join(OUT_DIR, "btr_infer_demo.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 对比图 ----
    T._setup_fonts()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 7), dpi=150, facecolor="black")
    ax = fig.add_subplot(111)
    tt = np.linspace(0, T_MAX, 800)
    px, py = geom.platform_xy(tt)
    l0 = geom.xy2ll(px[0], py[0])
    l1 = geom.xy2ll(px[-1], py[-1])
    ax.plot([l0[1], l1[1]], [l0[0], l1[0]], "w-", lw=2, label="平台航迹")
    tc = {"T1": "#ff9d00", "T2": "#ff00cc", "T3": "#00ff88", "T4": "#66aaff"}
    for name in TRUTH:
        X = truth_xy(name, tt, geom)
        ll = [geom.xy2ll(x, y) for x, y in X]
        ax.plot([p[1] for p in ll], [p[0] for p in ll], "-", color=tc[name],
                lw=2.4, alpha=0.95, label=f"{name} 真值")
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        r = rec["res"]
        for k, (rms, mx, is_main, s) in enumerate(rec["cands"]):
            X0 = geom.xy2ll(s[0], s[1])
            X1 = geom.xy2ll(s[0] + s[2] * r["t1"], s[1] + s[3] * r["t1"])
            ls = "--" if (is_main or k > 0) else "-"
            lbl = f"{r['name']} 反推({'主' if is_main else '镜像支' })" \
                if k <= 1 else None
            ax.plot([X0[1], X1[1]], [X0[0], X1[0]], ls,
                    color=tc.get(rec["name"], "#dddddd"), lw=1.6,
                    alpha=0.9, label=lbl)
        ins = r["instant"]
        ok = np.isfinite(ins[:, 0])
        if ok.any():
            ll = [geom.xy2ll(x, y) for x, y in ins[ok]]
            ax.plot([p[1] for p in ll], [p[0] for p in ll], ".",
                    color=tc.get(rec["name"], "#dddddd"), ms=3, alpha=0.35)
    ax.set_facecolor("#0a0f1f")
    ax.grid(color="gray", ls=":", lw=0.4, alpha=0.4)
    ax.set_xlabel("经度 Longitude (°E)")
    ax.set_ylabel("纬度 Latitude (°N)")
    ax.set_aspect(1.0 / math.cos(math.radians(geom.ref_lat)))
    ax.legend(loc="best", fontsize=8, facecolor="#101828",
              labelcolor="white")
    ax.set_title("hull+tow BTR 图像反推轨迹 vs 真值 (regen_A 场景)",
                 color="white")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "btr_infer_demo.png")
    fig.savefig(out_png, facecolor="black")
    print(f"OK -> {out_png}")


if __name__ == "__main__":
    main()

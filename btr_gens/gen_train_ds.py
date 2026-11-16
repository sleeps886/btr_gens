# -*- coding: utf-8 -*-
"""
场景化 BTR 训练数据集批量生成器 (走 btr_gui_server 平台服务)
=============================================================
两大类: 平台静止 / 平台移动
细化条件: 目标远近(近/远) × 目标数量(少=2/多=5) × 是否穿过平台(穿越/不穿越)
  -> 每大类 8 个场景, 共 16 个场景, 每场景 3 个训练数据 = 48 个数据集。

每个训练数据 = 一次平台仿真运行:
    btr_A_track.png/.txt  轨迹图 + 平台与目标信息
    btr_A_hull.png/.txt   舷侧阵 BTR + 目标舷角/CPA
    btr_A_tow.png/.txt    拖曳阵 BTR + 双站视差
    btr_A_hull_crop.png / btr_A_tow_crop.png   (后处理: 数据区裁剪, 1424x820,
                                               与 annotator 标注管线约定一致)
    meta.json             本样本的场景条件/参数/CPA 校验结果

用法:
    python gen_train_ds.py --smoke      # 冒烟测试: 只生成 1 个样本到 _smoke
    python gen_train_ds.py              # 全量生成 48 个样本
约定说明:
  - "穿过平台" = 存在目标轨迹穿越平台所在位置(静止)或与平台航迹相交(移动),
    穿越时刻与平台距离 < 0.6 km (目标名以 XC 开头)。
  - 远距离穿越场景: 背景目标全部保持在 8 km 以外, 穿越目标完成一次近距穿越,
    其余时间在远距离活动 (穿越目标必然经历近距离, 属场景内特例, 已在 meta 标注)。
"""
import argparse
import json
import math
import os
import re
import sys
import time
import random
import urllib.request

import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))            # btr_gens
# 输出树放在【工作区根目录】: 根目录带低完整性标签, 沙箱子进程才允许写入
# (btr_gens 等用户自建子目录缺少该标签, 子进程写入会被 Windows 拒绝)
WS = os.path.dirname(BASE)
ROOT = os.path.join(WS, "训练数据集")
SERVER = "http://127.0.0.1:8765/run"

MASTER_SEED = 20261109
REF = (30.0, 122.0)
KM_PER_DEG_LAT = 111.19
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(30.0))
BOUNDS = {"lat": [29.80, 30.30], "lon": [121.60, 122.50]}
PALETTE = ["#ff9d00", "#ff00cc", "#00ffff", "#aaff00",
           "#ff4d4d", "#7a5cff", "#00ff88", "#ffffff"]

# ----------------------------------------------------------------------
# 场景矩阵: (编号, 大类, 距离, 数量, 穿过平台)
# ----------------------------------------------------------------------
SCENARIOS = []
for cat in ("静止", "移动"):
    for dist in ("近", "远"):
        for cnt in ("少", "多"):
            for cross in (False, True):
                SCENARIOS.append((f"S{len(SCENARIOS)+1:02d}", cat, dist, cnt, cross))


def offset(ll, az_deg, d_km):
    """从 (lat,lon) 沿方位角 az 移动 d 公里"""
    return (ll[0] + d_km * math.cos(math.radians(az_deg)) / KM_PER_DEG_LAT,
            ll[1] + d_km * math.sin(math.radians(az_deg)) / KM_PER_DEG_LON)


def line_through(anchor, az, half_km, perp_km, side):
    """沿方位 az 的直线段, 距 anchor 垂距 perp_km (side=±1), 返回 (起点, 终点)"""
    c = offset(anchor, az + 90.0 * side, perp_km)
    s = offset(c, az + 180.0, half_km)
    e = offset(c, az, half_km)
    return s, e


# ----------------------------------------------------------------------
# 目标构造
# ----------------------------------------------------------------------
def mk_target(rng, name, start, speed, waypoints, amp, sh, st, t_on, t_off, color):
    return {"name": name, "start": [round(start[0], 6), round(start[1], 6)],
            "speed_kn": round(speed, 1), "path_type": "waypoints",
            "waypoints": [[round(w[0], 6), round(w[1], 6)] for w in waypoints],
            "curve_func": None, "amp": round(amp, 3),
            "sigma_hull": round(sh, 2), "sigma_tow": round(st, 2),
            "t_on": round(float(t_on), 1), "t_off": round(float(t_off), 1),
            "color": color}


def mk_curve_target(rng, name, anchor, v_ms, color, moving_platform):
    """曲线模式目标: 东向漂移 + 正弦纬度摆动 (速度由曲线导数决定)"""
    dlon_v = v_ms / (KM_PER_DEG_LON * 1000.0)          # deg/s
    amp_lat = rng.uniform(0.0015, 0.0040)              # 摆动幅度 (deg)
    period = rng.uniform(240.0, 420.0)
    phase = rng.uniform(0, 2 * math.pi)
    if not moving_platform:
        lat0, lon0 = offset(anchor, rng.uniform(0, 360),
                            rng.uniform(1.8, 3.4) if rng.random() < 0.5
                            else rng.uniform(9.8, 13.0))
    else:
        lat0, lon0 = offset(anchor, rng.uniform(0, 360),
                            rng.uniform(1.8, 3.2))
    expr = (f"({lat0:.6f} + {amp_lat:.6f}*np.sin(2*np.pi*t/{period:.1f} + {phase:.4f}), "
            f"{lon0:.6f} + {dlon_v:.9f}*t)")
    return {"name": name, "start": [round(lat0, 6), round(lon0, 6)],
            "speed_kn": round(v_ms / 0.514444, 1), "path_type": "curve",
            "waypoints": None, "curve_func": expr,
            "amp": round(rng.uniform(0.14, 0.26), 3),
            "sigma_hull": round(rng.uniform(3.8, 5.6), 2),
            "sigma_tow": round(rng.uniform(1.5, 2.2), 2),
            "t_on": round(rng.choice([0.0, rng.uniform(20, 100)]), 1),
            "t_off": round(rng.uniform(500, 600), 1), "color": color}


def bg_target(rng, idx, dist, anchor, moving, H, v_p, color):
    """背景目标 (不穿越): 近距横越 / 远距漂移; 移动平台时近似同向平行"""
    if dist == "近":
        r = rng.uniform(1.6, 3.6)
        amp = rng.uniform(0.16, 0.30)
    else:
        r = rng.uniform(9.5, 13.5)
        amp = rng.uniform(0.10, 0.20)
    t_on = rng.choice([0.0, rng.uniform(20, 120)])
    t_off = rng.uniform(500, 600)
    if moving:
        az_t = (H + rng.uniform(-18, 18)) % 360.0
        v = max(6.0, v_p + rng.uniform(-2.5, 4.5))
        l_half = min(v * 0.514444 * (t_off - t_on) * 0.62 / 1000.0,
                     2.8 if dist == "近" else 3.4)
    else:
        az_t = rng.uniform(0, 360)
        v = rng.uniform(10, 26) if dist == "近" else rng.uniform(8, 22)
        l_half = min(v * 0.514444 * (t_off - t_on) * 0.62 / 1000.0,
                     5.0 if dist == "近" else 4.5)
    s, e = line_through(anchor, az_t, l_half, r, rng.choice([-1, 1]))
    if rng.random() < 0.18:                            # 部分目标加一个转弯航点
        mid = offset(offset(s, az_t, l_half), az_t + rng.uniform(-40, 40),
                     rng.uniform(0.2, 0.8))
        return mk_target(rng, f"T{idx}", s, v, [mid, e], amp,
                         rng.uniform(3.8, 6.0), rng.uniform(1.4, 2.4),
                         t_on, t_off, color)
    return mk_target(rng, f"T{idx}", s, v, [e], amp,
                     rng.uniform(3.8, 6.0), rng.uniform(1.4, 2.4),
                     t_on, t_off, color)


def cross_target(rng, j, anchor, moving, H, v_p, tc_lo, tc_hi, color):
    """穿越目标: 轨迹在 t_c 时刻到达平台位置附近 (垂距 eps < 0.6 km)"""
    t_c = rng.uniform(tc_lo, tc_hi)
    eps = rng.uniform(0.15, 0.35)
    side = rng.choice([-1, 1])
    if moving:
        v = rng.uniform(20, 28)
        n_az = (H + 90.0 * side) % 360.0
        # 平台在 t_c 时的位置 (anchor = 平台 t=300 位置)
        X = offset(anchor, H, v_p * 0.514444 * (t_c - 300.0) / 1000.0)
        Xin = offset(X, n_az, eps)
        s = offset(Xin, (n_az + 180.0) % 360.0, v * 0.514444 * t_c / 1000.0)
        e = offset(Xin, n_az, v * 0.514444 * (600.0 - t_c) / 1000.0)
        t_off = 600.0
    else:
        v = rng.uniform(24, 29)
        az_c = rng.uniform(0, 360)
        half = v * 0.514444 * t_c / 1000.0             # 起点到穿越点距离
        s, e = line_through(anchor, az_c, half, eps, side)
        t_off = min(600.0, 2.0 * t_c + rng.uniform(0, 60))
    return mk_target(rng, f"XC{j}", s, v, [e], rng.uniform(0.22, 0.32),
                     rng.uniform(4.0, 5.6), rng.uniform(1.6, 2.2),
                     0.0, t_off, color)


# ----------------------------------------------------------------------
# 场景 payload
# ----------------------------------------------------------------------
def build_payload(code, cat, dist, cnt, cross, k, attempt):
    sc_idx = int(code[1:])
    seed = 1_000_000 + sc_idx * 1000 + k * 10 + attempt
    rng = random.Random(seed)
    moving = (cat == "移动")

    heading = (rng.choice(range(0, 360, 30)) + rng.uniform(-12, 12)) % 360.0
    if moving:
        v_p = rng.choice([8.0, 10.0, 12.0])
        ps = offset(REF, heading + 180.0, 1.5 + rng.uniform(0, 0.6))
        anchor = offset(ps, heading, v_p * 0.514444 * 300.0 / 1000.0)  # t=300 位置
    else:
        v_p = 0.0
        ps = offset(REF, rng.uniform(0, 360), rng.uniform(0, 1.2))
        anchor = ps
    platform = {"name": "A",
                "start_ll": [round(ps[0], 6), round(ps[1], 6)],
                "heading_deg": round(heading, 1),
                "speed_kn": v_p,
                "hull_m": [0.0, 0.0], "tow_m": [-150.0, 0.0],
                "bounds": BOUNDS, "color": "#ffd400"}

    targets, ti = [], 1
    if cross:
        n_cross = 2 if cnt == "多" else 1
        tcs = [(180.0, 280.0), (320.0, 440.0)]
        for j in range(n_cross):
            targets.append(cross_target(rng, j + 1, anchor, moving, heading,
                                        v_p, tcs[j][0], tcs[j][1],
                                        PALETTE[(j) % len(PALETTE)]))
        n_bg = 3 if cnt == "多" else 1
    else:
        n_bg = 5 if cnt == "多" else 2
    for i in range(n_bg):
        targets.append(bg_target(rng, ti, dist, anchor, moving, heading,
                                 v_p, PALETTE[(ti) % len(PALETTE)]))
        ti += 1
    if cnt == "多" and rng.random() < 0.5:             # 多目标场景掺一个曲线机动目标
        targets[-1] = mk_curve_target(rng, f"T{ti-1}", anchor,
                                      rng.uniform(8.0, 16.0),
                                      PALETTE[(ti - 1) % len(PALETTE)], moving)

    rng_noise = random.Random(seed + 777)
    payload = {
        "T_MAX": 600.0, "DT": 1.0, "B_MIN": 0.0, "B_MAX": 180.0, "DB_STEP": 0.25,
        "REF_LAT": REF[0], "REF_LON": REF[1],
        "NOISE_STD": round(rng_noise.uniform(0.024, 0.038), 4),
        "CLUTTER_LAM": round(rng_noise.uniform(0.009, 0.016), 4),
        "DB_RANGE": [-5.0, 16.0],
        "SEED": seed,
        "OUT_DIR": None,          # 由调用方填
        "platforms": [platform],
        "targets": targets,
        "_meta": {"code": code, "category": cat, "dist": dist, "count": cnt,
                  "cross": cross, "sample": k, "attempt": attempt, "seed": seed},
    }
    return payload


# ----------------------------------------------------------------------
# 平台服务调用 + CPA 校验
# ----------------------------------------------------------------------
CPA_RE = re.compile(r"^\s*(\S+)\s*\|.*?CPA弦测=\s*([0-9.]+)\s*km", re.M)


def call_server(payload):
    p = dict(payload)
    p.pop("_meta", None)
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(SERVER, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_cpas(log):
    return {m.group(1): float(m.group(2)) for m in CPA_RE.finditer(log)}


def check_cpas(meta, cpas):
    bad = []
    for name, km in cpas.items():
        if name.startswith("XC"):
            if km > 0.6:
                bad.append(f"{name}(穿越)CPA={km:.2f}km > 0.6")
        elif meta["dist"] == "近":
            if not (1.2 <= km <= 5.0):
                bad.append(f"{name}(近)CPA={km:.2f}km 超出[1.2,5.0]")
        else:
            if km < 8.0:
                bad.append(f"{name}(远)CPA={km:.2f}km < 8.0")
    return bad


# ----------------------------------------------------------------------
# BTR 数据区裁剪 (对齐 annotator 管线: 1424x820 黑底画布)
# ----------------------------------------------------------------------
def crop_btr(src, dst, canvas=(1424, 820), thresh=10):
    im = Image.open(src).convert("RGB")
    a = np.asarray(im)
    mask = a.max(axis=2) > thresh
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        im.save(dst)
        return None
    content = im.crop((int(cols.min()), int(rows.min()),
                       int(cols.max()) + 1, int(rows.max()) + 1))
    scale = min(1.0, canvas[0] / content.width, canvas[1] / content.height)
    if scale < 1.0:
        content = content.resize((int(content.width * scale),
                                  int(content.height * scale)), Image.LANCZOS)
    out = Image.new("RGB", canvas, (0, 0, 0))
    out.paste(content, ((canvas[0] - content.width) // 2,
                        (canvas[1] - content.height) // 2))
    out.save(dst)
    return {"content_wh": [content.width, content.height], "scale": round(scale, 3)}


# ----------------------------------------------------------------------
# 单样本生成
# ----------------------------------------------------------------------
def gen_sample(sc, k, out_root, smoke=False):
    code, cat, dist, cnt, cross = sc
    scen_dir_name = (f"{code}_{dist}距离_目标{'多' if cnt=='多' else '少'}_"
                     f"{'穿越' if cross else '不穿越'}")
    scen_dir = os.path.join(out_root, f"平台{cat}", scen_dir_name)
    sample_dir = os.path.join(scen_dir, f"样本{k+1}")
    os.makedirs(sample_dir, exist_ok=True)

    best, best_attempt, last = None, 0, -1
    for attempt in range(6):
        last = attempt
        payload = build_payload(code, cat, dist, cnt, cross, k, attempt)
        payload["OUT_DIR"] = sample_dir
        res = call_server(payload)
        if not res.get("ok"):
            raise RuntimeError(f"{code} 样本{k+1} 平台返回失败: {res.get('error')}")
        cpas = parse_cpas(res.get("log", ""))
        meta = payload["_meta"]
        bad = check_cpas(meta, cpas)
        rec = {"cpas": cpas, "bad": bad}
        if best is None or len(bad) < len(best["bad"]):
            best, best_attempt = rec, attempt
        if not bad:
            break
        print(f"    [retry] {code} 样本{k+1} attempt{attempt}: {'; '.join(bad)}",
              flush=True)
    if best_attempt != last:      # 磁盘文件须与最优尝试一致, 重跑最优 attempt
        payload = build_payload(code, cat, dist, cnt, cross, k, best_attempt)
        payload["OUT_DIR"] = sample_dir
        res = call_server(payload)
        cpas = parse_cpas(res.get("log", ""))
        bad = check_cpas(payload["_meta"], cpas)
    else:
        bad = best["bad"]
        cpas = best["cpas"]

    # 裁剪 hull / tow
    crop_info = {}
    for tag in ("hull", "tow"):
        src = os.path.join(sample_dir, f"btr_A_{tag}.png")
        dst = os.path.join(sample_dir, f"btr_A_{tag}_crop.png")
        if os.path.isfile(src):
            crop_info[tag] = crop_btr(src, dst)

    plat = payload["platforms"][0]
    meta_out = {
        "scenario_code": code,
        "category": f"平台{cat}",
        "scenario": scen_dir_name,
        "conditions": {
            "platform_motion": cat,
            "target_distance": "近距离(CPA 1.2~5km)" if dist == "近"
                               else "远距离(CPA ≥ 8km)",
            "target_count": f"{cnt}量(背景{sum(1 for t in payload['targets'] if not t['name'].startswith('XC'))}个)",
            "crossing": ("存在目标穿越平台(穿越时刻距离<0.6km)"
                         if cross else "所有目标不穿越平台"),
        },
        "sample": k + 1,
        "seed": payload["SEED"],
        "attempt_used": payload["_meta"]["attempt"],
        "sim_params": {k2: payload[k2] for k2 in
                       ("T_MAX", "DT", "NOISE_STD", "CLUTTER_LAM", "DB_RANGE")},
        "platform": {"name": plat["name"], "start_ll": plat["start_ll"],
                     "heading_deg": plat["heading_deg"],
                     "speed_kn": plat["speed_kn"],
                     "tow_offset_m": plat["tow_m"]},
        "targets": [{"name": t["name"],
                     "type": "穿越目标" if t["name"].startswith("XC") else "背景目标",
                     "speed_kn": t["speed_kn"], "amp": t["amp"],
                     "path_type": t["path_type"],
                     "t_on": t["t_on"], "t_off": t["t_off"]}
                    for t in payload["targets"]],
        "cpa_km_by_target": cpas,
        "label_check": "pass" if not bad else "warn: " + "; ".join(bad),
        "crop": crop_info,
        "files": sorted(os.listdir(sample_dir)),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "btr_gui_server.py /run (平台服务)",
    }
    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    return meta_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只生成 1 个样本做冒烟测试")
    ap.add_argument("--out", default=None, help="覆盖输出根目录")
    args = ap.parse_args()
    out_root = args.out or (os.path.join(WS, "训练数据集_smoke")
                            if args.smoke else ROOT)

    index = []
    if args.smoke:
        sc = SCENARIOS[3]          # 静止/近/多/穿越 —— 覆盖面最好
        m = gen_sample(sc, 0, out_root, smoke=True)
        index.append(m)
        print("[SMOKE]", json.dumps(m["conditions"], ensure_ascii=False))
        print("[SMOKE] cpa:", m["cpa_km_by_target"], "| check:", m["label_check"])
        print("[SMOKE] crop:", m["crop"])
    else:
        t0 = time.time()
        for si, sc in enumerate(SCENARIOS):
            for k in range(3):
                m = gen_sample(sc, k, out_root)
                index.append(m)
                print(f"[{len(index):2d}/48] {m['category']}/{m['scenario']}/"
                      f"样本{m['sample']} seed={m['seed']} "
                      f"check={m['label_check']}", flush=True)
        # 写 index.json
        idx_path = os.path.join(out_root, "index.json")
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "tool": "btr_gens/btr_gui_server.py (POST /run)",
                       "master_seed": MASTER_SEED,
                       "n_scenarios": len(SCENARIOS), "n_samples": len(index),
                       "samples": index}, f, ensure_ascii=False, indent=2)
        n_warn = sum(1 for m in index if not m["label_check"].startswith("pass"))
        print(f"\n完成: {len(index)} 个样本, {n_warn} 个含警告, "
              f"耗时 {time.time()-t0:.0f}s\nindex -> {idx_path}", flush=True)


if __name__ == "__main__":
    main()

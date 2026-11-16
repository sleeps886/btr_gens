# -*- coding: utf-8 -*-
"""按 20261026_141830 运行的 txt 参数重跑生成器, 验证图像与参数的一致性"""
import os
import sys

sys.path.insert(0, r"D:\Work\海洋声纳探测推论\btr_gens")
import btr_gui_server as S

params = {
    "T_MAX": 600.0, "DT": 1.0, "B_MIN": 0.0, "B_MAX": 180.0, "DB_STEP": 0.25,
    "EARTH_R": 6371000.0, "REF_LAT": 30.0, "REF_LON": 122.0,
    "NOISE_STD": 0.030, "CLUTTER_LAM": 0.012, "DB_RANGE": [-5, 16],
    "SEED": 20260911,
    "OUT_DIR": r"D:\Work\海洋声纳探测推论\infer_out\regen_A",
    "platforms": [{
        "name": "A",
        "start_ll": [30.0623, 122.0415],
        "heading_deg": 210.0, "speed_kn": 0.0,
        "hull_m": [0.0, 0.0], "tow_m": [-150.0, 0.0],
        "bounds": {"lat": [29.90, 30.15], "lon": [121.80, 122.30]},
        "color": "#ffd400",
    }],
    "targets": [
        {"name": "T1", "start": [30.0360, 122.1240], "speed_kn": 19.0,
         "path_type": "waypoints", "waypoints": [[30.0360, 122.0630]],
         "amp": 0.30, "sigma_hull": 3.5, "sigma_tow": 1.4,
         "t_on": 0, "t_off": 600, "color": "#ff9d00"},
        {"name": "T2", "start": [30.0225, 121.9690], "speed_kn": 23.0,
         "path_type": "curve",
         "curve_func": "30.0225 + 0.0045*np.sin(2*np.pi*t/400.0), "
                       "121.9690 + 0.0001227*t",
         "amp": 0.22, "sigma_hull": 4.0, "sigma_tow": 1.6,
         "t_on": 60, "t_off": 560, "color": "#ff00cc"},
        {"name": "T3", "start": [30.0045, 121.9780], "speed_kn": 29.0,
         "path_type": "waypoints", "waypoints": [[30.0045, 122.0580]],
         "amp": 0.18, "sigma_hull": 5.0, "sigma_tow": 2.0,
         "t_on": 150, "t_off": 450, "color": "#00ff88"},
        {"name": "T4", "start": [30.0630, 122.0620], "speed_kn": 10.0,
         "path_type": "waypoints", "waypoints": [[30.0630, 122.0090]],
         "amp": 0.10, "sigma_hull": 6.0, "sigma_tow": 2.4,
         "t_on": 100, "t_off": 560, "color": "#66aaff"},
    ],
}

res = S.run_simulation(params)
print("\n".join(res["log"][-14:]))
print("OK", res["ok"])
for pl in res["platforms"]:
    print(pl["imgs"])

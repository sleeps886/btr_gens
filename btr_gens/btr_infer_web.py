# -*- coding: utf-8 -*-
"""BTR 图像反推轨迹 —— Web 前端 (把 btr_infer_track 当库调用的演示)

纯标准库 HTTP 服务 + 单页前端, 推演逻辑完全复用 btr_infer_track 库:
    run_inference() -> draw_figure() -> write_txt()

用法 (任意工作目录):
    python btr_infer_web.py [--port 19477] [--host 127.0.0.1]
然后浏览器打开  http://127.0.0.1:19477

上传 hull/tow 两张 BTR 图 + 平台参数 -> "开始反演" -> 展示反演轨迹图、
目标清单与完整报告。也可点 "用示例图反演" 直接体验。
结果同时落盘到  <仓库>/infer_out/web_<时间戳>/ 。
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import btr_infer_track as T          # noqa: E402  (推演库)

OUT_ROOT = os.path.normpath(os.path.join(_HERE, "..", "infer_out"))
SAMPLE_DIR = os.path.join(OUT_ROOT, "regen_A")
MAX_BODY = 96 * 1024 * 1024

# ----------------------------------------------------------------------
# 推演执行 (库调用)
# ----------------------------------------------------------------------
_LOCK = threading.Lock()


def _targets_json(res_inf):
    """把库结果整理成前端友好的目标清单"""
    geom = res_inf["geom"]
    out = []
    for res in res_inf["results"]:
        states, mirrors, note = T.resolve_states(res)
        items = [("main", st) for st in states] + \
                [("mirror", st) for st in (mirrors or [])]
        branches = []
        for role, st in items:
            s = st["state"]
            spd = float(np.hypot(s[2], s[3]))
            course = (np.degrees(np.arctan2(s[2], s[3])) % 360.0
                      if spd > 1e-6 else 0.0)
            ll0 = geom.xy2ll(s[0], s[1])
            ll1 = geom.xy2ll(s[0] + s[2] * res["t1"], s[1] + s[3] * res["t1"])
            branches.append({
                "role": role, "rms": round(st["rms"], 3),
                "sh": st["sh"], "st": st["st"],
                "speed_kn": round(spd / T.KN2MS, 1),
                "course": round(course, 1),
                "t0_ll": [round(ll0[0], 4), round(ll0[1], 4)],
                "t1_ll": [round(ll1[0], 4), round(ll1[1], 4)],
            })
        out.append({
            "name": res["name"], "t0": res["t0"], "t1": res["t1"],
            "parallax": round(res["pair"]["dparallax"], 3),
            "ambiguous": res["fit"]["ambiguous"], "note": note,
            "branches": branches,
        })
    return out


def run_infer_request(payload):
    """执行一次反演: 校验参数 -> 落盘图片 -> 库调用 -> 组装响应"""
    p = payload.get("params") or {}
    f = lambda k, d: float(p.get(k, d))                       # noqa: E731
    start_ll = (f("lat", 30.0623), f("lon", 122.0415))
    heading = f("heading", 210.0) % 360.0
    speed_kn = f("speed_kn", 0.0)
    hull_m = (f("hull_x", 0.0), f("hull_y", 0.0))
    tow_m = (f("tow_x", -150.0), f("tow_y", 0.0))
    t_max = max(60.0, f("t_max", 600.0))
    db_thr = min(20.0, max(3.0, f("db_threshold", 7.0)))
    side = {"auto": "auto", "pos": "pos", "neg": "neg"}.get(
        str(p.get("side", "auto")).lower(), "auto")
    cmap = {"auto": "auto", "gui": "gui", "geo": "geo"}.get(
        str(p.get("cmap", "auto")).lower(), "auto")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"web_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    if payload.get("use_sample"):
        hull_path = os.path.join(SAMPLE_DIR, "btr_A_hull.png")
        tow_path = os.path.join(SAMPLE_DIR, "btr_A_tow.png")
        if not (os.path.isfile(hull_path) and os.path.isfile(tow_path)):
            raise FileNotFoundError("示例图不存在, 请先运行 regen_check.py "
                                    "或上传图片")
        tag = "A"
    else:
        for key in ("hull_b64", "tow_b64"):
            if not payload.get(key):
                raise ValueError("缺少图片: 请上传 hull 与 tow 两张 BTR 图")
        hull_path = os.path.join(out_dir, "upload_hull.png")
        tow_path = os.path.join(out_dir, "upload_tow.png")
        for key, path in (("hull_b64", hull_path), ("tow_b64", tow_path)):
            b64 = payload[key]
            if "," in b64 and b64.strip().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.save(path)
        tag = "U"

    log_lines = []

    def log(s):
        log_lines.append(str(s))
        print(s, flush=True)

    t0 = time.time()
    res_inf = T.run_inference(
        hull_path, tow_path, start_ll, heading, speed_kn,
        hull_m=hull_m, tow_m=tow_m, t_max=t_max,
        db_threshold=db_thr, side=side, cmap_style=cmap,
        log=log)
    tag_name = {"A": "A", "U": "U"}[tag]
    png_path = os.path.join(out_dir, f"btr_infer_{tag_name}.png")
    txt_path = os.path.join(out_dir, f"btr_infer_{tag_name}.txt")
    T.draw_figure(res_inf, png_path, tag=tag_name)
    T.write_txt(res_inf, txt_path, {
        "hull_img": hull_path, "tow_img": tow_path,
        "t_max": t_max, "b_min": 0.0, "b_max": 180.0,
        "db_threshold": db_thr}, tag=tag_name)
    elapsed = time.time() - t0

    with open(png_path, "rb") as fh:
        png_b64 = base64.b64encode(fh.read()).decode("ascii")
    with open(txt_path, "r", encoding="utf-8") as fh:
        txt = fh.read()

    return {
        "ok": True,
        "elapsed": round(elapsed, 1),
        "out_dir": out_dir,
        "png_b64": "data:image/png;base64," + png_b64,
        "txt": txt,
        "log": log_lines,
        "targets": _targets_json(res_inf),
        "counts": {"htracks": len(res_inf["htracks"]),
                   "ttracks": len(res_inf["ttracks"]),
                   "cmap": res_inf["cmap_style"]},
    }


# ----------------------------------------------------------------------
# HTTP 服务
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "BtrInferWeb/1.0"

    def log_message(self, fmt, *args):            # 安静一些
        sys.stderr.write("[%s] %s\n" % (self.address_string(), fmt % args))

    # -- GET --
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/api/health":
            self._json(200, {"ok": True, "lib": "btr_infer_track",
                             "time": datetime.now().isoformat(
                                 timespec="seconds")})
        else:
            self._json(404, {"ok": False, "error": f"no route {path}"})

    # -- POST --
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/infer":
            self._json(404, {"ok": False, "error": f"no route {path}"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                raise ValueError(f"请求体大小异常: {n}")
            raw = self.rfile.read(n)
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._json(400, {"ok": False, "error": f"请求解析失败: {exc}"})
            return
        with _LOCK:                                # 单线程跑推演, 避免争用
            try:
                result = run_infer_request(payload)
                self._json(200, result)
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"ok": False,
                                 "error": f"{exc}",
                                 "trace": traceback.format_exc()})

    # -- helpers --
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, "application/json; charset=utf-8", body)


# ----------------------------------------------------------------------
# 前端页面
# ----------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTR 图像反推轨迹</title>
<style>
  :root { --bg:#0a0f1f; --card:#101828; --line:#22304d; --fg:#e8eefc;
          --dim:#8ea3c8; --acc:#ffd400; --ok:#00ff88; --err:#ff5555; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.6 "Microsoft YaHei", system-ui, sans-serif; }
  .wrap { max-width:1180px; margin:0 auto; padding:22px 16px 60px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--dim); font-size:12.5px; margin-bottom:18px; }
  .grid { display:grid; grid-template-columns: 400px 1fr; gap:16px; }
  @media (max-width:900px){ .grid { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:10px; padding:14px 16px; }
  .card h2 { font-size:14px; margin:0 0 10px; color:var(--acc); }
  label { display:block; font-size:12px; color:var(--dim); margin:8px 0 3px; }
  input[type=text], input[type=number], select {
      width:100%; background:#0b1224; color:var(--fg);
      border:1px solid var(--line); border-radius:6px; padding:6px 8px;
      font-size:13px; }
  .row { display:flex; gap:10px; } .row > div { flex:1; }
  .drop { border:1.5px dashed var(--line); border-radius:8px; padding:10px;
          text-align:center; color:var(--dim); font-size:12px; cursor:pointer; }
  .drop:hover { border-color:var(--acc); color:var(--fg); }
  .drop img { max-width:100%; max-height:150px; display:block; margin:6px auto 0;
              border:1px solid var(--line); border-radius:4px; }
  button { background:var(--acc); color:#111; border:0; border-radius:7px;
           padding:10px 18px; font-size:14px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.45; cursor:wait; }
  button.ghost { background:transparent; color:var(--acc);
                 border:1px solid var(--acc); font-weight:400;
                 padding:7px 12px; font-size:12.5px; }
  .msg { margin-top:10px; font-size:12.5px; white-space:pre-wrap; }
  .msg.err { color:var(--err); } .msg.ok { color:var(--ok); }
  details { margin-top:10px; font-size:12.5px; color:var(--dim); }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th,td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }
  th { color:var(--dim); font-weight:500; }
  .b-main { color:var(--ok); } .b-mir { color:var(--dim); }
  #resultImg { max-width:100%; border:1px solid var(--line); border-radius:8px; }
  pre.txt { background:#0b1224; border:1px solid var(--line); border-radius:8px;
            padding:12px; font-size:12px; overflow:auto; max-height:420px;
            white-space:pre-wrap; }
  pre.log { background:#0b1224; border:1px solid var(--line); border-radius:8px;
            padding:12px; font-size:11.5px; color:var(--dim); overflow:auto;
            max-height:200px; }
  .spin { display:inline-block; width:13px; height:13px; border-radius:50%;
          border:2px solid var(--acc); border-top-color:transparent;
          animation:r 0.8s linear infinite; vertical-align:-2px; margin-right:6px; }
  @keyframes r { to { transform:rotate(360deg); } }
  .hidden { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>BTR 图像反推轨迹</h1>
  <div class="sub">上传 舷侧阵(hull) 与 拖曳阵(tow) 方位历程图, 输入平台与阵位参数,
     调用 <code>btr_infer_track</code> 推演库反演出目标轨迹。典型耗时 1~4 分钟。</div>

  <div class="grid">
    <div>
      <div class="card">
        <h2>① 输入图像</h2>
        <label>舷侧阵 BTR (hull)</label>
        <div class="drop" id="dropHull">点击选择 PNG 图像</div>
        <label>拖曳阵 BTR (tow)</label>
        <div class="drop" id="dropTow">点击选择 PNG 图像</div>
        <div class="row" style="margin-top:10px">
          <button class="ghost" id="btnSample">用示例图反演</button>
        </div>
      </div>

      <div class="card" style="margin-top:14px">
        <h2>② 平台与阵位参数</h2>
        <div class="row">
          <div><label>纬度 (°N)</label><input type="number" id="lat" value="30.0623" step="0.0001"></div>
          <div><label>经度 (°E)</label><input type="number" id="lon" value="122.0415" step="0.0001"></div>
        </div>
        <div class="row">
          <div><label>航向 (°, 北=0 顺时针)</label><input type="number" id="heading" value="210" step="1"></div>
          <div><label>航速 (kn)</label><input type="number" id="speed_kn" value="0" step="0.1"></div>
        </div>
        <div class="row">
          <div><label>舷侧阵 x 偏移 (m, +艏)</label><input type="number" id="hull_x" value="0" step="1"></div>
          <div><label>舷侧阵 y 偏移 (m, +左舷)</label><input type="number" id="hull_y" value="0" step="1"></div>
        </div>
        <div class="row">
          <div><label>拖曳阵 x 偏移 (m, +艏)</label><input type="number" id="tow_x" value="-150" step="1"></div>
          <div><label>拖曳阵 y 偏移 (m, +左舷)</label><input type="number" id="tow_y" value="0" step="1"></div>
        </div>
        <details>
          <summary>高级选项</summary>
          <div class="row">
            <div><label>时间轴总长 (s)</label><input type="number" id="t_max" value="600" step="10"></div>
            <div><label>检测门限 (dB)</label><input type="number" id="db_threshold" value="7" step="0.5"></div>
          </div>
          <div class="row">
            <div><label>左右舷</label><select id="side">
              <option value="auto">auto (输出两支候选)</option>
              <option value="pos">pos (强制 +1)</option>
              <option value="neg">neg (强制 -1)</option></select></div>
            <div><label>配色还原</label><select id="cmap">
              <option value="auto">auto</option>
              <option value="gui">gui</option>
              <option value="geo">geo</option></select></div>
          </div>
        </details>
        <div style="margin-top:14px">
          <button id="btnGo">开始反演</button>
        </div>
        <div class="msg" id="msg"></div>
      </div>
    </div>

    <div>
      <div class="card hidden" id="cardTargets">
        <h2>③ 反推目标清单</h2>
        <div id="targets"></div>
      </div>
      <div class="card hidden" id="cardFig" style="margin-top:14px">
        <h2>④ 反演轨迹图</h2>
        <img id="resultImg" alt="反演结果">
      </div>
      <div class="card hidden" id="cardTxt" style="margin-top:14px">
        <h2>⑤ 反演报告</h2>
        <pre class="txt" id="txtOut"></pre>
      </div>
      <div class="card hidden" id="cardLog" style="margin-top:14px">
        <h2>运行日志</h2>
        <pre class="log" id="logOut"></pre>
      </div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const files = { hull: null, tow: null };

function bindDrop(dropId, key) {
  const d = $(dropId);
  d.addEventListener("click", () => {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = "image/png,image/jpeg";
    inp.onchange = () => { if (inp.files[0]) setFile(key, inp.files[0], d); };
    inp.click();
  });
}
function setFile(key, file, d) {
  files[key] = file;
  d.innerHTML = (key === "hull" ? "舷侧阵: " : "拖曳阵: ") + file.name +
    ' <span style="color:#8ea3c8">(' + Math.round(file.size/1024) + " KB)</span>" +
    '<img src="' + URL.createObjectURL(file) + '">';
}
bindDrop("dropHull", "hull");
bindDrop("dropTow", "tow");

function toB64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

async function go(useSample) {
  const msg = $("msg");
  const btn = $("btnGo");
  if (!useSample && (!files.hull || !files.tow)) {
    msg.className = "msg err";
    msg.textContent = "请先上传两张 BTR 图 (或点击『用示例图反演』)";
    return;
  }
  btn.disabled = true;
  msg.className = "msg ok";
  msg.innerHTML = '<span class="spin"></span>反演中, 约 1~4 分钟, 请稍候…';
  try {
    const payload = { use_sample: !!useSample, params: {
      lat: +$("lat").value, lon: +$("lon").value,
      heading: +$("heading").value, speed_kn: +$("speed_kn").value,
      hull_x: +$("hull_x").value, hull_y: +$("hull_y").value,
      tow_x: +$("tow_x").value, tow_y: +$("tow_y").value,
      t_max: +$("t_max").value, db_threshold: +$("db_threshold").value,
      side: $("side").value, cmap: $("cmap").value } };
    if (!useSample) {
      payload.hull_b64 = await toB64(files.hull);
      payload.tow_b64 = await toB64(files.tow);
    }
    const r = await fetch("/api/infer", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || "服务器错误");
    msg.className = "msg ok";
    msg.textContent = "完成, 耗时 " + j.elapsed + " s, 结果已存 " + j.out_dir;

    // 目标清单
    const tb = $("targets");
    if (!j.targets.length) {
      tb.innerHTML = '<span style="color:#8ea3c8">未反推出目标 (可尝试降低检测门限)</span>';
    } else {
      let h = "<table><tr><th>目标</th><th>窗口</th><th>视差中位</th>" +
              "<th>支</th><th>位置 t0 → t1</th><th>航速/航向</th><th>RMS</th></tr>";
      for (const t of j.targets) {
        t.branches.forEach((b, i) => {
          h += "<tr>" + (i === 0
              ? '<td rowspan="' + t.branches.length + '">' + t.name +
                (t.ambiguous ? ' <span class="b-mir">(镜像模糊)</span>' : "") + "</td>" +
                '<td rowspan="' + t.branches.length + '">[' + Math.round(t.t0) + "," +
                Math.round(t.t1) + "]s</td>" +
                '<td rowspan="' + t.branches.length + '">' + t.parallax + "°</td>" : "")
            + '<td class="' + (b.role === "main" ? "b-main" : "b-mir") + '">' +
              (b.role === "main" ? "主输出" : "镜像候选") + "</td>"
            + "<td>(" + b.t0_ll[0] + "," + b.t0_ll[1] + ") → (" +
              b.t1_ll[0] + "," + b.t1_ll[1] + ")</td>"
            + "<td>" + b.speed_kn + " kn / " + b.course + "°</td>"
            + "<td>" + b.rms + "°</td></tr>";
        });
      }
      tb.innerHTML = h + "</table>";
    }
    $("cardTargets").classList.remove("hidden");
    $("resultImg").src = j.png_b64;
    $("cardFig").classList.remove("hidden");
    $("txtOut").textContent = j.txt;
    $("cardTxt").classList.remove("hidden");
    $("logOut").textContent = (j.log || []).join("\n");
    $("cardLog").classList.remove("hidden");
  } catch (e) {
    msg.className = "msg err";
    msg.textContent = "失败: " + e.message;
  } finally {
    btn.disabled = false;
  }
}
$("btnGo").addEventListener("click", () => go(false));
$("btnSample").addEventListener("click", () => go(true));
fetch("/api/health").then(r => r.json()).then(j => {
  if (!j.ok) throw 0;
  const m = document.querySelector(".sub");
  m.textContent += " | 推演库已连接: " + j.lib;
}).catch(() => {});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="BTR 图像反推轨迹 Web 前端 (调用 btr_infer_track 库)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19477)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"BTR 反演前端: http://{args.host}:{args.port}  "
          f"(推演库: {_HERE}\\btr_infer_track.py, 输出根: {OUT_ROOT})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

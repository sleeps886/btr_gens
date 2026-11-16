# -*- coding: utf-8 -*-
"""用 Ollama 服务上的新模型 (qwen3-vl:8b) 在同一验证集上做零样本推演评测。
用法: python eval_ollama.py [--model qwen3-vl:8b] [--out reports/qw38_zeroshot.json]
与 eval_btr.py 完全同口径: 同 96 张验证图、同指令、同归一化、同六项指标。
"""
import argparse
import base64
import json
import re
import time
import urllib.request

sys_path = "/data/sdb/btr/lfdata"
VAL = f"{sys_path}/sft_val.jsonl"
FIELDS = ["平台状态", "距离等级", "目标数量", "目标总数", "穿越", "穿越目标数"]

KEY_MAP = {
    "platform_status": "平台状态", "platform_motion_state": "平台状态",
    "platform_motion": "平台状态", "motion_state": "平台状态",
    "platform_state": "平台状态",
    "target_distance": "距离等级", "distance_grade": "距离等级",
    "distance_level": "距离等级", "distance": "距离等级",
    "target_count": "目标数量", "target_quantity": "目标数量",
    "total_targets": "目标总数", "target_total": "目标总数",
    "number_of_targets": "目标总数",
    "target_crossing_platform": "穿越", "cross_platform": "穿越",
    "crossing": "穿越", "trajectory_crosses_platform": "穿越",
    "crossing_platform": "穿越", "穿越平台": "穿越",
    "crossing_target_count": "穿越目标数", "crossing_targets": "穿越目标数",
}
VALUE_MAP = {"moving": "移动", "stationary": "静止", "static": "静止",
             "near": "近", "close": "近", "far": "远",
             "few": "少", "many": "多", True: "是", False: "否",
             "yes": "是", "no": "否", "crossing": "是"}


def normalize_pred(p):
    if not isinstance(p, dict):
        return None
    return {KEY_MAP.get(k, k): (VALUE_MAP.get(v, v) if isinstance(v, str) else v)
            for k, v in p.items()}


def extract_json(text):
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        return normalize_pred(json.loads(m.group(0)))
    except Exception:
        return None


def chat_ollama(model, instruction, img_b64, timeout=180):
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": instruction,
                      "images": [img_b64]}],
        "options": {"temperature": 0, "num_predict": 160},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--out", default="/data/sdb/btr/reports/qw38_zeroshot.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(VAL, encoding="utf-8")]
    print(f"模型: {args.model} | 验证样本: {len(rows)}")
    stats = {f: [0, 0] for f in FIELDS}
    all_ok = 0
    latencies = []
    details = []
    for i, row in enumerate(rows):
        gt = extract_json(row["messages"][1]["content"])
        img_path = f"{sys_path}/{row['images'][0]}"
        b64 = base64.b64encode(open(img_path, "rb").read()).decode()
        instruction = re.sub(r"^<image>", "", row["messages"][0]["content"])
        t0 = time.time()
        try:
            resp = chat_ollama(args.model, instruction, b64)
        except Exception as e:
            resp = f"ERROR: {e}"
        dt = time.time() - t0
        latencies.append(dt)
        pred = extract_json(resp)
        row_ok = pred is not None
        for f in FIELDS:
            stats[f][1] += 1
            if pred is not None and pred.get(f) == gt.get(f):
                stats[f][0] += 1
            else:
                row_ok = False
        all_ok += row_ok
        details.append({"image": row["images"][0], "pred": pred, "gt": gt,
                        "resp": resp[:200], "latency_s": round(dt, 2),
                        "all_fields_ok": bool(row_ok)})
        print(f"[{i+1}/{len(rows)}] {'PASS' if row_ok else 'DIFF'} "
              f"{dt:.1f}s {row['images'][0]}", flush=True)

    report = {
        "mode": f"ollama_zeroshot", "model": args.model, "n": len(rows),
        "field_accuracy": {f: round(stats[f][0] / stats[f][1], 3) for f in FIELDS},
        "all_fields_accuracy": round(all_ok / len(rows), 3),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "details": details,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("== 汇总 ==")
    for f in FIELDS:
        print(f"  {f}: {stats[f][0]}/{stats[f][1]}")
    print(f"  全对率: {report['all_fields_accuracy']}, 平均时延: "
          f"{report['avg_latency_s']}s")
    print("REPORT_SAVED", args.out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""BTR 快速推演批量评测: 基线(base) / 微调(base+LoRA) 用同一脚本对比。
用法:
  python eval_btr.py --mode base   --out /data/sdb/btr/reports/base.json
  python eval_btr.py --mode ft     --adapter /data/sdb/btr/saves/.../checkpoint-XX \
                                   --out /data/sdb/btr/reports/ft.json
对 sft_val.jsonl 的 32 张验证图逐张推演, 解析输出 JSON 与真值比对:
平台状态/距离等级/目标数量/目标总数/穿越/穿越目标数 六项准确率 + 全对率 + 平均时延。
"""
import argparse
import json
import re
import time

import torch
from PIL import Image

MODEL = "/data/sdb/models/Qwen/Qwen2.5-VL-7B-Instruct"
VAL = "/data/sdb/btr/lfdata/sft_val.jsonl"
IMGDIR = "/data/sdb/btr/lfdata"
FIELDS = ["平台状态", "距离等级", "目标数量", "目标总数", "穿越", "穿越目标数"]


def load_model(adapter=None):
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as VLModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4")
    model = VLModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                    quantization_config=bnb, device_map="cuda:0")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    return model, processor


def build_inputs(processor, instruction, image):
    """优先 qwen_vl_utils, 无则用 PIL 兜底"""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": instruction}]}]
    try:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        imgs, _ = process_vision_info(messages)
        return processor(text=[text], images=imgs, return_tensors="pt").to("cuda:0")
    except ImportError:
        text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": instruction}]}],
            tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=[image],
                         return_tensors="pt").to("cuda:0")


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
    "crossing_platform": "穿越",
    "crossing_target_count": "穿越目标数", "crossing_targets": "穿越目标数",
    "穿越平台": "穿越",
}
VALUE_MAP = {"moving": "移动", "stationary": "静止", "static": "静止",
             "near": "近", "close": "近", "far": "远",
             "few": "少", "many": "多", True: "是", False: "否",
             "yes": "是", "no": "否", "crossing": "是"}


def normalize_pred(p):
    if not isinstance(p, dict):
        return None
    out = {}
    for k, v in p.items():
        ck = KEY_MAP.get(k, k)
        cv = VALUE_MAP.get(v, v)
        if isinstance(cv, str):
            cv = cv.strip()
        out[ck] = cv
    return out


def extract_json(text):
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        return normalize_pred(json.loads(m.group(0)))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "ft"], required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, processor = load_model(args.adapter if args.mode == "ft" else None)
    rows = [json.loads(l) for l in open(VAL, encoding="utf-8")]
    print(f"验证样本: {len(rows)}")

    stats = {f: [0, 0] for f in FIELDS}
    all_ok = 0
    latencies = []
    details = []
    for i, row in enumerate(rows):
        gt = extract_json(row["messages"][1]["content"])
        img_path = row["images"][0]
        image = Image.open(f"{IMGDIR}/{img_path}").convert("RGB")
        instruction = re.sub(r"^<image>", "", row["messages"][0]["content"])
        inputs = build_inputs(processor, instruction, image)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        dt = time.time() - t0
        latencies.append(dt)
        resp = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]
        pred = extract_json(resp)
        row_ok = pred is not None
        for f in FIELDS:
            stats[f][1] += 1
            if pred is not None and pred.get(f) == gt.get(f):
                stats[f][0] += 1
            else:
                row_ok = False
        all_ok += row_ok
        details.append({"image": img_path, "pred": pred, "gt": gt,
                        "resp": resp[:200], "latency_s": round(dt, 2),
                        "all_fields_ok": bool(row_ok)})
        print(f"[{i+1}/{len(rows)}] {'PASS' if row_ok else 'DIFF'} "
              f"{dt:.1f}s {img_path}", flush=True)

    report = {
        "mode": args.mode, "adapter": args.adapter, "n": len(rows),
        "field_accuracy": {f: round(stats[f][0] / stats[f][1], 3) for f in FIELDS},
        "all_fields_accuracy": round(all_ok / len(rows), 3),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "details": details,
    }
    import os
    os.makedirs("/data/sdb/btr/reports", exist_ok=True)
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

#!/bin/bash
python3 - <<'PY'
import json, os
for d in ["/data/sdb/models/Qwen/Qwen2.5-VL-7B-Instruct",
          "/data/sdb/models/Qwen/Qwen2-VL-7B-Instruct"]:
    print("---", d, "---")
    try:
        cfg = json.load(open(os.path.join(d, "config.json")))
        print("model_type:", cfg.get("model_type"),
              "| layers:", cfg.get("num_hidden_layers"),
              "| arch:", cfg.get("architectures"))
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
            need = sorted(set(wm.values()))
            have = set(os.listdir(d))
            miss = [f for f in need if f not in have]
            print("分片:", len(need), "缺失:", miss or "无 → 完整可用")
        else:
            files = [f for f in os.listdir(d) if f.endswith(".safetensors")]
            print("无 index, 直接分片:", len(files))
    except Exception as e:
        print("ERROR:", e)
PY
echo "=== 文件级抽查(7B 应有 preprocessor/chat template) ==="
ls /data/sdb/models/Qwen/Qwen2.5-VL-7B-Instruct/ | grep -vE 'safetensors$'

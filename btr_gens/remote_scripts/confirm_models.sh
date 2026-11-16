#!/bin/bash
echo "=== 1) /data/sdb/models/Qwen/ 下所有子目录 ==="
ls -la /data/sdb/models/Qwen/
for d in /data/sdb/models/Qwen/*/; do
  echo "--- $d ($(du -sh "$d" 2>/dev/null | cut -f1)) ---"
  grep -o '"model_type": *"[^"]*"' "$d/config.json" 2>/dev/null | head -1
  grep -o '"architectures": *\[[^]]*\]' "$d/config.json" 2>/dev/null | head -1
  grep -o '"num_hidden_layers": *[0-9]*' "$d/config.json" 2>/dev/null | head -1
  grep -o '"quantization_config"' "$d/config.json" 2>/dev/null | head -1
done

echo "=== 2) Qwen2.5-VL-72B 完整性校验 ==="
D72=/data/sdb/models/Qwen/Qwen2.5-VL-72B-Instruct
if [ -f "$D72/model.safetensors.index.json" ]; then
  python3 - <<'PY'
import json, os
d = "/data/sdb/models/Qwen/Qwen2.5-VL-72B-Instruct"
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
have = set(os.listdir(d))
missing = [f for f in need if f not in have]
print("需分片:", len(need), "在盘:", len(need) - len(missing), "缺失:", missing[:5] or "无")
PY
else
  echo "无 index.json"
fi

echo "=== 3) OceanGPT-V 详情 ==="
D7=/data/sdb/models/oceangpt-v
grep -oE '"(model_type|architectures|hidden_size|num_hidden_layers)": *("[^"]*"|[0-9]+)' $D7/config.json | head -6
ls $D7 | grep -vE 'safetensors$' | head -12
python3 - <<'PY'
import json, os
d = "/data/sdb/models/oceangpt-v"
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    wm = json.load(open(idx))["weight_map"]
    need = sorted(set(wm.values()))
    have = set(os.listdir(d))
    print("OceanGPT-V 分片:", len(need), "缺失:", [f for f in need if f not in have] or "无")
PY

echo "=== 4) Qwen3-8B-FP8 详情 ==="
D8=/data/sdb/vllm/models/Qwen3-8B-FP8
grep -oE '"(model_type|architectures)": *("[^"]*"|\[[^]]*\])' $D8/config.json | head -4
grep -o '"quant_method": *"[^"]*"' $D8/config.json | head -1
ls $D8 | head -8

echo "=== 5) ollama-qw3.8 的 manifest 与半截 blob ==="
cat /data/sdb/ollama-qw3.8/manifests/registry.ollama.ai/hc/qwen3-8b-local/q4 2>/dev/null
echo ""
echo "=== 6) SFT 上传状态 ==="
ls -la /data/sdb/btr/*.part* 2>/dev/null | wc -l
ls -la /data/sdb/btr/btr_sft_20261111.tar.gz 2>/dev/null

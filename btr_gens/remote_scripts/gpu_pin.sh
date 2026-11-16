#!/bin/bash
echo 'zju123!@#' | sudo -S -p '' mkdir -p /etc/systemd/system/ollama.service.d
echo 'zju123!@#' | sudo -S -p '' tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="OLLAMA_KEEP_ALIVE=2h"
EOF
echo 'zju123!@#' | sudo -S -p '' systemctl daemon-reload
echo 'zju123!@#' | sudo -S -p '' systemctl restart ollama
sleep 8
pkill -f eval_ollama.py
# 预热: 让 qwen3-vl 加载到 GPU1
python3 - <<'PY'
import json, urllib.request, time
body = json.dumps({"model": "qwen3-vl:8b", "stream": False,
                   "messages": [{"role": "user", "content": "hi"}],
                   "options": {"num_gpu": 999}}).encode()
req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
urllib.request.urlopen(req, timeout=600).read()
print("warmup %.1fs" % (time.time() - t0))
PY
OLLAMA_MODELS=/data/sdb/ollama-qw3.8 ollama ps
nohup python3 /data/sdb/btr/eval_ollama.py --model qwen3-vl:8b \
    --out /data/sdb/btr/reports/qw38_zeroshot.json \
    > /data/sdb/btr/logs/eval_qw38.log 2>&1 &
echo eval_relaunched
sleep 60
tail -2 /data/sdb/btr/logs/eval_qw38.log

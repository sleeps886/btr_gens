#!/bin/bash
echo 'zju123!@#' | sudo -S -p '' tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_MODELS=/data/sdb/ollama-qw3.8"
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="OLLAMA_KEEP_ALIVE=2h"
EOF
echo 'zju123!@#' | sudo -S -p '' systemctl daemon-reload
echo 'zju123!@#' | sudo -S -p '' systemctl restart ollama
sleep 8
echo "=== 模型根与列表 ==="
echo 'zju123!@#' | sudo -S -p '' systemctl show ollama -p Environment
OLLAMA_MODELS=/data/sdb/ollama-qw3.8 ollama list 2>&1 | head -6
echo "=== 预热对话 ==="
python3 - <<'PY'
import json, urllib.request, time
body = json.dumps({"model": "qwen3-vl:8b", "stream": False,
                   "messages": [{"role": "user", "content": "hi"}],
                   "options": {"num_gpu": 999}}).encode()
req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
print("warmup %.1fs ->" % (time.time() - t0), r["message"]["content"][:60])
PY
OLLAMA_MODELS=/data/sdb/ollama-qw3.8 ollama ps

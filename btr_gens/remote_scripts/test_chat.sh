#!/bin/bash
echo "=== ollama list ==="
OLLAMA_MODELS=/data/sdb/ollama-qw3.8 ollama list 2>&1
echo "=== 有效环境 ==="
echo 'zju123!@#' | sudo -S -p '' systemctl show ollama -p Environment 2>&1
echo "=== 试对话 ==="
python3 - <<'PY'
import json, urllib.request
body = json.dumps({"model": "qwen3-vl:8b", "stream": False,
                   "messages": [{"role": "user", "content": "hi"}]}).encode()
req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                             headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=300)
    d = json.loads(r.read().decode())
    print("chat OK:", d["message"]["content"][:80])
except Exception as e:
    print("chat FAIL:", e)
    try:
        print(e.read().decode()[:300])
    except Exception:
        pass
PY

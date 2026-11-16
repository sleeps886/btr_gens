#!/bin/bash
# 公共用户第二 ollama 实例 (11435), 指向 qw3.8 模型根, 免 sudo
export OLLAMA_MODELS=/data/sdb/ollama-qw3.8
export OLLAMA_HOST=127.0.0.1:11435
export CUDA_VISIBLE_DEVICES=1
export OLLAMA_KEEP_ALIVE=2h
pkill -f "eval_ollama.py"
if ! curl -s --max-time 3 http://127.0.0.1:11435/api/version > /dev/null; then
  nohup /usr/local/bin/ollama serve > /data/sdb/btr/logs/ollama2.log 2>&1 &
  echo "ollama2_pid $!"
  sleep 8
fi
curl -s http://127.0.0.1:11435/api/version; echo ""
echo "=== tags ==="
curl -s http://127.0.0.1:11435/api/tags | python3 -c "import json,sys; d=json.load(sys.stdin); [print(m['name'], m['size']) for m in d['models']]" 2>&1
echo "=== 预热 ==="
python3 - <<'PY'
import json, urllib.request, time
body = json.dumps({"model": "qwen3-vl:8b", "stream": False,
                   "messages": [{"role": "user", "content": "hi"}],
                   "options": {"num_gpu": 999}}).encode()
req = urllib.request.Request("http://127.0.0.1:11435/api/chat", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
print("warmup %.1fs ->" % (time.time() - t0), r["message"]["content"][:60])
PY
OLLAMA_MODELS=/data/sdb/ollama-qw3.8 ollama ps 2>/dev/null | head -3
curl -s http://127.0.0.1:11435/api/ps | python3 -c "import json,sys; d=json.load(sys.stdin); [print(m['name'], '|', m.get('size_vram'), 'vram') for m in d['models']]" 2>&1

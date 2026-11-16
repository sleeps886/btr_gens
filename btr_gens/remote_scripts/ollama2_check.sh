#!/bin/bash
export OLLAMA_MODELS=/data/sdb/ollama-qw3.8
export OLLAMA_HOST=127.0.0.1:11435
export CUDA_VISIBLE_DEVICES=1
export OLLAMA_KEEP_ALIVE=2h
if ! curl -s --max-time 3 http://127.0.0.1:11435/api/version > /dev/null 2>&1; then
  echo "11435 未运行, 启动 ..."
  nohup /usr/local/bin/ollama serve > /data/sdb/btr/logs/ollama2.log 2>&1 &
  sleep 8
fi
curl -s --max-time 5 http://127.0.0.1:11435/api/version; echo ""
curl -s http://127.0.0.1:11435/api/tags | python3 -c "import json,sys; d=json.load(sys.stdin); [print(m['name']) for m in d['models']]" 2>&1
echo "=== 预热 + GPU 占用确认 ==="
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
curl -s http://127.0.0.1:11435/api/ps | python3 -c "import json,sys; d=json.load(sys.stdin); [print(m['name'], '| vram:', m.get('size_vram')) for m in d['models']]" 2>&1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3

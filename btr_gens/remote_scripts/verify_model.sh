#!/bin/bash
echo "=== 1) 现役模型的 ollama config blob ==="
cat /data/sdb/ollama-qw3.8/blobs/sha256-b2dd1377ec969b82f67561e2ec0c897788b00f4aaeaf6298a17e05f1f9290f97 2>/dev/null
echo ""
echo "=== 2) GGUF 头解析: 现役 q4 blob ==="
python3 /tmp/gguf_probe.py /data/sdb/ollama-qw3.8/blobs/sha256-d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
echo "=== 3) GGUF 头解析: 4.9G 半截 blob(中断的拉取) ==="
ls -lh /data/sdb/ollama-qw3.8/blobs/ | grep partial
python3 /tmp/gguf_probe.py /data/sdb/ollama-qw3.8/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f-partial
echo "=== 4) 全盘 qwen3.8 / qw3.8 相关文件 ==="
find /data/sdb -maxdepth 4 -iname "*qwen3.8*" 2>/dev/null | head
find /data/sdb -maxdepth 4 -iname "*qw3.8*" 2>/dev/null | head
echo "=== 5) 主 ollama 根(/data/sdb/ollama)的 manifests ==="
find /data/sdb/ollama -path "*manifests*" -type f 2>/dev/null | head -10
echo "=== 6) 运行中的 ollama 服务用的模型根 ==="
systemctl cat ollama 2>/dev/null | grep -iE 'models|environment' | head -5
ps aux | grep "ollama serve" | grep -v grep | head -2
echo "=== 7) LF 可用模板(找 qwen2_5_vl / qwen2_vl) ==="
source /data/sdb/btr/lfenv/bin/activate
python -c "
from llamafactory.data.template import TEMPLATES as T1
names = sorted(T1.keys())
print([n for n in names if 'qwen' in n])
" 2>&1 | tail -3

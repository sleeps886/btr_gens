#!/bin/bash
echo "=== /data/sdb/ollama-qwen3.8/ 目录结构 ==="
ls -la /data/sdb/ollama-qwen3.8/ 2>/dev/null | head -20
echo "=== 该目录下的模型文件(safetensors/gguf/bin) ==="
find /data/sdb/ollama-qwen3.8 -maxdepth 4 \( -name "*.safetensors" -o -name "*.gguf" -o -name "config.json" \) -exec ls -lh {} \; 2>/dev/null | head -30
echo "=== 各子目录大小 ==="
du -sh /data/sdb/ollama-qwen3.8/* 2>/dev/null | head -15
echo "=== /data/sdb 下其他模型目录(含 safetensors 的) ==="
find /data/sdb -maxdepth 3 -name "config.json" -path "*model*" 2>/dev/null | head -10
find /data/sdb -maxdepth 2 -type d 2>/dev/null | head -20
echo "=== 磁盘上的大 safetensors(>1GB, 全盘) ==="
find /data/sdb -maxdepth 5 -name "*.safetensors" -size +1G 2>/dev/null | head -15

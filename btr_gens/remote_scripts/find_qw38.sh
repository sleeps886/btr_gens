#!/bin/bash
echo "=== 1) ollama-qw3.8 根: manifests + 新 blob ==="
find /data/sdb/ollama-qw3.8/manifests -type f 2>/dev/null | head -8
ls -lht /data/sdb/ollama-qw3.8/blobs/ 2>/dev/null | head -6
echo "=== 2) 主 ollama 根 manifests ==="
find /data/sdb/ollama -path "*manifests*" -type f 2>/dev/null | head -10
echo "=== 3) 最近 24h 内新增/修改的大文件(>500MB) ==="
find /data/sdb -maxdepth 5 -size +500M -mmin -1440 2>/dev/null | head -15
echo "=== 4) models 目录最新状态 ==="
ls -lt /data/sdb/models/Qwen/ | head -6
ls -lt /data/sdb/vllm/models/ | head -5
echo "=== 5) ollama list(两个根) ==="
OLLAMA_MODELS=/data/sdb/ollama-qw3.8 ollama list 2>&1 | head -6
echo "=== 6) ollama library 里是否有 qwen3.8(测探, 5 秒超时) ==="
timeout 5 curl -s -o /dev/null -w '%{http_code}' 'https://registry.ollama.ai/v2/library/qwen3.8/manifests/latest' 2>&1; echo ""

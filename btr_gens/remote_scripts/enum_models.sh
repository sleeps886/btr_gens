#!/bin/bash
echo "=== /data/sdb/models/ 全部子目录及大小 ==="
du -sh /data/sdb/models/*/ 2>/dev/null | head -20
echo "=== /data/sdb/vllm/models/ ==="
du -sh /data/sdb/vllm/models/*/ 2>/dev/null | head -10
echo "=== 各候选模型的 config(model_type/架构) ==="
for d in /data/sdb/models/*/ /data/sdb/vllm/models/*/; do
  if [ -f "$d/config.json" ]; then
    mt=$(grep -o '"model_type": *"[^"]*"' "$d/config.json" | head -1)
    ha=$(grep -o '"architectures": *\[[^]]*\]' "$d/config.json" | head -1)
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "$d | $sz | $mt | $ha"
  fi
done
echo "=== ollama-qw3.8 manifests(该 ollama 根有哪些模型) ==="
find /data/sdb/ollama-qw3.8/manifests -type f 2>/dev/null | head -5
echo "=== ollama-qw3.8 blobs 大小 ==="
du -sh /data/sdb/ollama-qw3.8/blobs 2>/dev/null
ls -lh /data/sdb/ollama-qw3.8/blobs 2>/dev/null | sort -k5 -h | tail -5
echo "=== SFT 分块上传进度 ==="
ls -la /data/sdb/btr/btr_sft_20261111.tar.gz.part* 2>/dev/null | wc -l
ls -la /data/sdb/btr/btr_sft_20261111.tar.gz 2>/dev/null

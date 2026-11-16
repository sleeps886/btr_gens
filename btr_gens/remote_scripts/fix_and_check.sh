#!/bin/bash
echo "=== 1) 修正训练模板并重启 ==="
cd /data/sdb/btr
sed -i 's/^template: qwen2_5_vl/template: qwen2_vl/' qwen2.5vl-7b-qlora-btr.yaml
grep '^template:' qwen2.5vl-7b-qlora-btr.yaml
source lfenv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup llamafactory-cli train /data/sdb/btr/qwen2.5vl-7b-qlora-btr.yaml \
    > /data/sdb/btr/logs/train.log 2>&1 &
echo "train_pid $!"

echo "=== 2) hc 的 ollama 拉取: 半截 blob 是否在增长 ==="
S1=$(stat -c%s /data/sdb/ollama-qw3.8/blobs/sha256-06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439-partial 2>/dev/null)
sleep 30
S2=$(stat -c%s /data/sdb/ollama-qw3.8/blobs/sha256-06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439-partial 2>/dev/null)
echo "30 秒增量: $(( $S2 - $S1 )) bytes (当前 $S2)"

echo "=== 3) 该半截 blob 的 GGUF 头(若从头下载则有) ==="
python3 /tmp/gguf_probe.py /data/sdb/ollama-qw3.8/blobs/sha256-06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439-partial

sleep 25
echo "=== 4) train.log 头部 ==="
tail -6 /data/sdb/btr/logs/train.log
echo CHECK_DONE

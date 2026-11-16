#!/bin/bash
# Step E: v2 数据 -> 重训(GPU1) + 基线重评(GPU0) 并行
set -e
cd /data/sdb/btr

rm -rf lfdata && mkdir -p lfdata
cp sft_v2/sft_train.jsonl sft_v2/sft_val.jsonl lfdata/
cp -r sft_v2/images lfdata/
cp sft_v2/dataset_info.json lfdata/dataset_info.json
wc -l lfdata/sft_train.jsonl lfdata/sft_val.jsonl
ls lfdata/images | wc -l

# 训练输出目录改为 v2
sed -i 's|qwen2.5vl-7b-btr-qlora|qwen2.5vl-7b-btr-qlora-v2|g' qwen2.5vl-7b-qlora-btr.yaml
grep 'output_dir' qwen2.5vl-7b-qlora-btr.yaml

source lfenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 基线重评 (GPU0, 96 张验证图)
CUDA_VISIBLE_DEVICES=0 nohup python eval_btr.py --mode base \
    --out /data/sdb/btr/reports/base_v2.json \
    > /data/sdb/btr/logs/eval_base_v2.log 2>&1 &
echo "base_eval_pid $!"

# 重训 (GPU1)
CUDA_VISIBLE_DEVICES=1 nohup llamafactory-cli train qwen2.5vl-7b-qlora-btr.yaml \
    > /data/sdb/btr/logs/train_v2.log 2>&1 &
echo "train_v2_pid $!"

sleep 20
echo "--- eval_base_v2 ---"; tail -2 logs/eval_base_v2.log
echo "--- train_v2 ---"; tail -3 logs/train_v2.log | tr '\r' '\n' | tail -2
echo V2_LAUNCHED

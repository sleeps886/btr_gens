#!/bin/bash
# Step C(服务器端数据版): 组装 lfdata -> 依赖检查 -> 启动 QLoRA 训练
set -e
cd /data/sdb/btr

SFT=/home/public/data/datasets/btr_sft_20261111
rm -rf lfdata && mkdir -p lfdata
cp $SFT/sft_train.jsonl $SFT/sft_val.jsonl lfdata/
cp -r $SFT/images lfdata/
cp $SFT/dataset_info.json lfdata/dataset_info.json
wc -l lfdata/sft_train.jsonl lfdata/sft_val.jsonl
ls lfdata/images | wc -l

source lfenv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import bitsandbytes" 2>/dev/null && echo bnb_ok || {
  echo "安装 bitsandbytes ..."
  pip install bitsandbytes -q -i https://mirrors.aliyun.com/pypi/simple/ --timeout 180
}

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup llamafactory-cli train /data/sdb/btr/qwen2.5vl-7b-qlora-btr.yaml \
    > /data/sdb/btr/logs/train.log 2>&1 &
echo "train_pid $!"
sleep 30
echo "--- train.log 头部 ---"
tail -10 /data/sdb/btr/logs/train.log
echo TRAIN_LAUNCHED

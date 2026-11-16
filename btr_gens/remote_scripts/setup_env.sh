#!/bin/bash
# Step A: 工作目录 + authorized_keys 清理 + 后台下载(基座权重 / ollama 基线)
set -u
PW='zju123!@#'
echo "$PW" | sudo -S -p '' mkdir -p /data/sdb/btr/logs /data/sdb/btr/scripts /data/sdb/btr/models
echo "$PW" | sudo -S -p '' chown -R public:public /data/sdb/btr
ls -ld /data/sdb/btr

# 清理 authorized_keys 中此前损坏的行
sed -i "/^''ssh-ed25519n''/d" ~/.ssh/authorized_keys
wc -l ~/.ssh/authorized_keys

# modelscope 安装(user 级, 不污染系统)
pip3 install --user --break-system-packages -q modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m modelscope download --help >/dev/null 2>&1 && echo "modelscope: ok" || echo "modelscope: FAIL"

# 后台下载 1: 基座权重 Qwen3-VL-8B-Instruct (~17 GB)
nohup python3 -m modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
    --local_dir /data/sdb/btr/models/Qwen3-VL-8B-Instruct \
    > /data/sdb/btr/logs/dl_model.log 2>&1 &
echo "model download pid $!"

# 后台下载 2: ollama 基线模型 qwen3-vl:8b
nohup ollama pull qwen3-vl:8b > /data/sdb/btr/logs/pull_baseline.log 2>&1 &
echo "ollama pull pid $!"

sleep 8
echo "--- dl_model.log ---";   tail -3 /data/sdb/btr/logs/dl_model.log 2>/dev/null
echo "--- pull_baseline.log ---"; tail -3 /data/sdb/btr/logs/pull_baseline.log 2>/dev/null
echo STEP_A_DONE

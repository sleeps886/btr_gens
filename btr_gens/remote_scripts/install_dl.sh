#!/bin/bash
# Step A2: venv 安装 modelscope 并启动基座权重下载
set -u
echo "=== 诊断 user site(仅记录) ==="
ls ~/.local/lib/python3.12/site-packages/ 2>/dev/null | head -5

echo "=== venv 安装 modelscope ==="
python3 -m venv /data/sdb/btr/venv
/data/sdb/btr/venv/bin/pip install -q --timeout 120 \
    -i https://mirrors.aliyun.com/pypi/simple/ modelscope
/data/sdb/btr/venv/bin/python -c 'import modelscope; print("modelscope_ok", modelscope.__version__)' \
    || { echo modelscope_FAIL; exit 1; }

echo "=== 启动基座权重后台下载 ==="
nohup /data/sdb/btr/venv/bin/python -m modelscope download \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --local_dir /data/sdb/btr/models/Qwen3-VL-8B-Instruct \
    > /data/sdb/btr/logs/dl_model.log 2>&1 &
echo "model_dl_pid $!"

sleep 12
echo "--- dl_model.log 头部 ---"
head -5 /data/sdb/btr/logs/dl_model.log
echo "--- 已落盘大小 ---"
du -sh /data/sdb/btr/models/Qwen3-VL-8B-Instruct 2>/dev/null
echo STEP_A2_DONE

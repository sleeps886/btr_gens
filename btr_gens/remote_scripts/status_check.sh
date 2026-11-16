#!/bin/bash
echo "=== 网络现状 ==="
ip -br addr | grep -E 'eno|wlx|wg0|tailscale'
echo "=== LF 安装结果 ==="
tail -c 800 /data/sdb/btr/logs/install_lf.log | tr '\r' '\n' | grep -E 'LF_OK|LF_FAIL|INSTALL_LF_DONE|error|ERROR' | tail -5
ls /data/sdb/btr/lfenv/bin/llamafactory-cli 2>/dev/null && echo lf-cli-exists
echo "=== 网速复测: hf-mirror 20MB ==="
curl -sL -o /dev/null -w 'speed: %{speed_download} B/s\n' --max-time 30 -r 0-20000000 'https://hf-mirror.com/Qwen/Qwen3-VL-8B-Instruct/resolve/main/model-00004-of-00004.safetensors'
echo "=== 网速复测: modelscope 20MB ==="
curl -sL -o /dev/null -w 'speed: %{speed_download} B/s\n' --max-time 30 -r 0-20000000 'https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/master/model-00004-of-00004.safetensors'
echo "=== /data/sdb/btr 现状 ==="
ls /data/sdb/btr/
echo "=== 磁盘 ==="
df -h /data/sdb | tail -1

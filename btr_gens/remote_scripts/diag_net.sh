#!/bin/bash
echo "=== 网卡 5 秒流量 ==="
grep -E 'ens|eth|enp' /proc/net/dev
S1=$(grep -E 'ens|eth|enp' /proc/net/dev | awk -F'[: ]+' '{rx+=$3; tx+=$11} END{print rx, tx}')
sleep 5
grep -E 'ens|eth|enp' /proc/net/dev
S2=$(grep -E 'ens|eth|enp' /proc/net/dev | awk -F'[: ]+' '{rx+=$3; tx+=$11} END{print rx, tx}')
echo "5秒增量: rx=$(( $(echo $S2 | cut -d' ' -f1) - $(echo $S1 | cut -d' ' -f1) )) bytes, tx=$(( $(echo $S2 | cut -d' ' -f2) - $(echo $S1 | cut -d' ' -f2) )) bytes"

echo "=== modelscope/下载/ollama 进程 ==="
ps aux | grep -E 'modelscope|ollama' | grep -v grep | awk '{print $2, $3"%", $11, $12, $13}' | head -10

echo "=== TOP 进程 ==="
ps aux --sort=-%cpu | head -6 | awk '{printf "%s cpu=%s%% mem=%s%% %s\n", $2, $3, $4, $11}'

echo "=== 杀掉残留的 modelscope 下载进程 ==="
pkill -f "modelscope download" && echo killed || echo none-to-kill
sleep 1
ps aux | grep modelscope | grep -v grep | wc -l

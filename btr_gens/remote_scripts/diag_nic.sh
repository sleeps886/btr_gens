#!/bin/bash
echo "=== 全部网络接口与地址 ==="
ip -br addr
echo "=== 默认路由 ==="
ip route | head -5
echo "=== 非虚拟接口 5 秒流量 ==="
ls /sys/class/net | grep -v -E 'veth|docker|lo|br-' > /tmp/nics.txt
cat /tmp/nics.txt
S1=$(cat /tmp/nics.txt | while read n; do grep "$n" /proc/net/dev; done | awk -F'[: ]+' '{rx+=$3; tx+=$11} END{print rx, tx}')
sleep 5
S2=$(cat /tmp/nics.txt | while read n; do grep "$n" /proc/net/dev; done | awk -F'[: ]+' '{rx+=$3; tx+=$11} END{print rx, tx}')
echo "非虚拟接口5秒增量: rx=$(( $(echo $S2 | cut -d' ' -f1) - $(echo $S1 | cut -d' ' -f1) )) tx=$(( $(echo $S2 | cut -d' ' -f2) - $(echo $S1 | cut -d' ' -f2) ))"
echo "=== webui pip install 在装什么 ==="
ps aux | grep "webui/env/bin/pip" | grep -v grep | head -2
echo "=== LF 安装进度(尾部) ==="
tail -c 400 /data/sdb/btr/logs/install_lf.log | tr '\r' '\n' | tail -3

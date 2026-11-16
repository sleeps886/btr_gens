#!/bin/bash
echo "=== override.conf 内容 ==="
cat /etc/systemd/system/ollama.service.d/override.conf 2>&1
echo "=== ollama 服务状态 ==="
echo 'zju123!@#' | sudo -S -p '' systemctl status ollama --no-pager 2>&1 | head -8
echo "=== ollama 日志尾部 ==="
echo 'zju123!@#' | sudo -S -p '' journalctl -u ollama --since '10 min ago' --no-pager 2>&1 | tail -15
echo "=== 端口监听 ==="
ss -tlnp 2>/dev/null | grep 11434
echo "=== 直接探活 ==="
curl -s --max-time 5 http://127.0.0.1:11434/api/version 2>&1; echo ""
curl -s --max-time 5 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:11434/api/tags

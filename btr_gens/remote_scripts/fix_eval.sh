#!/bin/bash
pkill -f eval_ollama.py
sed -i 's|"options": {"temperature": 0, "num_predict": 160}|"options": {"temperature": 0, "num_predict": 160, "num_gpu": 999}|' /data/sdb/btr/eval_ollama.py
grep -n 'num_gpu' /data/sdb/btr/eval_ollama.py
nohup python3 /data/sdb/btr/eval_ollama.py --model qwen3-vl:8b \
    --out /data/sdb/btr/reports/qw38_zeroshot.json \
    > /data/sdb/btr/logs/eval_qw38.log 2>&1 &
echo relaunched
sleep 40
tail -2 /data/sdb/btr/logs/eval_qw38.log

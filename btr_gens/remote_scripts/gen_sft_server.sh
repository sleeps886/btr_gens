#!/bin/bash
cd /home/public/data/datasets
sed -i 's|WS = Path(__file__).resolve().parent.parent|WS = Path(__file__).resolve().parent|' make_sft_data.py
grep -n "^WS = \|^ROOT = " make_sft_data.py
python3 make_sft_data.py
echo "=== 产物校验 ==="
ls btr_sft_20261111/
wc -l btr_sft_20261111/sft_train.jsonl btr_sft_20261111/sft_val.jsonl
ls btr_sft_20261111/images | wc -l

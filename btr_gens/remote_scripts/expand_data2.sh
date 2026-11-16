#!/bin/bash
# Step D-v2: 平台端口 8766, 幂等版扩数据流水线
set -x
cd /data/sdb/btr
mkdir -p gen logs

# 1) 端口补丁
sed -i 's|PORT = 8765|PORT = 8766|' gen/btr_gui_server.py
sed -i 's|http://127.0.0.1:8765/run|http://127.0.0.1:8766/run|' gen/gen_train_ds.py
grep -n 'PORT = ' gen/btr_gui_server.py | head -1
grep -n '127.0.0.1:87' gen/gen_train_ds.py | head -1

# 2) 输出目录与样本数补丁
sed -i 's|ROOT = os.path.join(WS, "训练数据集")|ROOT = "/data/sdb/btr/ds_expanded"|' gen/gen_train_ds.py
sed -i 's|for k in range(3):|for k in range(15):|' gen/gen_train_ds.py

# 3) 启动平台服务
nohup /data/sdb/btr/lfenv/bin/python /data/sdb/btr/gen/btr_gui_server.py \
    > /data/sdb/btr/logs/gui_server.log 2>&1 &
sleep 6
curl -s -o /dev/null -w 'platform_api:%{http_code}\n' http://127.0.0.1:8766/
grep -n '8766' /data/sdb/btr/logs/gui_server.log | head -1

# 4) SFT v2 输出路径与验证集划分补丁
sed -i 's|DS = WS / "训练数据集"|DS = Path("/data/sdb/btr/ds_expanded")|' gen/make_sft_data.py
sed -i 's|OUT = WS / "btr_sft_20261111"|OUT = Path("/data/sdb/btr/sft_v2")|' gen/make_sft_data.py
sed -i 's|ARCHIVE = WS / "btr_sft_20261111.tar.gz"|ARCHIVE = Path("/data/sdb/btr/sft_v2.tar.gz")|' gen/make_sft_data.py
sed -i 's|(val if k == 3 else train)|(val if k % 5 == 0 else train)|' gen/make_sft_data.py

# 5) 批量生成 (后台)
nohup /data/sdb/btr/lfenv/bin/python /data/sdb/btr/gen/gen_train_ds.py \
    > /data/sdb/btr/logs/gen.log 2>&1 &
echo "gen_pid $!"

# 6) 完成后自动产出 v2 SFT
nohup bash -c 'while pgrep -f "gen/gen_train_ds.py" > /dev/null; do sleep 30; done; if grep -q "240 个样本" /data/sdb/btr/logs/gen.log; then /data/sdb/btr/lfenv/bin/python /data/sdb/btr/gen/make_sft_data.py > /data/sdb/btr/logs/make_sft_v2.log 2>&1; echo SFT_V2_DONE >> /data/sdb/btr/logs/gen.log; fi' > /dev/null 2>&1 &
echo EXPAND_V2_LAUNCHED

#!/bin/bash
# Step D: 扩数据(服务器本地 240 样本) -> v2 SFT 数据
set -x
mkdir -p /data/sdb/btr/gen /data/sdb/btr/logs
cd /data/sdb/btr

# 1) 启动本机仿真平台服务 (GUI 后端, 端口 8765)
nohup /data/sdb/btr/lfenv/bin/python /data/sdb/btr/gen/btr_gui_server.py \
    > /data/sdb/btr/logs/gui_server.log 2>&1 &
sleep 5
curl -s -o /dev/null -w 'platform_api:%{http_code}\n' http://127.0.0.1:8765/

# 2) 补丁: 输出目录与样本数 (16 场景 x 15 样本 = 240)
sed -i 's|ROOT = os.path.join(WS, "训练数据集")|ROOT = "/data/sdb/btr/ds_expanded"|' gen/gen_train_ds.py
sed -i 's|for k in range(3):|for k in range(15):|' gen/gen_train_ds.py
grep -n 'ROOT = \|range(15)' gen/gen_train_ds.py | head -3

# 3) 批量生成 (后台, 约 20-30 分钟)
nohup /data/sdb/btr/lfenv/bin/python /data/sdb/btr/gen/gen_train_ds.py \
    > /data/sdb/btr/logs/gen.log 2>&1 &
echo "gen_pid $!"

# 4) make_sft_data v2 补丁: 数据源/输出/验证集划分
sed -i 's|DS = WS / "训练数据集"|DS = Path("/data/sdb/btr/ds_expanded")|' gen/make_sft_data.py
sed -i 's|OUT = WS / "btr_sft_20261111"|OUT = Path("/data/sdb/btr/sft_v2")|' gen/make_sft_data.py
sed -i 's|ARCHIVE = WS / "btr_sft_20261111.tar.gz"|ARCHIVE = Path("/data/sdb/btr/sft_v2.tar.gz")|' gen/make_sft_data.py
sed -i 's|(val if k == 3 else train)|(val if k % 5 == 0 else train)|' gen/make_sft_data.py
grep -n 'DS = \|OUT = \|ARCHIVE = \|k % 5' gen/make_sft_data.py | head -5

# 5) 等生成完成后自动产出 v2 SFT(轮询 gen 进程)
nohup bash -c '
  while pgrep -f "gen_train_ds.py" > /dev/null; do sleep 30; done
  if grep -q "完成: 240 个样本" /data/sdb/btr/logs/gen.log; then
    /data/sdb/btr/lfenv/bin/python /data/sdb/btr/gen/make_sft_data.py \
      > /data/sdb/btr/logs/make_sft_v2.log 2>&1
    echo SFT_V2_DONE >> /data/sdb/btr/logs/gen.log
  fi
' > /dev/null 2>&1 &
echo "watcher_pid $!"
echo EXPAND_LAUNCHED

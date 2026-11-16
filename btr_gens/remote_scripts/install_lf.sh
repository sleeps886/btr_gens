#!/bin/bash
# Step B: LLaMA-Factory 安装(后台, 日志到 install_lf.log)
set -x
cd /data/sdb/btr
python3 -m venv lfenv
source lfenv/bin/activate
pip install -U pip -q -i https://mirrors.aliyun.com/pypi/simple/
pip install llamafactory -i https://mirrors.aliyun.com/pypi/simple/ --timeout 180
python -c "import llamafactory, sys; print('LF_OK', llamafactory.__version__)" || echo LF_FAIL
echo INSTALL_LF_DONE

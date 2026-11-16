# -*- coding: utf-8 -*-
"""一次性 SSH 引导: 用密码登录 LLM-Zhoushan, 装入免密公钥并采集环境信息。
之后所有操作走普通 ssh/scp 密钥认证, 不再使用密码。"""
import sys
import paramiko

sys.stdout.reconfigure(encoding="utf-8")
HOST, USER, PWD = "192.168.0.145", "public", "zju123!@#"
PUBKEY = ("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBTT/ah8hesHuqYOcZ0z6HSILA"
          "BikzeQVJmr+B+MTQaA z5626302@ad.unsw.edu.au")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PWD, timeout=45,
          banner_timeout=90, auth_timeout=60)
print("[connect] OK")


def run(cmd, timeout=90):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    rc = out.channel.recv_exit_status()
    return rc, o, e


# 1) 安装免密公钥(已存在则跳过)
cmd = ("mkdir -p ~/.ssh && "
       f"grep -qF '{PUBKEY}' ~/.ssh/authorized_keys 2>/dev/null || "
       f"echo '{PUBKEY}' >> ~/.ssh/authorized_keys; "
       "chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys; echo KEY_INSTALLED")
rc, o, e = run(cmd)
print("[key]", o.strip() or e.strip())

# 2) 环境采集
checks = [
    ("主机", "hostname; whoami; uname -r"),
    ("sudo", "echo 'zju123!@#' | sudo -S -p '' true 2>/dev/null && echo SUDO_OK || echo SUDO_NO"),
    ("GPU", "nvidia-smi -L"),
    ("ollama", "ollama list 2>&1; systemctl is-active ollama 2>&1"),
    ("磁盘", "df -h /data/sdb | tail -1; ls -ld /data/sdb"),
    ("写权限", "touch /data/sdb/.wtest 2>&1 && echo WRITE_OK && rm -f /data/sdb/.wtest"),
    ("python", "python3 --version 2>&1; pip3 --version 2>&1"),
    ("内存CPU", "free -g | head -2; nproc"),
    ("已有数据", "ls ~/data/datasets/训练数据集 2>&1 | head -3"),
    ("外网", "curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://modelscope.cn 2>&1"),
    ("home空间", "df -h ~ | tail -1"),
]
for tag, cmd in checks:
    rc, o, e = run(cmd)
    print(f"--- {tag} (rc={rc}) ---")
    print((o + e).strip() or "(空)")

c.close()
print("[done]")

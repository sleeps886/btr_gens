# -*- coding: utf-8 -*-
"""LLM-Zhoushan 远程操作助手(paramiko 通道)。
用法:
    python ssh_run.py run   "<remote command>"
    python ssh_run.py put   <local> <remote>
    python ssh_run.py get   <remote> <local>
说明: 该服务器 SSH banner 响应慢(最长约 60s), 且 sshd 对 authorized_keys
的信任受家目录权限影响 —— 诊断与修复命令见首次运行输出。
"""
import os
import sys
import paramiko

sys.stdout.reconfigure(encoding="utf-8")
HOST, USER, PWD = "192.168.0.145", "public", "zju123!@#"


def connect():
    import time
    last = None
    for attempt in range(4):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, username=USER, password=PWD, timeout=60,
                      banner_timeout=150, auth_timeout=60)
            return c
        except Exception as e:                       # 服务器 sshd 会间歇重置
            last = e
            print(f"[retry {attempt+1}/4] 连接失败: {e}", file=sys.stderr)
            time.sleep(15)
    raise last


def main():
    mode = sys.argv[1]
    c = connect()
    try:
        if mode == "run":
            cmd = sys.argv[2]
            _, out, err = c.exec_command(cmd, timeout=None)
            o = out.read().decode("utf-8", "replace")
            e = err.read().decode("utf-8", "replace")
            rc = out.channel.recv_exit_status()
            sys.stdout.write(o)
            if e.strip():
                sys.stdout.write("\n[stderr]\n" + e)
            sys.exit(rc)
        if mode == "put":
            local, remote = sys.argv[2], sys.argv[3]
            sftp = c.open_sftp()
            sftp.put(local, remote)
            st = sftp.stat(remote)
            print(f"uploaded {local} -> {remote} ({st.st_size/1048576:.1f} MB)")
            sys.exit(0)
        if mode == "get":
            remote, local = sys.argv[2], sys.argv[3]
            sftp = c.open_sftp()
            sftp.get(remote, local)
            print(f"downloaded {remote} -> {local} "
                  f"({os.path.getsize(local)/1048576:.1f} MB)")
            sys.exit(0)
        raise SystemExit(f"unknown mode: {mode}")
    finally:
        c.close()


if __name__ == "__main__":
    main()

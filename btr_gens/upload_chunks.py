# -*- coding: utf-8 -*-
"""分块断点续传上传: 应对不稳定 SSH 链路。
用法: python upload_chunks.py <local> <remote> [chunk_mb] [workers]
- 每块 64MB(默认), 逐块 SFTP 上传 + 服务器端 sha256 校验, 不匹配/断连自动重试
- 已上传且校验通过的块跳过(可反复重跑实现断点续传)
- 全部完成后服务器端 cat 合并并做整文件 sha256 校验
"""
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from ssh_run import HOST, USER, PWD, connect  # noqa: E402
import paramiko  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def put_chunk_with_retry(local_chunk, remote_chunk, expect_hash, tries=6):
    last = None
    for attempt in range(tries):
        try:
            c = connect()
            sftp = c.open_sftp()
            sftp.put(local_chunk, remote_chunk)
            _, out, _ = c.exec_command(f"sha256sum '{remote_chunk}'", timeout=120)
            rh = out.read().decode().split()[0]
            c.close()
            if rh == expect_hash:
                return True
            last = RuntimeError(f"hash mismatch {rh} != {expect_hash}")
        except Exception as e:
            last = e
            time.sleep(8)
    raise last


def main():
    local = Path(sys.argv[1])
    remote = sys.argv[2]
    chunk_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    chunk = chunk_mb * 1024 * 1024
    total = local.stat().st_size
    n = (total + chunk - 1) // chunk
    print(f"[upload] {local.name}: {total/1048576:.1f} MB -> {n} 块 x {chunk_mb}MB")

    tmpdir = Path(local.parent / f".upchunks_{local.name}")
    tmpdir.mkdir(exist_ok=True)

    c0 = connect()
    rdir = os.path.dirname(remote)
    _, out, _ = c0.exec_command(f"mkdir -p '{rdir}' && echo ok")
    out.read()

    results = [None] * n
    t0 = time.time()
    done_bytes = 0

    def work(i):
        start = i * chunk
        size = min(chunk, total - start)
        cpath = tmpdir / f"chunk_{i:04d}"
        with open(local, "rb") as f:
            f.seek(start)
            data = f.read(size)
        cpath.write_bytes(data)
        ch = hashlib.sha256(data).hexdigest()
        remote_chunk = f"{remote}.part{i:04d}"
        # 断点续传: 已存在且校验通过则跳过
        try:
            _, out, _ = c0.exec_command(
                f"sha256sum '{remote_chunk}' 2>/dev/null | cut -d' ' -f1",
                timeout=60)
            if out.read().decode().strip() == ch:
                return i, size, True
        except Exception:
            pass
        ok = put_chunk_with_retry(str(cpath), remote_chunk, ch)
        return i, size, ok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in range(n)}
        for fut in as_completed(futs):
            i, size, ok = fut.result()
            results[i] = ok
            done_bytes += size
            rate = done_bytes / 1048576 / max(time.time() - t0, 1)
            print(f"  块 {i+1}/{n} {'OK' if ok else 'FAIL'} "
                  f"({done_bytes/1048576:.0f}MB, {rate:.2f} MB/s)", flush=True)

    if not all(results):
        raise SystemExit("存在失败块, 重跑本命令可续传")

    print("[assemble] 服务器端合并 + 整体校验 ...")
    parts = " ".join(f"'{remote}.part{i:04d}'" for i in range(n))
    cmd = (f"cat {parts} > '{remote}.assembling' && "
           f"sha256sum '{remote}.assembling'")
    _, out, err = c0.exec_command(cmd, timeout=1800)
    assembled = out.read().decode().split()[0]
    expect = sha256_file(local)
    if assembled != expect:
        raise SystemExit(f"整文件校验不符: {assembled} != {expect}")
    c0.exec_command(f"mv '{remote}.assembling' '{remote}' && rm -f {parts}",
                    timeout=300)
    time.sleep(1)
    print(f"[done] {remote} 上传并校验通过, 耗时 {time.time()-t0:.0f}s")
    c0.close()


if __name__ == "__main__":
    main()

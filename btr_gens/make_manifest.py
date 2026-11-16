# -*- coding: utf-8 -*-
"""为 训练数据集 生成 Linux sha256sum 兼容清单并打包。

用法:
    python make_manifest.py                 # 生成 manifest.sha256 + tar.gz
    python make_manifest.py --check         # 仅本地模拟 sha256sum -c 全量校验

要点:
    - 路径统一用 pathlib.as_posix() 转成正斜杠(上一版用 replace('\\\\','/')
      在 PowerShell 传递时转义错误, 导致清单里出现反斜杠, Linux 端校验全部失败);
    - manifest.sha256 本身不入清单;
    - 打包后输出压缩包自身 SHA256, 供服务器端比对。
"""
import hashlib
import os
import sys
import tarfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
WS = Path(__file__).resolve().parent.parent          # 工作区根
ROOT = WS / "训练数据集"
ARCHIVE = WS / "btr_train_ds_20261110.tar.gz"


def iter_files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.name != "manifest.sha256":
            yield p


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest():
    t0 = time.time()
    lines = [f"{sha256_of(p)}  {p.relative_to(ROOT).as_posix()}"
             for p in iter_files()]
    out = ROOT / "manifest.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"manifest.sha256: {len(lines)} 个文件, 耗时 {time.time()-t0:.1f}s, "
          f"反斜杠行数: {sum(1 for l in lines if chr(92) in l)}")
    print("样例:", lines[0][:60] + "...")


def check_manifest():
    """本地模拟 sha256sum -c, 必须全部 OK"""
    t0 = time.time()
    bad = ok = 0
    for line in (ROOT / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        if sha256_of(ROOT / rel) == digest:
            ok += 1
        else:
            bad += 1
            print("MISMATCH:", rel)
    print(f"本地校验: {ok} OK, {bad} 不符, 耗时 {time.time()-t0:.1f}s")
    return bad == 0


def build_archive():
    t0 = time.time()
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=4) as tf:
        tf.add(ROOT, arcname="训练数据集")
    print(f"打包完成: {ARCHIVE} ({ARCHIVE.stat().st_size/1048576:.1f} MB, "
          f"{time.time()-t0:.0f}s)")
    print("ARCHIVE_SHA256:", sha256_of(ARCHIVE))


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check_manifest() else 1)
    build_manifest()
    if not check_manifest():
        sys.exit("manifest 自校验失败, 未打包")
    build_archive()

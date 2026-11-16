# -*- coding: utf-8 -*-
"""解析 GGUF 文件头元数据, 确认模型真实身份 (不依赖文件名)。"""
import struct
import sys


def read_gguf_kv(path):
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        return {"error": "不是 GGUF 文件: " + repr(magic)}
    struct.unpack("<I", f.read(4))              # version
    struct.unpack("<Q", f.read(8))              # tensor count
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def read_str():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def read_val(t):
        if t == 0:
            return struct.unpack("<B", f.read(1))[0]
        if t == 1:
            return struct.unpack("<b", f.read(1))[0]
        if t == 2:
            return struct.unpack("<H", f.read(2))[0]
        if t == 3:
            return struct.unpack("<h", f.read(2))[0]
        if t == 4:
            return struct.unpack("<I", f.read(4))[0]
        if t == 5:
            return struct.unpack("<i", f.read(4))[0]
        if t == 6:
            return struct.unpack("<f", f.read(4))[0]
        if t == 7:
            return f.read(8)
        if t == 8:
            return read_str()
        if t in (9, 10):
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [read_val(et) for _ in range(min(n, 8))]
        raise ValueError(t)

    kv = {}
    for _ in range(min(n_kv, 300)):
        k = read_str()
        t = struct.unpack("<I", f.read(4))[0]
        try:
            kv[k] = read_val(t)
        except Exception as e:
            return {"partial_kv": kv, "error": str(e)}
    return kv


def show(path):
    print("=====", path.split("/")[-1][:60], "=====")
    try:
        kv = read_gguf_kv(path)
    except Exception as e:
        print("读取失败:", e)
        return
    if "error" in kv:
        print(kv["error"])
        return
    keys = ["general.name", "general.architecture", "general.size_label",
            "general.finetune", "general.basename", "general.file_type",
            "qwen3.context_length", "qwen2.context_length",
            "qwen3.embedding_length", "qwen2.embedding_length",
            "qwen3.block_count", "qwen2.block_count",
            "qwen3.attention.head_count", "qwen2.attention.head_count",
            "tokenizer.ggml.model", "general.license",
            "general.organization", "general.quantization_version"]
    for k in keys:
        if k in kv:
            print(f"  {k} = {kv[k]}")
    others = [k for k in kv if k.startswith("general.") and k not in keys]
    for k in others[:10]:
        print(f"  {k} = {kv[k]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        show(p)

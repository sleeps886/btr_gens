# -*- coding: utf-8 -*-
"""用网格线(灰色点线)精确标定 BTR 图的像素<->坐标映射, 并 dump 方位剖面。"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btr_infer_track as B

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "btr_out", "20261105_152635")

for tag in ("hull", "tow"):
    img = np.asarray(Image.open(os.path.join(SRC, f"btr_A_{tag}.png")).convert("RGB"))
    R, G, Bl = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    gray = (np.abs(R - G) < 12) & (np.abs(G - Bl) < 12) & (R > 25) & (R < 130)
    cnt_col = gray.sum(0)
    cnt_row = gray.sum(1)
    base_c = np.median(cnt_col)
    base_r = np.median(cnt_row)
    pk_c = [i for i in range(1, len(cnt_col) - 1)
            if cnt_col[i] > base_c + 25 and cnt_col[i] >= cnt_col[i - 1]
            and cnt_col[i] > cnt_col[i + 1]]
    pk_r = [i for i in range(1, len(cnt_row) - 1)
            if cnt_row[i] > base_r + 25 and cnt_row[i] >= cnt_row[i - 1]
            and cnt_row[i] > cnt_row[i + 1]]
    # 合并相邻
    def group(idxs):
        out, cur = [], [idxs[0]]
        for v in idxs[1:]:
            if v - cur[-1] <= 3:
                cur.append(v)
            else:
                out.append(sum(cur) / len(cur)); cur = [v]
        out.append(sum(cur) / len(cur))
        return out
    gc = group(pk_c) if pk_c else []
    gr = group(pk_r) if pk_r else []
    print(f"=== {tag}: 图像 {img.shape}")
    print(f"  竖直网格线列: {[round(v,1) for v in gc]}")
    if len(gc) > 1:
        print(f"  间距: {[round(gc[i+1]-gc[i],1) for i in range(len(gc)-1)]}")
    print(f"  水平网格线行: {[round(v,1) for v in gr]}")
    if len(gr) > 1:
        print(f"  间距: {[round(gr[i+1]-gr[i],1) for i in range(len(gr)-1)]}")

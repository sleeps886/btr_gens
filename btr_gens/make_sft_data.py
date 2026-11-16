# -*- coding: utf-8 -*-
"""把 训练数据集/ 转成多模态 SFT 训练数据(LLaMA-Factory sharegpt 视觉格式)。

输出:
    btr_sft_20261111/
    ├─ README.md            数据格式与用法说明
    ├─ dataset_info.json    LLaMA-Factory 数据集注册片段
    ├─ sft_train.jsonl      训练集(每场景样本1/2)
    ├─ sft_val.jsonl        验证集(每场景样本3)
    └─ images/              舷侧阵/拖曳阵 crop 图(ASCII 文件名)

设计:
    - 每个样本的 hull_crop / tow_crop 各生成一条图文对(48 样本 → 96 条);
    - 指令模板随机变化增强鲁棒性;答案只含可从图像泛化推演的场景级结论
      (平台状态/距离等级/数量等级/穿越/目标数/穿越目标数), 短答案利于"快速推演";
    - 逐目标精确 CPA/速度不出现在答案里(避免教模型死记硬背), 留在 meta.json 供核对。
"""
import json
import random
import shutil
import sys
import tarfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
WS = Path(__file__).resolve().parent.parent
DS = WS / "训练数据集"
OUT = WS / "btr_sft_20261111"
ARCHIVE = WS / "btr_sft_20261111.tar.gz"
rng = random.Random(20261111)

TEMPLATES = {
    "hull": [
        "<image>这是舷侧阵(舰壳)声呐方位历程图(BTR),横轴为相对船头舷角0~180°,纵轴为时间0~600s。请快速推演并输出JSON:平台状态(静止/移动)、距离等级(近/远)、目标数量(少/多)、目标总数、是否存在目标穿越平台、穿越目标数,最后给一行结论。",
        "<image>根据这张舷侧阵BTR图,快速推演场景:平台是静止还是移动?目标处于近距还是远距?目标数量是多少?有无目标轨迹穿越平台位置?以JSON+一行结论回答。",
        "<image>请以声呐分析员身份快速判读此舷侧阵方位历程图,给出场景四要素(平台运动/目标远近/目标多少/是否穿越)与目标计数,JSON格式,结论一句话。",
    ],
    "tow": [
        "<image>这是拖曳线列阵声呐方位历程图(BTR),横轴为舷角0~180°,纵轴为时间0~600s。请快速推演并输出JSON:平台状态(静止/移动)、距离等级(近/远)、目标数量(少/多)、目标总数、是否存在目标穿越平台、穿越目标数,最后给一行结论。",
        "<image>根据这张拖曳阵BTR图快速推演:平台运动状态?目标远近?目标数量?是否穿越平台?请以JSON和一句话结论作答。",
        "<image>请快速判读此拖曳阵方位历程图的场景要素(平台运动/目标远近/目标多少/是否穿越)与目标计数,JSON格式加一行结论。",
    ],
}


def build_answer(meta):
    cond = meta["conditions"]
    dist = "近" if cond["target_distance"].startswith("近") else "远"
    crossing = cond["crossing"].startswith("存在")
    n_xc = sum(1 for t in meta["targets"] if t["name"].startswith("XC"))
    n_total = len(meta["targets"])
    motion = "静止" if cond["platform_motion"] == "静止" else "移动"
    count = "多" if "目标多" in meta["scenario"] else "少"
    ans = {
        "平台状态": motion, "距离等级": dist, "目标数量": count,
        "目标总数": n_total, "穿越": "是" if crossing else "否",
        "穿越目标数": n_xc,
    }
    summary = (f"结论:平台{motion};{dist}距{count}目标场景;"
               + ("存在目标穿越平台,穿越目标 %d 个。" % n_xc if crossing
                  else "无目标穿越平台。"))
    return json.dumps(ans, ensure_ascii=False) + "\n" + summary


def main():
    index = json.loads((DS / "index.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    train, val = [], []
    for m in index["samples"]:
        code, k = m["scenario_code"], m["sample"]
        src_hull = DS / m["category"] / m["scenario"] / f"样本{k}" / "btr_A_hull_crop.png"
        src_tow = DS / m["category"] / m["scenario"] / f"样本{k}" / "btr_A_tow_crop.png"
        answer = build_answer(m)
        for tag, src in (("hull", src_hull), ("tow", src_tow)):
            img_name = f"{code}_{k}_{tag}_crop.png"
            shutil.copyfile(src, OUT / "images" / img_name)
            entry = {
                "messages": [
                    {"role": "user", "content": rng.choice(TEMPLATES[tag])},
                    {"role": "assistant", "content": answer},
                ],
                "images": [f"images/{img_name}"],
            }
            (val if k == 3 else train).append(entry)

    for name, data in (("sft_train.jsonl", train), ("sft_val.jsonl", val)):
        with open(OUT / name, "w", encoding="utf-8", newline="\n") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} 条")

    info = {
        "btr_sft_train": {
            "file_name": "sft_train.jsonl", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant",
                     "system_tag": "system"},
        },
        "btr_sft_val": {
            "file_name": "sft_val.jsonl", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant",
                     "system_tag": "system"},
        },
    }
    (OUT / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "README.md").write_text(
        "# BTR 快速推演 SFT 数据(LLaMA-Factory 视觉 sharegpt 格式)\n\n"
        "- sft_train.jsonl 64 条 / sft_val.jsonl 32 条(48 样本 × hull/tow 两图)\n"
        "- 答案仅含可泛化的场景级结论(平台状态/距离/数量/穿越/计数),"
        "不含逐样本 CPA/速度,避免死记硬背\n"
        "- 划分:每个场景的样本1/2 训练,样本3 验证(验证覆盖全部 16 个场景)\n"
        "- 图像统一为 1424×820 裁剪版 BTR,文件名 ASCII,位于 images/\n\n"
        "## LLaMA-Factory 用法\n"
        "1. 把整个目录拷到 LLaMA-Factory/data/ 下(即 data/sft_train.jsonl、"
        "data/images/…);或保持本目录结构,将 dataset_info.json 内容合并进"
        " LLaMA-Factory/data/dataset_info.json 并相应调整 file_name 路径;\n"
        "2. 训练 yaml 中 `dataset: btr_sft_train`, `eval_dataset: btr_sft_val`;\n"
        "3. 基座模型路径指向本地 qwen3.8(与 Ollama 同源的 HF 权重)。\n",
        encoding="utf-8", newline="\n")

    n_img = len(list((OUT / "images").glob("*.png")))
    print(f"images: {n_img} 张")
    t0 = time.time()
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=4) as tf:
        tf.add(OUT, arcname=OUT.name)
    print(f"打包: {ARCHIVE} ({ARCHIVE.stat().st_size/1048576:.1f} MB, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

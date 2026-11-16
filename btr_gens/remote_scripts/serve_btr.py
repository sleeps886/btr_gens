# -*- coding: utf-8 -*-
"""BTR 快速推演服务(轻量 HTTP API, 无额外依赖)。
  GET  /health          -> {"ok": true, "model": ..., "adapter": ...}
  POST /predict         -> body: {"image_b64": "<png base64>", "max_tokens": 160}
                         <- {"ok": true, "result": "<模型原始输出>", "latency_s": x}
启动: CUDA_VISIBLE_DEVICES=0 nohup python serve_btr.py \
        [--adapter /data/sdb/btr/saves/qwen2.5vl-7b-btr-qlora-v2] \
        [--port 8000] > /data/sdb/btr/logs/serve.log 2>&1 &
"""
import base64
import io
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

MODEL = "/data/sdb/models/Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER = None
PORT = 8000

for i, a in enumerate(__import__("sys").argv):
    if a == "--adapter":
        ADAPTER = __import__("sys").argv[i + 1]
    if a == "--port":
        PORT = int(__import__("sys").argv[i + 1])


def load():
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VL
    except ImportError:
        from transformers import AutoModelForVision2Seq as VL
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4")
    m = VL.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                           quantization_config=bnb, device_map="cuda:0")
    if ADAPTER:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ADAPTER)
        print("[serve] adapter loaded:", ADAPTER)
    m.eval()
    return m, AutoProcessor.from_pretrained(MODEL)


MODEL_OBJ, PROCESSOR = load()
print("[serve] ready")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model": MODEL, "adapter": ADAPTER,
                             "port": PORT})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            self._json(404, {"ok": False, "error": "use /predict"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            img = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")
            instruction = req.get("instruction") or (
                "这是声呐方位历程图(BTR),横轴为相对船头舷角0~180°,纵轴为时间0~600s。"
                "请快速推演并输出JSON:平台状态(静止/移动)、距离等级(近/远)、目标数量(少/多)、"
                "目标总数、是否存在目标穿越平台、穿越目标数,最后给一行结论。")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": instruction}]}]
            text = PROCESSOR.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            try:
                from qwen_vl_utils import process_vision_info
                imgs, _ = process_vision_info(messages)
            except ImportError:
                imgs = [img]
            inputs = PROCESSOR(text=[text], images=imgs, return_tensors="pt").to("cuda:0")
            t0 = time.time()
            with torch.no_grad():
                out = MODEL_OBJ.generate(**inputs,
                                         max_new_tokens=int(req.get("max_tokens", 160)),
                                         do_sample=False)
            resp = PROCESSOR.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0]
            self._json(200, {"ok": True, "result": resp,
                             "latency_s": round(time.time() - t0, 2)})
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})


if __name__ == "__main__":
    print(f"[serve] listening 0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

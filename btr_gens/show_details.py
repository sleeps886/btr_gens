import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
r = json.load(open("/data/sdb/btr/reports/base.json"))
print("n =", r["n"], "| all_fields =", r["all_fields_accuracy"],
      "| avg_latency =", r["avg_latency_s"])
for d in r["details"][:4]:
    print("IMG :", d["image"])
    print("RESP:", repr(d["resp"][:200]))
    print("PRED:", d["pred"])
    print("GT  :", {k: d["gt"].get(k) for k in d["gt"]})
    print("-" * 60)

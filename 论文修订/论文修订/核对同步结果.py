"""只读核对现有 Q4 结果；JSON 输出至标准输出，也可用 --output 同步保存。"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.utils.datetime import from_excel

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("q4_independent", ROOT / "scripts/scripts/verify_q4.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
v.DATA = ROOT / "data/data"
v.OUT = ROOT / "q4/q4"
log = io.StringIO()
with contextlib.redirect_stdout(log):
    p = v.load_pmat()
    v.LOAD_PV = v.load_actual_load_pv()
    rows = {k: v.verify_variant(k, p, 30) for k in ("2", "3")}
summaries = {}
workbooks = {}
selected = {}
for k in ("2", "3"):
    summaries[k] = json.loads((v.OUT / f"summary_q4-{k}_K30.json").read_text(encoding="utf-8"))
    days = json.loads((v.OUT / f"payload_q4-{k}_K30.json").read_text(encoding="utf-8"))["days"]
    selected[k] = [{a: d[a] for a in ("date", "total_cost_yuan", "emergency_total_kwh")} for d in days if d["date"] in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")]
    path = v.DATA / "附件5" / f"result4-{k}.xlsx"
    book = load_workbook(path, read_only=True, data_only=True)
    expected = {"计划购电量": [[datetime.fromisoformat(d["date"]), *d["plan_kwh"], d["plan_total_kwh"], d["plan_cost_yuan"]] for d in days]}
    if k == "3":
        expected["调整购电量"] = [[datetime.fromisoformat(d["date"]), *d["adjusted_kwh"], d["adjusted_total_kwh"], d["adjusted_cost_yuan"]] for d in days]
    expected["充放电量"] = []
    expected["紧急购电量"] = []
    for d in days:
        for i, b in enumerate(d["storage_blocks"]):
            expected["充放电量"].append([datetime.fromisoformat(d["date"]) if i == 0 else None, b["time_range"], b["charge_kwh"], b["discharge_kwh"], "0:00" if i == 0 else "24:00" if i == 1 else None, d["soc_start_kwh"] if i == 0 else d["soc_end_kwh"] if i == 1 else None])
        for i, b in enumerate(d["emergency_segments"] or [{"time_range": None, "energy_kwh": 0}]):
            expected["紧急购电量"].append([datetime.fromisoformat(d["date"]) if i == 0 else None, b["time_range"], b["energy_kwh"]])
    workbooks[k] = {}
    for name, erows in expected.items():
        actual = list(book[name].values)[1:]
        if len(actual) != len(erows):
            v.failures.append(f"{k}/{name}: row count mismatch")
        gap = 0.0
        labels = 0
        normalized_dates = 0
        for ar, er in zip(actual, erows):
            if len(ar) != len(er):
                v.failures.append(f"{k}/{name}: column count mismatch")
            for a, e in zip(ar, er):
                if isinstance(e, datetime) and isinstance(a, (int, float)):
                    a = from_excel(a, book.epoch)
                    normalized_dates += 1
                if isinstance(e, (int, float)):
                    gap = max(gap, abs(float(a) - e))
                elif a != e:
                    labels += 1
        if gap > 1e-6 or labels:
            v.failures.append(f"{k}/{name}: numeric gap={gap}, label mismatches={labels}")
        workbooks[k][name] = {"data_rows": len(actual), "max_numeric_gap": gap, "label_mismatches": labels, "excel_serial_dates_normalized": normalized_dates}
    book.close()
source_paths = [v.DATA / name for name in ("附件1.xlsx", "附件2.xlsx", "附件4.xlsx")]
source_paths += [v.OUT / f"{kind}_q4-{k}_K30.{ext}" for k in ("2", "3") for kind, ext in (("detail", "npz"), ("payload", "json"), ("summary", "json"))]
source_paths += [ROOT / "src/src" / name for name in ("q4_solver.py", "q4_q2_solver.py", "q3_multistage.py", "q2_solver.py")]
source_paths += [Path(__file__).resolve(), ROOT / "scripts/scripts/verify_q4.py"]
source_paths += [v.DATA / "附件5" / f"result4-{k}.xlsx" for k in ("2", "3")]
report = {
    "scope": "现有结果独立结算、物理约束及 Q4-3 实时重放；不重建预测、不重解 LP、不验证最优性",
    "variants": rows,
    "worst": v.worst_seen,
    "failures": v.failures,
    "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    "totals": {k: s["totals"] for k, s in summaries.items()},
    "workbook_payload_checks": workbooks,
    "selected_days": selected,
    "log": log.getvalue(),
}
rendered = json.dumps(report, ensure_ascii=False, indent=2)
print(rendered)
if args.output:
    args.output.write_text(rendered + "\n", encoding="utf-8")
sys.exit(bool(v.failures))

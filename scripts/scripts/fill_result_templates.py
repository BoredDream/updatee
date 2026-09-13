"""Fill the supplied result workbooks in place in memory; never create sheets.

python scripts/fill_result_templates.py --question all
On another machine, pass --template-dir pointing to the original 附件5.
The original templates are read-only inputs; filled copies go to outputs/.
Only existing payload values are transferred. No solver is imported or run.
"""
from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_TEMPLATES = Path("C:/Users/Lenovo/Desktop/数模/zz/problem/data/附件5")
QUESTIONS = ("q1", "q2", "q3", "q4-2", "q4-3")
EMERGENCY_DATE_FORMAT = "yyyy/m/d"
EMERGENCY_DATE_COLUMN_WIDTH = 13


def row_pattern(sheet, row: int, columns: int):
    """Snapshot before filling so later rows cannot contaminate the example."""
    return (
        [copy(sheet.cell(row, col)._style) for col in range(1, columns + 1)],
        copy(sheet.row_dimensions[row]),
    )


def apply_row_pattern(sheet, row: int, pattern) -> None:
    styles, dimension = pattern
    for col, style in enumerate(styles, start=1):
        sheet.cell(row, col)._style = copy(style)
    dimension = copy(dimension)
    dimension.index = row
    sheet.row_dimensions[row] = dimension


def clear_example_values(sheet, columns: int) -> None:
    # Keep the original header and formatting, including rows with example dots.
    for cells in sheet.iter_rows(min_row=2, max_col=columns):
        for cell in cells:
            cell.value = None


def write_row(sheet, row: int, values) -> None:
    for col, value in enumerate(values, start=1):
        # Setting a datetime may automatically change number_format; preserve it.
        cell = sheet.cell(row, col)
        style = copy(cell._style)
        cell.value = value
        cell._style = style


def fill_storage(sheet, days) -> None:
    patterns = [row_pattern(sheet, row, 6) for row in range(2, 8)]
    clear_example_values(sheet, 6)
    for day_index, day in enumerate(days):
        if len(day["storage_blocks"]) != 6:
            raise ValueError(f"{day['date']}: expected six storage blocks")
        for index, block in enumerate(day["storage_blocks"]):
            row = 2 + day_index * 6 + index
            apply_row_pattern(sheet, row, patterns[index])
            write_row(sheet, row, [
                datetime.fromisoformat(day["date"]) if index == 0 else None,
                block["time_range"], block["charge_kwh"], block["discharge_kwh"],
                "0:00" if index == 0 else "24:00" if index == 1 else None,
                day["soc_start_kwh"] if index == 0 else day["soc_end_kwh"] if index == 1 else None,
            ])


def fill_emergency(sheet, days) -> None:
    first, middle, last = [row_pattern(sheet, row, 3) for row in range(2, 5)]
    bottom_edges = [copy(sheet.cell(4, col).border.bottom) for col in range(1, 4)]
    clear_example_values(sheet, 3)
    sheet.column_dimensions["A"].width = EMERGENCY_DATE_COLUMN_WIDTH
    row = 2
    for day in days:
        segments = day["emergency_segments"] or [{"time_range": None, "energy_kwh": 0}]
        for index, segment in enumerate(segments):
            pattern = first if index == 0 else last if index == len(segments) - 1 else middle
            apply_row_pattern(sheet, row, pattern)
            if len(segments) == 1:
                # A one-row day needs both edges from the original three-row example.
                # Retain the first row's date format, font, alignment and side borders.
                for col, bottom in enumerate(bottom_edges, start=1):
                    border = copy(sheet.cell(row, col).border)
                    border.bottom = copy(bottom)
                    sheet.cell(row, col).border = border
            write_row(sheet, row, [
                datetime.fromisoformat(day["date"]) if index == 0 else None,
                segment["time_range"], segment["energy_kwh"],
            ])
            if index == 0:
                # Do not rely on a copied template style: a General style exposes
                # Excel's date serial (for example 45813) instead of a date.
                sheet.cell(row, 1).number_format = EMERGENCY_DATE_FORMAT
            row += 1


def fill_price(sheet, days, vector: str, total: str, cost: str) -> None:
    for row, day in enumerate(days, start=2):
        if len(day[vector]) != 144:
            raise ValueError(f"{day['date']}: expected 144 purchase intervals")
        write_row(sheet, row, [datetime.fromisoformat(day["date"]),
                              *day[vector], day[total], day[cost]])


def fill_workbook(workbook, payload, question: str) -> None:
    """Write into the original worksheet objects without tables or restyling."""
    if question == "q1":
        if len(payload["template_grid_kwh"]) != 144 or len(payload["storage_blocks"]) != 6:
            raise ValueError("Q1 requires 144 intervals and six storage blocks")
        plan, storage = workbook["计划购电量"], workbook["充放电量"]
        for row, value in enumerate(payload["template_grid_kwh"], start=2):
            plan.cell(row, 2).value = value
        for row, block in enumerate(payload["storage_blocks"], start=2):
            storage.cell(row, 2).value = block["charge_kwh"]
            storage.cell(row, 3).value = block["discharge_kwh"]
        storage["E2"] = payload["soc_start_kwh"]
        storage["E3"] = payload["soc_end_kwh"]
        return

    days = payload["days"]
    if len(days) != 334:
        raise ValueError(f"Expected 334 output days, got {len(days)}")
    keys = ("template_grid_kwh", "template_grid_total_kwh", "template_grid_cost_yuan") \
        if question == "q2" else ("plan_kwh", "plan_total_kwh", "plan_cost_yuan")
    fill_price(workbook["计划购电量"], days, *keys)
    if question in ("q3", "q4-3"):
        fill_price(workbook["调整购电量"], days, "adjusted_kwh", "adjusted_total_kwh", "adjusted_cost_yuan")
    fill_storage(workbook["充放电量"], days)
    fill_emergency(workbook["紧急购电量"], days)


def paths_for(question: str, tag: str, scenarios: int):
    if question in ("q1", "q2"):
        return question, "solver_payload.json", f"result{question[1:]}.xlsx"
    if question == "q3":
        return "q3_multistage", f"payload_stages{tag}_K{scenarios}.json", "result3.xlsx"
    variant = question[-1]
    return "q4", f"payload_q4-{variant}_K{scenarios}.json", f"result4-{variant}.xlsx"


def build_result(question: str, template_dir: Path, output_root: Path,
                 payload_root: Path, tag: str = "0123", scenarios: int = 30) -> Path:
    folder, payload_name, result_name = paths_for(question, tag, scenarios)
    template = template_dir / result_name
    target = output_root / folder / result_name
    if template.resolve() == target.resolve():
        raise ValueError("The original template must not be overwritten")
    payload = json.loads((payload_root / folder / payload_name).read_text(encoding="utf-8"))
    # Never fall back to problem/data/附件5, which may contain previously filled results.
    workbook = load_workbook(template)
    sheet_names = workbook.sheetnames[:]
    fill_workbook(workbook, payload, question)
    if workbook.sheetnames != sheet_names:
        raise ValueError("Worksheet names/order changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".result-fill-", suffix=".xlsx", dir=target.parent)
    os.close(handle)
    try:
        workbook.save(temporary)
        workbook.close()
        # Validate that the saved package can be opened before replacing an old result.
        saved = load_workbook(temporary, read_only=True)
        try:
            if saved.sheetnames != sheet_names:
                raise ValueError("Saved worksheet names/order changed")
        finally:
            saved.close()
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", choices=(*QUESTIONS, "all"), default="all")
    parser.add_argument("--template-dir", type=Path,
                        default=Path(os.environ.get("RESULT_TEMPLATE_DIR", str(ORIGINAL_TEMPLATES))))
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("RESULT_OUTPUT_ROOT", str(ROOT / "outputs"))))
    parser.add_argument("--payload-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--tag", default="0123")
    parser.add_argument("--K", type=int, default=30)
    args = parser.parse_args()
    for question in QUESTIONS if args.question == "all" else (args.question,):
        target = build_result(question, args.template_dir, args.output_root,
                              args.payload_root, args.tag, args.K)
        print(f"Filled original template: {target}", flush=True)


if __name__ == "__main__":
    main()

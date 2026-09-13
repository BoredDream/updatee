// Keep existing Node entry points; Python fills the original workbook directly.
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

export function fillResultTemplate(question, { tag = "0123", K = "30" } = {}) {
  const root = path.resolve(import.meta.dirname, "..");
  const localPython = path.join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  const python = process.env.RESULT_PYTHON ?? (fs.existsSync(localPython) ? localPython : "python");
  const result = spawnSync(python, ["-X", "utf8", path.join(root, "scripts", "fill_result_templates.py"),
    "--question", question, "--tag", tag, "--K", String(K)], { stdio: "inherit", windowsHide: true });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`Original-template fill failed: ${question} (exit ${result.status})`);
}

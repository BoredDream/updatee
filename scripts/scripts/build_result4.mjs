// 用官方附件5模板生成 result4-2.xlsx / result4-3.xlsx（波动电价模型）。
// 数据来自 scripts/export_q4.py 落盘的 payload。
// 表2 采用"模板行框"：六个段为 t=0..23,24..47,...,120..143，
// 分别覆盖 0:10-4:10 / 4:10-8:10 / 8:10-12:10 / 12:10-16:10 / 16:10-20:10 / 20:10-0:10+1。
//
// 用法： node scripts/build_result4.mjs <2|3> [K]
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import { fillResultTemplate } from "./fill_result_templates.mjs";

const repoRoot = path.resolve(import.meta.dirname, "..");
const variant = process.argv[2] ?? "3";
const K = process.argv[3] ?? "30";
if (!["2", "3"].includes(variant)) throw new Error(`variant must be 2 or 3, got ${variant}`);

const payloadPath = path.join(repoRoot, "outputs", "q4", `payload_q4-${variant}_K${K}.json`);
const outputDir = path.join(process.env.RESULT_OUTPUT_ROOT ?? path.join(repoRoot, "outputs"), "q4");
const outputPath = path.join(outputDir, `result4-${variant}.xlsx`);
const previewDir = path.join(outputDir, "previews");

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
if (!Array.isArray(payload.days) || payload.days.length !== 334) {
  throw new Error(`Expected 334 output days, received ${payload.days?.length ?? "none"}`);
}
console.log(`meta: ${JSON.stringify(payload.meta)}`);

const emergencyEndRow = 1 + payload.days.reduce((count, day) => count + Math.max(1, day.emergency_segments.length), 0);
fillResultTemplate(`q4-${variant}`, { K });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
workbook.recalculate();

for (const [sheetName, maxCols] of [["计划购电量", 147], ["调整购电量", 147], ["充放电量", 6], ["紧急购电量", 3]]) {
  if (!workbook.worksheets.items.some((ws) => ws.name === sheetName)) continue;
  const check = await workbook.inspect({
    kind: "table", range: `${sheetName}!A1`,
    include: "values", tableMaxRows: 2, tableMaxCols: maxCols, maxChars: 1500,
  });
  console.log(`--- ${sheetName} ---\n${check.ndjson}`);
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, range, filename] of [
  ["计划购电量", "A1:H6", `plan_q4-${variant}.png`],
  ["充放电量", "A1:F8", `storage_q4-${variant}.png`],
  ["紧急购电量", `A1:C${Math.min(emergencyEndRow, 20)}`, `emergency_q4-${variant}.png`],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(previewDir, filename), new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });

console.log(`Saved ${outputPath}`);

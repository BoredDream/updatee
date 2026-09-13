// 用官方附件5模板生成 result3.xlsx（多阶段随机规划模型）。
// 计划/调整表保留模板行；实际充放电、SOC与紧急购电按自然日跨行重组。
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import { fillResultTemplate } from "./fill_result_templates.mjs";

const repoRoot = path.resolve(import.meta.dirname, "..");
const tag = process.argv[2] ?? "0123";
const K = process.argv[3] ?? "30";
const payloadPath = path.join(repoRoot, "outputs", "q3_multistage", `payload_stages${tag}_K${K}.json`);
const outputDir = path.join(process.env.RESULT_OUTPUT_ROOT ?? path.join(repoRoot, "outputs"), "q3_multistage");
const outputPath = path.join(outputDir, "result3.xlsx");
const previewDir = path.join(outputDir, "previews");

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
if (!Array.isArray(payload.days) || payload.days.length !== 334) {
  throw new Error(`Expected 334 output days, received ${payload.days?.length ?? "none"}`);
}
console.log(`meta: ${JSON.stringify(payload.meta)}`);

const emergencyEndRow = 1 + payload.days.reduce((count, day) => count + Math.max(1, day.emergency_segments.length), 0);
fillResultTemplate("q3", { tag, K });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
workbook.recalculate();

for (const [sheetName, range, maxRows, maxCols] of [
  ["计划购电量", "A1:EQ6", 6, 147],
  ["调整购电量", "A1:EQ6", 6, 147],
  ["充放电量", "A1:F15", 15, 6],
  ["紧急购电量", `A1:C${Math.min(emergencyEndRow, 20)}`, 20, 3],
]) {
  const check = await workbook.inspect({
    kind: "table", range: `${sheetName}!${range}`,
    include: "values", tableMaxRows: maxRows, tableMaxCols: maxCols, maxChars: 6000,
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
  ["计划购电量", "A1:H12", "plan_start.png"],
  ["调整购电量", "A1:H12", "adjust_start.png"],
  ["充放电量", "A1:F14", "storage_start.png"],
  ["紧急购电量", `A1:C${Math.min(emergencyEndRow, 30)}`, "emergency_start.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(previewDir, filename), new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });

console.log(`Saved ${outputPath}`);

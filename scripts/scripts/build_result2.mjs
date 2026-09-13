import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import { fillResultTemplate } from "./fill_result_templates.mjs";

const repoRoot = path.resolve(import.meta.dirname, "..");
const payloadPath = path.join(repoRoot, "outputs", "q2", "solver_payload.json");
const outputDir = path.join(process.env.RESULT_OUTPUT_ROOT ?? path.join(repoRoot, "outputs"), "q2");
const outputPath = path.join(outputDir, "result2.xlsx");
const previewDir = path.join(outputDir, "previews");

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
if (!Array.isArray(payload.days) || payload.days.length !== 334) {
  throw new Error(`Expected 334 output days, received ${payload.days?.length ?? "none"}`);
}

const emergencyEndRow = 1 + payload.days.reduce((count, day) => count + Math.max(1, day.emergency_segments.length), 0);
fillResultTemplate("q2");
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
workbook.recalculate();

const planCheck = await workbook.inspect({
  kind: "table",
  range: "计划购电量!A1:EQ6",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 147,
  maxChars: 12000,
});
console.log(planCheck.ndjson);

const storageCheck = await workbook.inspect({
  kind: "table",
  range: "充放电量!A1:F15",
  include: "values,formulas",
  tableMaxRows: 15,
  tableMaxCols: 6,
  maxChars: 6000,
});
console.log(storageCheck.ndjson);

const emergencyCheck = await workbook.inspect({
  kind: "table",
  range: `紧急购电量!A1:C${Math.min(emergencyEndRow, 20)}`,
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 3,
  maxChars: 6000,
});
console.log(emergencyCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
const previews = [
  ["计划购电量", "A1:H12", "plan_start.png"],
  ["计划购电量", "EN1:EQ12", "plan_totals.png"],
  ["充放电量", "A1:F20", "storage_start.png"],
  ["紧急购电量", `A1:C${Math.min(emergencyEndRow, 30)}`, "emergency_start.png"],
];
for (const [sheetName, range, filename] of previews) {
  const preview = await workbook.render({ sheetName, range, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(previewDir, filename), new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });


const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const savedCheck = await saved.inspect({
  kind: "sheet,table",
  maxChars: 6000,
  tableMaxRows: 3,
  tableMaxCols: 8,
});
console.log(savedCheck.ndjson);
console.log(`Saved ${outputPath}`);


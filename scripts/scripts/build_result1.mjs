import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import { fillResultTemplate } from "./fill_result_templates.mjs";

const repoRoot = path.resolve(import.meta.dirname, "..");
const payloadPath = path.join(repoRoot, "outputs", "q1", "solver_payload.json");
const outputDir = path.join(process.env.RESULT_OUTPUT_ROOT ?? path.join(repoRoot, "outputs"), "q1");
const outputPath = path.join(outputDir, "result1.xlsx");
const previewDir = path.join(outputDir, "previews");

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
if (!Array.isArray(payload.template_grid_kwh) || payload.template_grid_kwh.length !== 144) {
  throw new Error(`Expected 144 template intervals, received ${payload.template_grid_kwh?.length ?? "none"}`);
}
if (!Array.isArray(payload.storage_blocks) || payload.storage_blocks.length !== 6) {
  throw new Error(`Expected 6 storage blocks, received ${payload.storage_blocks?.length ?? "none"}`);
}

fillResultTemplate("q1");
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
workbook.recalculate();

const planCheck = await workbook.inspect({
  kind: "table",
  range: "计划购电量!A1:B12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 2,
  maxChars: 4000,
});
console.log(planCheck.ndjson);

const planTail = await workbook.inspect({
  kind: "table",
  range: "计划购电量!A140:B145",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 2,
  maxChars: 4000,
});
console.log(planTail.ndjson);

const storageCheck = await workbook.inspect({
  kind: "table",
  range: "充放电量!A1:E7",
  include: "values,formulas",
  tableMaxRows: 7,
  tableMaxCols: 5,
  maxChars: 4000,
});
console.log(storageCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
const previews = [
  ["计划购电量", "A1:B12", "plan_start.png"],
  ["计划购电量", "A134:B145", "plan_end.png"],
  ["充放电量", "A1:F7", "storage.png"],
];
for (const [sheetName, range, filename] of previews) {
  const preview = await workbook.render({ sheetName, range, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(previewDir, filename), new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });


const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const savedCheck = await saved.inspect({
  kind: "sheet,table",
  maxChars: 4000,
  tableMaxRows: 3,
  tableMaxCols: 6,
});
console.log(savedCheck.ndjson);
console.log(`Saved ${outputPath}`);

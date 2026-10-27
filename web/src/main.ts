import "./style.css";
import Papa from "papaparse";
import { drawCv, drawPath, featureColor } from "./chart";
import { syntheticCsv } from "./data.mjs";
import type { CrossValidation, Inspection, Loaded, ModelPath, Reply, Request } from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const hide = (id: string, value = true) => { $(id).hidden = value; };
const text = (id: string, value: string) => { $(id).textContent = value; };
const value = (id: string) => $<HTMLInputElement>(id).value;
const format = (v: number) => v === 0 ? "0" : Math.abs(v) < .001 || Math.abs(v) >= 10000 ? v.toExponential(3) : v.toPrecision(5);
let worker: Worker;
let busy = false;
let loaded: Loaded | undefined;
let inspection: Inspection | undefined;
let model: ModelPath | undefined;
let cv: CrossValidation | undefined;
let sourceName = "";
let automatic = false;
let settings = { alpha: 1, nlambda: 100, eps: 1e-7, folds: 0 };
let pending = settings;
let exampleAbort: AbortController | undefined;

function makeWorker() {
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (event: MessageEvent<Reply>) => receive(event.data);
  worker.onerror = (event) => fail(event.message || "The computation worker stopped unexpectedly. Try a smaller dataset.");
}

function setBusy(next: boolean) {
  busy = next;
  for (const id of ["load-example", "file", "example", "family", "alpha", "nlambda", "eps", "folds", "go-best", "go-1se", "target", "positive", "import"]) {
    ($<HTMLInputElement>(id)).disabled = next;
  }
  $("fit").toggleAttribute("disabled", next || !loaded);
  document.querySelectorAll<HTMLButtonElement>("[data-alpha]").forEach((button) => { button.disabled = next; });
  hide("cancel", !next); hide("progress", !next);
  if (next) $<HTMLProgressElement>("progress").removeAttribute("value");
}

function post(request: Request) { setBusy(true); hide("error"); worker.postMessage(request); }
function status(message: string) { text("status", message); }
function fail(message: string) { setBusy(false); automatic = false; text("error", message); hide("error", false); status("Could not complete that step."); }

function clearModel() {
  model = undefined; cv = undefined;
  hide("model-results"); hide("chart-panel"); hide("cv-panel"); hide("lambda-controls"); hide("empty", false);
  text("timing", "Ready when you are"); text("path-title", "From sparse to expressive.");
}

async function inspectFile(file: File, auto = false) {
  if (busy) return;
  loaded = undefined; inspection = undefined; sourceName = file.name; automatic = auto;
  hide("dataset-summary"); clearModel(); hide("import-panel"); status("Inspecting file locally…");
  post({ kind: "inspect", file });
}

function chosenFeatures() {
  return [...document.querySelectorAll<HTMLInputElement>("#features input:checked")].map((c) => c.value);
}

function updateFeatureCount() { text("feature-count", `Choose predictors · ${chosenFeatures().length.toLocaleString()} selected`); }

function populateColumns() {
  if (!inspection) return;
  hide("positive-field", value("family") === "gaussian");
  hide("class-hint", value("family") === "gaussian");
  const target = value("target");
  const targetIndex = inspection.columns.indexOf(target);
  const classes = [...new Set(inspection.preview.map((r) => r[targetIndex]).filter((v) => v.trim() !== ""))].sort();
  const options = $("class-options"); options.replaceChildren();
  for (const label of classes) { const option = document.createElement("option"); option.value = label; options.append(option); }
  $<HTMLInputElement>("positive").value = classes.includes("1") ? "1" : classes.at(-1) ?? "1";
  const features = $("features"); features.replaceChildren();
  inspection.columns.forEach((column, i) => {
    if (column === target) return;
    const label = document.createElement("label"); label.dataset.name = column.toLowerCase();
    const input = document.createElement("input"); input.type = "checkbox"; input.value = column;
    input.checked = inspection!.numeric[i]; input.dataset.numeric = String(inspection!.numeric[i]);
    input.addEventListener("change", updateFeatureCount);
    label.append(input, document.createTextNode(column)); features.append(label);
  });
  updateFeatureCount();
  const table = $("preview-table"); table.replaceChildren();
  const shown = [targetIndex, ...inspection.columns.map((_, i) => i).filter((i) => i !== targetIndex).slice(0, 5)];
  const head = document.createElement("thead"), tr = document.createElement("tr");
  for (const i of shown) { const th = document.createElement("th"); th.textContent = inspection.columns[i]; tr.append(th); }
  head.append(tr); table.append(head);
  const body = document.createElement("tbody");
  for (const values of inspection.preview.slice(0, 5)) {
    const row = document.createElement("tr");
    for (const i of shown) { const td = document.createElement("td"); td.textContent = values[i]; row.append(td); }
    body.append(row);
  }
  table.append(body);
}

function loadSelection() {
  const features = chosenFeatures();
  if (!features.length) { fail("Select at least one numeric predictor."); return; }
  status("Loading selected columns…");
  post({ kind: "load", target: value("target"), features, positive: value("positive").trim(), family: value("family") as "binomial" | "gaussian" });
}

function fit() {
  if (!loaded || busy) return;
  pending = { alpha: Number(value("alpha")), nlambda: Number(value("nlambda")), eps: Number(value("eps")), folds: Number(value("folds")) };
  status("Starting the Rust solver…"); post({ kind: "fit", ...pending });
}

function receive(reply: Reply) {
  if (reply.kind === "progress") {
    status(reply.message);
    if (reply.total) { const progress = $<HTMLProgressElement>("progress"); progress.max = reply.total; progress.value = reply.done ?? 0; }
    return;
  }
  setBusy(false);
  if (reply.kind === "error") { fail(reply.message); return; }
  if (reply.kind === "inspected") {
    inspection = reply.info;
    text("file-name", `${inspection.name} · ${inspection.columns.length.toLocaleString()} columns${inspection.rows ? ` · ${inspection.rows.toLocaleString()} rows` : ""}`);
    const target = $<HTMLSelectElement>("target"); target.replaceChildren();
    for (const column of inspection.columns) { const option = document.createElement("option"); option.value = column; option.textContent = column; target.append(option); }
    target.value = inspection.columns.find((c) => /^(response|target|y|label|class|diagnosis)$/i.test(c)) ?? inspection.columns.at(-1)!;
    populateColumns();
    if (automatic) { automatic = false; loadSelection(); }
    else { hide("import-panel", false); status("Choose a response and predictors, then load the columns."); }
  } else if (reply.kind === "loaded") {
    loaded = reply.info;
    const response = loaded.family === "gaussian" ? "Gaussian · continuous response" : `Positive class: ${loaded.classes[1]} (${loaded.positives.toLocaleString()} rows)`;
    text("dataset-summary", `${loaded.rows.toLocaleString()} rows × ${loaded.features.length.toLocaleString()} predictors · ${(loaded.bytes / 1024 ** 2).toFixed(2)} MiB\n${response}`);
    hide("dataset-summary", false); hide("import-panel"); fit();
  } else {
    if (!reply.path.lambdas.length) { fail(reply.path.warnings.join("\n") || "The path stopped before producing a model."); return; }
    model = reply.path; cv = reply.cv; settings = pending;
    hide("empty"); hide("chart-panel", false); hide("lambda-controls", false); hide("model-results", false);
    text("path-title", `${settings.alpha === 1 ? "Lasso" : "Elastic net"} · α = ${settings.alpha.toFixed(2)}`);
    text("timing", `${model.lambdas.length} models · ${reply.milliseconds < 1000 ? `${Math.round(reply.milliseconds)} ms` : `${(reply.milliseconds / 1000).toFixed(2)} s`}`);
    const slider = $<HTMLInputElement>("lambda"); slider.max = String(model.lambdas.length - 1); slider.value = String(cv ? cv.oneSe : Math.floor((model.lambdas.length - 1) * .6));
    hide("cv-panel", !cv);
    text("training-note", cv ? `Cross-validated over ${cv.folds} folds: the slider starts at λ1se, the largest penalty whose held-out loss is within one standard error of the lowest.` : "These are training fits, not an estimate of predictive performance. Turn on cross-validation for one.");
    text("lambda-max", `λ ${format(model.lambdas[0])}`); text("lambda-min", `λ ${format(model.lambdas.at(-1)!)}`);
    text("warnings", model.warnings.join("\n")); hide("warnings", model.warnings.length === 0);
    renderModel(); status(cv ? `Cross-validated over ${cv.folds} folds. Move the λ slider, or jump to λmin or λ1se.` : "Path ready. Move the λ slider to explore a model.");
  }
}

function renderModel() {
  if (!model || !loaded) return;
  const k = Number(value("lambda"));
  text("lambda-value", format(model.lambdas[k]));
  text("active-count", String(model.indices[k].length)); text("iterations", model.iterations[k].toLocaleString());
  text("rejections", model.n_rejected[k].toLocaleString()); text("deviance", model.deviance[k].toFixed(2));
  text("loss-label", model.family === "gaussian" ? "Training RSS" : "Training deviance");
  const body = $("coefficient-body"); body.replaceChildren();
  const append = (name: string, raw: number, std: number | undefined, index?: number) => {
    const row = document.createElement("tr"), feature = document.createElement("td");
    if (index !== undefined) { const dot = document.createElement("span"); dot.className = "swatch"; dot.style.background = featureColor(index); feature.append(dot); }
    feature.append(document.createTextNode(name)); row.append(feature);
    for (const v of [raw, std]) { const td = document.createElement("td"); td.textContent = v === undefined ? "—" : format(v); row.append(td); }
    body.append(row);
  };
  append("(intercept)", model.intercept[k], undefined);
  const entries = model.indices[k].map((j, i) => ({ j, b: model!.coefficients[k][i], s: model!.standardized_coefficients[k][i] })).sort((a, b) => Math.abs(b.b) - Math.abs(a.b));
  for (const entry of entries.slice(0, 100)) append(loaded.features[entry.j], entry.b, entry.s, entry.j);
  text("table-note", entries.length === 0 ? "All feature coefficients are zero at this lambda." : entries.length > 100 ? `Showing 100 of ${entries.length.toLocaleString()} active features. Download all coefficients below.` : `${entries.length.toLocaleString()} active features of ${loaded.features.length.toLocaleString()}.`);
  drawPath($<HTMLCanvasElement>("chart"), model, k, cv);
  if (cv) {
    drawCv($<HTMLCanvasElement>("cv-chart"), model, cv, k);
    const loss = model.family === "gaussian" ? "squared error" : "deviance";
    text("cv-note", (k < cv.mean.length ? `Held-out ${loss} per row at this λ: ${format(cv.mean[k])} ± ${format(cv.se[k])}. ` : "Not every fold reached this λ. ")
      + `Lowest at λmin = ${format(model.lambdas[cv.best])} (${model.indices[cv.best].length} features); λ1se = ${format(model.lambdas[cv.oneSe])} (${model.indices[cv.oneSe].length} features). Solid orange marks λmin, dashed orange λ1se, and dotted black the λ shown; bars are ±1 standard error across folds.`);
  }
}

function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("load-example").addEventListener("click", async () => {
  if (busy) return;
  const controller = new AbortController(); exampleAbort = controller;
  try {
    const key = value("example");
    $<HTMLSelectElement>("family").value = key === "gaussian" ? "gaussian" : "binomial";
    if (key === "synthetic" || key === "gaussian") { await inspectFile(new File([syntheticCsv(key === "gaussian" ? "gaussian" : "binomial")], "correlated-features.csv"), true); return; }
    setBusy(true); status("Loading example…");
    const name = key === "colon" ? "colon.csv" : "breast-cancer.csv";
    const response = await fetch(new URL(`examples/${name}`, document.baseURI), { signal: controller.signal });
    if (!response.ok) throw new Error("The example could not be loaded. You can still choose a local file.");
    const file = new File([await response.blob()], name);
    if (controller.signal.aborted) return;
    setBusy(false); await inspectFile(file, true);
  } catch (error) { if (!controller.signal.aborted) fail(error instanceof Error ? error.message : String(error)); }
  finally { if (exampleAbort === controller) exampleAbort = undefined; }
});
$("example").addEventListener("change", () => {
  const descriptions: Record<string, string> = {
    colon: "Tumor versus normal tissue. A high-dimensional example shipped with R biglasso.",
    breast: "Malignant versus benign tumors. A small example with 30 measured features; 1 means malignant.",
    synthetic: "Six groups of correlated predictors. Compare lasso with elastic net to explore how coefficients are shared.",
    gaussian: "A continuous response with six groups of correlated predictors. Explore least-squares lasso and elastic net.",
  };
  text("example-description", descriptions[value("example")]);
});
$<HTMLInputElement>("file").addEventListener("change", (event) => { const file = (event.target as HTMLInputElement).files?.[0]; if (file) void inspectFile(file); });
$("dropzone").addEventListener("dragover", (event) => { event.preventDefault(); $("dropzone").classList.add("dragging"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragging"));
$("dropzone").addEventListener("drop", (event) => { event.preventDefault(); $("dropzone").classList.remove("dragging"); const file = event.dataTransfer?.files[0]; if (file) void inspectFile(file); });
$("target").addEventListener("change", populateColumns);
$("family").addEventListener("change", () => {
  loaded = undefined; clearModel(); hide("dataset-summary");
  $("fit").toggleAttribute("disabled", true);
  if (inspection) { populateColumns(); hide("import-panel", false); }
  status("Choose the response and reload columns for the selected family.");
});
$("import").addEventListener("click", loadSelection);
$("fit").addEventListener("click", fit);
$("lambda").addEventListener("input", renderModel);
for (const [id, pick] of [["go-best", (c: CrossValidation) => c.best], ["go-1se", (c: CrossValidation) => c.oneSe]] as const) {
  $(id).addEventListener("click", () => { if (cv) { $<HTMLInputElement>("lambda").value = String(pick(cv)); renderModel(); } });
}
$("feature-search").addEventListener("input", () => {
  const query = value("feature-search").toLowerCase();
  document.querySelectorAll<HTMLElement>("#features label").forEach((label) => { label.hidden = !label.dataset.name!.includes(query); });
});
for (const id of ["select-all", "select-none"]) $(id).addEventListener("click", () => {
  document.querySelectorAll<HTMLInputElement>("#features input").forEach((input) => { input.checked = id === "select-all" && input.dataset.numeric === "true"; }); updateFeatureCount();
});
function alphaChanged() {
  text("alpha-value", Number(value("alpha")).toFixed(2));
  document.querySelectorAll<HTMLElement>("[data-alpha]").forEach((button) => button.classList.toggle("selected", Number(button.dataset.alpha) === Number(value("alpha"))));
  if (model) status("Settings changed. Fit the path to apply them.");
}
$("alpha").addEventListener("input", alphaChanged);
document.querySelectorAll<HTMLElement>("[data-alpha]").forEach((button) => button.addEventListener("click", () => { $<HTMLInputElement>("alpha").value = button.dataset.alpha!; alphaChanged(); }));
for (const id of ["nlambda", "eps", "folds"]) $(id).addEventListener("change", () => { if (model) status("Settings changed. Fit the path to apply them."); });
$("cancel").addEventListener("click", () => {
  exampleAbort?.abort(); exampleAbort = undefined;
  worker.onmessage = null; worker.onerror = null; worker.terminate(); makeWorker();
  loaded = undefined; inspection = undefined; automatic = false;
  setBusy(false); clearModel(); hide("import-panel"); hide("dataset-summary"); status("Cancelled. Choose the example or file again to reload its data.");
});
$("download-csv").addEventListener("click", () => {
  if (!model || !loaded) return;
  const k = Number(value("lambda")), coefs = new Map(model.indices[k].map((j, i) => [j, model!.coefficients[k][i]]));
  const rows = [["feature", "coefficient"], ["(intercept)", model.intercept[k]], ...loaded.features.map((name, j) => [name, coefs.get(j) ?? 0])];
  download("pybiglasso-coefficients.csv", Papa.unparse(rows, { escapeFormulae: true }), "text/csv;charset=utf-8");
});
$("download-json").addEventListener("click", () => {
  if (!model || !loaded) return;
  download("pybiglasso-path.json", JSON.stringify({ format_version: 1, source: sourceName, response: value("target"), features: loaded.features, classes: loaded.family === "binomial" ? loaded.classes : undefined, settings, selected_index: Number(value("lambda")), ...model, cross_validation: cv }, null, 2), "application/json");
});
new ResizeObserver(() => { if (model) renderModel(); }).observe($("chart-panel"));
makeWorker();

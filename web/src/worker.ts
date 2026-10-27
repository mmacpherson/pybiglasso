import Papa from "papaparse";
import { parquetMetadataAsync, parquetRead, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import init, { Dataset } from "../pkg/solver.js";
import wasmUrl from "../pkg/solver_bg.wasm?url";
import { binaryClasses, checkSize, numericValue, responseValue } from "./data.mjs";
import type { Inspection, Loaded, ModelPath, Reply, Request } from "./types";

const send = (message: Reply) => self.postMessage(message);
const host = globalThis as typeof globalThis & { pybiglassoProgress: (done: number, total: number) => void };
host.pybiglassoProgress = (done, total) => send({ kind: "progress", message: `Fitting λ ${done} of ${total}`, done, total });
const ready = init({ module_or_path: wasmUrl });
let file: File | undefined;
let inspection: Inspection | undefined;
let dataset: Dataset | undefined;
let family: "binomial" | "gaussian" = "binomial";
let busy = false;

function parquetFile(input: File) {
  return { byteLength: input.size, slice: (start: number, end?: number) => input.slice(start, end).arrayBuffer() };
}

function validateHeaders(columns: string[], renamed?: Record<string, string>) {
  if (!columns.length || columns.some((s) => !s.trim()) || new Set(columns).size !== columns.length
      || (renamed && Object.keys(renamed).length > 0)) {
    throw new Error("Use nonempty, unique column names in the file header.");
  }
}

async function inspect(input: File): Promise<Inspection> {
  let columns: string[];
  let values: Record<string, unknown>[];
  let rows: number | undefined;
  if (/\.parquet$/i.test(input.name)) {
    const source = parquetFile(input);
    const metadata = await parquetMetadataAsync(source);
    rows = Number(metadata.num_rows);
    if (!Number.isSafeInteger(rows)) throw new Error("Parquet row count is too large.");
    // Nested tables have no unambiguous mapping to one scalar per feature.
    if (metadata.schema.some((s, i) => i > 0 && s.num_children)) {
      throw new Error("Select a flat Parquet table; nested fields are not supported.");
    }
    columns = metadata.schema.slice(1).map((s) => s.name);
    validateHeaders(columns);
    values = await parquetReadObjects({ file: source, metadata, columns, rowEnd: 20, compressors });
  } else {
    const parsed = await new Promise<Papa.ParseResult<Record<string, string>>>((resolve, reject) => {
      Papa.parse<Record<string, string>>(input, {
        header: true, preview: 20, skipEmptyLines: "greedy",
        complete: resolve, error: reject,
      });
    });
    if (parsed.errors.length) throw new Error(parsed.errors[0].message);
    columns = parsed.meta.fields ?? [];
    validateHeaders(columns, parsed.meta.renamedHeaders);
    values = parsed.data;
  }
  if (!values.length) throw new Error("The table has no data rows.");
  const numeric = columns.map((column) => {
    try { values.forEach((row, i) => numericValue(row[column], column, i)); return true; }
    catch { return false; }
  });
  return { name: input.name, columns, rows, numeric, preview: values.map((row) => columns.map((c) => row[c] == null ? "" : String(row[c]))) };
}

async function csvRows(input: File, onChunk: (rows: Record<string, string>[]) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    let failed = false;
    Papa.parse<Record<string, string>>(input, {
      header: true, skipEmptyLines: "greedy", chunkSize: 256 * 1024,
      chunk(result, parser) {
        try {
          if (result.errors.length) throw new Error(result.errors[0].message);
          onChunk(result.data);
        } catch (error) { failed = true; parser.abort(); reject(error); }
      },
      complete() { if (!failed) resolve(); }, error: reject,
    });
  });
}

async function load(target: string, features: string[], positive: string, selectedFamily: "binomial" | "gaussian"): Promise<Loaded> {
  if (!file || !inspection) throw new Error("Choose a file first.");
  if (!inspection.columns.includes(target) || !features.length || features.includes(target)
      || new Set(features).size !== features.length || features.some((c) => !inspection!.columns.includes(c))) {
    throw new Error("Choose a response and at least one different feature.");
  }
  dataset?.free(); dataset = undefined;
  const labels = new Set<string>();
  let rows = inspection.rows ?? 0;
  send({ kind: "progress", message: "Checking rows and memory…" });
  if (!/\.parquet$/i.test(file.name)) {
    rows = 0;
    await csvRows(file, (chunk) => {
      rows += chunk.length;
      checkSize(Math.max(2, rows), features.length);
      for (const [i, row] of chunk.entries()) {
        if (selectedFamily === "gaussian") { numericValue(row[target], target, rows - chunk.length + i); continue; }
        labels.add(responseValue(row[target]));
        if (labels.size > 2) throw new Error("The response must have exactly two classes. Choose Gaussian for continuous outcomes.");
      }
    });
  }
  const bytes = checkSize(rows, features.length);
  await ready;
  const next = new Dataset(rows, features.length);
  let positives = 0;
  try {
    if (/\.parquet$/i.test(file.name)) {
      await parquetRead({ file: parquetFile(file), columns: [...features, target], compressors,
        onChunk(chunk) {
          if (chunk.columnName === target) {
            const values = Float64Array.from(chunk.columnData, (value: unknown, i: number) => {
              if (selectedFamily === "gaussian") return numericValue(value, target, chunk.rowStart + i);
              const label = responseValue(value); labels.add(label);
              if (labels.size > 2) throw new Error("The response must have exactly two classes.");
              const y = Number(label === positive); positives += y; return y;
            });
            next.write_response(chunk.rowStart, values);
          } else {
            const index = features.indexOf(chunk.columnName);
            const values = Float64Array.from(chunk.columnData, (v: unknown, i: number) => numericValue(v, chunk.columnName, chunk.rowStart + i));
            next.write_column(index, chunk.rowStart, values);
          }
        },
      });
    } else {
      let start = 0;
      await csvRows(file, (chunk) => {
        for (const [j, name] of features.entries()) {
          next.write_column(j, start, Float64Array.from(chunk, (r, i) => numericValue(r[name], name, start + i)));
        }
        next.write_response(start, Float64Array.from(chunk, (row, i) => {
          if (selectedFamily === "gaussian") return numericValue(row[target], target, start + i);
          const y = Number(responseValue(row[target]) === positive); positives += y; return y;
        }));
        start += chunk.length;
        send({ kind: "progress", message: `Reading ${start.toLocaleString()} of ${rows.toLocaleString()} rows` });
      });
    }
    const classes: [string, string] = selectedFamily === "binomial" ? binaryClasses(labels, positive) as [string, string] : ["", ""];
    family = selectedFamily;
    dataset = next;
    return { family, rows, features, classes, positives, bytes };
  } catch (error) { next.free(); throw error; }
}

self.onmessage = async (event: MessageEvent<Request>) => {
  if (busy) { send({ kind: "error", message: "A task is already running." }); return; }
  busy = true;
  try {
    const request = event.data;
    if (request.kind === "inspect") {
      dataset?.free(); dataset = undefined; file = request.file;
      inspection = await inspect(file);
      send({ kind: "inspected", info: inspection });
    } else if (request.kind === "load") {
      send({ kind: "loaded", info: await load(request.target, request.features, request.positive, request.family) });
    } else {
      if (!dataset) throw new Error("Load the selected columns first.");
      const start = performance.now();
      const path = dataset.fit({ family, alpha: request.alpha, nlambda: request.nlambda, eps: request.eps }) as ModelPath;
      send({ kind: "result", path, milliseconds: performance.now() - start });
    }
  } catch (error) {
    send({ kind: "error", message: error instanceof Error ? error.message : String(error) });
  } finally { busy = false; }
};

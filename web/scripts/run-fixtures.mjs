// Input matrices and independent R expectations are prepared by check_wasm.py.
// This runner only returns Wasm fits; it never generates expected values.
import { readFileSync } from "node:fs";
import { initSync, Dataset } from "../pkg/solver.js";

globalThis.pybiglassoProgress = () => {};
initSync({ module: readFileSync(new URL("../pkg/solver_bg.wasm", import.meta.url)) });
const cases = JSON.parse(readFileSync(0, "utf8"));
const engine = process.argv[2] ?? "node";
if (engine !== "node") {
  const { chromium, firefox, webkit } = await import("@playwright/test");
  const browsers = { chromium, firefox, webkit };
  if (!(engine in browsers)) throw new Error(`Unknown engine: ${engine}`);
  const browser = await browsers[engine].launch();
  try {
    const page = await browser.newPage();
    const result = await page.evaluate(async ({ javascript, bytes, cases }) => {
      globalThis.pybiglassoProgress = () => {};
      const url = URL.createObjectURL(new Blob([javascript], { type: "text/javascript" }));
      const { initSync, Dataset } = await import(url);
      initSync({ module: Uint8Array.from(atob(bytes), (c) => c.charCodeAt(0)) });
      URL.revokeObjectURL(url);
      return cases.map((entry) => {
        const n = entry.y.length, p = entry.X[0].length;
        const dataset = new Dataset(n, p);
        try {
          for (let j = 0; j < p; j++) dataset.write_column(j, 0, Float64Array.from(entry.X, (row) => row[j]));
          dataset.write_response(0, Float64Array.from(entry.y));
          return dataset.fit(entry.options);
        } finally { dataset.free(); }
      });
    }, {
      javascript: readFileSync(new URL("../pkg/solver.js", import.meta.url), "utf8"),
      bytes: readFileSync(new URL("../pkg/solver_bg.wasm", import.meta.url)).toString("base64"),
      cases,
    });
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
} else {
const results = [];
for (const entry of cases) {
  const n = entry.y.length, p = entry.X[0].length;
  const dataset = new Dataset(n, p);
  try {
    for (let j = 0; j < p; j++) dataset.write_column(j, 0, Float64Array.from(entry.X, (row) => row[j]));
    dataset.write_response(0, Float64Array.from(entry.y));
    results.push(dataset.fit(entry.options));
  } finally { dataset.free(); }
}
console.log(JSON.stringify(results));
}

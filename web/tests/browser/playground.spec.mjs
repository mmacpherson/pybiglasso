import { test, expect } from "@playwright/test";
import { parquetWriteBuffer } from "hyparquet-writer";
import { gzipSync, zstdCompressSync } from "node:zlib";

const rows = Array.from({ length: 48 }, (_, i) => [Number((i % 3 === 2 && i % 7 !== 0) || (i % 3 === 1 && i % 2 === 0) || i % 11 === 0), i % 3 - 1, Math.floor(i / 3) % 3 - 1]);
const csv = "response,x,z\n" + rows.map((r) => r.join(",")).join("\n");

async function upload(page, name, buffer, family = "binomial") {
  await page.locator("#family").selectOption(family);
  await page.locator("#file").setInputFiles({ name, mimeType: "application/octet-stream", buffer: Buffer.from(buffer) });
  await expect(page.locator("#import-panel")).toBeVisible();
  await page.locator("#import").click();
  await expect(page.locator("#model-results")).toBeVisible();
  await expect(page.locator("#cancel")).toBeHidden();
}

async function pathDownload(page) {
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#download-json").click();
  const download = await downloadPromise;
  const stream = await download.createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString());
}

test("colon fits locally, lambda changes the model, and enet refits", async ({ page }) => {
  const errors = []; page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.locator("#load-example").click();
  await expect(page.locator("#dataset-summary")).toContainText("62 rows × 2,000 predictors");
  await expect(page.locator("#model-results")).toBeVisible();
  await expect(page.locator("#cancel")).toBeHidden();
  await page.locator("#lambda").fill("0");
  await expect(page.locator("#active-count")).toHaveText("0");
  const nullIntercept = await page.locator("#coefficient-body tr").first().textContent();
  expect(nullIntercept).toContain((Math.log(40 / 22)).toPrecision(5));
  const last = await page.locator("#lambda").getAttribute("max");
  await page.locator("#lambda").fill(last);
  await expect(page.locator("#active-count")).not.toHaveText("0");
  await page.locator('[data-alpha="0.5"]').click(); await page.locator("#fit").click();
  await expect(page.locator("#path-title")).toContainText("α = 0.50");
  const result = await pathDownload(page);
  expect(result.features).toHaveLength(2000);
  expect(result.settings.alpha).toBe(.5);
  expect(result.iterations.length).toBe(result.lambdas.length);
  expect(errors).toEqual([]);
});

test("CSV and multi-row-group Parquet produce identical paths, with no data requests", async ({ page }) => {
  const network = [];
  page.on("request", (r) => {
    if (r.url().startsWith("blob:")) return;
    const url = new URL(r.url());
    if (!["GET", "HEAD"].includes(r.method()) || url.origin !== "http://127.0.0.1:4174"
        || url.search || !/^\/(?:assets\/[^/]+\.(?:js|css|wasm))?$/.test(url.pathname)) network.push(r.url());
  });
  await page.goto("/");
  await upload(page, "local.csv", csv);
  const a = await pathDownload(page);
  expect(a.indices.some((row) => row.length > 0)).toBe(true);
  for (const codec of ["SNAPPY", "GZIP", "ZSTD"]) {
  const buffer = parquetWriteBuffer({ columnData: [
    { name: "response", data: rows.map((r) => r[0]), type: "INT32" },
    { name: "x", data: rows.map((r) => r[1]), type: "DOUBLE" },
    { name: "z", data: rows.map((r) => r[2]), type: "DOUBLE" },
  ], rowGroupSize: 7, codec, compressors: { GZIP: gzipSync, ZSTD: zstdCompressSync } });
  await upload(page, "local.parquet", new Uint8Array(buffer));
  const b = await pathDownload(page);
  for (const field of ["lambdas", "intercept", "coefficients", "iterations", "n_rejected", "center", "scale"]) expect(a[field]).toEqual(b[field]);
  }
  expect(network).toEqual([]);
});

test("missing values after the preview and continuous responses give clear errors", async ({ page }) => {
  await page.goto("/");
  await page.locator("#family").selectOption("binomial");
  const bad = rows.map((r) => [...r]); bad[30][1] = "";
  await page.locator("#file").setInputFiles({ name: "missing.csv", mimeType: "text/csv", buffer: Buffer.from("response,x,z\n" + bad.map((r) => r.join(",")).join("\n")) });
  await expect(page.locator("#import-panel")).toBeVisible(); await page.locator("#import").click();
  await expect(page.locator("#error")).toContainText("Missing value");
  const continuous = "response,x\n" + Array.from({ length: 30 }, (_, i) => `${i + .5},${i}`).join("\n");
  await page.locator("#file").setInputFiles({ name: "continuous.csv", mimeType: "text/csv", buffer: Buffer.from(continuous) });
  await expect(page.locator("#import-panel")).toBeVisible(); await page.locator("#import").click();
  await expect(page.locator("#error")).toContainText("Gaussian");
});

test("phone layout fits the viewport and a new dataset replaces the old model", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/"); await page.locator("#example").selectOption("synthetic"); await page.locator("#load-example").click();
  await expect(page.locator("#model-results")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.locator("#example").selectOption("breast"); await page.locator("#load-example").click();
  await expect(page.locator("#dataset-summary")).toContainText("569 rows × 30 predictors");
  await expect(page.locator("#model-results")).toBeVisible();
});

test("Gaussian imports CSV and Parquet, reports RSS, and exports the fitted family", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#family")).toHaveValue("gaussian");
  const data = Array.from({ length: 48 }, (_, i) => [3 + 2 * (i % 3 - 1) + i % 5 / 10, i % 3 - 1, Math.floor(i / 3) % 3 - 1]);
  await upload(page, "regression.csv", "response,x,z\n" + data.map((r) => r.join(",")).join("\n"), "gaussian");
  await expect(page.locator("#loss-label")).toHaveText("Training RSS");
  const a = await pathDownload(page);
  expect(a.family).toBe("gaussian");
  expect(a.classes).toBeUndefined();
  expect(a.lambdas.length).toBe(100);
  expect(a.deviance.at(-1)).toBeLessThan(a.deviance[0]);
  const buffer = parquetWriteBuffer({ columnData: [
    { name: "response", data: data.map((r) => r[0]), type: "DOUBLE" },
    { name: "x", data: data.map((r) => r[1]), type: "DOUBLE" },
    { name: "z", data: data.map((r) => r[2]), type: "DOUBLE" },
  ], rowGroupSize: 7 });
  await upload(page, "regression.parquet", new Uint8Array(buffer), "gaussian");
  const b = await pathDownload(page);
  for (const field of ["lambdas", "intercept", "coefficients", "deviance", "iterations", "n_rejected"]) expect(a[field]).toEqual(b[field]);
  await page.locator("#example").selectOption("gaussian");
  await page.locator("#load-example").click();
  await expect(page.locator("#dataset-summary")).toContainText("Gaussian");
  await expect(page.locator("#model-results")).toBeVisible();
  expect((await pathDownload(page)).family).toBe("gaussian");
});

test("cancelling an example download cannot restart a fit when the response arrives", async ({ page }) => {
  let release;
  const waiting = new Promise((resolve) => { release = resolve; });
  await page.route("**/examples/colon.csv", async (route) => { await waiting; await route.continue().catch(() => {}); });
  await page.goto("/"); await page.locator("#load-example").click();
  await page.locator("#cancel").click(); release();
  // Let the pending response and its continuations settle before checking
  // cancellation; starting another task immediately could mask a late restart.
  await page.waitForLoadState("networkidle");
  await expect(page.locator("#status")).toContainText("Cancelled");
  await expect(page.locator("#model-results")).toBeHidden();
  await page.locator("#example").selectOption("synthetic"); await page.locator("#load-example").click();
  await expect(page.locator("#dataset-summary")).toContainText("160 rows × 48 predictors");
  await expect(page.locator("#model-results")).toBeVisible();
});

test("cross-validation chooses λ on held-out rows for both families", async ({ page }) => {
  const errors = []; page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  for (const example of ["breast", "gaussian"]) {
    await page.locator("#example").selectOption(example);
    await page.locator("#folds").selectOption("5");
    await page.locator("#load-example").click();
    await expect(page.locator("#model-results")).toBeVisible();
    await expect(page.locator("#cancel")).toBeHidden();
    await expect(page.locator("#cv-panel")).toBeVisible();
    const result = await pathDownload(page);
    const cv = result.cross_validation;
    expect(cv.folds).toBe(5);
    expect(cv.mean.length).toBeGreaterThan(1);
    expect(cv.best).toBeGreaterThanOrEqual(cv.oneSe);
    // λ1se is the largest penalty within one standard error of the lowest loss
    expect(cv.mean[cv.oneSe]).toBeLessThanOrEqual(cv.mean[cv.best] + cv.se[cv.best]);
    expect(cv.mean.slice(0, cv.oneSe).every((m) => m > cv.mean[cv.best] + cv.se[cv.best])).toBe(true);
    expect(result.selected_index).toBe(cv.oneSe);
    await page.locator("#go-best").click();
    await expect(page.locator("#active-count")).toHaveText(String(result.indices[cv.best].length));
    await expect(page.locator("#cv-note")).toContainText(example === "gaussian" ? "squared error" : "deviance");
  }
  await page.locator("#folds").selectOption("0"); await page.locator("#fit").click();
  await expect(page.locator("#cv-panel")).toBeHidden();
  expect(errors).toEqual([]);
});

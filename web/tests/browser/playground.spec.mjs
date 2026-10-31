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

// Record the actual canvas marker strokes, including the device-pixel transform.
async function observeLambdaMarkers(page) {
  await page.addInitScript(() => {
    const proto = CanvasRenderingContext2D.prototype;
    const moveTo = proto.moveTo, stroke = proto.stroke;
    proto.moveTo = function (x, y) { this.__startX = x; return moveTo.call(this, x, y); };
    proto.stroke = function (...args) {
      const kind = this.strokeStyle === "#ba783b" && this.globalAlpha === 1 && this.getLineDash().length === 0 ? "best"
        : this.strokeStyle === "#283f30" && this.getLineDash().join() === "3,4" ? "selected" : undefined;
      if (kind && ["chart", "cv-chart"].includes(this.canvas.id)) {
        const transform = this.getTransform(), bounds = this.canvas.getBoundingClientRect();
        (window.__lambdaMarkers ??= {})[`${this.canvas.id}/${kind}`] = bounds.left
          + (transform.a * this.__startX + transform.e) * bounds.width / this.canvas.width;
      }
      return stroke.apply(this, args);
    };
  });
}

async function lambdaAlignment(page) {
  const slider = page.locator("#lambda");
  const bounds = await slider.boundingBox();
  const pixels = await slider.screenshot({ scale: "css" });
  // Locate the thumb itself above the track, rather than assuming a browser's
  // default inset or deriving the answer from the application's layout rules.
  return page.evaluate(async ({ bounds, png }) => {
    const blob = await (await fetch(`data:image/png;base64,${png}`)).blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement("canvas"); canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d"); ctx.drawImage(bitmap, 0, 0);
    const row = ctx.getImageData(0, 2, canvas.width, 1).data, green = [];
    for (let x = 0; x < canvas.width; x++) {
      const [r, g, b] = row.slice(4 * x, 4 * x + 3);
      if (g > r * 1.2 && g > b * 1.1) green.push(x);
    }
    if (!green.length) throw new Error("Could not locate the range thumb above the track");
    return { thumb: bounds.x + (green[0] + green.at(-1) + 1) / 2, markers: window.__lambdaMarkers };
  }, { bounds, png: pixels.toString("base64") });
}

for (const width of [1446, 390]) {
  test(`lambda thumb aligns with both plots at ${width}px and after resize`, async ({ page }, testInfo) => {
    await observeLambdaMarkers(page);
    await page.setViewportSize({ width, height: 1000 });
    await page.goto("/");
    await page.locator("#example").selectOption("colon");
    await page.getByText("Cross-validate to choose λ").click();
    await page.locator("#load-example").click();
    await expect(page.locator("#cv-panel")).toBeVisible();
    await expect(page.locator("#cancel")).toBeHidden();
    await page.locator("#go-best").click();
    const best = await page.locator("#lambda").inputValue();
    for (const viewport of [width, width === 390 ? 1446 : 390]) {
      await page.setViewportSize({ width: viewport, height: 1000 });
      // Wait for the existing plot resize observer; don't move the slider or
      // otherwise force a render that could conceal stale resize geometry.
      await expect.poll(() => page.evaluate(() => ["chart", "cv-chart"].every((id) => {
        const canvas = document.getElementById(id);
        return canvas.width === Math.round(canvas.clientWidth * (window.devicePixelRatio || 1));
      }))).toBe(true);
      for (const index of [best, "0", await page.locator("#lambda").getAttribute("max")]) {
        if (index !== best) await page.locator("#lambda").fill(index);
        const measurement = { viewport, index, ...await lambdaAlignment(page) };
        await testInfo.attach(`lambda-alignment-${viewport}-${index}`, { body: JSON.stringify(measurement, null, 2), contentType: "application/json" });
        console.log(`lambda alignment ${testInfo.project.name}: ${JSON.stringify(measurement)}`);
        const kind = index === best ? "best" : "selected";
        for (const id of ["chart", "cv-chart"]) {
          expect(Math.abs(measurement.thumb - measurement.markers[`${id}/${kind}`])).toBeLessThanOrEqual(2);
        }
        if (index !== best) {
          const label = await page.locator(index === "0" ? "#lambda-max" : "#lambda-min").boundingBox();
          const edge = index === "0" ? label.x : label.x + label.width;
          expect(Math.abs(edge - measurement.markers["chart/selected"])).toBeLessThanOrEqual(2);
        }
      }
      await page.locator("#lambda").fill(best);
    }
  });
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
  // the table opens sorted by standardized size; headings re-sort it
  const column = async (i) => (await page.locator(`#coefficient-body tr:not(:first-child) td:nth-child(${i})`).allTextContents()).map(Number);
  const nonincreasing = (v) => v.every((x, i) => i === 0 || Math.abs(v[i - 1]) >= Math.abs(x));
  expect(nonincreasing(await column(3))).toBe(true);
  await page.locator('th[data-sort="raw"] button').click();
  await expect(page.locator('th[data-sort="raw"]')).toHaveAttribute("aria-sort", "descending");
  expect(nonincreasing(await column(2))).toBe(true);
  await page.locator('th[data-sort="raw"] button').click();
  await expect(page.locator('th[data-sort="raw"]')).toHaveAttribute("aria-sort", "ascending");
  expect(nonincreasing((await column(2)).reverse())).toBe(true);
  await expect(page.locator("#coefficient-body tr").first()).toContainText("(intercept)");
  await page.locator('th[data-sort="feature"] button').press("Enter");
  await expect(page.locator('th[data-sort="feature"]')).toHaveAttribute("aria-sort", "ascending");
  const names = await page.locator("#coefficient-body tr:not(:first-child) td:first-child").allTextContents();
  expect(names).toEqual([...names].sort((a, b) => a.localeCompare(b, undefined, { numeric: true })));
  // penalty changes refit on their own; the Fit button stays away while fits are quick
  await page.locator('[data-alpha="0.2"]').click(); await page.locator('[data-alpha="0.5"]').click();
  await expect(page.locator("#fit")).toBeHidden();
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
    const cvSwitch = page.getByLabel("Cross-validate to choose λ");
    if (!await cvSwitch.isChecked()) await page.getByText("Cross-validate to choose λ").click();
    await expect(cvSwitch).toBeChecked();
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
    await expect(page.locator("#cv-note sub")).toHaveText(["min", "1se", "min", "1se"]);
  }
  await page.getByText("Cross-validate to choose λ").click();
  await expect(page.getByLabel("Cross-validate to choose λ")).not.toBeChecked();
  await expect(page.locator("#cv-panel")).toBeHidden();
  expect(errors).toEqual([]);
});

test("metric explainers open on hover, focus and tap, and close on Escape", async ({ page }) => {
  await page.goto("/");
  await page.locator("#example").selectOption("breast");
  await page.locator("#load-example").click();
  await expect(page.locator("#model-results")).toBeVisible();
  const tip = page.locator("#tip-rejections");
  const button = page.getByRole("button", { name: "About strong-rule rejections" });
  await expect(tip).toBeHidden();
  // hover, then onto the text itself, which stays open; Escape closes it
  // while the pointer is still there
  await button.hover();
  await expect(tip).toBeVisible();
  await expect(tip).toContainText("screening never changes the answer");
  await tip.hover();
  await expect(tip).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(tip).toBeHidden();
  await page.mouse.move(0, 0);
  // keyboard focus opens it; Escape closes it with focus still there
  await button.focus();
  await expect(tip).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(tip).toBeHidden();
  await expect(button).toBeFocused();
  // a tap opens it and leaving the pointer elsewhere keeps it; a click away closes it
  await page.getByRole("button", { name: "About training deviance" }).click();
  await page.mouse.move(0, 0);
  await expect(page.locator("#tip-loss")).toBeVisible();
  await expect(page.locator("#tip-loss")).toContainText("deviance");
  await page.locator("h1").click();
  await expect(page.locator("#tip-loss")).toBeHidden();
});

test("Gaussian names its loss explainer for RSS", async ({ page }) => {
  await page.goto("/");
  await page.locator("#example").selectOption("gaussian");
  await page.locator("#load-example").click();
  await expect(page.locator("#model-results")).toBeVisible();
  await expect(page.getByRole("button", { name: "About training rss" })).toBeVisible();
  await expect(page.locator("#loss-label")).toHaveText("Training RSS");
});

test("a change just before loading another dataset doesn't refit the new one twice", async ({ page }) => {
  const fits = [];
  await page.goto("/");
  await page.locator("#example").selectOption("breast");
  await page.locator("#load-example").click();
  await expect(page.locator("#model-results")).toBeVisible();
  await page.evaluate(() => {
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (message, ...rest) { (window.__posts ??= []).push(message.kind); return post.call(this, message, ...rest); };
  });
  await page.route("**/examples/colon.csv", async (route) => { await new Promise((r) => setTimeout(r, 800)); await route.continue(); });
  await page.locator('[data-alpha="0.5"]').click();
  await page.locator("#example").selectOption("colon");
  await page.locator("#load-example").click();
  await expect(page.locator("#dataset-summary")).toContainText("62 rows");
  await expect(page.locator("#model-results")).toBeVisible();
  await page.waitForTimeout(500);
  fits.push(...await page.evaluate(() => window.__posts.filter((kind) => kind === "fit")));
  expect(fits).toHaveLength(1);
});

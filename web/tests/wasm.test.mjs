import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, Dataset } from "../pkg/solver.js";

globalThis.pybiglassoProgress = () => {};
initSync({ module: readFileSync(new URL("../pkg/solver_bg.wasm", import.meta.url)) });

function sample() {
  const d = new Dataset(8, 2);
  d.write_column(0, 0, Float64Array.of(-1, 1, -1, 1, -1, 1, -1, 1));
  d.write_column(1, 0, Float64Array.of(-1, -1, 1, 1, -1, -1, 1, 1));
  d.write_response(0, Float64Array.of(0, 1, 0, 1, 1, 0, 0, 1));
  return d;
}

test("lambda above the maximum gives the balanced null model", () => {
  const d = sample();
  try {
    const p = d.fit({ family: "binomial", alpha: 1, lambdas: [1] });
    assert.deepEqual(p.indices, [[]]);
    assert.deepEqual(p.coefficients, [[]]);
    assert.deepEqual(p.intercept, [0]);
    // Eight balanced Bernoulli contributions, true deviance = 16 ln 2.
    assert.equal(p.deviance[0], 16 * Math.LN2);
    assert.equal(p.lambdas.length, 1);
  } finally { d.free(); }
});

test("invalid dimensions, incomplete data, invalid options and nonbinary y are clean errors", () => {
  assert.throws(() => new Dataset(0, 1));
  assert.throws(() => new Dataset(100000, 10000));
  const empty = new Dataset(8, 2);
  try { assert.throws(() => empty.fit({})); } finally { empty.free(); }
  const d = sample();
  try {
    assert.throws(() => d.write_column(2, 0, Float64Array.of(1)));
    assert.throws(() => d.write_column(0, 7, Float64Array.of(1, 2)));
    assert.throws(() => d.write_column(0, 0, Float64Array.of(NaN)));
    assert.throws(() => d.write_response(0, Float64Array.of(NaN)));
    d.write_response(0, Float64Array.of(2));
    assert.throws(() => d.fit({ family: "binomial" }));
    d.write_response(0, Float64Array.of(0));
    for (const options of [{ family: "invalid" }, { alpha: 0 }, { alpha: NaN }, { nlambda: 1 }, { nlambda: 201 }, { eps: -1 }, { lambdas: [1, 2] }, { penalty_factor: [1] }, { max_iter: 0 }]) {
      assert.throws(() => d.fit(options));
    }
  } finally { d.free(); }
});

test("Gaussian uses the analytic squared-loss elastic-net optimum and RSS", () => {
  const d = new Dataset(4, 2);
  try {
    d.write_column(0, 0, Float64Array.of(-1, -1, 1, 1));
    d.write_column(1, 0, Float64Array.of(-1, 1, -1, 1));
    d.write_response(0, Float64Array.of(1, 1, 5, 5));
    const p = d.fit({ family: "gaussian", alpha: .5, lambdas: [.5], eps: 1e-14 });
    assert.equal(p.family, "gaussian");
    assert.deepEqual(p.indices, [[0]]);
    assert(Math.abs(p.coefficients[0][0] - 1.4) <= 16 * Number.EPSILON);
    assert.equal(p.intercept[0], 3);
    assert(Math.abs(p.deviance[0] - 4 * .6 ** 2) <= 32 * Number.EPSILON);
    assert.deepEqual(d.fit({ family: "gaussian", alpha: .5, lambdas: [.5], eps: 1e-14 }), p);
  } finally { d.free(); }
});

test("one dataset can fit lasso and enet repeatedly without changing its input", () => {
  const d = sample();
  try {
    const lasso = d.fit({ alpha: 1, nlambda: 10 });
    d.fit({ alpha: .5, nlambda: 10 });
    assert.deepEqual(d.fit({ alpha: 1, nlambda: 10 }), lasso);
  } finally { d.free(); }
});

test("default elastic net satisfies independent one-feature logistic KKT", () => {
  const d = new Dataset(8, 1);
  try {
    d.write_column(0, 0, Float64Array.of(-1, -1, -1, -1, 1, 1, 1, 1));
    d.write_response(0, Float64Array.of(0, 0, 0, 1, 0, 1, 1, 1));
    const lam = 1 / 16, alpha = .5;
    const fit = d.fit({ alpha, lambdas: [lam], eps: 1e-14 });
    const a = fit.intercept[0], b = fit.coefficients[0][0];
    assert(b > 0);
    const p = (eta) => 1 / (1 + Math.exp(-eta));
    // x is centered with population variance 1. The two x groups have
    // positive shares 1/4 and 3/4; these are their exact mean gradients.
    const interceptGradient = (p(a + b) + p(a - b)) / 2 - .5;
    const betaGradient = (p(a + b) - p(a - b)) / 2 - .25 + lam * (alpha + (1 - alpha) * b);
    // The same stated KKT ceiling as the independent native checks. Tight
    // eps makes this a useful fixture; no universal eps-to-KKT claim is made.
    assert(Math.max(Math.abs(interceptGradient), Math.abs(betaGradient)) < 1e-6);
  } finally { d.free(); }
});

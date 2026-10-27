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
    assert.deepEqual(d.fit({ alpha: .5, lambdas: [.5], eps: 1e-14 }), p);
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

test("Gaussian rejects an overflowing automatic grid but accepts finite supplied penalties", () => {
  const d = new Dataset(4, 2);
  try {
    d.write_column(0, 0, Float64Array.of(-1, -1, 1, 1));
    d.write_column(1, 0, Float64Array.of(-1, 1, -1, 1));
    d.write_response(0, Float64Array.of(0, 0, 2, 2));
    for (const r_compatible of [false, true]) {
      const options = { family: "gaussian", alpha: Number.MIN_VALUE, eps: 1e-14, r_compatible };
      assert.throws(() => d.fit({ ...options, nlambda: 3 }));
      const p = d.fit({ ...options, lambdas: [1, .5] });
      assert.deepEqual(p.lambdas, [1, .5]);
      assert.deepEqual(p.intercept, [1, 1]);
      assert.deepEqual(p.indices, [[0], [0]]);
      assert(Math.abs(p.coefficients[0][0] - .5) <= 8 * Number.EPSILON);
      assert(Math.abs(p.coefficients[1][0] - 2 / 3) <= 8 * Number.EPSILON);
    }
  } finally { d.free(); }
});

test("default elastic net satisfies independent one-feature logistic KKT", () => {
  const d = new Dataset(8, 1);
  try {
    d.write_column(0, 0, Float64Array.of(-1, -1, -1, -1, 1, 1, 1, 1));
    d.write_response(0, Float64Array.of(0, 0, 0, 1, 0, 1, 1, 1));
    const lam = 1 / 16, alpha = .5;
    const fit = d.fit({ family: "binomial", alpha, lambdas: [lam], eps: 1e-14 });
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

function design(nrow, ncol, family) {
  let state = 5;
  const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 2 ** 32; };
  const x = Array.from({ length: ncol }, () => Float64Array.from({ length: nrow }, () => 4 * random() - 2));
  const y = Float64Array.from({ length: nrow }, (_, i) => {
    const signal = 1.5 * x[0][i] - x[1][i] + random() - .5;
    return family === "gaussian" ? 3 + signal : Number(signal > 0);
  });
  return { x, y };
}

function dataset({ x, y }, rows) {
  const d = new Dataset(rows.length, x.length);
  x.forEach((column, j) => d.write_column(j, 0, Float64Array.from(rows, (i) => column[i])));
  d.write_response(0, Float64Array.from(rows, (i) => y[i]));
  return d;
}

// A fold read through a mask (most rows), through its row indices (a few), and
// through its row indices because a left-out value is too large to mask safely
const selections = {
  mask: (i) => i % 4 !== 1,
  indices: (i) => i % 4 === 0,
  "indices, unsafe mask": (i) => i % 4 !== 1 && i !== 1,
};

for (const family of ["binomial", "gaussian"]) {
  for (const [how, inTrain] of Object.entries(selections)) {
    test(`a ${family} fold through ${how} fits its training rows alone and scores the held-out rows`, () => {
      const data = design(60, 6, family), all = [...Array(60).keys()];
      if (how === "indices, unsafe mask") data.x[2][1] = 1e200;
      const train = all.filter(inTrain), test = all.filter((i) => !inTrain(i) && data.x[2][i] !== 1e200);
      const whole = dataset(data, all), part = dataset(data, train);
      try {
        const lambdas = dataset(data, all.filter((i) => i !== 1)).fit({ family, alpha: .8, nlambda: 20 }).lambdas;
        const fold = whole.fit({ family, alpha: .8, lambdas, train_rows: train, test_rows: test });
        if (family === "gaussian") {
          assert.deepEqual(whole.fit({ alpha: .8, lambdas, train_rows: train, test_rows: test }), fold);
        }
        const alone = part.fit({ family, alpha: .8, lambdas });
        assert.deepEqual(fold.indices, alone.indices);
        assert.deepEqual(fold.iterations, alone.iterations);
        fold.coefficients.forEach((b, k) => b.forEach((v, i) => assert(Math.abs(v - alone.coefficients[k][i]) <= 1e-12 * (1 + Math.abs(v)))));
        // the held-out loss, from the returned coefficients
        assert.equal(fold.holdout.length, fold.lambdas.length);
        fold.holdout.forEach((loss, k) => {
          const losses = test.map((i) => {
            const eta = fold.intercept[k] + fold.indices[k].reduce((s, j, n) => s + data.x[j][i] * fold.coefficients[k][n], 0);
            return family === "gaussian" ? (data.y[i] - eta) ** 2 : 2 * (Math.log1p(Math.exp(eta)) - data.y[i] * eta);
          });
          const expected = losses.reduce((a, b) => a + b, 0) / test.length;
          assert(Math.abs(loss - expected) <= 1e-12 * expected, `${loss} vs ${expected}`);
        });
        assert.equal(alone.holdout, undefined);
      } finally { whole.free(); part.free(); }
    });
  }
}

test("a fold's automatic grid ends where its training rows alone would", () => {
  // 12 rows by 6 features is tall (lambda_min 0.001); its 6 training rows are
  // not (0.05)
  const data = design(12, 6, "gaussian"), all = [...Array(12).keys()], train = all.slice(0, 6);
  const whole = dataset(data, all), part = dataset(data, train);
  try {
    const fold = whole.fit({ family: "gaussian", nlambda: 5, train_rows: train }).lambdas;
    const alone = part.fit({ family: "gaussian", nlambda: 5 }).lambdas;
    assert(Math.abs(fold.at(-1) / fold[0] - .05) < 1e-12);
    fold.forEach((l, k) => assert(Math.abs(l - alone[k]) <= 1e-12 * l));
  } finally { whole.free(); part.free(); }
});

test("fold rows out of range, repeated, or empty are clean errors", () => {
  const d = sample();
  try {
    for (const rows of [[8], [0, 0, 1], []]) {
      assert.throws(() => d.fit({ lambdas: [.1], train_rows: rows }), /Fold rows/);
      assert.throws(() => d.fit({ lambdas: [.1], test_rows: rows }), /Fold rows/);
    }
  } finally { d.free(); }
});

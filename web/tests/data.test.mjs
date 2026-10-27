import test from "node:test";
import assert from "node:assert/strict";
import { numericValue, checkSize, responseValue, binaryClasses, syntheticCsv, assignFolds, chooseLambda } from "../src/data.mjs";

test("import preserves finite values and rejects lossy integers and missing data", () => {
  for (const [input, expected] of [["1.25", 1.25], ["-2e-3", -.002], ["0", 0], [3n, 3]]) {
    assert.equal(numericValue(input, "x", 0), expected);
  }
  for (const value of [null, undefined, "", "NaN", "Infinity", "2nd", true, [2], 9007199254740993n, "9007199254740993"]) {
    assert.throws(() => numericValue(value, "x", 4), /row 5/);
  }
  assert.equal(numericValue("9007199254740992", "x", 0), 2 ** 53);
});

test("memory guard uses decoded rows and columns", () => {
  assert.equal(checkSize(62, 2000), 992000);
  for (const shape of [[0, 10], [10, 0], [1, 3], [100000, 2000], [10, 30000], [NaN, 2]]) {
    assert.throws(() => checkSize(...shape));
  }
});

test("class mapping requires two explicit scalar labels", () => {
  assert.deepEqual(binaryClasses(new Set(["normal", "tumor"]), "tumor"), ["normal", "tumor"]);
  assert.throws(() => binaryClasses(new Set(["a"]), "a"));
  assert.throws(() => binaryClasses(new Set(["a", "b", "c"]), "a"));
  assert.throws(() => binaryClasses(new Set(["a", "b"]), "absent"));
  assert.throws(() => responseValue(null));
  assert.throws(() => responseValue([1]));
});

test("synthetic input is reproducible with the documented shape and both classes", () => {
  assert.equal(syntheticCsv(), syntheticCsv());
  const lines = syntheticCsv().split("\n").map((r) => r.split(","));
  assert.equal(lines.length, 161);
  assert(lines.every((r) => r.length === 49));
  assert.deepEqual([...new Set(lines.slice(1).map((r) => r[0]))].sort(), ["0", "1"]);
});

test("folds partition the rows, balance sizes, and stratify the classes", () => {
  const y = Array.from({ length: 23 }, (_, i) => Number(i % 3 === 0));
  const fold = assignFolds(y, 5, true);
  const sizes = [0, 0, 0, 0, 0], positives = [0, 0, 0, 0, 0];
  fold.forEach((f, i) => { sizes[f]++; positives[f] += y[i]; });
  assert(Math.max(...sizes) - Math.min(...sizes) <= 1);
  assert(Math.max(...positives) - Math.min(...positives) <= 1);
  assert.deepEqual(assignFolds(y, 5, true), fold);
  assert.notDeepEqual(assignFolds(y, 5, true, 2), fold);
  assert.throws(() => assignFolds([0, 0, 0, 1], 2, true), /Each class/);
  assert.throws(() => assignFolds([1, 2, 3], 4, false), /at least 4 rows/);
  assert.throws(() => assignFolds([1, 2, 3], 1, false));
});

test("lambda choice: the minimum, the one-standard-error rule, and incomplete folds", () => {
  // fold losses at four lambdas; the third fold's path stopped after three
  const holdout = [[4, 2, 1.0, 0.5], [4, 2.2, 1.4], [4, 1.8, 1.0]];
  const { mean, se, best, oneSe } = chooseLambda(holdout, 4);
  assert.equal(mean.length, 3);
  assert.equal(best, 2);
  // losses 1, 1.4, 1 at the best: mean 1.4/1.2, sd 0.2309.., se 0.1333..
  assert(Math.abs(se[2] - 0.4 / Math.sqrt(3) / Math.sqrt(3)) < 1e-15);
  assert.equal(oneSe, 2);
  assert.equal(chooseLambda([[3, 1.05, 1], [3, 1.05, 1]], 3).oneSe, 2);
  assert.equal(chooseLambda([[3, 1.1, 1], [3, 0.9, 1.2]], 3).oneSe, 1);
  assert.throws(() => chooseLambda([[NaN], [1]], 1), /every fold/);
});

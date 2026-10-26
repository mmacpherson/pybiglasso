import test from "node:test";
import assert from "node:assert/strict";
import { numericValue, checkSize, responseValue, binaryClasses, syntheticCsv } from "../src/data.mjs";

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

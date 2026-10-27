// Shared by the import worker and its parser tests. Keep text intact until a
// selected column is converted, so large CSV integers cannot silently round.
export const MAX_VALUES = 16 * 1024 * 1024;

export function numericValue(value, column, row) {
  const where = `column “${column}”, row ${row + 1}`;
  if (value === null || value === undefined || String(value).trim() === "") {
    throw new Error(`Missing value in ${where}. Fill or remove missing values before importing.`);
  }
  if (!["string", "number", "bigint"].includes(typeof value)) throw new Error(`Nonnumeric value in ${where}.`);
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`Nonnumeric or nonfinite value in ${where}.`);
  const text = String(value).trim();
  if (typeof value === "bigint" || /^[+-]?\d+$/.test(text)) {
    if (BigInt(number) !== BigInt(text)) {
      throw new Error(`Integer in ${where} cannot be represented exactly as float64.`);
    }
  }
  return number;
}

export function checkSize(rows, columns) {
  if (!Number.isSafeInteger(rows) || rows < 2 || rows > 1_000_000 || !Number.isSafeInteger(columns)
      || columns < 1 || columns > 20_000 || rows * columns > MAX_VALUES) {
    throw new Error("Select at least two rows and one feature, with no more than 128 MiB of decoded float64 features. Compressed file size does not measure this.");
  }
  return rows * columns * 8;
}

export function responseValue(value) {
  if (value === null || value === undefined || String(value).trim() === "") {
    throw new Error("The response has missing values. Fill or remove them before importing.");
  }
  if (!["string", "number", "bigint", "boolean"].includes(typeof value)) {
    throw new Error("The response must contain scalar class labels, not nested values.");
  }
  return String(value).trim();
}

export function binaryClasses(labels, positive) {
  const classes = [...labels].sort();
  if (classes.length !== 2 || !labels.has(positive)) {
    throw new Error("Choose a response with exactly two classes and a positive class present in that column. This demo fits logistic regression; continuous outcomes require the future Gaussian solver.");
  }
  return [classes.find((c) => c !== positive), positive];
}

export function syntheticCsv(family = "binomial") {
  let state = 17;
  const random = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return state >>> 0;
  };
  const lines = ["response," + Array.from({ length: 48 }, (_, j) => `group_${Math.floor(j / 8) + 1}_${j % 8 + 1}`).join(",")];
  for (let i = 0; i < 160; i++) {
    const groups = Array.from({ length: 6 }, () => random() % 3);
    const noise = random() % 5;
    const y = family === "gaussian" ? 2.5 + 2 * groups[0] - groups[1] + noise / 5 : Number(groups[0] + groups[1] + noise > 4);
    const x = Array.from({ length: 48 }, (_, j) => groups[Math.floor(j / 8)] + (random() % 4 === 0 ? (random() % 3) - 1 : 0));
    lines.push([y, ...x].join(","));
  }
  return lines.join("\n");
}

// Each row's fold, from a seeded shuffle; with `stratify` (0/1 responses),
// each class is dealt across the folds in turn, as StratifiedKFold balances
// them, so every training fold sees both classes.
export function assignFolds(y, folds, stratify, seed = 1) {
  if (!Number.isInteger(folds) || folds < 2) throw new Error("Use at least two folds.");
  const groups = stratify ? [0, 1].map((c) => [...y.keys()].filter((i) => y[i] === c)) : [[...y.keys()]];
  if (groups.some((rows) => rows.length < folds)) {
    throw new Error(stratify
      ? `Each class needs at least ${folds} rows for ${folds}-fold cross-validation; use fewer folds.`
      : `Cross-validation needs at least ${folds} rows; use fewer folds.`);
  }
  let state = seed;
  const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 2 ** 32; };
  const fold = new Int32Array(y.length);
  let next = 0;
  for (const rows of groups) {
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [rows[i], rows[j]] = [rows[j], rows[i]];
    }
    for (const row of rows) { fold[row] = next; next = (next + 1) % folds; }
  }
  return fold;
}

// The held-out loss's mean and standard error at each lambda every fold
// reached (a path can stop early), and the lambdas chosen by the minimum and
// by the one-standard-error rule, as BiglassoClassifierCV chooses them.
export function chooseLambda(holdout, nlambda) {
  const folds = holdout.length;
  const mean = [], se = [];
  for (let k = 0; k < nlambda; k++) {
    const losses = holdout.map((h) => h[k]);
    if (losses.some((v) => v === undefined || !Number.isFinite(v))) break;
    const m = losses.reduce((a, b) => a + b, 0) / folds;
    const variance = losses.reduce((a, b) => a + (b - m) ** 2, 0) / (folds - 1);
    mean.push(m); se.push(Math.sqrt(variance / folds));
  }
  if (!mean.length) throw new Error("No lambda was fit in every fold; try a longer path or a larger α.");
  let best = 0;
  mean.forEach((m, k) => { if (m < mean[best]) best = k; });
  // lambdas decrease, so the first within one standard error is the largest
  const oneSe = mean.findIndex((m) => m <= mean[best] + se[best]);
  return { mean, se, best, oneSe };
}

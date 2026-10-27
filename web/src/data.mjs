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

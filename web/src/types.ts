export interface Inspection {
  name: string;
  columns: string[];
  preview: string[][];
  numeric: boolean[];
  rows?: number;
}

export interface ModelPath {
  family: "binomial" | "gaussian";
  lambdas: number[];
  intercept: number[];
  indices: number[][];
  coefficients: number[][];
  standardized_coefficients: number[][];
  center: number[];
  scale: number[];
  iterations: number[];
  n_rejected: number[];
  deviance: number[];
  warnings: string[];
}

export interface Loaded {
  family: "binomial" | "gaussian";
  rows: number;
  features: string[];
  classes: [string, string];
  positives: number;
  bytes: number;
}

export type Request =
  | { kind: "inspect"; file: File }
  | { kind: "load"; target: string; features: string[]; positive: string; family: "binomial" | "gaussian" }
  | { kind: "fit"; alpha: number; nlambda: number; eps: number };

export type Reply =
  | { kind: "inspected"; info: Inspection }
  | { kind: "loaded"; info: Loaded }
  | { kind: "progress"; message: string; done?: number; total?: number }
  | { kind: "result"; path: ModelPath; milliseconds: number }
  | { kind: "error"; message: string };

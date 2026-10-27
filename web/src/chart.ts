import type { CrossValidation, ModelPath } from "./types";

const colors = ["#346b52", "#ba783b", "#547ba6", "#a96976", "#7b71a5", "#548c87", "#947d42", "#849755"];
export const featureColor = (index: number) => colors[index % colors.length];

// A canvas sized for the screen, and the x position of each lambda (log
// scale), shared by the path and cross-validation charts so they line up.
function frame(canvas: HTMLCanvasElement, path: ModelPath) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return undefined;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  ctx.scale(ratio, ratio);
  const left = 47, right = width - 14, top = 14, bottom = height - 28;
  const first = path.lambdas[0], last = path.lambdas.at(-1)!;
  const x = (k: number) => {
    const fraction = last > 0 && first > last
      ? (Math.log(first) - Math.log(path.lambdas[k])) / (Math.log(first) - Math.log(last))
      : k / Math.max(1, path.lambdas.length - 1);
    return left + fraction * (right - left);
  };
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = "#657165"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const k = Math.round(i * (path.lambdas.length - 1) / 4);
    ctx.fillText(path.lambdas[k].toExponential(1), x(k), bottom + 10);
  }
  return { ctx, left, right, top, bottom, x };
}

// λ_min solid and λ_1se dashed, in the accent color; then the selected λ
function marks(ctx: CanvasRenderingContext2D, x: (k: number) => number, top: number, bottom: number, selected: number, cv?: CrossValidation) {
  const line = (k: number, color: string, dash: number[]) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(x(k), top); ctx.lineTo(x(k), bottom); ctx.stroke();
  };
  ctx.globalAlpha = 1;
  if (cv) { line(cv.best, "#ba783b", []); line(cv.oneSe, "#ba783b", [6, 3]); }
  line(selected, "#283f30", [3, 4]); ctx.setLineDash([]);
}

export function drawPath(canvas: HTMLCanvasElement, path: ModelPath, selected: number, cv?: CrossValidation) {
  const f = frame(canvas, path);
  if (!f) return;
  const { ctx, left, right, top, bottom, x } = f;
  const strength = new Map<number, number>();
  for (let k = 0; k < path.lambdas.length; k++) {
    path.indices[k].forEach((j, i) => strength.set(j, Math.max(strength.get(j) ?? 0, Math.abs(path.standardized_coefficients[k][i]))));
  }
  const shown = [...strength].sort((a, b) => b[1] - a[1]).slice(0, 24).map(([j]) => j);
  const curves = new Map(shown.map((j) => [j, new Float64Array(path.lambdas.length)]));
  for (let k = 0; k < path.lambdas.length; k++) {
    path.indices[k].forEach((j, i) => { const c = curves.get(j); if (c) c[k] = path.standardized_coefficients[k][i]; });
  }
  const maxY = Math.max(0.05, ...shown.map((j) => strength.get(j)!)) * 1.1;
  const y = (v: number) => top + (maxY - v) / (2 * maxY) * (bottom - top);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = -2; i <= 2; i++) {
    const v = i * maxY / 2, py = y(v);
    ctx.strokeStyle = i === 0 ? "#c0cbbd" : "#e8ece3"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
    ctx.fillStyle = "#657165"; ctx.fillText(v.toPrecision(2), left - 9, py);
  }
  for (const [j, values] of curves) {
    ctx.strokeStyle = featureColor(j); ctx.globalAlpha = 0.75; ctx.lineWidth = 1.6;
    ctx.beginPath(); values.forEach((v, k) => k === 0 ? ctx.moveTo(x(k), y(v)) : ctx.lineTo(x(k), y(v))); ctx.stroke();
  }
  marks(ctx, x, top, bottom, selected, cv);
  canvas.setAttribute("aria-label", `Standardized coefficient paths for ${shown.length} features across ${path.lambdas.length} lambdas. Selected lambda ${path.lambdas[selected].toPrecision(4)}.`);
}

export function drawCv(canvas: HTMLCanvasElement, path: ModelPath, cv: CrossValidation, selected: number) {
  const f = frame(canvas, path);
  if (!f) return;
  const { ctx, left, right, top, bottom, x } = f;
  const low = Math.min(...cv.mean.map((m, k) => m - cv.se[k])), high = Math.max(...cv.mean.map((m, k) => m + cv.se[k]));
  const pad = (high - low) * .08 || Math.abs(high) * .05 || 1;
  const y = (v: number) => top + (high + pad - v) / (high - low + 2 * pad) * (bottom - top);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 3; i++) {
    const v = low - pad + i * (high - low + 2 * pad) / 3, py = y(v);
    ctx.strokeStyle = "#e8ece3"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
    ctx.fillStyle = "#657165"; ctx.fillText(v.toPrecision(3), left - 9, py);
  }
  ctx.strokeStyle = "#9fb0a0"; ctx.lineWidth = 1;
  cv.mean.forEach((m, k) => { ctx.beginPath(); ctx.moveTo(x(k), y(m - cv.se[k])); ctx.lineTo(x(k), y(m + cv.se[k])); ctx.stroke(); });
  ctx.fillStyle = "#346b52";
  cv.mean.forEach((m, k) => { ctx.beginPath(); ctx.arc(x(k), y(m), 2.4, 0, 2 * Math.PI); ctx.fill(); });
  marks(ctx, x, top, bottom, Math.min(selected, cv.mean.length - 1), cv);
  canvas.setAttribute("aria-label", `Held-out loss, mean and standard error over ${cv.folds} folds, across ${cv.mean.length} lambdas. Lowest at lambda ${path.lambdas[cv.best].toPrecision(4)}; one-standard-error choice ${path.lambdas[cv.oneSe].toPrecision(4)}.`);
}

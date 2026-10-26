import type { ModelPath } from "./types";

const colors = ["#346b52", "#ba783b", "#547ba6", "#a96976", "#7b71a5", "#548c87", "#947d42", "#849755"];
export const featureColor = (index: number) => colors[index % colors.length];

export function drawPath(canvas: HTMLCanvasElement, path: ModelPath, selected: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  ctx.scale(ratio, ratio);
  const left = 47, right = width - 14, top = 14, bottom = height - 28;
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
  const first = path.lambdas[0], last = path.lambdas.at(-1)!;
  const x = (k: number) => {
    const fraction = last > 0 && first > last
      ? (Math.log(first) - Math.log(path.lambdas[k])) / (Math.log(first) - Math.log(last))
      : k / Math.max(1, path.lambdas.length - 1);
    return left + fraction * (right - left);
  };
  const y = (v: number) => top + (maxY - v) / (2 * maxY) * (bottom - top);
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = -2; i <= 2; i++) {
    const v = i * maxY / 2, py = y(v);
    ctx.strokeStyle = i === 0 ? "#c0cbbd" : "#e8ece3"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
    ctx.fillStyle = "#657165"; ctx.fillText(v.toPrecision(2), left - 9, py);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const k = Math.round(i * (path.lambdas.length - 1) / 4);
    ctx.fillText(path.lambdas[k].toExponential(1), x(k), bottom + 10);
  }
  for (const [j, values] of curves) {
    ctx.strokeStyle = featureColor(j); ctx.globalAlpha = 0.75; ctx.lineWidth = 1.6;
    ctx.beginPath(); values.forEach((v, k) => k === 0 ? ctx.moveTo(x(k), y(v)) : ctx.lineTo(x(k), y(v))); ctx.stroke();
  }
  ctx.globalAlpha = 1; ctx.strokeStyle = "#283f30"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
  ctx.beginPath(); ctx.moveTo(x(selected), top); ctx.lineTo(x(selected), bottom); ctx.stroke(); ctx.setLineDash([]);
  canvas.setAttribute("aria-label", `Standardized coefficient paths for ${shown.length} features across ${path.lambdas.length} lambdas. Selected lambda ${path.lambdas[selected].toPrecision(4)}.`);
}

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const result = spawnSync("wasm-pack", [
  "build", "../rust", "--target", "web", "--release",
  "--out-dir", "../web/pkg", "--out-name", "solver",
  "--", "--no-default-features", "--features", "wasm",
], { cwd: root, stdio: "inherit" });
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;

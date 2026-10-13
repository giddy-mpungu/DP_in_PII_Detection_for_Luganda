// Checks demo/site/rdp.js against the epsilon Opacus reported for every DP run in results/.
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { epsilon } from "./site/rdp.js";
let worst = 0;
for (const d of readdirSync("results")) {
  const p = `results/${d}/metrics.json`;
  if (!existsSync(p)) continue;
  const pr = JSON.parse(readFileSync(p, "utf8")).privacy;
  if (!pr || pr.epsilon_rdp == null) continue;
  const { epsilon: e } = epsilon(pr.sample_rate, pr.noise_multiplier, pr.optimizer_steps, pr.delta);
  worst = Math.max(worst, Math.abs(e - pr.epsilon_rdp));
  console.log(d.padEnd(28), "opacus", pr.epsilon_rdp.toFixed(6), "js", e.toFixed(6));
}
console.log("max abs difference", worst.toExponential(2));

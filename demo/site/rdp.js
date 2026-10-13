// Renyi-DP accountant for the Poisson-subsampled Gaussian mechanism (DP-SGD), ported line by line from
// Opacus 1.6 (opacus/accountants/analysis/rdp.py; Mironov, Talwar & Zhang 2019, arXiv:1908.10530).
// Same orders as Opacus' RDPAccountant, so the epsilon matches what training reported.

const ALPHAS = [...Array(99).keys()].map((x) => 1 + (x + 1) / 10).concat([...Array(52).keys()].map((x) => x + 12));

function logAdd(x, y) {
  const a = Math.min(x, y), b = Math.max(x, y);
  if (a === -Infinity) return b;
  return Math.log1p(Math.exp(a - b)) + b;
}

function logSub(x, y) {
  if (x < y) throw new Error("negative result");
  if (y === -Infinity) return x;
  if (x === y) return -Infinity;
  const d = Math.expm1(x - y);
  return Number.isFinite(d) ? Math.log(d) + y : x;
}

// log(erfc(z)) for any z, accurate in the far tail (Numerical Recipes erfcc, relative error < 1.2e-7)
function logErfc(z) {
  const az = Math.abs(z), t = 1 / (1 + 0.5 * az);
  const p = -az * az - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
  const logTail = Math.log(t) + p; // log erfc(|z|)
  return z >= 0 ? logTail : Math.log(2 - Math.exp(logTail));
}

function logBinomInt(n, k) { // log C(n, k) for integers
  let s = 0;
  for (let j = 1; j <= k; j++) s += Math.log(n - k + j) - Math.log(j);
  return s;
}

function logAInt(q, sigma, alpha) {
  let la = -Infinity;
  for (let i = 0; i <= alpha; i++) {
    const c = logBinomInt(alpha, i) + i * Math.log(q) + (alpha - i) * Math.log(1 - q);
    la = logAdd(la, c + (i * i - i) / (2 * sigma * sigma));
  }
  return la;
}

function logAFrac(q, sigma, alpha) {
  let a0 = -Infinity, a1 = -Infinity, i = 0;
  let logAbsCoef = 0, sign = 1; // C(alpha, 0) = 1, updated multiplicatively
  const z0 = sigma * sigma * Math.log(1 / q - 1) + 0.5;
  for (;;) {
    const j = alpha - i;
    const t0 = logAbsCoef + i * Math.log(q) + j * Math.log(1 - q);
    const t1 = logAbsCoef + j * Math.log(q) + i * Math.log(1 - q);
    const e0 = Math.log(0.5) + logErfc((i - z0) / (Math.SQRT2 * sigma));
    const e1 = Math.log(0.5) + logErfc((z0 - j) / (Math.SQRT2 * sigma));
    const s0 = t0 + (i * i - i) / (2 * sigma * sigma) + e0;
    const s1 = t1 + (j * j - j) / (2 * sigma * sigma) + e1;
    if (sign > 0) { a0 = logAdd(a0, s0); a1 = logAdd(a1, s1); } else { a0 = logSub(a0, s0); a1 = logSub(a1, s1); }
    const f = (alpha - i) / (i + 1); // C(alpha, i+1) = C(alpha, i) * (alpha - i) / (i + 1)
    logAbsCoef += Math.log(Math.abs(f));
    if (f < 0) sign = -sign;
    i += 1;
    if (Math.max(s0, s1) < -30) break;
  }
  return logAdd(a0, a1);
}

function rdpOrder(q, sigma, alpha) {
  if (q === 0) return 0;
  if (sigma === 0) return Infinity;
  if (q === 1) return alpha / (2 * sigma * sigma);
  const la = Number.isInteger(alpha) ? logAInt(q, sigma, alpha) : logAFrac(q, sigma, alpha);
  return la / (alpha - 1);
}

// epsilon of `steps` DP-SGD steps with sampling rate q and noise multiplier sigma, at the given delta
export function epsilon(q, sigma, steps, delta) {
  let best = Infinity, bestAlpha = null;
  for (const a of ALPHAS) {
    const e = steps * rdpOrder(q, sigma, a) - (Math.log(delta) + Math.log(a)) / (a - 1) + Math.log((a - 1) / a);
    if (e < best) { best = e; bestAlpha = a; }
  }
  return { epsilon: best, order: bestAlpha };
}

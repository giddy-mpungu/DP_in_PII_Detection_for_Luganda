"""Empirical lower bound on epsilon from one training run (Steinke, Nasr & Jagielski, 2023).

Each of m canaries was included in training independently with probability 1/2 (dp_pii_luganda.make_canaries).
The auditor ranks canaries by the trained model's loss on them, guesses "included" for the k lowest-loss and
"excluded" for the k highest-loss canaries, and abstains on the rest. If training is eps-DP, the number of correct
guesses among the r = 2k made is stochastically dominated by Binomial(r, e^eps / (1 + e^eps)) (their Theorem 3.1).
The lower bound is the largest eps that this observation rejects at confidence 1 - beta.

We use the pure eps-DP form of the bound. The (eps, delta) form subtracts a term of order delta * m (1e-5 * 1000 here),
which would make the reported lower bounds very slightly smaller; it cannot make them larger. The guess sizes k are
fixed in advance and the confidence is split across them (Bonferroni), so taking the best k is valid.

usage: python canary_audit.py <results dir>      -> <results>/canary_audit.json
"""
import json
import math
import os
import sys

from scipy.stats import binom

KS = (10, 25, 50, 100, 250)
BETA = 0.05


def eps_lower_bound(r, v, beta):
    """Largest eps with P[Binomial(r, p_eps) >= v] <= beta, p_eps = e^eps / (1 + e^eps); 0 if none."""
    tail = lambda e: binom.sf(v - 1, r, math.exp(e) / (1 + math.exp(e)))   # P[X >= v]
    if tail(0.0) > beta:
        return 0.0
    lo, hi = 0.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if tail(mid) <= beta:
            lo = mid
        else:
            hi = mid
    return lo


def audit(path):
    c = json.load(open(path))
    inc, loss = c["included"], c["loss"]
    order = sorted(range(len(loss)), key=lambda i: loss[i])           # lowest loss first = most likely included
    rows = []
    for k in KS:
        guess_in, guess_out = order[:k], order[-k:]
        v = sum(inc[i] for i in guess_in) + sum(not inc[i] for i in guess_out)
        rows.append({"k": k, "guesses": 2 * k, "correct": v, "accuracy": v / (2 * k),
                     "eps_lower_bound": eps_lower_bound(2 * k, v, BETA / len(KS))})
    best = max(rows, key=lambda r: r["eps_lower_bound"])
    return {"canaries": len(inc), "included": sum(inc), "beta": BETA, "by_k": rows,
            "eps_lower_bound": best["eps_lower_bound"], "best_k": best["k"]}


def main(res):
    out = {}
    for run in sorted(os.listdir(res)):
        p = os.path.join(res, run, "canaries.json")
        if os.path.exists(p):
            out[run] = audit(p)
            m = json.load(open(os.path.join(res, run, "metrics.json")))
            out[run]["epsilon_upper_bound"] = m["privacy"].get("epsilon_prv")
            print(f"{run}: empirical eps >= {out[run]['eps_lower_bound']:.2f} (95% confidence; best k={out[run]['best_k']}, "
                  f"accuracy {max(r['accuracy'] for r in out[run]['by_k']):.2f}); "
                  f"theoretical eps <= {out[run]['epsilon_upper_bound']}")
    with open(os.path.join(res, "canary_audit.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

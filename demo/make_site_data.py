"""Collects the numbers the demo site shows into demo/site/data.json (no corpus text, only aggregates).

usage: python demo/make_site_data.py [results dir]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ["PERSON", "NORP", "DATE", "USERID", "ORG", "LOCATION"]


def load(res, name):
    p = os.path.join(res, name, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def seeds(res, prefix):
    return [m for m in (load(res, f"{prefix}_seed{s}") for s in (42, 1234, 2024)) if m]


def hist(losses, edges):
    h, _ = np.histogram(np.clip(losses, edges[0], edges[-1]), bins=edges)
    return (h / max(len(losses), 1)).round(4).tolist()


def main(res):
    hy = json.load(open(os.path.join(res, "hybrid", "hybrid.json")))
    onnx_scores = json.load(open(os.path.join(HERE, "onnx_test_scores.json")))
    out = {"models": {}, "reference": {}, "mia": {}}
    for eps in ("8", "3", "1", "0.5"):
        m = load(res, f"big_dp_eps{eps}_seed42")       # Experiment E, seed 42: the released v2 model
        g = seeds(res, f"big_dp_eps{eps}")
        pr = m["privacy"]
        h = hy["runs"][f"big_dp_eps{eps}_seed42"]
        out["models"][eps] = {
            "repo": f"e4gl33y3/luganda-pii-dp-v2-eps{eps}",
            "privacy": {"noise_multiplier": pr["noise_multiplier"], "max_grad_norm": pr["max_grad_norm"],
                        "sample_rate": pr["sample_rate"], "steps": pr["optimizer_steps"], "delta": pr["delta"],
                        "epsilon_rdp": pr["epsilon_rdp"], "epsilon_prv": pr.get("epsilon_prv"),
                        "accountant": pr.get("accountant"),
                        "expected_batch": m["config"]["batch_size"], "epochs": m["config"]["epochs"],
                        "lr": m["config"]["lr"], "train_sentences": 2085},
            "test": {"f1": m["test"]["entity_micro"]["f1"], "recall": m["test"]["entity_micro"]["recall"],
                     "precision": m["test"]["entity_micro"]["precision"],
                     "f1_seeds_mean": float(np.mean([x["test"]["entity_micro"]["f1"] for x in g])),
                     "f1_seeds_sd": float(np.std([x["test"]["entity_micro"]["f1"] for x in g], ddof=1)) if len(g) > 1 else 0.0,
                     "per_type_recall": {t: m["test"]["per_type"][t]["recall"] for t in TYPES},
                     "with_rules_f1": h["model_plus_rules"]["entity_micro"]["f1"],
                     "with_rules_recall": h["model_plus_rules"]["entity_micro"]["recall"],
                     "with_rules_per_type_recall": {t: h["model_plus_rules"]["per_type"][t]["recall"] for t in TYPES},
                     "browser_model_f1": onnx_scores[eps]["f1"]},
            "mia_auc": m["mia_auc_train_vs_test"],
        }
    base = seeds(res, "pub_nonprivate")
    out["reference"]["nonprivate"] = {
        "f1": float(np.mean([x["test"]["entity_micro"]["f1"] for x in base])),
        "recall": float(np.mean([x["test"]["entity_micro"]["recall"] for x in base])),
        "mia_auc": float(np.mean([x["mia_auc_train_vs_test"] for x in base])),
        "per_type_recall": {t: float(np.mean([x["test"]["per_type"][t]["recall"] for x in base])) for t in TYPES}}
    # loss distributions of training (member) vs test (non-member) sentences, for the membership-inference chart
    edges = np.linspace(0, 3, 31)
    for key, run in (("nonprivate", "pub_nonprivate_seed42"), ("dp8", "big_dp_eps8_seed42"), ("dp1", "big_dp_eps0.5_seed42")):
        L = json.load(open(os.path.join(res, run, "losses.json")))
        out["mia"][key] = {"edges": edges.round(2).tolist(), "members": hist(L["train"], edges),
                           "nonmembers": hist(L["test"], edges), "auc": load(res, run)["mia_auc_train_vs_test"]}
    with open(os.path.join(HERE, "site", "data.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote demo/site/data.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results"))

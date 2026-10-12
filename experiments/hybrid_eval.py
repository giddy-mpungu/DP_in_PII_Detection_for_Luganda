"""
Hybrid PII detection: a DP-trained model plus rule-based recognisers for USERID and DATE.

Under DP the model detects almost no USERID or DATE spans, yet both categories are highly regular
(phone numbers, ID / plate numbers, years, weekday and month names). Rules use no training data, so
adding them costs no privacy budget. The rules were written by looking only at the train and
validation splits; the test split is used once, for the numbers reported.

Usage
    # 1. test-set predictions for a run that has no predictions.json (e.g. the released Hub models)
    python hybrid_eval.py predict --model e4gl33y3/luganda-pii-dp-eps8 --run results/pub_dp_eps8_seed42
    # 2. rules alone on every split, and DP-only vs DP+rules for every run with predictions.json
    python hybrid_eval.py score results            # writes results/hybrid/hybrid.json

Note on span convention: lg-ner is stored as word pieces, and in multi-piece numbers the final piece is
labelled O (e.g. `202/U-DATE ##2/O`, `0/B-USERID ... ##3/L-USERID ##8/O`). Exact-match scoring therefore
expects that convention, so the rules reproduce it; an overlap (boundary-lenient) score is reported too.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_pii_luganda import ENTITY_TYPES, ID2LABEL, evaluate_tags, load_splits, prf, spans  # noqa: E402

RULE_TYPES = ("USERID", "DATE")

# --- lexicons (Luganda + English) ---------------------------------------------------------------
WEEKDAY = re.compile(r"^(o?lw(a|o)?(kubiri|kusatu|kuna|kutaano|mukaaga)|ssande|sande|mmande|bbalaza|"
                     r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)$")
ORDINAL = re.compile(r"^(okubiri|okusatu|okuna|okutaano|omukaaga|omusanvu|omunaana|omwenda|ekkumi|olubereberye)$")
MONTH = re.compile(r"^(january|february|march|april|may|june|july|august|september|october|november|december|"
                   r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|decembera|febr?uary|"
                   r"januari|febwali|maaki|apuli|maayi|juuni|j?julaayi|agusito|sebutemba|okitobba|novemba|"
                   r"desemba|museenene)$")
YEAR = re.compile(r"^(19[5-9]\d|20[0-3]\d)$")
DAY = re.compile(r"^(0?[1-9]|[12]\d|3[01])$")
CURRENCY = {"ssiringi", "shs", "ugx", "ssente", "obukadde", "zokka", "doola"}
MODIFIER = {"eno", "ejja", "ewedde", "eyise"}
WEEK = {"wiiki", "ssabbiiti", "sabbiiti", "wiikendi"}

# A date is a run of word classes (one letter per word, assigned in word_class) matching DATE_PATTERN.
#   W weekday  M month  Y year  D day number  X ordinal after gw ' / lw '  T th/st/nd/rd  P / . -  C ,
#   K mwezi/omwezi  G gwa/ogwa  Q gw/lw  A '  E eno/ejja/ewedde  R oluwedde  L lwa  V wiiki  N nga  U mu  ( )
_WD = r"(?:W|QAX)"
_MON = r"(?:KG?M|K?QAX|M(?:\(M)?)"
_DM = r"(?:DT?(?:P(?:DT?|M))?C?U?(?:KG)?" + _MON + r")"
_NUM = r"(?:DP(?:D|M)P(?:Y|D)|DPM)"
_CORE = r"(?:" + _NUM + "|" + _DM + "|" + _MON + "|" + _WD + r"|Y(?:PY)?)"
_TAIL = r"(?:LVE|R|C?Y|N(?:" + _NUM + "|" + _DM + r"|MD|DT?)|U" + _MON + r")*"
DATE_PATTERN = re.compile(_CORE + _TAIL)


def words(tokens):
    """Group word pieces into words: [(text, first_piece, last_piece)]."""
    out = []
    for i, t in enumerate(tokens):
        if t.startswith("##") and out:
            text, s, _ = out[-1]
            out[-1] = (text + t[2:], s, i)
        else:
            out.append((t, i, i))
    return out


def word_class(ws, i):
    w = ws[i][0].lower()
    prev = ws[i - 1][0].lower() if i else ""
    nxt = ws[i + 1][0].lower() if i + 1 < len(ws) else ""
    if WEEKDAY.match(w):
        return "W"
    if MONTH.match(w):
        return "M"
    if YEAR.match(w) and prev not in CURRENCY and nxt not in CURRENCY:
        return "Y"
    if DAY.match(w):
        return "D"
    if ORDINAL.match(w) and i >= 2 and ws[i - 1][0] == "'" and ws[i - 2][0].lower() in ("gw", "lw"):
        return "X"
    return {"th": "T", "st": "T", "nd": "T", "rd": "T", "/": "P", ".": "P", "-": "P", ",": "C",
            "mwezi": "K", "omwezi": "K", "omwenzi": "K", "gwa": "G", "ogwa": "G", "gw": "Q", "lw": "Q", "'": "A",
            "lwa": "L", "nga": "N", "mu": "U", "oluwedde": "R", "(": "(", ")": ")"}.get(
        w, "E" if w in MODIFIER else "V" if w in WEEK else "_")


def date_spans(ws):
    """Word ranges (a, b) of dates: DATE_PATTERN over the string of word classes."""
    cls = "".join(word_class(ws, i) for i in range(len(ws)))
    return [(m.start(), m.end() - 1) for m in DATE_PATTERN.finditer(cls) if any(c in "WMYX" for c in m.group())]


def userid_spans(ws):
    low = [w[0].lower() for w in ws]
    out, i = [], 0
    while i < len(ws):
        # national ID  <2 letters><8 digits><4 letters>  (one string, split by the tokenizer)
        if (i + 2 < len(ws) and re.fullmatch(r"[a-z]{2}", low[i]) and re.fullmatch(r"\d{8}", low[i + 1])
                and re.fullmatch(r"[a-z]{4}", low[i + 2])):
            out.append((i, i + 2))
            i += 3
        # number plate  <3 letters><3 digits>[letter]
        elif i + 1 < len(ws) and re.fullmatch(r"u[a-z]{2}", low[i]) and re.fullmatch(r"\d{3}", low[i + 1]):
            j = i + 2 if i + 2 < len(ws) and re.fullmatch(r"[a-z]", low[i + 2]) else i + 1
            out.append((i, j))
            i = j + 1
        # phone / ID / plate digits (6+ digits)
        elif re.fullmatch(r"\+?\d{6,13}", low[i]):
            out.append((i, i))
            i += 1
        else:
            i += 1
    return out


def rule_spans(tokens):
    """Rule-detected spans as piece indices. lg-ner's gold spans end one character early (an offset slip
    in the source annotations), so a final word piece of one character is labelled O: `202 ##2` is tagged
    `202`, `es ##d ##m` is tagged `es ##d`, but `in ##sa` is tagged whole. The spans follow that convention."""
    ws = words(tokens)
    found = set()
    for typ, ranges in (("USERID", userid_spans(ws)), ("DATE", date_spans(ws))):
        for a, b in ranges:
            s, e = ws[a][1], ws[b][2]
            if e > s and len(tokens[e].lstrip("#")) == 1:
                e -= 1
            found.add((typ, s, e))
    return found


def to_tags(n, span_set):
    tags = ["O"] * n
    for t, s, e in sorted(span_set, key=lambda x: x[1]):
        if s == e:
            tags[s] = f"U-{t}"
        else:
            tags[s] = f"B-{t}"
            for k in range(s + 1, e):
                tags[k] = f"I-{t}"
            tags[e] = f"L-{t}"
    return tags


def combine(model_tags, rule_set, strategy):
    """DP+rules for the rule types. 'add': a rule span replaces any model span it overlaps, other model spans
    are kept. 'replace': as 'add', and the model's own USERID/DATE spans are dropped (rules own those types).
    The strategy is chosen on the validation split (cmd_score)."""
    keep = {x for x in spans(model_tags) if not any(not (x[2] < r[1] or x[1] > r[2]) for r in rule_set)}
    if strategy == "replace":
        keep = {x for x in keep if x[0] not in RULE_TYPES}
    return to_tags(len(model_tags), keep | rule_set)


STRATEGIES = ("add", "replace")


def lenient(gold, pred, types=ENTITY_TYPES):
    """Overlap match: a gold span counts as found if a predicted span of the same type overlaps it."""
    out = {}
    for t in list(types) + ["all"]:
        ng = npred = cg = cp = 0
        for g, p in zip(gold, pred):
            gs = [x for x in spans(g) if t == "all" or x[0] == t]
            ps = [x for x in spans(p) if t == "all" or x[0] == t]
            ov = lambda a, b: a[0] == b[0] and not (a[2] < b[1] or a[1] > b[2])
            ng, npred = ng + len(gs), npred + len(ps)
            cg += sum(any(ov(x, y) for y in ps) for x in gs)
            cp += sum(any(ov(y, x) for x in gs) for y in ps)
        p = cp / npred if npred else 0.0
        r = cg / ng if ng else 0.0
        out[t] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return out


def summary(gold, pred):
    m = evaluate_tags(gold, pred)
    return {"entity_micro": m["entity_micro"], "entity_macro_f1": m["entity_macro_f1"],
            "entity_micro_f1_ci95": m.get("entity_micro_f1_ci95"),
            "per_type": m["per_type"], "lenient": lenient(gold, pred)}


def only_types(tags, types):
    return to_tags(len(tags), {x for x in spans(tags) if x[0] in types})


# --- commands -----------------------------------------------------------------------------------
def cmd_predict(a):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from dp_pii_luganda import TokenDataset, fix_truncation, make_collate, predict

    sp = load_splits()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model, torch_dtype=torch.float32).to(device)
    out = {"source": a.model}
    for split in ("validation", "test"):
        loader = DataLoader(TokenDataset(sp[split], tok), batch_size=32, shuffle=False,
                            collate_fn=make_collate(tok.pad_token_id))
        g, p, _ = predict(model, loader, device)
        out[split] = dict(zip(("gold", "pred"), fix_truncation(g, p, sp[split])))
        m = evaluate_tags(out[split]["gold"], out[split]["pred"], n_boot=0)
        print(f"{a.run}: {split} entity micro-F1 {m['entity_micro']['f1']:.4f} (from {a.model})")
    os.makedirs(a.run, exist_ok=True)
    with open(os.path.join(a.run, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)


def micro_f1(gold, pred):
    return evaluate_tags(gold, pred, n_boot=0)["entity_micro"]["f1"]


def cmd_score(a):
    sp = load_splits()
    out = {"rule_types": list(RULE_TYPES), "rules_only": {}, "runs": {}}
    for split in ("train", "validation", "test"):
        gold = [[ID2LABEL[x] for x in t] for t in sp[split]["ner_tags"]]
        rules = [to_tags(len(g), rule_spans(toks)) for g, toks in zip(gold, sp[split]["tokens"])]
        gold_rt = [only_types(g, RULE_TYPES) for g in gold]
        m = summary(gold_rt, rules)
        out["rules_only"][split] = {t: {**m["per_type"][t], "lenient_f1": m["lenient"][t]["f1"]} for t in RULE_TYPES}
        print("rules only,", split, {t: round(out["rules_only"][split][t]["f1"], 3) for t in RULE_TYPES})
    rule_sets = {k: [rule_spans(t) for t in sp[k]["tokens"]] for k in ("validation", "test")}
    preds = {}
    for path in sorted(glob.glob(os.path.join(a.results, "*", "predictions.json"))):
        with open(path, encoding="utf-8") as f:
            preds[os.path.basename(os.path.dirname(path))] = json.load(f)
    # combination strategy: the one with the higher mean validation micro-F1 over runs that have validation
    # predictions (the test split plays no part in this choice)
    val = {st: [micro_f1(p["validation"]["gold"], [combine(x, r, st) for x, r in
                                                   zip(p["validation"]["pred"], rule_sets["validation"])])
                for p in preds.values() if "validation" in p] for st in STRATEGIES}
    strategy = max(STRATEGIES, key=lambda st: sum(val[st]) / max(len(val[st]), 1))
    out["strategy"] = {"chosen": strategy, "validation_micro_f1": val,
                       "validation_runs": [n for n, p in preds.items() if "validation" in p]}
    print("validation micro-F1 by strategy:", {k: [round(x, 3) for x in v] for k, v in val.items()}, "->", strategy)
    for name, p in preds.items():
        gold, pred = p["test"]["gold"], p["test"]["pred"]
        hyb = [combine(x, r, strategy) for x, r in zip(pred, rule_sets["test"])]
        out["runs"][name] = {"model_only": summary(gold, pred), "model_plus_rules": summary(gold, hyb)}
        mo, mh = out["runs"][name]["model_only"], out["runs"][name]["model_plus_rules"]
        print(f"{name}: test micro-F1 {mo['entity_micro']['f1']:.3f} -> {mh['entity_micro']['f1']:.3f} | "
              + " ".join(f"{t} {mo['per_type'][t]['f1']:.2f}->{mh['per_type'][t]['f1']:.2f}" for t in RULE_TYPES))
    os.makedirs(os.path.join(a.results, "hybrid"), exist_ok=True)
    with open(os.path.join(a.results, "hybrid", "hybrid.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--run", required=True)
    s = sub.add_parser("score")
    s.add_argument("results")
    s.add_argument("--dev", action="store_true", help="print rule errors on train/validation (rule development)")
    a = ap.parse_args()
    if a.cmd == "predict":
        cmd_predict(a)
    elif a.dev:
        cmd_dev()
    else:
        cmd_score(a)


def cmd_dev():
    """Rule development aid: misses and false alarms on train + validation only (never test)."""
    sp = load_splits()
    for split in ("train", "validation"):
        for toks, tags in zip(sp[split]["tokens"], sp[split]["ner_tags"]):
            gold = {x for x in spans([ID2LABEL[t] for t in tags]) if x[0] in RULE_TYPES}
            rs = rule_spans(toks)
            for t, s, e in sorted(gold - rs):
                print(f"[{split} MISS {t}] {' '.join(toks[s:e + 1])}  || {' '.join(toks[max(0, s - 4):e + 3])}")
            for t, s, e in sorted(rs - gold):
                print(f"[{split} FA   {t}] {' '.join(toks[s:e + 1])}  || {' '.join(toks[max(0, s - 4):e + 3])}")


if __name__ == "__main__":
    main()

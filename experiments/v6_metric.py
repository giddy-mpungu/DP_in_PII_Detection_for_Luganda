"""Scores models with the metric used for luganda-ner-v6 (its training notebook, LugandaNER.ipynb):
every XLM-R sub-token carries its word's tag (label_all_tokens=True) and seqeval (mode="strict", scheme=BILOU)
is computed over those sub-token sequences, so a word split into k sub-tokens tagged U-X counts as k entities.
Reported next to this study's word-level entity F1 so the two scales can be compared.

usage: python v6_metric.py <out.json> <name=model> [<name=model> ...]
"""
import json
import sys

import numpy as np
import torch
from seqeval.metrics import f1_score
from seqeval.scheme import BILOU
from transformers import AutoModelForTokenClassification, AutoTokenizer

from dp_pii_luganda import ID2LABEL, evaluate_tags, load_splits

test = load_splits()["test"]
out = {}
for arg in sys.argv[2:]:
    key, name = arg.split("=", 1)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForTokenClassification.from_pretrained(name, dtype=torch.float32).eval()
    sub_gold, sub_pred, w_gold, w_pred = [], [], [], []
    for toks, tags in zip(test["tokens"], test["ner_tags"]):
        enc = tok(list(toks), is_split_into_words=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            p = model(**enc).logits[0].argmax(-1).tolist()
        g_sub, p_sub, p_word, prev = [], [], ["O"] * len(toks), None
        for i, w in enumerate(enc.word_ids()):
            if w is None:
                continue
            g_sub.append(ID2LABEL[tags[w]]); p_sub.append(model.config.id2label[p[i]])
            if w != prev:
                p_word[w] = model.config.id2label[p[i]]
            prev = w
        sub_gold.append(g_sub); sub_pred.append(p_sub)
        w_gold.append([ID2LABEL[t] for t in tags]); w_pred.append(p_word)
    v6_way = f1_score(sub_gold, sub_pred, mode="strict", scheme=BILOU)
    ours = evaluate_tags(w_gold, w_pred, n_boot=0)["entity_micro"]["f1"]
    out[key] = {"v6_metric_f1": v6_way, "word_level_f1": ours}
    print(f"{name}: v6 metric (sub-token, strict BILOU) F1 = {v6_way:.3f} | this study (word-level entity) F1 = {ours:.3f}")
json.dump(out, open(sys.argv[1], "w"), indent=1)

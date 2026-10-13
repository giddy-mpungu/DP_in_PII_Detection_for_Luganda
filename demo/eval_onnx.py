"""Test-set entity F1 of the int8 ONNX demo models (corpus pieces, exact match), to quote on the site."""
import json, os, sys
import numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from dp_pii_luganda import ID2LABEL, evaluate_tags, load_splits

test = load_splits()["test"]
res = {}
for eps in ("8", "3", "1", "0.5"):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "models", f"eps{eps}")
    tok = AutoTokenizer.from_pretrained(d)
    sess = ort.InferenceSession(os.path.join(d, "onnx", "model_quantized.onnx"))
    gold, pred = [], []
    for toks, tags in zip(test["tokens"], test["ner_tags"]):
        enc = tok(list(toks), is_split_into_words=True, truncation=True, max_length=128, return_tensors="np")
        am = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                             "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0].argmax(-1)
        p, prev = ["O"] * len(toks), None
        for i, w in enumerate(enc.word_ids()):
            if w is not None and w != prev:
                p[w] = ID2LABEL[int(am[i])]
            prev = w
        gold.append([ID2LABEL[x] for x in tags]); pred.append(p)
    m = evaluate_tags(gold, pred, n_boot=0)
    res[eps] = {"f1": m["entity_micro"]["f1"], "recall": m["entity_micro"]["recall"]}
    print(eps, res[eps])
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_test_scores.json"), "w"), indent=1)

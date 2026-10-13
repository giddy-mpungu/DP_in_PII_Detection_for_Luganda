"""Export the released DP models to int8 ONNX for the in-browser demo (transformers.js layout).

usage: python demo/export_onnx.py      -> demo/site/models/eps{8,3,1,0.5}/{config.json, tokenizer*.json, onnx/model_quantized.onnx}
"""
import json
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "models")


def weight_only_int8(src, dst, keep=("classifier",)):
    """Weight-only int8: every large MatMul/Gather weight becomes int8 + per-channel scale, dequantised by a standard
    DequantizeLinear node; activations stay float. Dynamic int8 (which also quantises activations) cost ~0.04 test F1,
    because DP models make low-margin predictions; the 8-bit MatMulNBits kernel is missing from onnxruntime-web."""
    m = onnx.load(src)
    inits = {i.name: i for i in m.graph.initializer}
    users = {}
    for n in m.graph.node:
        for k, x in enumerate(n.input):
            users.setdefault(x, []).append((n, k))
    new_nodes = []
    for name, init in list(inits.items()):
        w = numpy_helper.to_array(init)
        if w.ndim != 2 or w.size < 65536 or any(s in name for s in keep):
            continue
        uses = users.get(name, [])
        if not uses or not all(n.op_type in ("MatMul", "Gather") for n, _ in uses):
            continue
        axis = 0 if uses[0][0].op_type == "Gather" else 1        # per embedding row / per output channel
        amax = np.abs(w).max(axis=1 - axis, keepdims=True)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        wq = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        m.graph.initializer.remove(init)
        m.graph.initializer.extend([numpy_helper.from_array(wq, name + "_q"),
                                    numpy_helper.from_array(scale.reshape(-1), name + "_scale")])
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale"], [name], axis=axis,
                                          name=name + "_dq"))
    for n in reversed(new_nodes):
        m.graph.node.insert(0, n)
    onnx.save(m, dst)


# Experiment E models (synthetic warm start + improved DP-SGD), seed 42, half-precision release copies
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
EPS = ("8", "3", "1", "0.5")


def export(eps):
    name = os.path.join(RESULTS, f"big_dp_eps{eps}_seed42", "release_model")
    out = os.path.join(OUT, f"eps{eps}")
    os.makedirs(os.path.join(out, "onnx"), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForTokenClassification.from_pretrained(name, torch_dtype=torch.float32).eval()
    model.config.save_pretrained(out)
    tok.save_pretrained(out)
    # transformers.js reads the legacy `add_prefix_space` flag of Metaspace, not `prepend_scheme`; without it every
    # word loses its leading "▁" and the browser tokens differ from those the model was trained on
    tj = os.path.join(out, "tokenizer.json")
    t = json.load(open(tj, encoding="utf-8"))
    for part in (t["pre_tokenizer"], t["decoder"]):
        for pt in part.get("pretokenizers", [part]):
            if pt.get("type") == "Metaspace":
                pt["add_prefix_space"] = True
    json.dump(t, open(tj, "w", encoding="utf-8"), ensure_ascii=False)
    for f in ("sentencepiece.bpe.model", "special_tokens_map.json"):   # transformers.js needs tokenizer.json only
        p = os.path.join(out, f)
        if os.path.exists(p):
            os.remove(p)
    fp32 = os.path.join(out, "onnx", "model.onnx")
    enc = tok(["omusajja", "ono", "ayitibwa", "musoke"], is_split_into_words=True, return_tensors="pt")
    torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), fp32, input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "logits": {0: "batch", 1: "seq"}})
    q = os.path.join(out, "onnx", "model_quantized.onnx")
    weight_only_int8(fp32, q)
    os.remove(fp32)
    # agreement check against PyTorch on a sample input
    with torch.no_grad():
        ref = model(**enc).logits.argmax(-1).numpy()
    got = ort.InferenceSession(q).run(None, {k: enc[k].numpy() for k in ("input_ids", "attention_mask")})[0].argmax(-1)
    print(f"eps{eps}: {os.path.getsize(q) / 1e6:.0f} MB, argmax agreement on sample {np.mean(ref == got):.2f}")


if __name__ == "__main__":
    for e in EPS:
        export(e)

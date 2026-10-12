"""
Differentially private PII detection for Luganda -- experiment pipeline.

Every run writes   <out>/<run_name>/metrics.json   (config, environment, per-epoch history,
privacy accounting, entity-level and token-level test metrics, confusion matrix, bootstrap CI,
membership-inference AUC) and   errors.json   (false negatives / false positives on the test set).
Runs whose metrics.json already exists are skipped, so the script can be re-launched after a
Colab/Kaggle disconnect and it will resume.

Usage
    python dp_pii_luganda.py --plan audit   --out results      # data audit + figures (CPU, ~1 min)
    python dp_pii_luganda.py --plan leakage --out results      # v6 evaluated on train/val/test (no training)
    python dp_pii_luganda.py --plan main    --out results      # Experiment B (clean xlm-roberta-base backbone)
    python dp_pii_luganda.py --plan repro   --out results      # Experiment A (original setup, corrected metrics)
    python dp_pii_luganda.py --plan all     --out results
    python dp_pii_luganda.py --plan main --only dp_eps8_seed42 --out results
"""
import argparse
import gc
import json
import os

# many runs share one process: let the CUDA allocator reuse fragmented memory between them
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import platform
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATASET = "Conrad747/lg-ner"
V6_MODEL = "Conrad747/luganda-ner-v6"   # xlm-roberta-base already fine-tuned (non-privately) on lg-ner train
CLEAN_BACKBONE = "xlm-roberta-base"     # pretrained only on unlabelled CC-100 text, never saw lg-ner

LABELS = ["O",
          "B-PERSON", "I-PERSON", "L-PERSON", "U-PERSON",
          "B-NORP", "I-NORP", "L-NORP", "U-NORP",
          "B-DATE", "I-DATE", "L-DATE", "U-DATE",
          "B-USERID", "I-USERID", "L-USERID", "U-USERID",
          "B-ORG", "I-ORG", "L-ORG", "U-ORG",
          "B-LOCATION", "I-LOCATION", "L-LOCATION", "U-LOCATION"]
ENTITY_TYPES = ["PERSON", "NORP", "DATE", "USERID", "ORG", "LOCATION"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
DELTA = 1e-5          # < 1/N (N = 2,085 training sentences); used for every DP configuration
MAX_LEN = 128


# ----------------------------------------------------------------------------------------------
# Experiment plans
# ----------------------------------------------------------------------------------------------
MAIN_LRS = (2e-4, 5e-4, 1e-3)
PUB_LRS = (5e-5, 2e-4, 5e-4)
PE_LRS = (1e-3, 3e-3)


def plan_runs(plan, dp_lr=None, out="results", pub_lr=None):
    seeds = [42, 1234, 2024]
    runs = []
    if plan in ("public", "all"):
        # Experiment C: non-private warm start on PUBLIC Luganda NER data (MasakhaNER 2.0, overlap with lg-ner
        # removed), then DP fine-tuning on lg-ner. DP protects lg-ner; the public data needs no protection.
        warm = os.path.join(out, "public_warmstart", "model")
        runs.append(dict(name="public_warmstart", init=CLEAN_BACKBONE, mode="nonprivate", train_data="public",
                         lr=5e-5, batch_size=16, physical_batch=16, epochs=5, freeze_embeddings=True, seed=42,
                         save_model=True))
        for s in seeds:
            runs.append(dict(name=f"pub_nonprivate_seed{s}", init=warm, mode="nonprivate", lr=5e-5, batch_size=16,
                             physical_batch=16, epochs=10, freeze_embeddings=True, seed=s))
        for lr in PUB_LRS:
            runs.append(dict(name=f"pub_lrsweep_eps8_lr{lr:g}", init=warm, mode="dp", target_epsilon=8.0,
                             max_grad_norm=1.0, lr=lr, batch_size=64, physical_batch=16, epochs=15,
                             freeze_embeddings=True, seed=42, sweep=True))
        if pub_lr is not None:
            for eps in (8.0, 3.0, 1.0):
                for s in seeds:
                    runs.append(dict(name=f"pub_dp_eps{eps:g}_seed{s}", init=warm, mode="dp", target_epsilon=eps,
                                     max_grad_norm=1.0, lr=pub_lr, batch_size=64, physical_batch=16, epochs=15,
                                     freeze_embeddings=True, seed=s, release=(s == 42)))
    if plan in ("public_pe", "all"):
        # Experiment D: as C, but DP updates only the top transformer layers and the classifier head.
        # Fewer trainable coordinates means less total noise for the same (eps, delta) (Yu et al., 2022).
        warm = os.path.join(out, "public_warmstart", "model")
        pe = dict(init=warm, mode="dp", max_grad_norm=1.0, batch_size=128, physical_batch=32, epochs=20,
                  freeze_embeddings=True, train_top_layers=2)
        for lr in PE_LRS:
            runs.append(dict(pe, name=f"pub_pe_lrsweep_eps8_lr{lr:g}", target_epsilon=8.0, lr=lr, seed=42,
                             sweep=True))
        if pub_lr is not None:
            for eps in (8.0, 3.0, 1.0):
                for s in seeds:
                    runs.append(dict(pe, name=f"pub_pe_dp_eps{eps:g}_seed{s}", target_epsilon=eps, lr=pub_lr,
                                     seed=s, release=(s == 42)))
    if plan in ("leakage", "repro", "all"):
        # The model used in the submitted thesis, evaluated without any further training.
        runs.append(dict(name="v6_no_further_training", init=V6_MODEL, mode="eval_only", seed=42))
    if plan in ("repro", "all"):
        # Experiment A: faithful re-run of the submitted configuration (v6 start, batch 8, lr 5e-5,
        # 5 epochs, full model trainable), now seeded and evaluated with entity-level metrics.
        # one seed: this experiment explains the submitted numbers; the main experiments carry the 3-seed evidence
        orig = [("hu", 1.0, 1.0), ("bp", 2.0, 0.5), ("sp", 3.0, 0.5)]
        for tag, sigma, clip in orig:
            for s in seeds[:1]:
                runs.append(dict(name=f"repro_{tag}_seed{s}", init=V6_MODEL, mode="dp", noise_multiplier=sigma,
                                 # physical batch 2 as in the original code: per-example gradients of the
                                 # trainable 192M-parameter embedding matrix do not fit a T4 at 8
                                 max_grad_norm=clip, lr=5e-5, batch_size=8, physical_batch=2, epochs=5,
                                 freeze_embeddings=False, seed=s))
    if plan in ("main", "all"):
        # Experiment B: clean backbone. Non-private baseline + three privacy levels set by target epsilon.
        for s in seeds:
            runs.append(dict(name=f"nonprivate_seed{s}", init=CLEAN_BACKBONE, mode="nonprivate", lr=5e-5,
                             batch_size=16, physical_batch=16, epochs=10, freeze_embeddings=True, seed=s))
        # learning-rate selection on the validation set (High Utility, seed 42)
        for lr in MAIN_LRS:
            runs.append(dict(name=f"lrsweep_eps8_lr{lr:g}", init=CLEAN_BACKBONE, mode="dp", target_epsilon=8.0,
                             max_grad_norm=1.0, lr=lr, batch_size=64, physical_batch=16, epochs=15,
                             freeze_embeddings=True, seed=42, sweep=True))
        if dp_lr is not None:
            for eps in (8.0, 3.0, 1.0):
                for s in seeds:
                    runs.append(dict(name=f"dp_eps{eps:g}_seed{s}", init=CLEAN_BACKBONE, mode="dp",
                                     target_epsilon=eps, max_grad_norm=1.0, lr=dp_lr, batch_size=64,
                                     physical_batch=16, epochs=15, freeze_embeddings=True, seed=s))
    return runs


def select_dp_lr(out, prefix="lrsweep", lrs=MAIN_LRS):
    """Best learning rate from the sweep by validation entity-F1 (final epoch)."""
    best = None
    for lr in lrs:
        p = os.path.join(out, f"{prefix}_eps8_lr{lr:g}", "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            v = m["history"][-1]["val_entity_f1"]
            if best is None or v > best[1]:
                best = (lr, v)
    return best


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def load_splits():
    from datasets import load_dataset
    # the Hub's auto-converted Parquet copy: identical rows, and works with datasets>=3 (no loading script)
    base = f"hf://datasets/{DATASET}@refs%2Fconvert%2Fparquet/lug"
    ds = load_dataset("parquet", data_files={k: f"{base}/{k}/0000.parquet" for k in ("train", "validation", "test")})
    return {k: ds[k] for k in ("train", "validation", "test")}


PUBLIC_DATASET = "masakhane/masakhaner2"   # Luganda subset; PER/ORG/LOC/DATE in IOB2
PUBLIC_TYPES = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION", "DATE": "DATE"}


def _norm_words(tokens):
    """Comparable word sequence: merge '##' pieces (lg-ner), lower-case, split off punctuation/apostrophes."""
    words = []
    for t in tokens:
        if t.startswith("##") and words:
            words[-1] += t[2:]
        else:
            words.append(t)
    text = " ".join(words).lower()
    return "".join(c if c.isalnum() else " " for c in text).split()


def _ngrams(words, n=8):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)} or ({tuple(words)} if words else set())


def iob2_to_bilou(tags):
    out = []
    for i, t in enumerate(tags):
        if t == "O":
            out.append("O")
            continue
        p, typ = t.split("-", 1)
        typ = PUBLIC_TYPES[typ]
        cont = i + 1 < len(tags) and tags[i + 1] == f"I-{t.split('-', 1)[1]}"
        start = p == "B" or i == 0 or tags[i - 1] == "O" or tags[i - 1].split("-", 1)[1] != t.split("-", 1)[1]
        out.append(("B-" if cont else "U-") + typ if start else ("I-" if cont else "L-") + typ)
    return out


def load_public(private_splits, report_path=None):
    """MasakhaNER 2.0 Luganda train/validation, mapped to the lg-ner label space, lower-cased like lg-ner,
    with every sentence that shares an 8-word sequence with ANY lg-ner sentence removed."""
    from datasets import Dataset as HFDataset, load_dataset
    base = f"hf://datasets/{PUBLIC_DATASET}@refs%2Fconvert%2Fparquet/lug"
    ds = load_dataset("parquet", data_files={k: f"{base}/{k}/0000.parquet" for k in ("train", "validation")})
    names = ds["train"].features["ner_tags"].feature.names
    private = set()
    for s in private_splits.values():
        for toks in s["tokens"]:
            private |= _ngrams(_norm_words(list(toks)))
    out, rep = {}, {}
    for k in ("train", "validation"):
        toks_all, tags_all, dropped = [], [], 0
        for toks, tags in zip(ds[k]["tokens"], ds[k]["ner_tags"]):
            if _ngrams(_norm_words(list(toks))) & private:
                dropped += 1
                continue
            toks_all.append([t.lower() for t in toks])
            tags_all.append([LABEL2ID[x] for x in iob2_to_bilou([names[i] for i in tags])])
        out[k] = HFDataset.from_dict({"tokens": toks_all, "ner_tags": tags_all})
        rep[k] = {"original": len(ds[k]), "dropped_overlap_with_lg_ner": dropped, "kept": len(toks_all)}
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(rep, f, indent=1)
    print("public data:", rep)
    return out


class TokenDataset(Dataset):
    """Tuples (input_ids, attention_mask, labels) so Opacus' empty-batch handling works."""

    def __init__(self, split, tokenizer):
        words = [list(x) for x in split["tokens"]]   # datasets>=4 returns a lazy Column, not a list
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
        self.items, self.n_truncated = [], 0
        for i, tags in enumerate(split["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)
            labels, prev = [], None
            for w in word_ids:
                # label only the first sub-token of each word; special tokens and continuations -> -100
                labels.append(-100 if w is None or w == prev else tags[w])
                prev = w
            seen = {w for w in word_ids if w is not None}
            if len(seen) < len(tags):
                self.n_truncated += 1
            self.items.append((torch.tensor(enc["input_ids"][i]), torch.tensor(enc["attention_mask"][i]),
                               torch.tensor(labels)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        if len(batch) == 0:
            return None
        ids, mask, lab = zip(*batch)
        pad = torch.nn.utils.rnn.pad_sequence
        return (pad(ids, batch_first=True, padding_value=pad_id), pad(mask, batch_first=True, padding_value=0),
                pad(lab, batch_first=True, padding_value=-100))
    return collate


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def to_iobes(tag):
    return tag.replace("L-", "E-").replace("U-", "S-")


def _end_of_chunk(prev_tag, tag, prev_type, type_):
    return (prev_tag in ("E", "S") or (prev_tag == "B" and tag in ("B", "S", "O"))
            or (prev_tag == "I" and tag in ("B", "S", "O"))
            or (prev_tag != "O" and prev_tag != "." and prev_type != type_))


def _start_of_chunk(prev_tag, tag, prev_type, type_):
    return (tag in ("B", "S") or (prev_tag in ("E", "S", "O") and tag in ("E", "I"))
            or (tag != "O" and tag != "." and prev_type != type_))


def get_entities(seq):
    """conlleval-style chunk extraction; identical to seqeval.metrics.sequence_labeling.get_entities
    (default, lenient mode) for IOBES tags. Vendored because seqeval only ships a setup.py sdist,
    which fails to build on some current Python/setuptools versions (e.g. Google Colab)."""
    chunks, prev_tag, prev_type, begin = [], "O", "", 0
    for i, chunk in enumerate(list(seq) + ["O"]):
        tag, type_ = chunk[0], chunk[1:].split("-", maxsplit=1)[-1] or "_"
        if _end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin, i - 1))
        if _start_of_chunk(prev_tag, tag, prev_type, type_):
            begin = i
        prev_tag, prev_type = tag, type_
    return chunks


def spans(tags):
    return {(t, s, e) for t, s, e in get_entities([to_iobes(x) for x in tags])}


def entity_counts(gold, pred):
    """Per-sentence (n_gold, n_pred, n_correct) overall and per type -- exact span+type match."""
    rows = []
    for g, p in zip(gold, pred):
        gs, ps = spans(g), spans(p)
        row = {"all": (len(gs), len(ps), len(gs & ps))}
        for t in ENTITY_TYPES:
            gt = {x for x in gs if x[0] == t}
            pt = {x for x in ps if x[0] == t}
            row[t] = (len(gt), len(pt), len(gt & pt))
        rows.append(row)
    return rows


def prf(ng, npred, nc):
    p = nc / npred if npred else 0.0
    r = nc / ng if ng else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate_tags(gold, pred, n_boot=1000, seed=0):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    rows = entity_counts(gold, pred)
    out = {"per_type": {}}
    tot = np.array([r["all"] for r in rows])
    p, r, f = prf(*tot.sum(0))
    out["entity_micro"] = {"precision": p, "recall": r, "f1": f, "n_gold": int(tot[:, 0].sum()),
                           "n_pred": int(tot[:, 1].sum()), "n_correct": int(tot[:, 2].sum())}
    fs = []
    for t in ENTITY_TYPES:
        a = np.array([x[t] for x in rows]).sum(0)
        tp, tr, tf = prf(*a)
        out["per_type"][t] = {"precision": tp, "recall": tr, "f1": tf, "support": int(a[0]),
                              "missed": int(a[0] - a[2]), "spurious": int(a[1] - a[2])}
        if a[0] > 0:
            fs.append(tf)
    out["entity_macro_f1"] = float(np.mean(fs))
    try:  # optional cross-check against seqeval when it is installed
        from seqeval.metrics import f1_score
        from seqeval.scheme import IOBES
        g_io = [[to_iobes(x) for x in s] for s in gold]
        p_io = [[to_iobes(x) for x in s] for s in pred]
        out["entity_micro_f1_seqeval"] = f1_score(g_io, p_io)
        out["entity_micro_f1_strict"] = f1_score(g_io, p_io, mode="strict", scheme=IOBES)
    except ImportError:
        pass

    # bootstrap 95% CI over sentences
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), len(rows))
        boots.append(prf(*tot[idx].sum(0))[2])
    if boots:
        out["entity_micro_f1_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # token level (first sub-token of each word)
    gf = [x for s in gold for x in s]
    pf = [x for s in pred for x in s]
    wp, wr, wf, _ = precision_recall_fscore_support(gf, pf, average="weighted", zero_division=0)
    out["token_weighted_incl_O"] = {"precision": wp, "recall": wr, "f1": wf}   # metric used in the submitted thesis
    ent_labels = [l for l in LABELS if l != "O"]
    mp, mr, mf, _ = precision_recall_fscore_support(gf, pf, labels=ent_labels, average="macro", zero_division=0)
    out["token_macro_excl_O"] = {"precision": mp, "recall": mr, "f1": mf}
    out["token_accuracy"] = float(np.mean([a == b for a, b in zip(gf, pf)]))
    typ = lambda x: x.split("-", 1)[1] if "-" in x else "O"
    cm_labels = ["O"] + ENTITY_TYPES
    out["confusion_matrix"] = {"labels": cm_labels,
                               "matrix": confusion_matrix([typ(x) for x in gf], [typ(x) for x in pf],
                                                          labels=cm_labels).tolist()}
    return out


def error_examples(tokens, gold, pred):
    errs = []
    for i, (toks, g, p) in enumerate(zip(tokens, gold, pred)):
        gs, ps = spans(g), spans(p)
        for t, s, e in sorted(gs - ps):
            overlap = [x for x in ps if not (x[2] < s or x[1] > e)]
            kind = ("type_confusion" if any(x[1] == s and x[2] == e for x in overlap)
                    else "boundary" if overlap else "missed")
            errs.append({"sent": i, "kind": "FN", "subtype": kind, "type": t, "span": " ".join(toks[s:e + 1]),
                         "pred": [list(x) for x in overlap], "sentence": " ".join(toks)})
        for t, s, e in sorted(ps - gs):
            overlap = [x for x in gs if not (x[2] < s or x[1] > e)]
            kind = ("type_confusion" if any(x[1] == s and x[2] == e for x in overlap)
                    else "boundary" if overlap else "spurious")
            errs.append({"sent": i, "kind": "FP", "subtype": kind, "type": t, "span": " ".join(toks[s:e + 1]),
                         "gold": [list(x) for x in overlap], "sentence": " ".join(toks)})
    return errs


# ----------------------------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------------------------
def per_sample_loss(logits, labels):
    """Mean token loss per sentence -> (B,). Keeps each sentence's gradient independent of the others,
    which the DP-SGD sensitivity argument requires (a batch-wide token mean would not)."""
    tok = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    n = (labels != -100).sum(1).clamp(min=1)
    return tok.sum(1) / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, golds, losses = [], [], []
    for batch in loader:
        ids, mask, lab = (x.to(device) for x in batch)
        logits = model(input_ids=ids, attention_mask=mask).logits
        losses.extend(per_sample_loss(logits, lab).cpu().tolist())
        am = logits.argmax(-1)
        for pr, lb in zip(am.cpu().tolist(), lab.cpu().tolist()):
            keep = [i for i, l in enumerate(lb) if l != -100]
            preds.append([ID2LABEL[pr[i]] for i in keep])
            golds.append([ID2LABEL[lb[i]] for i in keep])
    return golds, preds, losses


def fix_truncation(golds, preds, split):
    """Words lost to truncation count as predicted 'O' so gold entities are never silently dropped."""
    out_g, out_p = [], []
    for g, p, tags in zip(golds, preds, split["ner_tags"]):
        full = [ID2LABEL[t] for t in tags]
        out_g.append(full)
        out_p.append(p + ["O"] * (len(full) - len(p)))
    return out_g, out_p


def mia_auc(train_losses, test_losses):
    """Loss-threshold membership inference (Yeom et al., 2018): AUC of telling train from test
    sentences by their loss. 0.5 = no signal; higher = the model behaves differently on members."""
    from sklearn.metrics import roc_auc_score
    y = [1] * len(train_losses) + [0] * len(test_losses)
    s = [-x for x in train_losses] + [-x for x in test_losses]
    return float(roc_auc_score(y, s))


def environment():
    import transformers, datasets
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
           "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu_threads": torch.get_num_threads()}
    try:
        import opacus
        env["opacus"] = opacus.__version__
    except ImportError:
        pass
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------------------------
def run(cfg, splits, out, device, public=None):
    try:
        _run(cfg, splits, out, device, public)
    finally:
        # release the model, optimizer and Opacus state before the next run in this process
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run(cfg, splits, out, device, public=None):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    rdir = os.path.join(out, cfg["name"])
    if os.path.exists(os.path.join(rdir, "metrics.json")):
        print(f"[skip] {cfg['name']} (already done)")
        return
    os.makedirs(rdir, exist_ok=True)
    print(f"\n=== {cfg['name']} ===\n{json.dumps(cfg)}")
    set_seed(cfg["seed"])
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(cfg["init"])
    data = {k: TokenDataset(v, tok) for k, v in splits.items()}
    # training data: lg-ner (private) by default; MasakhaNER Luganda (public) for the warm-start model.
    # Validation curves, test metrics and the membership-inference test always use lg-ner.
    train_ds = TokenDataset(public["train"], tok) if cfg.get("train_data") == "public" else data["train"]
    collate = make_collate(tok.pad_token_id)
    eval_loader = lambda k: DataLoader(data[k], batch_size=32, shuffle=False, collate_fn=collate)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["init"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID).to(device)
    if cfg.get("freeze_embeddings"):
        for p in model.base_model.embeddings.parameters():
            p.requires_grad = False
    if cfg.get("train_top_layers"):
        for p in model.parameters():
            p.requires_grad = False
        for p in model.base_model.encoder.layer[-cfg["train_top_layers"]:].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    history, privacy = [], {}
    if cfg["mode"] != "eval_only":
        g = torch.Generator().manual_seed(cfg["seed"])
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                  collate_fn=collate, generator=g)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
        engine = None
        if cfg["mode"] == "dp":
            from opacus import PrivacyEngine
            engine = PrivacyEngine(accountant="rdp")
            model.train()  # from_pretrained returns eval mode; Opacus validates train mode
            common = dict(module=model, optimizer=optimizer, data_loader=train_loader,
                          max_grad_norm=cfg["max_grad_norm"], poisson_sampling=True)
            if "target_epsilon" in cfg:
                model, optimizer, train_loader = engine.make_private_with_epsilon(
                    target_epsilon=cfg["target_epsilon"], target_delta=DELTA, epochs=cfg["epochs"], **common)
            else:
                model, optimizer, train_loader = engine.make_private(
                    noise_multiplier=cfg["noise_multiplier"], **common)
            privacy.update(noise_multiplier=optimizer.noise_multiplier, max_grad_norm=cfg["max_grad_norm"],
                           sample_rate=1 / len(train_loader), delta=DELTA, accountant="RDP (Opacus)")

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            tr_loss, n_steps = 0.0, 0
            if engine is not None:
                from opacus.utils.batch_memory_manager import BatchMemoryManager
                ctx = BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=cfg["physical_batch"],
                                         optimizer=optimizer)
            else:
                import contextlib
                ctx = contextlib.nullcontext(train_loader)
            with ctx as loader:
                for batch in loader:
                    if batch is None or batch[0].numel() == 0:
                        continue
                    ids, mask, lab = (x.to(device) for x in batch)
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    loss = per_sample_loss(logits, lab).mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    tr_loss += loss.item()
                    n_steps += 1
            core = model._module if hasattr(model, "_module") else model
            vg, vp, vl = predict(core, eval_loader("validation"), device)
            vg, vp = fix_truncation(vg, vp, splits["validation"])
            vm = evaluate_tags(vg, vp, n_boot=0)
            h = {"epoch": epoch, "train_loss": tr_loss / max(n_steps, 1), "val_loss": float(np.mean(vl)),
                 "val_entity_f1": vm["entity_micro"]["f1"], "val_entity_macro_f1": vm["entity_macro_f1"],
                 "val_token_weighted_f1": vm["token_weighted_incl_O"]["f1"],
                 "elapsed_min": (time.time() - t0) / 60}
            if engine is not None:
                h["epsilon_cumulative"] = engine.get_epsilon(DELTA)
            history.append(h)
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()}, flush=True)

        if engine is not None:
            from opacus.accountants import PRVAccountant
            privacy["epsilon_rdp"] = engine.get_epsilon(DELTA)
            privacy["optimizer_steps"] = sum(n for _, _, n in engine.accountant.history)
            try:
                prv = PRVAccountant()
                prv.history = list(engine.accountant.history)
                privacy["epsilon_prv"] = prv.get_epsilon(delta=DELTA)
            except Exception as e:  # PRV accountant can fail for some (sigma, q) combinations
                privacy["epsilon_prv_error"] = repr(e)

    core = model._module if hasattr(model, "_module") else model
    if cfg.get("save_model"):
        core.save_pretrained(os.path.join(rdir, "model"))
        tok.save_pretrained(os.path.join(rdir, "model"))
    if cfg.get("release"):
        # fp16 copy of a privately trained model, for publication on the Hugging Face Hub
        import copy
        rel = copy.deepcopy(core).half()
        rel.save_pretrained(os.path.join(rdir, "release_model"))
        tok.save_pretrained(os.path.join(rdir, "release_model"))
        del rel
    results = {}
    for split in ("train", "validation", "test"):
        gg, pp, ll = predict(core, eval_loader(split), device)
        gg, pp = fix_truncation(gg, pp, splits[split])
        results[split] = {"losses": ll, "gold": gg, "pred": pp}
    test_metrics = evaluate_tags(results["test"]["gold"], results["test"]["pred"])
    train_metrics = evaluate_tags(results["train"]["gold"], results["train"]["pred"], n_boot=0)
    val_metrics = evaluate_tags(results["validation"]["gold"], results["validation"]["pred"], n_boot=0)

    metrics = {
        "config": cfg, "environment": environment(), "device": str(device), "trainable_params": n_trainable,
        "truncated_sentences": {k: v.n_truncated for k, v in data.items()},
        "history": history, "privacy": privacy,
        "test": test_metrics,
        "train_summary": {"entity_micro_f1": train_metrics["entity_micro"]["f1"],
                          "token_weighted_f1": train_metrics["token_weighted_incl_O"]["f1"],
                          "mean_loss": float(np.mean(results["train"]["losses"]))},
        "validation_summary": {"entity_micro_f1": val_metrics["entity_micro"]["f1"],
                               "entity_macro_f1": val_metrics["entity_macro_f1"],
                               "mean_loss": float(np.mean(results["validation"]["losses"]))},
        "mia_auc_train_vs_test": mia_auc(results["train"]["losses"], results["test"]["losses"]),
        "runtime_min": (time.time() - t0) / 60,
    }
    with open(os.path.join(rdir, "losses.json"), "w") as f:   # per-sentence losses: membership-inference plots
        json.dump({k: [round(x, 5) for x in results[k]["losses"]] for k in ("train", "test")}, f)
    with open(os.path.join(rdir, "errors.json"), "w", encoding="utf-8") as f:
        json.dump(error_examples(splits["test"]["tokens"], results["test"]["gold"], results["test"]["pred"]),
                  f, ensure_ascii=False, indent=1)
    with open(os.path.join(rdir, "predictions.json"), "w", encoding="utf-8") as f:   # input to hybrid_eval.py
        json.dump({k: {"gold": results[k]["gold"], "pred": results[k]["pred"]} for k in ("validation", "test")}, f)
    with open(os.path.join(rdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    m = test_metrics
    print(f"[done] {cfg['name']}: test entity F1={m['entity_micro']['f1']:.4f} "
          f"macro={m['entity_macro_f1']:.4f} token-weighted={m['token_weighted_incl_O']['f1']:.4f} "
          f"eps={privacy.get('epsilon_rdp')} MIA-AUC={metrics['mia_auc_train_vs_test']:.3f} "
          f"({metrics['runtime_min']:.1f} min)")


# ----------------------------------------------------------------------------------------------
# Data audit
# ----------------------------------------------------------------------------------------------
def audit(splits, out):
    from transformers import AutoTokenizer
    d = os.path.join(out, "data_audit")
    os.makedirs(d, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CLEAN_BACKBONE)
    rep = {"splits": {}}
    seqs = {}
    for k, s in splits.items():
        tags = [[LABELS[t] for t in x] for x in s["ner_tags"]]
        ents = Counter(t for x in tags for t, _, _ in spans(x))
        tag_counts = Counter(t for x in tags for t in x)
        lens = [len(x) for x in s["tokens"]]
        sub = [len(tok(x, is_split_into_words=True)["input_ids"]) for x in s["tokens"]]
        invalid = 0
        for x in tags:
            for a, b in zip(["O"] + x, x + ["O"]):
                # BILOU validity: B/I must be followed by I/L of the same type
                if a[:2] in ("B-", "I-") and not (b[:2] in ("I-", "L-") and b[2:] == a[2:]):
                    invalid += 1
        seqs[k] = [" ".join(x) for x in s["tokens"]]
        rep["splits"][k] = {
            "sentences": len(lens), "words": int(sum(lens)), "mean_words": float(np.mean(lens)),
            "max_words": int(max(lens)), "mean_subwords_xlmr": float(np.mean(sub)), "max_subwords_xlmr": int(max(sub)),
            "sentences_over_128_subwords": int(sum(x > MAX_LEN for x in sub)),
            "entities_by_type": {t: ents.get(t, 0) for t in ENTITY_TYPES},
            "sentences_without_entities": int(sum(not spans(x) for x in tags)),
            "tag_counts": dict(tag_counts), "O_token_share": tag_counts["O"] / sum(tag_counts.values()),
            "words_with_##_prefix": int(sum(w.startswith("##") for x in s["tokens"] for w in x)),
            "invalid_bilou_transitions": invalid,
            "duplicate_sentences_within_split": len(seqs[k]) - len(set(seqs[k])),
        }
    st = {k: set(v) for k, v in seqs.items()}
    rep["overlap"] = {"train_test": len(st["train"] & st["test"]), "train_validation": len(st["train"] & st["validation"]),
                      "validation_test": len(st["validation"] & st["test"])}
    with open(os.path.join(d, "data_audit.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="all", choices=["audit", "leakage", "repro", "main", "public", "public_pe", "all"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--dp_lr", type=float, default=None, help="override the learning rate picked by the sweep")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny subsets, 1 epoch: checks the pipeline end to end")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, json.dumps(environment()))
    splits = load_splits()
    full_splits = splits
    if a.smoke:
        splits = {k: v.select(range(min(len(v), 96 if k == "train" else 32))) for k, v in splits.items()}
        _plan = plan_runs
        globals()["plan_runs"] = lambda p, **kw: [dict(r, epochs=1) for r in _plan(p, **kw)]
    if a.plan in ("audit", "all"):
        audit(splits, a.out)
        if a.plan == "audit":
            return
    only = set(a.only.split(",")) if a.only else None
    public = None
    if a.plan in ("public", "all"):
        # overlap is always checked against the FULL lg-ner corpus
        public = load_public(full_splits, os.path.join(a.out, "data_audit", "public_overlap.json"))
        if a.smoke:
            public = {k: v.select(range(min(len(v), 96))) for k, v in public.items()}

    def go(runs):
        for cfg in runs:
            if only is None or cfg["name"] in only:
                run(cfg, splits, a.out, device, public)

    def dp_stage(prefix, lrs, plan, kw, run_prefix):
        lr = a.dp_lr
        if lr is None:
            best = select_dp_lr(a.out, prefix, lrs)
            if best is None:
                print(f"{prefix} sweep incomplete; DP runs not started.")
                return
            lr = best[0]
            print(f"Selected DP learning rate for {plan}: {lr:g} (validation entity F1 {best[1]:.4f})")
        collapsed = a.dp_lr is None and best[1] < 0.01
        with open(os.path.join(a.out, f"selected_lr_{plan}.json"), "w") as f:
            json.dump({"lr": lr, "best_val_entity_f1": None if a.dp_lr else best[1], "collapsed": collapsed}, f)
        runs = [r for r in plan_runs(plan, out=a.out, **{kw: lr}) if r["name"].startswith(run_prefix)]
        if collapsed:
            # every learning rate left the model predicting no entities at the loosest budget (eps=8); stricter
            # budgets cannot do better, so one seed per budget documents the collapse without wasting GPU time
            print(f"{plan}: DP training collapsed at every learning rate; running seed 42 only.")
            runs = [r for r in runs if r["seed"] == 42]
        go(runs)

    if a.plan in ("main", "all"):
        go(plan_runs("main", out=a.out))
        dp_stage("lrsweep", MAIN_LRS, "main", "dp_lr", "dp_")
    if a.plan in ("public", "all"):
        go(plan_runs("public", out=a.out))
        dp_stage("pub_lrsweep", PUB_LRS, "public", "pub_lr", "pub_dp_")
    if a.plan in ("public_pe", "all"):
        go(plan_runs("public_pe", out=a.out))
        dp_stage("pub_pe_lrsweep", PE_LRS, "public_pe", "pub_lr", "pub_pe_dp_")
    if a.plan in ("leakage", "repro", "all"):
        go(plan_runs("repro" if a.plan == "all" else a.plan, out=a.out))


if __name__ == "__main__":
    main()

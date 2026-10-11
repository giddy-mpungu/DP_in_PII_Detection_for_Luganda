"""Inter-annotator agreement for lg-ner, using the upstream 'data with duplicates' files
(github.com/conradsuuna/luganda-ner-data), which keep every annotator's copy of a sentence.

usage: python annotation_agreement.py <path to luganda-ner-data> <out.json>
"""
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

from seqeval.metrics.sequence_labeling import get_entities
from sklearn.metrics import cohen_kappa_score


def read_conll(path):
    sents, toks, tags = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if toks:
                sents.append((tuple(toks), tuple(tags)))
            toks, tags = [], []
            continue
        parts = line.split(" ")
        toks.append(parts[0])
        tags.append(parts[-1])
    if toks:
        sents.append((tuple(toks), tuple(tags)))
    return sents


def spans(tags):
    return {tuple(x) for x in get_entities([t.replace("L-", "E-").replace("U-", "S-") for t in tags])}


def main(root, out):
    rep = {}
    groups_all = defaultdict(list)
    for split in ("train", "val", "test"):
        dup = read_conll(os.path.join(root, "data with duplicates", f"{split}.txt"))
        clean = read_conll(os.path.join(root, "data", f"{split}.txt"))
        g = defaultdict(list)
        for toks, tags in dup:
            g[toks].append(tags)
            groups_all[toks].append(tags)
        sizes = Counter(len(v) for v in g.values())
        rep[split] = {"sentences_with_duplicates_file": len(dup), "unique_sentences": len(g),
                      "copies_per_sentence": dict(sorted(sizes.items())),
                      "released_sentences": len(clean)}

    # pairwise agreement over sentences annotated more than once
    multi = {k: v for k, v in groups_all.items() if len(v) >= 2}
    tok_a, tok_b, ent_tp, ent_a, ent_b, identical, pairs = [], [], 0, 0, 0, 0, 0
    typ = lambda t: t.split("-", 1)[1] if "-" in t else "O"
    for toks, versions in multi.items():
        for a, b in itertools.combinations(versions, 2):
            pairs += 1
            identical += a == b
            tok_a += [typ(x) for x in a]
            tok_b += [typ(x) for x in b]
            sa, sb = spans(a), spans(b)
            ent_tp += len(sa & sb)
            ent_a += len(sa)
            ent_b += len(sb)
    p = ent_tp / ent_b if ent_b else 0
    r = ent_tp / ent_a if ent_a else 0
    rep["agreement"] = {
        "sentences_annotated_more_than_once": len(multi),
        "annotation_pairs": pairs,
        "identical_pairs_share": identical / pairs if pairs else None,
        "token_type_cohen_kappa": float(cohen_kappa_score(tok_a, tok_b)) if pairs else None,
        "entity_pairwise_f1": 2 * p * r / (p + r) if p + r else None,
        "note": "kappa over word-level entity type (O/PERSON/...); entity F1 = exact span+type match between copies",
    }
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

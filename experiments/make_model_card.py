"""Writes README.md (Hugging Face model card) for a released DP model from its metrics.json.

usage: python make_model_card.py <run dir with metrics.json and release_model/> <repo id>
"""
import json
import os
import sys

TYPES = ["PERSON", "NORP", "DATE", "USERID", "ORG", "LOCATION"]


def main(run_dir, repo_id):
    m = json.load(open(os.path.join(run_dir, "metrics.json")))
    t, p, c = m["test"], m["privacy"], m["config"]
    rows = "\n".join(f"| {k} | {t['per_type'][k]['support']} | {t['per_type'][k]['precision']:.3f} | "
                     f"{t['per_type'][k]['recall']:.3f} | {t['per_type'][k]['f1']:.3f} |" for k in TYPES)
    card = f"""---
language:
- lg
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: token-classification
base_model: xlm-roberta-base
datasets:
- Conrad747/lg-ner
- masakhane/masakhaner2
tags:
- ner
- pii-detection
- luganda
- differential-privacy
- opacus
---

# Luganda PII detector trained with differential privacy (ε = {c['target_epsilon']:g})

Token-classification model that detects personally identifiable information (PII) in Luganda text:
PERSON, USERID (telephone numbers, email addresses, account and ID numbers), LOCATION, DATE, ORG and NORP,
in the BILOU scheme.

Developed by **Gideon Mpungu** for the MSc dissertation *Differentially Private Model for PII Detection in
Low-Resource Languages* (Makerere University, 2026). Code:
https://github.com/giddy-mpungu/DP_in_PII_Detection_for_Luganda

## Privacy guarantee

This model satisfies **(ε, δ) = ({p['epsilon_rdp']:.2f}, {p['delta']:g})-differential privacy with respect to
each individual sentence of the `lg-ner` training split** (2,085 sentences).

| | |
|---|---|
| Mechanism | DP-SGD (Opacus {m['environment'].get('opacus', '')}), per-example clipping and Gaussian noise |
| Noise multiplier σ | {p['noise_multiplier']:.3f} |
| Clipping norm C | {p['max_grad_norm']} |
| Sampling | Poisson, rate q = {p['sample_rate']:.4f} (expected batch {c['batch_size']}) |
| Steps | {p['optimizer_steps']} ({c['epochs']} epochs) |
| Accountant | RDP (ε = {p['epsilon_rdp']:.2f}); PRV cross-check ε = {p.get('epsilon_prv', float('nan')):.2f} |

Before private fine-tuning, the model was trained **non-privately on public data only**: `xlm-roberta-base`
fine-tuned on the Luganda part of MasakhaNER 2.0, with every sentence that shares an 8-word sequence with any
`lg-ner` sentence removed. The guarantee therefore covers the `lg-ner` sentences in full.

Limits of the guarantee: it is per sentence, not per person (a person appearing in k sentences is protected at
roughly kε); the learning rate was selected on the validation split without accounting for that selection; and
it says nothing about documents the model is *applied* to: PII that the model misses remains exposed.

## Evaluation (lg-ner test split, entity level, exact match)

Micro-F1 **{t['entity_micro']['f1']:.3f}** (95% bootstrap CI {t['entity_micro_f1_ci95'][0]:.3f}–{t['entity_micro_f1_ci95'][1]:.3f}),
precision {t['entity_micro']['precision']:.3f}, recall {t['entity_micro']['recall']:.3f}, macro-F1 {t['entity_macro_f1']:.3f}.
Membership-inference (loss-threshold) AUC on train vs. test sentences: {m['mia_auc_train_vs_test']:.3f} (0.5 = no signal).

| Category | Test mentions | Precision | Recall | F1 |
|---|---|---|---|---|
{rows}

Results for all privacy levels, three seeds each, are in the dissertation and the code repository.

## Intended use

Assisting human reviewers to find PII in Luganda text before it is shared or analysed. **Do not** use it as the
only safeguard: check the per-category recall above, and combine it with rule-based recognisers for structured
identifiers (telephone numbers, NINs, email addresses). Evaluated only on news-style, lower-cased, pre-tokenised
text; performance on conversational, SMS or transcribed speech is unknown.

## Usage

```python
from transformers import pipeline
ner = pipeline("token-classification", model="{repo_id}", aggregation_strategy="simple")
ner("ssemakula yategese ekivvulu mu buddu")
```

The training text was lower-cased; lower-case your input for best results.
"""
    with open(os.path.join(run_dir, "release_model", "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print("wrote", os.path.join(run_dir, "release_model", "README.md"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

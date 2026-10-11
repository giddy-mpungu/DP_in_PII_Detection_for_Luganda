# Differentially Private PII Detection for Luganda

Code and results for the MSc dissertation *Differentially Private Model for PII Detection in Low-Resource
Languages* (Gideon Mpungu, Makerere University, 2026).

The study fine-tunes an XLM-RoBERTa-base token classifier to detect personally identifiable information (PII)
in Luganda text, without privacy and with DP-SGD (Opacus) at ε = 8, 3 and 1 (δ = 10⁻⁵). It evaluates utility
at entity level and per PII category, alongside the formal privacy guarantee and an empirical
membership-inference test.

## Results (lg-ner test split, entity level)

| Model | ε | Micro-F1 | Macro-F1 | Membership-inference AUC |
|---|---|---|---|---|
| `Conrad747/luganda-ner-v6` as released (token-weighted F1: 0.943) | – | 0.751 | 0.740 | 0.661 |
| Non-private, from `xlm-roberta-base` | ∞ | 0.732 ± 0.015 | 0.723 | 0.696 |
| DP-SGD, from `xlm-roberta-base` | 8 / 3 / 1 | 0.000 (collapses) | 0.000 | 0.50 |
| Public warm start only (MasakhaNER 2.0 Luganda) | – | 0.264 | 0.162 | 0.500 |
| Warm start + non-private | ∞ | 0.767 ± 0.023 | 0.783 | 0.719 |
| Warm start + DP-SGD | 8 | 0.373 ± 0.001 | 0.213 | 0.515 |
| Warm start + DP-SGD | 3 | 0.341 ± 0.004 | 0.193 | 0.512 |
| Warm start + DP-SGD | 1 | 0.307 (one seed) | 0.177 | 0.508 |

Main findings:
* DP-SGD started from a raw multilingual backbone learns nothing at any budget; from a public warm start it
  works, with utility falling monotonically as privacy tightens.
* The loss of utility is uneven across categories: USERID (phone numbers, emails, IDs), NORP and DATE are lost
  entirely under DP, because the public data does not cover them.
* DP reduces membership-inference success from AUC ≈ 0.70 to chance.

## Released models

Private models trained from the public warm start, each with a model card stating its guarantee and per-category
scores: [luganda-pii-dp-eps8](https://huggingface.co/e4gl33y3/luganda-pii-dp-eps8) ·
[luganda-pii-dp-eps3](https://huggingface.co/e4gl33y3/luganda-pii-dp-eps3) ·
[luganda-pii-dp-eps1](https://huggingface.co/e4gl33y3/luganda-pii-dp-eps1).

## Contents

| Path | What it is |
|---|---|
| `experiments/dp_pii_luganda.py` | Single script: data audit, preprocessing, non-private and DP-SGD training, privacy accounting, evaluation, error extraction, membership inference |
| `experiments/run_experiments_colab.ipynb` | Runs every experiment on a Colab GPU (resumable; results saved to Google Drive) |
| `experiments/annotation_agreement.py` | Inter-annotator agreement from the upstream multi-annotator files |
| `experiments/analyze.py` | Builds every table, figure and number in the dissertation from `results/` |
| `experiments/make_model_card.py` | Writes the Hugging Face model card of a released private model from its results |
| `results/` | `metrics.json` (configuration, environment, history, privacy accounting, all metrics) and `losses.json` for every run |
| `original_notebooks/` | Notebooks of the earlier submitted version (kept for reference; see note below) |

`errors.json` files (test-set errors with their sentences) are produced by the script but not published here,
because they contain text from the lg-ner corpus.

## Experiments

| Plan | Runs |
|---|---|
| `main` | Non-private baseline (3 seeds); DP learning-rate sweep at ε = 8; DP at ε = 8, 3, 1. All start from `xlm-roberta-base`, which has never seen `lg-ner` labels. |
| `public` | Non-private warm start on public Luganda NER data (MasakhaNER 2.0, with every sentence sharing an 8-gram with `lg-ner` removed), then non-private and DP fine-tuning on `lg-ner` |
| `public_pe` | Optional: as `public`, but DP updates only the top two transformer layers and the classifier |
| `leakage` | `Conrad747/luganda-ner-v6` evaluated as released (entity and token metrics, train-split F1, membership inference) |
| `audit` | Dataset statistics, split overlap, truncation |

```bash
pip install -r experiments/requirements.txt
python experiments/dp_pii_luganda.py --plan main   --out results
python experiments/dp_pii_luganda.py --plan public --out results
python experiments/dp_pii_luganda.py --plan leakage --out results
python experiments/dp_pii_luganda.py --plan audit  --out results
python experiments/annotation_agreement.py <path-to-luganda-ner-data> results/annotation_agreement.json
python experiments/analyze.py results <latex-dir>
```

Runs whose `metrics.json` exists are skipped, so an interrupted run can be resumed. Every run records its
configuration, package versions and hardware. Seeds: 42, 1234, 2024. The reported runs used a Tesla T4 GPU,
Python 3.13, PyTorch 2.11, Transformers 4.57.6 and Opacus 1.6.0.

## Data

* **lg-ner** ([Conrad747/lg-ner](https://huggingface.co/datasets/Conrad747/lg-ner), sourced from
  [conradsuuna/luganda-ner-data](https://github.com/conradsuuna/luganda-ner-data)): 2,085 / 358 / 536 Luganda
  sentences; PERSON, NORP, DATE, USERID, ORG, LOCATION in BILOU format. No licence is stated, so the data is not
  redistributed here; the scripts download it.
* **MasakhaNER 2.0 Luganda** ([masakhane/masakhaner2](https://huggingface.co/datasets/masakhane/masakhaner2)),
  used only as public warm-start data.

## Note on the earlier version

The notebooks in `original_notebooks/` fine-tuned `Conrad747/luganda-ner-v6` with DP-SGD. That checkpoint was
already fine-tuned **non-privately on the same lg-ner training data**, so the DP guarantee of those models does
not cover the lg-ner training sentences. They also reported token-level F1 weighted by label frequency
(including `O`), which is dominated by the non-entity class: a model that predicts no PII at all scores 0.69 on
it. The revised experiments fix both issues.

## Privacy statement

Models trained with the `main` and `public` plans satisfy (ε, δ)-differential privacy with respect to individual
**sentences** of the lg-ner training set (Opacus RDP accountant, Poisson sampling, δ = 10⁻⁵). The guarantee does
not cover a person who appears in several sentences beyond the usual group-privacy bound, and it says nothing
about the documents a deployed detector is applied to: missed PII in those documents remains exposed.

# Differentially Private PII Detection for Luganda

Code and results for the MSc dissertation *Differentially Private Model for PII Detection in Low-Resource
Languages* (Mpungu Gideon, Makerere University, 2026).

The study fine-tunes an XLM-RoBERTa-base token classifier to detect personally identifiable information (PII)
in Luganda text, without privacy and with DP-SGD (Opacus) at ε = 8, 3 and 1 (δ = 10⁻⁵), and evaluates utility
at entity level and per PII category, alongside the formal privacy guarantee and an empirical
membership-inference test.

## Contents

| Path | What it is |
|---|---|
| `experiments/dp_pii_luganda.py` | Single script: data audit, preprocessing, non-private and DP-SGD training, privacy accounting, evaluation, error extraction, membership inference |
| `experiments/run_experiments_colab.ipynb` | Runs every experiment on a free Colab GPU (resumable; results saved to Google Drive) |
| `experiments/annotation_agreement.py` | Inter-annotator agreement from the upstream multi-annotator files |
| `experiments/analyze.py` | Builds every table, figure and number in the dissertation from `results/` |
| `results/` | Raw `metrics.json` / `errors.json` for every run |
| `original_notebooks/` | Notebooks of the earlier submitted version (kept for reference; see note below) |

## Experiments

| Plan | Runs |
|---|---|
| `main` | Non-private baseline (3 seeds); DP learning-rate sweep at ε = 8; DP at ε = 8, 3, 1 (3 seeds each). All start from `xlm-roberta-base`, which has never seen `lg-ner` labels. |
| `public` | Non-private warm start on public Luganda NER data (MasakhaNER 2.0, with every sentence sharing an 8-gram with `lg-ner` removed), then DP fine-tuning on `lg-ner` |
| `repro` | Re-run of the submitted configuration (DP fine-tuning from `Conrad747/luganda-ner-v6`), plus that model evaluated as released |
| `audit` | Dataset statistics, split overlap, truncation |

```bash
pip install -r experiments/requirements.txt
python experiments/dp_pii_luganda.py --plan all --out results     # ~4-6 h on a T4 GPU
python experiments/annotation_agreement.py <path-to-luganda-ner-data> results/annotation_agreement.json
python experiments/analyze.py results <latex-dir>
```

Every run records its configuration, package versions and hardware in `metrics.json`. Seeds: 42, 1234, 2024.

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
(including `O`), which is dominated by the non-entity class. The revised experiments fix both issues; the
`repro` plan quantifies the difference.

## Privacy statement

Models trained with the `main` and `public` plans satisfy (ε, δ)-differential privacy with respect to individual
**sentences** of the lg-ner training set (Opacus RDP accountant, Poisson sampling, δ = 10⁻⁵). The guarantee does
not cover a person who appears in several sentences beyond the usual group-privacy bound, and it says nothing
about the documents a deployed detector is applied to: missed PII in those documents remains exposed.

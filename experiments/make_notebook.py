"""Builds run_experiments_colab.ipynb with dp_pii_luganda.py embedded, so only one file needs uploading."""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "dp_pii_luganda.py"), encoding="utf-8").read()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# DP PII detection for Luganda: full experiment run\n"
       "1. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.\n"
       "2. Approve the Google Drive prompt. Results are written to `MyDrive/dp_pii_results`, so if the session "
       "disconnects just **Run all** again: finished runs are skipped.\n"
       "3. When the last cell finishes, download `dp_pii_results.zip` (also saved in Drive) and put it in the thesis folder."),
    code("!nvidia-smi -L || echo 'NO GPU: switch runtime to T4 GPU'"),
    # versions pinned to those the pipeline was tested with
    code("!pip -q install \"transformers==4.57.6\" \"opacus==1.6.0\" \"datasets>=3,<6\"\n"
         "import torch, transformers, opacus; print(torch.__version__, transformers.__version__, opacus.__version__, torch.cuda.is_available())"),
    code("from google.colab import drive\ndrive.mount('/content/drive')\n"
         "OUT = '/content/drive/MyDrive/dp_pii_results'\nimport os; os.makedirs(OUT, exist_ok=True)"),
    code("%%writefile dp_pii_luganda.py\n" + src),
    md("Experiment B (clean `xlm-roberta-base` backbone): non-private baseline, learning-rate sweep, then the three "
       "privacy levels × 3 seeds. Roughly 1.5–3 h on a T4."),
    code("!python dp_pii_luganda.py --plan main --out $OUT"),
    md("Experiment C: warm start on PUBLIC Luganda NER data (MasakhaNER 2.0, overlap with lg-ner removed), then DP "
       "fine-tuning on lg-ner. Roughly 1–2 h."),
    code("!python dp_pii_luganda.py --plan public --out $OUT"),
    md("Experiment A: the submitted configuration (v6 start) re-run with one seed and entity-level metrics, plus the "
       "leakage check. Roughly 30–45 min."),
    code("!python dp_pii_luganda.py --plan repro --out $OUT"),
    code("!python dp_pii_luganda.py --plan audit --out $OUT\n"
         "# the warm-start checkpoint (~1 GB) is not needed for the thesis: leave it out of the zip\n"
         "!cd /content/drive/MyDrive && zip -qr dp_pii_results.zip dp_pii_results -x '*/model/*' && ls -lh dp_pii_results.zip\n"
         "from google.colab import files; files.download('/content/drive/MyDrive/dp_pii_results.zip')"),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(os.path.join(here, "run_experiments_colab.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote run_experiments_colab.ipynb")

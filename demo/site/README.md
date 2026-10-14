---
title: Luganda PII Detector (DP)
emoji: 🔒
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: Luganda PII detection with DP-trained models, in-browser
---

# Luganda PII detector, trained with differential privacy

A browser-only demo of the private models from the MSc dissertation *Differentially Private Model for PII Detection
in Low-Resource Languages* (Gideon Mpungu, Makerere University, 2026).

* Detection runs in the visitor's browser (transformers.js + ONNX Runtime Web); typed text is never uploaded.
* The page loads the browser copies (`onnx/model_quantized.onnx`) of the released v2 models
  [e4gl33y3/luganda-pii-dp-v2-eps8](https://huggingface.co/e4gl33y3/luganda-pii-dp-v2-eps8),
  [eps3](https://huggingface.co/e4gl33y3/luganda-pii-dp-v2-eps3), [eps1](https://huggingface.co/e4gl33y3/luganda-pii-dp-v2-eps1)
  and [eps0.5](https://huggingface.co/e4gl33y3/luganda-pii-dp-v2-eps0.5): weights stored as int8 plus per-channel
  scales (activations in float); test F1 within 0.003 of the full-precision models.
* `rdp.js` is a line-by-line port of the Opacus RDP accountant; it reproduces the RDP ε of every DP training run
  to within 1e-7 (the certified ε comes from the tighter PRV accountant used to calibrate the noise).
* `rules.js` holds the rule-based ID and date recognisers.

Code and results: https://github.com/giddy-mpungu/DP_in_PII_Detection_for_Luganda (folder `demo/`).

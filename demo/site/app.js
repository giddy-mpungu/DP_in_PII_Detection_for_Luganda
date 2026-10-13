import { AutoTokenizer, AutoModelForTokenClassification, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";
import { epsilon } from "./rdp.js";
import { ruleSpans } from "./rules.js";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("./models/", location.href).href;

const TYPES = ["PERSON", "NORP", "DATE", "USERID", "ORG", "LOCATION"];
const TYPE_LABEL = { PERSON: "Person", NORP: "Group / affiliation", DATE: "Date", USERID: "ID / phone / email",
  ORG: "Organisation", LOCATION: "Location" };
const RULE_TYPES = new Set(["USERID", "DATE"]);
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));
const pct = (x) => `${Math.round(100 * x)}%`;

let DATA, labels, eps = "8";
const loaded = {}; // eps -> {tok, model}

// ---------------------------------------------------------------- model
async function getModel(e) {
  if (!loaded[e]) {
    loaded[e] = (async () => {
      const opts = { dtype: "q8", device: "wasm", progress_callback: (p) => {
        if (p.status === "progress" && p.file.endsWith(".onnx")) setStatus(`Downloading the ε = ${e} model… ${Math.round(p.progress)}% (once; then cached)`);
      } };
      const tok = await AutoTokenizer.from_pretrained(`eps${e}`);
      const model = await AutoModelForTokenClassification.from_pretrained(`eps${e}`, opts);
      labels = model.config.id2label;
      return { tok, model };
    })();
  }
  return loaded[e];
}

// lower-case and split off punctuation, as in the training corpus
function splitWords(text) {
  return [...text.toLowerCase().matchAll(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu)].map((m) => ({ text: m[0], start: m.index, end: m.index + m[0].length }));
}

// word-level BILOU tags -> [{type, start, end}] in characters
function decode(words, tags) {
  const out = [];
  let cur = null;
  tags.forEach((tag, i) => {
    const [p, t] = tag === "O" ? ["O", null] : tag.split("-");
    if (cur && (p === "O" || p === "B" || p === "U" || t !== cur.type)) { out.push(cur); cur = null; }
    if (p !== "O") {
      if (!cur) cur = { type: t, start: words[i].start, end: words[i].end, source: "model" };
      else cur.end = words[i].end;
      if (p === "L" || p === "U") { out.push(cur); cur = null; }
    }
  });
  if (cur) out.push(cur);
  return out;
}

async function modelSpans(text, words) {
  const { tok, model } = await getModel(eps);
  const tags = new Array(words.length).fill("O");
  // chunks of whole words, at most ~400 subword tokens each (XLM-R takes 512)
  let i = 0;
  while (i < words.length) {
    const ids = [0], first = [];
    let j = i;
    while (j < words.length) {
      const pieces = tok.encode(words[j].text, { add_special_tokens: false });
      if (ids.length + pieces.length > 400 && j > i) break;
      first.push(ids.length);
      ids.push(...pieces);
      j++;
    }
    ids.push(2);
    const shape = [1, ids.length];
    const { logits } = await model({
      input_ids: new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), shape),
      attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), shape),
    });
    const n = logits.dims[2], data = logits.data;
    first.forEach((pos, k) => {
      let best = 0;
      for (let c = 1; c < n; c++) if (data[pos * n + c] > data[pos * n + best]) best = c;
      tags[i + k] = labels[best];
    });
    i = j;
  }
  return decode(words, tags);
}

// "replace" strategy chosen on validation: rules own USERID and DATE; a rule span replaces overlapping model spans
function combine(model, rules) {
  const keep = model.filter((m) => !RULE_TYPES.has(m.type) && !rules.some((r) => m.start < r.end && m.end > r.start));
  return keep.concat(rules).sort((a, b) => a.start - b.start);
}

// ---------------------------------------------------------------- detector UI
function setStatus(msg, busy = true) {
  $("status").textContent = msg;
  $("status").dataset.busy = busy ? "1" : "";
}

function render(text, spans) {
  const out = $("output");
  out.replaceChildren();
  let pos = 0;
  for (const s of spans) {
    if (s.start > pos) out.append(document.createTextNode(text.slice(pos, s.start)));
    const mark = document.createElement("mark");
    mark.className = `t-${s.type}`;
    mark.textContent = text.slice(s.start, s.end);
    mark.title = `${TYPE_LABEL[s.type]} (${s.source === "rule" ? "rule" : "DP model"})`;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = s.type + (s.source === "rule" ? " · rule" : "");
    mark.append(tag);
    out.append(mark);
    pos = s.end;
  }
  out.append(document.createTextNode(text.slice(pos)));
  const list = $("found");
  list.replaceChildren();
  if (!spans.length) {
    const li = document.createElement("li");
    li.textContent = "No PII detected. That does not mean there is none: the private models miss more than half of PII.";
    list.append(li);
  }
  for (const s of spans) {
    const li = document.createElement("li");
    const b = document.createElement("b");
    b.textContent = text.slice(s.start, s.end);
    li.append(b, ` — ${TYPE_LABEL[s.type]}, found by ${s.source === "rule" ? "a rule (no training data)" : `the ε = ${eps} DP model`}`);
    list.append(li);
  }
  $("redacted").textContent = spans.reduceRight((t, s) => t.slice(0, s.start) + `[${s.type}]` + t.slice(s.end), text);
}

async function detect() {
  const text = $("input").value;
  if (!text.trim()) return;
  $("run").disabled = true;
  setStatus(`Running the ε = ${eps} model in your browser…`);
  try {
    const words = splitWords(text);
    const model = await modelSpans(text, words);
    const spans = $("rules").checked ? combine(model, ruleSpans(text.toLowerCase(), words)) : model;
    render(text, spans);
    $("result").hidden = false;
    setStatus("Done. Nothing you typed left this device.", false);
  } catch (err) {
    console.error(err);
    setStatus(`Could not run the model: ${err.message}`, false);
  } finally {
    $("run").disabled = false;
  }
}

// ---------------------------------------------------------------- privacy certificate
function certificate() {
  const m = DATA.models[eps], p = m.privacy;
  $("cert-eps").textContent = fmt(p.epsilon_prv, 2);
  $("cert-rdp").textContent = fmt(p.epsilon_rdp, 2);
  $("cert-repo").textContent = m.repo;
  $("cert-repo").href = `https://huggingface.co/${m.repo}`;
  const rows = [
    ["Privacy unit", "one training sentence (add or remove)"],
    ["Training set", `${p.train_sentences.toLocaleString()} lg-ner sentences`],
    ["Sampling", `Poisson, rate q = ${fmt(p.sample_rate, 4)} (expected batch ${p.expected_batch})`],
    ["Noise multiplier σ", fmt(p.noise_multiplier, 4)],
    ["Clipping norm C", fmt(p.max_grad_norm, 1)],
    ["Optimiser steps T", `${p.steps} (${p.epochs} epochs)`],
    ["δ", p.delta.toExponential(0)],
    ["ε, PRV accountant (certified)", fmt(p.epsilon_prv, 4)],
    ["ε, RDP accountant (looser bound)", fmt(p.epsilon_rdp, 4)],
  ];
  const tb = $("cert-table");
  tb.replaceChildren(...rows.map(([k, v]) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.textContent = k;
    const td = document.createElement("td"); td.textContent = v;
    tr.append(th, td);
    return tr;
  }));
  $("calc-sigma").value = String(p.noise_multiplier);
  $("calc-q").value = String(p.sample_rate);
  $("calc-steps").value = p.steps;
  $("calc-delta").value = p.delta;
  recompute();
}

function recompute() {
  const s = +$("calc-sigma").value, q = +$("calc-q").value, T = Math.round(+$("calc-steps").value), d = +$("calc-delta").value;
  if (!(s > 0 && q > 0 && q < 1 && T > 0 && d > 0 && d < 1)) { $("calc-out").textContent = "–"; return; }
  const t0 = performance.now();
  const { epsilon: e, order } = epsilon(q, s, T, d);
  $("calc-out").textContent = fmt(e, 4);
  $("calc-note").textContent = `best Rényi order α = ${order}; computed in ${Math.round(performance.now() - t0)} ms in your browser`;
}

// ---------------------------------------------------------------- evidence: membership inference + accuracy
function miaChart(el, h, title) {
  const W = 320, H = 150, P = 24, n = h.members.length;
  const max = Math.max(...h.members, ...h.nonmembers);
  const x = (i) => P + (i / n) * (W - 2 * P), y = (v) => H - P - (v / max) * (H - 2 * P);
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)} L${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  el.innerHTML = `<figcaption><b>${title}</b> · attack AUC ${fmt(h.auc, 2)}</figcaption>
  <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}: loss distribution of training sentences versus unseen sentences, attack AUC ${fmt(h.auc, 2)}">
    <line x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}" class="axis"/>
    <path d="${path(h.nonmembers)}" class="nonmem"/><path d="${path(h.members)}" class="mem"/>
    <text x="${P}" y="${H - 6}" class="lbl">loss 0</text><text x="${W - P}" y="${H - 6}" class="lbl" text-anchor="end">3+</text>
  </svg>`;
}

function accuracy() {
  const tb = $("acc-table"), np = DATA.reference.nonprivate;
  const row = (cells) => { const tr = document.createElement("tr"); cells.forEach((c, i) => { const td = document.createElement(i ? "td" : "th"); td.textContent = c; tr.append(td); }); return tr; };
  tb.replaceChildren(
    row(["Non-private (not released)", "∞", fmt(np.f1), pct(np.recall), "–", fmt(np.mia_auc, 2)]),
    ...["8", "3", "1", "0.5"].map((e) => {
      const m = DATA.models[e];
      return row([`DP model, ε = ${e}`, e, fmt(m.test.f1), pct(m.test.recall), `${fmt(m.test.with_rules_f1)} / ${pct(m.test.with_rules_recall)}`, fmt(m.mia_auc, 2)]);
    }));
  const pt = $("type-table");
  const m = DATA.models[eps].test;
  pt.replaceChildren(...TYPES.map((t) => row([TYPE_LABEL[t], pct(np.per_type_recall[t]), pct(m.per_type_recall[t]), pct(m.with_rules_per_type_recall[t])])));
  $("type-eps").textContent = eps;
  $("browser-f1").textContent = fmt(DATA.models[eps].test.browser_model_f1);
  $("full-f1").textContent = fmt(DATA.models[eps].test.f1);
}

// ---------------------------------------------------------------- wiring
function selectEps(e) {
  eps = e;
  document.querySelectorAll("[data-eps]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.eps === e)));
  certificate();
  accuracy();
  if (!$("result").hidden) detect();
}

async function main() {
  DATA = await (await fetch("data.json")).json();
  document.querySelectorAll("[data-eps]").forEach((b) => b.addEventListener("click", () => selectEps(b.dataset.eps)));
  document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => { $("input").value = b.dataset.example; detect(); }));
  $("run").addEventListener("click", detect);
  $("rules").addEventListener("change", () => { if (!$("result").hidden) detect(); });
  ["calc-sigma", "calc-q", "calc-steps", "calc-delta"].forEach((id) => $(id).addEventListener("input", recompute));
  $("calc-reset").addEventListener("click", certificate);
  miaChart($("mia-np"), DATA.mia.nonprivate, "Non-private model");
  miaChart($("mia-dp8"), DATA.mia.dp8, "DP model, ε = 8");
  miaChart($("mia-dp1"), DATA.mia.dp1, "DP model, ε = 0.5");
  selectEps("8");
  setStatus("The model (about 290 MB) downloads on first use and is then cached by your browser.", false);
}

main();

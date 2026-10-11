"""Aggregate run results into the thesis: LaTeX macros, tables and figures.

usage: python analyze.py <results dir> <latex dir>
writes <latex>/generated/*.tex and <latex>/figures/*.pdf
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPES = ["PERSON", "NORP", "DATE", "USERID", "ORG", "LOCATION"]
CONFIGS = [("nonprivate", "Non-private", None), ("dp_eps8", "High Utility", 8), ("dp_eps3", "Better Privacy", 3),
           ("dp_eps1", "Strong Privacy", 1)]
# reference palette: ordinal blue ramp for privacy level, secondary ink for the baseline
COLORS = {"nonprivate": "#52514e", "dp_eps8": "#184f95", "dp_eps3": "#3987e5", "dp_eps1": "#86b6ef"}
SPLIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def load(res):
    runs = {}
    for p in glob.glob(os.path.join(res, "*", "metrics.json")):
        with open(p) as f:
            runs[os.path.basename(os.path.dirname(p))] = json.load(f)
    return runs


def group(runs, prefix):
    return [m for n, m in sorted(runs.items()) if n.startswith(prefix + "_seed")]


def ms(vals, nd=3, pct=False):
    v = np.array(vals, dtype=float) * (100 if pct else 1)
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{v.mean():.{nd}f} $\\pm$ {v.std(ddof=1):.{nd}f}"


def mean(vals):
    return float(np.mean(vals)) if len(vals) else float("nan")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def fmt_int(n):
    return f"{int(n):,}"


def main(res, tex):
    gen, figs = os.path.join(tex, "generated"), os.path.join(tex, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    runs = load(res)
    G = {k: group(runs, k) for k, _, _ in CONFIGS}
    macros = {}

    # ---------------------------------------------------------------- data audit
    audit = json.load(open(os.path.join(res, "data_audit", "data_audit.json")))
    sp = audit["splits"]
    ents = {k: sum(sp[k]["entities_by_type"].values()) for k in sp}
    macros.update(AuditTrainEnts=fmt_int(ents["train"]), AuditValEnts=fmt_int(ents["validation"]),
                  AuditTestEnts=fmt_int(ents["test"]), AuditAllEnts=fmt_int(sum(ents.values())),
                  AuditOShare=f"{100 * sp['train']['O_token_share']:.1f}\\%",
                  AuditTruncTrain=str(sp["train"]["sentences_over_128_subwords"]),
                  AuditTruncTest=str(sp["test"]["sentences_over_128_subwords"]),
                  AuditNoEntTrain=fmt_int(sp["train"]["sentences_without_entities"]),
                  AuditHashTrain=f"{100 * sp['train']['words_with_##_prefix'] / sp['train']['words']:.1f}\\%")
    agp = os.path.join(res, "annotation_agreement.json")
    if os.path.exists(agp):
        ag = json.load(open(agp))["agreement"]
        macros.update(AgreeKappa=f"{ag['token_type_cohen_kappa']:.3f}", AgreeF=f"{ag['entity_pairwise_f1']:.3f}",
                      AgreeSents=fmt_int(ag["sentences_annotated_more_than_once"]),
                      AgreePairs=fmt_int(ag["annotation_pairs"]),
                      AgreeIdentical=f"{100 * ag['identical_pairs_share']:.1f}\\%")
    rows = []
    for t in TYPES:
        c = [sp[k]["entities_by_type"][t] for k in ("train", "validation", "test")]
        rows.append(f"{t} & {' & '.join(fmt_int(x) for x in c)} & {100 * c[0] / ents['train']:.1f}\\% \\\\")
    write(os.path.join(gen, "audit_table.tex"),
          "\\begin{table}[ht]\n\\centering\n\\caption{Entity mentions by type and split (data audit).}\n"
          "\\label{tab:audit}\n\\begin{tabular}{lrrrr}\n\\toprule\nType & Train & Validation & Test & Share of train \\\\\n\\midrule\n"
          + "\n".join(rows) + f"\n\\midrule\nAll entities & {fmt_int(ents['train'])} & {fmt_int(ents['validation'])} & "
          f"{fmt_int(ents['test'])} & 100\\% \\\\\nSentences with no entity & {fmt_int(sp['train']['sentences_without_entities'])} & "
          f"{fmt_int(sp['validation']['sentences_without_entities'])} & {fmt_int(sp['test']['sentences_without_entities'])} & \\\\\n"
          f"Words labelled \\texttt{{O}} & {100 * sp['train']['O_token_share']:.1f}\\% & {100 * sp['validation']['O_token_share']:.1f}\\% & "
          f"{100 * sp['test']['O_token_share']:.1f}\\% & \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    x = np.arange(len(TYPES))
    w = 0.26
    for i, (k, lab) in enumerate([("train", "Training"), ("validation", "Validation"), ("test", "Test")]):
        v = [sp[k]["entities_by_type"][t] for t in TYPES]
        ax.bar(x + (i - 1) * w, v, w - 0.03, color=SPLIT_COLORS[i], label=lab)
    ax.set_xticks(x, TYPES)
    ax.set_ylabel("Entity mentions")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "entity_distribution.pdf"))
    plt.close(fig)

    # ---------------------------------------------------------------- main results
    any_dp = next((m for k in ("dp_eps8", "dp_eps3", "dp_eps1") for m in G[k]), None)
    if any_dp:
        pr = any_dp["privacy"]
        macros.update(SampleRate=f"{pr['sample_rate']:.4f}", DPSteps=fmt_int(pr["optimizer_steps"]),
                      DPEpochs=str(any_dp["config"]["epochs"]), SelectedLR=sci(any_dp["config"]["lr"]),
                      NTrainable=f"{any_dp['trainable_params'] / 1e6:.1f} million")
    for k, name in (("dp_eps8", "SigmaEpsEight"), ("dp_eps3", "SigmaEpsThree"), ("dp_eps1", "SigmaEpsOne")):
        macros[name] = f"{G[k][0]['privacy']['noise_multiplier']:.3f}" if G[k] else "--"

    def f1(m):
        return m["test"]["entity_micro"]["f1"]

    rows, summary = [], {}
    for k, label, eps in CONFIGS:
        g = G[k]
        if not g:
            continue
        e_rdp = [m["privacy"]["epsilon_rdp"] for m in g] if eps else []
        s = {"P": [m["test"]["entity_micro"]["precision"] for m in g],
             "R": [m["test"]["entity_micro"]["recall"] for m in g],
             "F": [f1(m) for m in g], "M": [m["test"]["entity_macro_f1"] for m in g],
             "T": [m["test"]["token_weighted_incl_O"]["f1"] for m in g],
             "A": [m["mia_auc_train_vs_test"] for m in g],
             "CI": [m["test"].get("entity_micro_f1_ci95", [np.nan, np.nan]) for m in g]}
        summary[k] = s
        eps_txt = "$\\infty$" if eps is None else f"{np.mean(e_rdp):.2f}"
        rows.append(f"{label} & {eps_txt} & {ms(s['P'])} & {ms(s['R'])} & {ms(s['F'])} & {ms(s['M'])} & {ms(s['A'])} \\\\")
        tag = {"nonprivate": "Base", "dp_eps8": "EpsEight", "dp_eps3": "EpsThree", "dp_eps1": "EpsOne"}[k]
        v = np.array(s["F"])
        macros[f"{tag}F"] = f"{v.mean():.3f}"
        macros[f"{tag}Fsd"] = f"{v.std(ddof=1):.3f}" if len(v) > 1 else "0"
        macros[f"{tag}Macro"] = f"{np.mean(s['M']):.3f}"
        macros[f"{tag}P"] = f"{np.mean(s['P']):.3f}"
        macros[f"{tag}R"] = f"{np.mean(s['R']):.3f}"
        macros[f"{tag}Missed"] = f"{100 * (1 - np.mean(s['R'])):.1f}\\%"
        macros[f"{tag}Tok"] = f"{np.mean(s['T']):.3f}"
        macros[f"{tag}MIA"] = f"{np.mean(s['A']):.3f}"
        ci = np.array(s["CI"], dtype=float)
        macros[f"{tag}CI"] = f"[{np.nanmean(ci[:, 0]):.3f}, {np.nanmean(ci[:, 1]):.3f}]"
        if eps:
            macros[f"{tag}Achieved"] = f"{np.mean(e_rdp):.2f}"
            prv = [m["privacy"].get("epsilon_prv") for m in g if m["privacy"].get("epsilon_prv")]
            macros[f"{tag}PRV"] = f"{np.mean(prv):.2f}" if prv else "--"
            macros[f"{tag}Drop"] = f"{100 * (1 - v.mean() / np.mean(summary['nonprivate']['F'])):.1f}\\%" if "nonprivate" in summary else "--"
        macros[f"{tag}Time"] = f"{np.mean([m['runtime_min'] for m in g]):.0f}"
    write(os.path.join(gen, "main_results_table.tex"),
          "\\begin{table}[ht]\n\\centering\n\\caption{Test-set results of the main experiment (entity level, mean $\\pm$ "
          "standard deviation over three seeds). $\\varepsilon$ is the achieved value at $\\delta = 10^{-5}$; MIA AUC is the "
          "membership-inference attack success (0.5 = chance).}\n\\label{tab:main_results}\n\\begin{FitTable}\n"
          "\\begin{tabular}{lcccccc}\n\\toprule\nModel & $\\varepsilon$ & Precision & Recall & Micro-F1 & Macro-F1 & MIA AUC \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")

    # per-type F1 and recall
    for metric, cap, lab in (("f1", "F1-score", "tab:per_type_f1"), ("recall", "recall (share of PII found)", "tab:per_type_recall")):
        rows = []
        for t in TYPES:
            sup = next((m["test"]["per_type"][t]["support"] for g in G.values() for m in g), 0)
            cells = [ms([m["test"]["per_type"][t][metric] for m in G[k]], 2) for k, _, _ in CONFIGS]
            rows.append(f"{t} ({sup}) & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule\nMacro average & " + " & ".join(
            ms([np.mean([m["test"]["per_type"][t][metric] for t in TYPES]) for m in G[k]], 2) for k, _, _ in CONFIGS) + " \\\\")
        write(os.path.join(gen, f"per_type_{metric}_table.tex"),
              f"\\begin{{table}}[ht]\n\\centering\n\\caption{{Per-category test {cap} (mean $\\pm$ s.d.\\ over three seeds; "
              f"test support in brackets).}}\n\\label{{{lab}}}\n\\begin{{FitTable}}\n\\begin{{tabular}}{{lcccc}}\n\\toprule\n"
              "PII category & Non-private & $\\varepsilon=8$ & $\\varepsilon=3$ & $\\varepsilon=1$ \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")
    for k, _, _ in CONFIGS:
        tag = {"nonprivate": "Base", "dp_eps8": "EpsEight", "dp_eps3": "EpsThree", "dp_eps1": "EpsOne"}[k]
        for t in TYPES:
            if G[k]:
                macros[f"{tag}{t.title()}F"] = f"{np.mean([m['test']['per_type'][t]['f1'] for m in G[k]]):.2f}"
                macros[f"{tag}{t.title()}R"] = f"{np.mean([m['test']['per_type'][t]['recall'] for m in G[k]]):.2f}"

    # privacy accounting table
    rows = []
    for k, label, eps in CONFIGS[1:]:
        if not G[k]:
            continue
        m0 = G[k][0]["privacy"]
        rows.append(f"{label} & {eps} & {m0['noise_multiplier']:.3f} & {m0['max_grad_norm']} & {m0['sample_rate']:.4f} & "
                    f"{fmt_int(m0['optimizer_steps'])} & {ms([m['privacy']['epsilon_rdp'] for m in G[k]], 2)} & "
                    f"{ms([m['privacy'].get('epsilon_prv', np.nan) for m in G[k]], 2)} \\\\")
    write(os.path.join(gen, "privacy_table.tex"),
          "\\begin{table}[ht]\n\\centering\n\\caption{Privacy accounting of the private models ($\\delta = 10^{-5}$, "
          "$N = 2{,}085$). $\\sigma$ is calibrated by Opacus to the target $\\varepsilon$; the achieved $\\varepsilon$ is "
          "computed with the RDP accountant and cross-checked with the PRV accountant.}\n\\label{tab:privacy}\n"
          "\\begin{FitTable}\n\\begin{tabular}{lccccccc}\n\\toprule\nModel & Target $\\varepsilon$ & $\\sigma$ & $C$ & $q$ & "
          "Steps $T$ & $\\varepsilon$ (RDP) & $\\varepsilon$ (PRV) \\\\\n\\midrule\n" + "\n".join(rows)
          + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")

    # epsilon by epoch (seed 42) + learning curves
    seed42 = {k: runs.get(f"{k}_seed42") for k, _, _ in CONFIGS}
    if all(seed42[k] for k in ("dp_eps8", "dp_eps3", "dp_eps1")):
        ep = len(seed42["dp_eps8"]["history"])
        rows = []
        for i in range(ep):
            cells = []
            for k in ("dp_eps8", "dp_eps3", "dp_eps1"):
                h = seed42[k]["history"][i]
                cells += [f"{h['epsilon_cumulative']:.2f}", f"{h['val_entity_f1']:.3f}"]
            rows.append(f"{i + 1} & " + " & ".join(cells) + " \\\\")
        write(os.path.join(gen, "epsilon_epochs_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Cumulative privacy loss $\\varepsilon$ ($\\delta=10^{-5}$) and "
              "validation entity micro-F1 after each epoch (seed 42). $\\varepsilon$ grows with every step and the final "
              "row is the guarantee of the released model.}\n\\label{tab:eps_epochs}\n\\begin{FitTable}\n"
              "\\begin{tabular}{ccccccc}\n\\toprule\n & \\multicolumn{2}{c}{High Utility} & \\multicolumn{2}{c}{Better Privacy} "
              "& \\multicolumn{2}{c}{Strong Privacy} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
              "Epoch & $\\varepsilon$ & Val.\\ F1 & $\\varepsilon$ & Val.\\ F1 & $\\varepsilon$ & Val.\\ F1 \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for k, label, eps in CONFIGS:
        g = G[k]
        if not g:
            continue
        n = min(len(m["history"]) for m in g)
        xs = np.arange(1, n + 1)
        name = label if eps is None else f"$\\varepsilon$={eps}"
        for ax, key in ((axes[0], "val_entity_f1"), (axes[1], "val_loss")):
            arr = np.array([[m["history"][i][key] for i in range(n)] for m in g])
            ax.plot(xs, arr.mean(0), color=COLORS[k], lw=2, label=name)
            if len(g) > 1:
                ax.fill_between(xs, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=COLORS[k], alpha=0.15, lw=0)
        if eps:
            arr = np.array([[m["history"][i]["epsilon_cumulative"] for i in range(n)] for m in g])
            axes[2].plot(xs, arr.mean(0), color=COLORS[k], lw=2, label=name)
    axes[0].set(title="Validation entity micro-F1", xlabel="Epoch")
    axes[1].set(title="Validation loss", xlabel="Epoch")
    axes[2].set(title="Cumulative $\\varepsilon$ ($\\delta=10^{-5}$)", xlabel="Epoch")
    for ax in axes:
        ax.title.set_size(9)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "learning_curves.pdf"))
    plt.close(fig)

    # ---------------------------------------------------------------- public warm start (Experiment C)
    PUB = [("pub_nonprivate", "Non-private", None), ("pub_dp_eps8", "High Utility", 8), ("pub_dp_eps3", "Better Privacy", 3),
           ("pub_dp_eps1", "Strong Privacy", 1)]
    GP = {k: group(runs, k) for k, _, _ in PUB}
    pub_summary = {}
    warm = runs.get("public_warmstart")
    rows = []
    if warm:
        macros.update(WarmF=f"{f1(warm):.3f}", WarmMacro=f"{warm['test']['entity_macro_f1']:.3f}",
                      WarmMIA=f"{warm['mia_auc_train_vs_test']:.3f}")
        rows.append(f"Public warm start only (no \\texttt{{lg-ner}} training) & -- & {f1(warm):.3f} & "
                    f"{warm['test']['entity_macro_f1']:.3f} & {warm['test']['per_type']['USERID']['f1']:.2f} & "
                    f"{warm['mia_auc_train_vs_test']:.3f} \\\\")
    for k, label, eps in PUB:
        g = GP[k]
        if not g:
            continue
        pub_summary[k] = {"F": [f1(m) for m in g], "M": [m["test"]["entity_macro_f1"] for m in g]}
        tag = {"pub_nonprivate": "PubBase", "pub_dp_eps8": "PubEpsEight", "pub_dp_eps3": "PubEpsThree",
               "pub_dp_eps1": "PubEpsOne"}[k]
        macros[f"{tag}F"] = f"{np.mean(pub_summary[k]['F']):.3f}"
        macros[f"{tag}Fsd"] = f"{np.std(pub_summary[k]['F'], ddof=1):.3f}" if len(g) > 1 else "0"
        macros[f"{tag}Macro"] = f"{np.mean(pub_summary[k]['M']):.3f}"
        macros[f"{tag}R"] = f"{np.mean([m['test']['entity_micro']['recall'] for m in g]):.3f}"
        macros[f"{tag}MIA"] = f"{np.mean([m['mia_auc_train_vs_test'] for m in g]):.3f}"
        for t in TYPES:
            macros[f"{tag}{t.title()}F"] = f"{np.mean([m['test']['per_type'][t]['f1'] for m in g]):.2f}"
        if eps:
            macros[f"{tag}Achieved"] = f"{np.mean([m['privacy']['epsilon_rdp'] for m in g]):.2f}"
            macros[f"{tag}Sigma"] = f"{g[0]['privacy']['noise_multiplier']:.3f}"
            macros[f"{tag}LR"] = sci(g[0]["config"]["lr"])
        eps_txt = "$\\infty$" if eps is None else f"{np.mean([m['privacy']['epsilon_rdp'] for m in g]):.2f}"
        rows.append(f"Warm start + {label} & {eps_txt} & {ms(pub_summary[k]['F'])} & {ms(pub_summary[k]['M'])} & "
                    f"{ms([m['test']['per_type']['USERID']['f1'] for m in g], 2)} & {ms([m['mia_auc_train_vs_test'] for m in g])} \\\\")
    if rows:
        write(os.path.join(gen, "public_results_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Test-set results with a non-private warm start on public "
              "Luganda NER data (MasakhaNER 2.0) before fine-tuning on \\texttt{lg-ner} (mean $\\pm$ s.d.\\ over three "
              "seeds). $\\varepsilon$ ($\\delta=10^{-5}$) covers the \\texttt{lg-ner} training sentences.}\n"
              "\\label{tab:public_results}\n\\begin{FitTable}\n\\begin{tabular}{lccccc}\n\\toprule\n"
              "Model & $\\varepsilon$ & Micro-F1 & Macro-F1 & USERID F1 & MIA AUC \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")
    rows = []
    for n, m in sorted(runs.items()):
        if n.startswith("pub_lrsweep"):
            rows.append(f"{m['config']['lr']:g} & {m['history'][-1]['val_entity_f1']:.3f} & {m['privacy']['epsilon_rdp']:.2f} \\\\")
    if rows:
        write(os.path.join(gen, "pub_lrsweep_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Learning-rate selection for DP-SGD from the public warm start "
              "($\\varepsilon=8$, seed 42, validation split).}\n\\label{tab:pub_lrsweep}\n\\begin{tabular}{ccc}\n\\toprule\n"
              "Learning rate & Validation entity micro-F1 & $\\varepsilon$ \\\\\n\\midrule\n" + "\n".join(rows)
              + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # privacy-utility trade-off figure: micro-F1 vs epsilon for both starting points
    if summary:
        keys = [k for k, _, _ in CONFIGS[::-1] if k in summary]   # eps=1, 3, 8, inf
        xl = {"dp_eps1": "1", "dp_eps3": "3", "dp_eps8": "8", "nonprivate": "$\\infty$\n(non-private)"}
        fig, ax = plt.subplots(figsize=(5.4, 3.1))
        xs = np.arange(len(keys))
        series = [("From XLM-R", summary, "", "#2a78d6", "-", "o")]
        if pub_summary:
            series.append(("From public warm start", pub_summary, "pub_", "#eb6834", "--", "s"))
        for lab, summ, pre, col, ls, mk in series:
            ks = [pre + k for k in keys]
            if not all(k in summ for k in ks):
                continue
            mu = [np.mean(summ[k]["F"]) for k in ks]
            sd = [np.std(summ[k]["F"], ddof=1) if len(summ[k]["F"]) > 1 else 0 for k in ks]
            ax.errorbar(xs, mu, yerr=sd, color=col, ls=ls, marker=mk, ms=6, lw=2, capsize=3, label=lab)
            for xi, yi in zip(xs, mu):
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center",
                            fontsize=7, color=INK)
        ax.set_xticks(xs, [xl[k] for k in keys])
        ax.set_xlabel("Privacy budget $\\varepsilon$ (left = stronger privacy)")
        ax.set_ylabel("Test entity micro-F1")
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "privacy_utility.pdf"))
        plt.close(fig)

        # per-type F1 grouped bars
        fig, ax = plt.subplots(figsize=(6.4, 2.9))
        x = np.arange(len(TYPES))
        present = [(k, l, e) for k, l, e in CONFIGS if G[k]]
        w = 0.8 / len(present)
        for i, (k, label, eps) in enumerate(present):
            v = [np.mean([m["test"]["per_type"][t]["f1"] for m in G[k]]) for t in TYPES]
            ax.bar(x + (i - (len(present) - 1) / 2) * w, v, w - 0.02, color=COLORS[k],
                   label=label if eps is None else f"$\\varepsilon$={eps}")
        ax.set_xticks(x, TYPES)
        ax.set_ylabel("Test entity F1")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "per_type_f1.pdf"))
        plt.close(fig)

    # confusion matrices (seed 42), row-normalised
    cm_keys = [k for k in ("nonprivate", "dp_eps1") if seed42.get(k)]
    if cm_keys:
        fig, axes = plt.subplots(1, len(cm_keys), figsize=(3.4 * len(cm_keys), 3.1))
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, cm_keys):
            cm = np.array(seed42[k]["test"]["confusion_matrix"]["matrix"], dtype=float)
            labels = seed42[k]["test"]["confusion_matrix"]["labels"]
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#3987e5", "#0d366b"]),
                      vmin=0, vmax=1)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if norm[i, j] >= 0.005:
                        ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if norm[i, j] > 0.55 else INK)
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.grid(False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Gold")
            ax.set_title(dict((c[0], c[1] if c[2] is None else f"{c[1]} ($\\varepsilon$={c[2]})") for c in CONFIGS)[k], fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "confusion_matrices.pdf"))
        plt.close(fig)

    # error analysis counts (seed 42)
    rows = []
    for k, label, eps in CONFIGS:
        p = os.path.join(res, f"{k}_seed42", "errors.json")
        if not os.path.exists(p):
            continue
        errs = json.load(open(p, encoding="utf-8"))
        c = lambda kind, sub: sum(1 for e in errs if e["kind"] == kind and e["subtype"] == sub)
        rows.append(f"{label} & {c('FN', 'missed')} & {c('FN', 'boundary')} & {c('FN', 'type_confusion')} & "
                    f"{c('FP', 'spurious')} & {c('FP', 'boundary')} \\\\")
    if rows:
        write(os.path.join(gen, "errors_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Test-set errors by kind (seed 42). Missed entities are PII left "
              "completely undetected; type confusions are found but mislabelled.}\n\\label{tab:errors}\n\\begin{FitTable}\n"
              "\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{3}{c}{False negatives (gold not matched)} & "
              "\\multicolumn{2}{c}{False positives} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\nModel & Missed & Boundary & "
              "Type confusion & Spurious & Boundary \\\\\n\\midrule\n" + "\n".join(rows)
              + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")

    # lr sweep
    rows = []
    for n, m in sorted(runs.items()):
        if n.startswith("lrsweep"):
            rows.append(f"{m['config']['lr']:g} & {m['history'][-1]['val_entity_f1']:.3f} & {m['privacy']['epsilon_rdp']:.2f} \\\\")
    if rows:
        write(os.path.join(gen, "lrsweep_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Learning-rate selection for DP-SGD ($\\varepsilon=8$, seed 42, "
              "validation split).}\n\\label{tab:lrsweep}\n\\begin{tabular}{ccc}\n\\toprule\nLearning rate & "
              "Validation entity micro-F1 & $\\varepsilon$ \\\\\n\\midrule\n" + "\n".join(rows)
              + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------------------------------------------------------- re-run of submitted configuration
    v6 = runs.get("v6_no_further_training")
    rows = []
    if v6:
        macros.update(VsixTestF=f"{v6['test']['entity_micro']['f1']:.3f}", VsixTestMacro=f"{v6['test']['entity_macro_f1']:.3f}",
                      VsixTok=f"{v6['test']['token_weighted_incl_O']['f1']:.3f}",
                      VsixTrainF=f"{v6['train_summary']['entity_micro_f1']:.3f}",
                      VsixValF=f"{v6['validation_summary']['entity_micro_f1']:.3f}",
                      VsixMIA=f"{v6['mia_auc_train_vs_test']:.3f}")
        rows.append(f"\\texttt{{luganda-ner-v6}} as released & -- & {v6['test']['token_weighted_incl_O']['f1']:.3f} & "
                    f"{v6['test']['entity_micro']['f1']:.3f} & {v6['test']['entity_macro_f1']:.3f} & "
                    f"{v6['train_summary']['entity_micro_f1']:.3f} & {v6['mia_auc_train_vs_test']:.3f} \\\\")
    for tag, label, sig, clip in (("hu", "High Utility", 1.0, 1.0), ("bp", "Better Privacy", 2.0, 0.5), ("sp", "Strong Privacy", 3.0, 0.5)):
        g = group(runs, f"repro_{tag}")
        if not g:
            continue
        macros[f"Repro{tag.upper()}Tok"] = f"{np.mean([m['test']['token_weighted_incl_O']['f1'] for m in g]):.3f}"
        macros[f"Repro{tag.upper()}F"] = f"{np.mean([f1(m) for m in g]):.3f}"
        macros[f"Repro{tag.upper()}Eps"] = f"{np.mean([m['privacy']['epsilon_rdp'] for m in g]):.2f}"
        macros[f"Repro{tag.upper()}MIA"] = f"{np.mean([m['mia_auc_train_vs_test'] for m in g]):.3f}"
        rows.append(f"{label} ($\\sigma$={sig}, $C$={clip}) & {ms([m['privacy']['epsilon_rdp'] for m in g], 2)} & "
                    f"{ms([m['test']['token_weighted_incl_O']['f1'] for m in g])} & "
                    f"{ms([f1(m) for m in g])} & {ms([m['test']['entity_macro_f1'] for m in g])} & "
                    f"{ms([m['train_summary']['entity_micro_f1'] for m in g])} & {ms([m['mia_auc_train_vs_test'] for m in g])} \\\\")
    if rows:
        write(os.path.join(gen, "repro_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Re-run of the submitted configuration (DP fine-tuning starting "
              "from \\texttt{luganda-ner-v6}), mean $\\pm$ s.d.\\ over three seeds. Token-weighted F1 is the metric reported "
              "in the submitted version; entity-level scores are on the test split; train F1 is entity micro-F1 on the "
              "training split.}\n\\label{tab:repro}\n\\begin{FitTable}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
              "Model & $\\varepsilon$ & Token-weighted F1 & Entity micro-F1 & Entity macro-F1 & Train F1 & MIA AUC \\\\\n"
              "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")

    # ---------------------------------------------------------------- environment
    env_src = any_dp or v6 or next(iter(runs.values()), None)
    if env_src:
        e = env_src["environment"]
        rows = [("Python", e["python"]), ("PyTorch", e["torch"]), ("Transformers", e["transformers"]),
                ("Datasets", e["datasets"]), ("Opacus", e.get("opacus", "--")), ("CUDA", e.get("cuda") or "--"),
                ("GPU", e.get("gpu") or "none (CPU)"), ("Platform", e["platform"].replace("_", "\\_")),
                ("Random seeds", "42, 1234, 2024")]
        write(os.path.join(gen, "environment_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Software and hardware used for the reported experiments "
              "(recorded automatically in each result file).}\n\\label{tab:environment}\n\\begin{tabular}{ll}\n\\toprule\n"
              "Component & Version / type \\\\\n\\midrule\n" + "\n".join(f"{a} & {b} \\\\" for a, b in rows)
              + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        macros["RunGPU"] = (e.get("gpu") or "CPU").replace("_", " ")

    # every macro the chapters use exists even before all runs finish; missing ones show as "??"
    stems = {"Base": "", "EpsEight": "", "EpsThree": "", "EpsOne": "", "PubBase": "", "PubEpsEight": "",
             "PubEpsThree": "", "PubEpsOne": ""}
    for s in stems:
        for suf in ("F", "Fsd", "Macro", "P", "R", "Missed", "Tok", "MIA", "CI", "Achieved", "PRV", "Drop", "Time",
                    "Sigma", "LR") + tuple(f"{t.title()}{m}" for t in TYPES for m in ("F", "R")):
            macros.setdefault(s + suf, "??")
    for tag in ("HU", "BP", "SP"):
        for suf in ("Tok", "F", "Eps", "MIA"):
            macros.setdefault(f"Repro{tag}{suf}", "??")
    for k in ("WarmF", "WarmMacro", "WarmMIA", "SigmaEpsEight", "SigmaEpsThree", "SigmaEpsOne", "SampleRate",
              "DPSteps", "DPEpochs", "SelectedLR", "NTrainable", "VsixTestF", "VsixTestMacro", "VsixTok",
              "VsixTrainF", "VsixValF", "VsixMIA", "RunGPU"):
        macros.setdefault(k, "??")
    write(os.path.join(gen, "numbers.tex"),
          "% generated by experiments/analyze.py -- do not edit\n"
          + "\n".join(f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(runs)} runs; {len(macros)} macros written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

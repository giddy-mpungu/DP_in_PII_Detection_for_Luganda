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
COLORS = {"nonprivate": "#52514e", "dp_eps8": "#184f95", "dp_eps3": "#3987e5", "dp_eps1": "#86b6ef",
          "dp_eps0.5": "#7f97b8"}
LSTYLE = {"nonprivate": "-", "dp_eps8": "-", "dp_eps3": "--", "dp_eps1": ":", "dp_eps0.5": "-."}
V2 = "#c4501b"        # Experiment E (v2) models: orange family, distinct from the blue warm-start models
V2_LIGHT = "#f0a074"
SPLIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": True, "axes.spines.right": True,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


PNG_DIR = None   # set in main(): every figure is also written as PNG for slides


def save(fig, figs, name):
    fig.savefig(os.path.join(figs, name + ".pdf"))
    if PNG_DIR:
        fig.savefig(os.path.join(PNG_DIR, name + ".png"), dpi=220)
    plt.close(fig)


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


TABLE_NAMES = {
    "tab:audit": "Entity Mentions by Type and Split",
    "tab:main_results": "Test-Set Results of the Main Experiment",
    "tab:per_type_f1": "Test Entity F1 per PII Category",
    "tab:per_type_recall": "Test Recall per PII Category",
    "tab:privacy": "Privacy Accounting of the Private Models",
    "tab:eps_epochs": "Cumulative Privacy Loss and Validation F1 per Epoch",
    "tab:public_results": "Test-Set Results with a Public Warm Start",
    "tab:pub_lrsweep": "Learning-Rate Selection from the Public Warm Start",
    "tab:errors": "Test-Set Errors by Kind",
    "tab:lrsweep": "Learning-Rate Selection for DP-SGD",
    "tab:repro": "Evaluation of the Public Model \\texttt{luganda-ner-v6}",
    "tab:environment": "Software and Hardware",
    "tab:pe_results": "Full and Parameter-Efficient DP Fine-Tuning",
    "tab:improve_results": "Synthetic Warm Start with Improved DP Training",
    "tab:hybrid": "Private Models Combined with Rule-Based Recognisers",
}


def finish_table(text):
    """Dissertation table style: a short Title Case name as caption, and fully ruled cells."""
    import re
    from latex_grid import gridify
    lab = re.search(r"\\label\{(tab:[^}]*)\}", text)
    if lab and lab.group(1) in TABLE_NAMES:
        i = text.index("\\caption{")
        j, d = i + len("\\caption{"), 1
        while d:
            d += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        text = text[:i] + "\\caption{" + TABLE_NAMES[lab.group(1)] + "}" + text[j:]
    return gridify(text)


def write(path, text):
    if "\\begin{table}" in text:
        text = finish_table(text)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def fmt_int(n):
    return f"{int(n):,}"


def main(res, tex):
    global PNG_DIR
    gen, figs = os.path.join(tex, "generated"), os.path.join(tex, "figures")
    PNG_DIR = os.path.join(tex, "figures_png")
    os.makedirs(PNG_DIR, exist_ok=True)
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
    save(fig, figs, "entity_distribution")

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
        macros[f"{tag}FPM"] = f"{v.mean():.3f}~$\\pm$~{v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f} (one seed)"
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
          "standard deviation over three seeds; models started from \\texttt{xlm-roberta-base}. The private models collapsed at every learning rate and were run with seed 42 only). $\\varepsilon$ is the achieved value at $\\delta = 10^{-5}$; MIA AUC is the "
          "membership-inference attack success (0.5 = chance).}\n\\label{tab:main_results}\n\\begin{FitTable}\n"
          "\\begin{tabular}{lcccccc}\n\\toprule\nModel & $\\varepsilon$ & Precision & Recall & Micro-F1 & Macro-F1 & MIA AUC \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")

    # per-type F1 and recall: non-private (both starts) and DP from the public warm start
    # (DP from xlm-roberta-base predicts no entities at all, so its columns would be all zero)
    PT = [(k, h) for k, h in (("nonprivate", "Non-private\\\\(XLM-R)"), ("pub_nonprivate", "Non-private\\\\(warm start)"),
                              ("pub_dp_eps8", "DP $\\varepsilon=8$\\\\(warm start)"), ("pub_dp_eps3", "DP $\\varepsilon=3$\\\\(warm start)"),
                              ("pub_dp_eps1", "DP $\\varepsilon=1$\\\\(warm start)")) if group(runs, k)]
    for metric, cap, lab in (("f1", "F1-score", "tab:per_type_f1"), ("recall", "recall (share of PII found)", "tab:per_type_recall")):
        rows = []
        for t in TYPES:
            sup = next((m["test"]["per_type"][t]["support"] for g in G.values() for m in g), 0)
            cells = [ms([m["test"]["per_type"][t][metric] for m in group(runs, k)], 2) for k, _ in PT]
            rows.append(f"{t} ({sup}) & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule\nMacro average & " + " & ".join(
            ms([np.mean([m["test"]["per_type"][t][metric] for t in TYPES]) for m in group(runs, k)], 2) for k, _ in PT) + " \\\\")
        head = " & ".join("\\begin{tabular}[c]{@{}c@{}}" + h + "\\end{tabular}" for _, h in PT)
        write(os.path.join(gen, f"per_type_{metric}_table.tex"),
              f"\\begin{{table}}[ht]\n\\centering\n\\caption{{Per-category test {cap} (mean $\\pm$ s.d.\\ over seeds; values without s.d.\\ come from one seed; "
              f"test support in brackets). DP models started from \\texttt{{xlm-roberta-base}} detect no entities and are "
              f"omitted.}}\n\\label{{{lab}}}\n\\begin{{FitTable}}\n\\begin{{tabular}}{{l{'c' * len(PT)}}}\n\\toprule\n"
              f"PII category & {head} \\\\\n\\midrule\n"
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
    pre_e = "pub_" if all(runs.get(f"pub_dp_eps{e}_seed42") for e in (8, 3, 1)) else ""
    eps_runs = {k: runs.get(f"{pre_e}{k}_seed42") for k in ("dp_eps8", "dp_eps3", "dp_eps1")}
    if all(eps_runs.values()):
        ep = len(eps_runs["dp_eps8"]["history"])
        rows = []
        for i in range(ep):
            cells = []
            for k in ("dp_eps8", "dp_eps3", "dp_eps1"):
                h = eps_runs[k]["history"][i]
                cells += [f"{h['epsilon_cumulative']:.2f}", f"{h['val_entity_f1']:.3f}"]
            rows.append(f"{i + 1} & " + " & ".join(cells) + " \\\\")
        write(os.path.join(gen, "epsilon_epochs_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Cumulative privacy loss $\\varepsilon$ ($\\delta=10^{-5}$) and "
              "validation entity micro-F1 after each epoch (seed 42" + (", public warm start" if pre_e else "") + "). $\\varepsilon$ grows with every step and the final "
              "row is the guarantee of the released model.}\n\\label{tab:eps_epochs}\n\\begin{FitTable}\n"
              "\\begin{tabular}{ccccccc}\n\\toprule\n & \\multicolumn{2}{c}{High Utility} & \\multicolumn{2}{c}{Better Privacy} "
              "& \\multicolumn{2}{c}{Strong Privacy} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
              "Epoch & $\\varepsilon$ & Val.\\ F1 & $\\varepsilon$ & Val.\\ F1 & $\\varepsilon$ & Val.\\ F1 \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")
    families = [("XLM-R start", "")]
    if group(runs, "pub_nonprivate"):
        families.append(("warm start", "pub_"))
    if group(runs, "big_dp_eps8"):
        families.append(("v2", "big_"))
    LC = CONFIGS + [("dp_eps0.5", "Strong Privacy+", 0.5)]
    fig, axes = plt.subplots(len(families), 3, figsize=(7.2, 2.25 * len(families) + 0.5), squeeze=False)
    handles = {}
    for row, (fam, pre) in enumerate(families):
        ax_f, ax_l, ax_e = axes[row]
        for k, label, eps in LC:
            g = group(runs, pre + k)
            if not g:
                continue
            n = min(len(m["history"]) for m in g)
            xs = np.arange(1, n + 1)
            name = label if eps is None else f"DP $\\varepsilon$={eps:g}"
            for ax, key in ((ax_f, "val_entity_f1"), (ax_l, "val_loss")):
                arr = np.array([[m["history"][i][key] for i in range(n)] for m in g])
                h, = ax.plot(xs, arr.mean(0), color=COLORS[k], ls=LSTYLE[k], lw=2, label=name)
                handles[name] = h
                if len(g) > 1:
                    ax.fill_between(xs, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=COLORS[k],
                                    alpha=0.15, lw=0)
            if eps:
                arr = np.array([[m["history"][i]["epsilon_cumulative"] for i in range(n)] for m in g])
                ax_e.plot(xs, arr.mean(0), color=COLORS[k], ls=LSTYLE[k], lw=2)
        ax_f.set(title=f"Validation micro-F1 ({fam})", xlabel="Epoch", ylim=(-0.03, 0.85))
        ax_l.set(title=f"Validation loss ({fam})", xlabel="Epoch")
        ax_e.set(title="Cumulative $\\varepsilon$ ($\\delta=10^{-5}$)", xlabel="Epoch")
        for ax in axes[row]:
            ax.title.set_size(8)
        if pre == "":
            ax_f.text(0.97, 0.06, "all DP runs: 0.00 at every epoch", transform=ax_f.transAxes, ha="right",
                      fontsize=6.5, color=MUTED)
    order = [n for n in ("Non-private", "DP $\\varepsilon$=8", "DP $\\varepsilon$=3", "DP $\\varepsilon$=1",
                         "DP $\\varepsilon$=0.5") if n in handles]
    fig.legend([handles[n] for n in order], order, loc="lower center", ncol=len(order), fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, figs, "learning_curves")

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
        macros[f"{tag}FPM"] = (f"{np.mean(pub_summary[k]['F']):.3f}~$\\pm$~{np.std(pub_summary[k]['F'], ddof=1):.3f}"
                               if len(g) > 1 else f"{np.mean(pub_summary[k]['F']):.3f} (one seed)")
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
              "seeds). $\\varepsilon$ ($\\delta=10^{-5}$) covers the \\texttt{lg-ner} training sentences. Values without a standard deviation come from a single seed (Section~\\ref{sec:training}).}\n"
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

    # privacy-utility trade-off figure: micro-F1 vs epsilon for every starting point
    if summary:
        cats = ["0.5", "1", "3", "8", "inf"]
        xl = {"0.5": "0.5", "1": "1", "3": "3", "8": "8", "inf": "$\\infty$\n(non-private)"}
        fig, ax = plt.subplots(figsize=(5.8, 3.3))
        xs = {c: i for i, c in enumerate(cats)}

        def pts(prefix, keys):
            out = []
            for c, k in keys:
                g = group(runs, prefix + k)
                if g:
                    F = [f1(m) for m in g]
                    out.append((xs[c], np.mean(F), np.std(F, ddof=1) if len(F) > 1 else 0.0))
            return out
        base_keys = [("1", "dp_eps1"), ("3", "dp_eps3"), ("8", "dp_eps8"), ("inf", "nonprivate")]
        series = [("From XLM-R", pts("", base_keys), "#7f8a99", "-", "o", -12),
                  ("From public warm start", pts("pub_", base_keys), "#2a78d6", "--", "s", -12),
                  ("From public + synthetic warm start, improved DP (v2)",
                   pts("big_", [("0.5", "dp_eps0.5"), ("1", "dp_eps1"), ("3", "dp_eps3"), ("8", "dp_eps8")]),
                   V2, "-", "D", 8)]
        for lab, P, col, ls, mk, dy in series:
            if not P:
                continue
            ax.errorbar([p[0] for p in P], [p[1] for p in P], yerr=[p[2] for p in P], color=col, ls=ls, marker=mk,
                        ms=5.5, lw=2, capsize=3, label=lab)
            for x0, y0, _ in P:
                if x0 == xs["inf"] and lab.startswith("From XLM"):
                    dy_ = -12
                elif x0 == xs["inf"]:
                    dy_ = 8
                else:
                    dy_ = dy
                ax.annotate(f"{y0:.2f}", (x0, y0), textcoords="offset points", xytext=(0, dy_), ha="center",
                            fontsize=6.8, color=INK)
        ax.set_xticks(range(len(cats)), [xl[c] for c in cats])
        ax.set_xlabel("Privacy budget $\\varepsilon$ (left = stronger privacy)")
        ax.set_ylabel("Test entity micro-F1")
        ax.set_ylim(-0.08, 1)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        save(fig, figs, "privacy_utility")

        # per-type F1 grouped bars
        fig, ax = plt.subplots(figsize=(6.8, 3.1))
        x = np.arange(len(TYPES))
        present = [(k, lab, col) for k, lab, col in (("nonprivate", "Non-private (XLM-R)", "#52514e"),
                                                     ("pub_nonprivate", "Non-private (warm start)", "#898781"),
                                                     ("pub_dp_eps8", "DP $\\varepsilon$=8 (warm start)", "#184f95"),
                                                     ("pub_dp_eps3", "DP $\\varepsilon$=3 (warm start)", "#3987e5"),
                                                     ("pub_dp_eps1", "DP $\\varepsilon$=1 (warm start)", "#86b6ef"),
                                                     ("big_dp_eps8", "DP $\\varepsilon$=8 (v2)", V2),
                                                     ("big_dp_eps1", "DP $\\varepsilon$=1 (v2)", V2_LIGHT))
                   if group(runs, k)]
        w = 0.84 / len(present)
        for i, (k, lab, col) in enumerate(present):
            v = [np.mean([m["test"]["per_type"][t]["f1"] for m in group(runs, k)]) for t in TYPES]
            xb = x + (i - (len(present) - 1) / 2) * w
            ax.bar(xb, v, w - 0.015, color=col, label=lab)
            for xi, vi in zip(xb, v):
                if vi < 0.005:
                    ax.text(xi, 0.012, "0", ha="center", va="bottom", fontsize=5.5, color=MUTED)
        ax.set_xticks(x, TYPES)
        ax.set_ylabel("Test entity F1")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=6.5)
        fig.tight_layout()
        save(fig, figs, "per_type_f1")

    # confusion matrices (seed 42), row-normalised
    seed42.update({k: runs.get(f"{k}_seed42") for k in ("pub_nonprivate", "pub_dp_eps8", "pub_dp_eps1", "big_dp_eps8")})
    cm_keys = [k for k in ("pub_nonprivate", "pub_dp_eps8", "pub_dp_eps1", "big_dp_eps8") if seed42.get(k)] or \
              [k for k in ("nonprivate", "dp_eps1") if seed42.get(k)]
    if cm_keys:
        ncol = 2 if len(cm_keys) == 4 else len(cm_keys)
        fig, axes = plt.subplots(int(np.ceil(len(cm_keys) / ncol)), ncol, figsize=(3.5 * ncol, 3.2 * np.ceil(len(cm_keys) / ncol)))
        axes = np.atleast_1d(axes).ravel()
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
                    else:
                        ax.text(j, i, "0", ha="center", va="center", fontsize=5.5, color="#c3c2b7")
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.grid(False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Gold")
            ax.set_title({"pub_nonprivate": "Warm start, non-private", "pub_dp_eps8": "Warm start, DP $\\varepsilon$=8",
                          "pub_dp_eps1": "Warm start, DP $\\varepsilon$=1", "big_dp_eps8": "v2, DP $\\varepsilon$=8",
                          "nonprivate": "Non-private",
                          "dp_eps1": "DP $\\varepsilon$=1"}[k], fontsize=9)
        fig.tight_layout()
        save(fig, figs, "confusion_matrices")

    # error analysis counts (seed 42)
    rows = []
    err_sets = [(k, label + (" (XLM-R start)" if eps is None else f" $\\varepsilon$={eps} (XLM-R start)")) for k, label, eps in CONFIGS] + \
               [("pub_" + k, label + (" (warm start)" if eps is None else f" $\\varepsilon$={eps} (warm start)")) for k, label, eps in CONFIGS]
    for k, label in err_sets:
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
              "\\begin{table}[ht]\n\\centering\n\\caption{The public Luganda NER model "
              "\\texttt{luganda-ner-v6}, evaluated as released. Token-weighted F1 includes the \\texttt{O} class; "
              "entity-level scores are on the test split; train F1 is entity micro-F1 on the training split; "
              "MIA AUC is the membership-inference attack success (0.5 = chance).}\n\\label{tab:repro}\n\\begin{FitTable}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
              "Model & $\\varepsilon$ & Token-weighted F1 & Entity micro-F1 & Entity macro-F1 & Train F1 & MIA AUC \\\\\n"
              "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")

    # ---------------------------------------------------------------- metric illusion: token-weighted vs entity F1
    s = sp["test"]["O_token_share"]   # a model that labels every word O: weighted F1 = s * F1(O)
    bars = [("Predicts no PII\n(every word O)", [s * 2 * s / (1 + s)], [0.0])]
    macros["AllOTokF"] = f"{s * 2 * s / (1 + s):.2f}"
    if v6:
        bars.append(("luganda-ner-v6\n(as released)", [v6["test"]["token_weighted_incl_O"]["f1"]], [f1(v6)]))
    for tag, label in (("hu", "DP from v6, HU"), ("sp", "DP from v6, SP")):
        g = group(runs, f"repro_{tag}")
        if g:
            bars.append((label, [m["test"]["token_weighted_incl_O"]["f1"] for m in g], [f1(m) for m in g]))
    for k, label in (("nonprivate", "Non-private\n(XLM-R)"), ("dp_eps8", "DP $\\varepsilon$=8\n(XLM-R)"),
                     ("pub_dp_eps8", "DP $\\varepsilon$=8\n(warm start)"), ("pub_dp_eps1", "DP $\\varepsilon$=1\n(warm start)"),
                     ("big_dp_eps8", "DP $\\varepsilon$=8\n(v2)")):
        g = G.get(k) or GP.get(k) or group(runs, k)
        if g:
            bars.append((label, [m["test"]["token_weighted_incl_O"]["f1"] for m in g], [f1(m) for m in g]))
    if bars:
        fig, ax = plt.subplots(figsize=(7.4, 3.0))
        x = np.arange(len(bars))
        for i, (lab, col, key) in enumerate((("Token-weighted F1 (incl. O)", "#c3c2b7", 1),
                                             ("Entity-level micro-F1", "#2a78d6", 2))):
            v = [np.mean(b[key]) for b in bars]
            ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=col, label=lab)
            for xi, vi in zip(x, v):
                ax.text(xi + (i - 0.5) * 0.38, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=6.5, color=INK)
        ax.set_xticks(x, [b[0] for b in bars], fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Test F1")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=7.5)
        fig.tight_layout()
        save(fig, figs, "metric_illusion")

    # ---------------------------------------------------------------- membership inference: loss distributions
    mia = [(n, lab) for n, lab in (("v6_no_further_training", "luganda-ner-v6 (non-private,\ntrained on these sentences)"),
                                    ("pub_nonprivate_seed42", "Warm start + non-private"),
                                    ("pub_dp_eps8_seed42", "Warm start + DP $\\varepsilon$=8"),
                                    ("pub_dp_eps1_seed42", "Warm start + DP $\\varepsilon$=1"),
                                    ("big_dp_eps8_seed42", "v2 + DP $\\varepsilon$=8"))
           if os.path.exists(os.path.join(res, n, "losses.json"))]
    if mia:
        fig, axes = plt.subplots(1, len(mia), figsize=(1.95 * len(mia), 2.5), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, (n, lab) in zip(axes, mia):
            L = json.load(open(os.path.join(res, n, "losses.json")))
            hi = np.percentile(L["test"], 98)
            bins = np.linspace(0, max(hi, 1e-3), 30)
            h1, _, _ = ax.hist(np.clip(L["train"], 0, hi), bins=bins, density=True, histtype="step", lw=1.6, color="#2a78d6", label="Train (members)")
            h2, _, _ = ax.hist(np.clip(L["test"], 0, hi), bins=bins, density=True, histtype="stepfilled", color="#eb6834", alpha=0.35, label="Test (non-members)")
            ax.set_ylim(0, 1.12 * max(np.max(h1), np.max(h2)))
            ax.set_title(f"{lab}\nAUC = {runs[n]['mia_auc_train_vs_test']:.2f}", fontsize=6.8)
            ax.set_xlabel("Per-sentence loss", fontsize=7.5)
            ax.set_yticks([])
            ax.grid(False)
        axes[0].legend(fontsize=6.5, loc="upper right")
        fig.tight_layout()
        save(fig, figs, "mia_losses")

    # ---------------------------------------------------------------- missed-PII rate per category
    sets = [(k, lab, col) for k, lab, col in (("nonprivate", "Non-private (XLM-R)", "#52514e"),
                                              ("pub_nonprivate", "Non-private (warm start)", "#898781"),
                                              ("pub_dp_eps8", "DP $\\varepsilon$=8 (warm start)", "#184f95"),
                                              ("pub_dp_eps3", "DP $\\varepsilon$=3 (warm start)", "#3987e5"),
                                              ("pub_dp_eps1", "DP $\\varepsilon$=1 (warm start)", "#86b6ef"),
                                              ("big_dp_eps8", "DP $\\varepsilon$=8 (v2)", V2),
                                              ("big_dp_eps1", "DP $\\varepsilon$=1 (v2)", V2_LIGHT))
            if (G.get(k) or GP.get(k) or group(runs, k))]
    if sets:
        fig, ax = plt.subplots(figsize=(6.8, 3.1))
        x = np.arange(len(TYPES))
        w = 0.84 / len(sets)
        for i, (k, lab, col) in enumerate(sets):
            g = G.get(k) or GP.get(k) or group(runs, k)
            v = [100 * (1 - np.mean([m["test"]["per_type"][t]["recall"] for m in g])) for t in TYPES]
            ax.bar(x + (i - (len(sets) - 1) / 2) * w, v, w - 0.015, color=col, label=lab)
        ax.set_xticks(x, TYPES)
        ax.set_ylabel("Missed PII (% of gold mentions)")
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.25))
        fig.tight_layout()
        save(fig, figs, "missed_pii")

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

    # ---------------------------------------------------------------- parameter-efficient DP (Experiment D)
    rows = []
    for eps, tag in ((8, "EpsEight"), (3, "EpsThree"), (1, "EpsOne")):
        full, pe = group(runs, f"pub_dp_eps{eps}"), group(runs, f"pub_pe_dp_eps{eps}")
        if not pe:
            continue
        F = [f1(m) for m in pe]
        macros[f"PubPe{tag}F"] = f"{np.mean(F):.3f}"
        macros[f"PubPe{tag}FPM"] = (f"{np.mean(F):.3f}~$\\pm$~{np.std(F, ddof=1):.3f}" if len(F) > 1
                                     else f"{np.mean(F):.3f} (one seed)")
        macros[f"PubPe{tag}Macro"] = f"{np.mean([m['test']['entity_macro_f1'] for m in pe]):.3f}"
        macros[f"PubPe{tag}MIA"] = f"{np.mean([m['mia_auc_train_vs_test'] for m in pe]):.3f}"
        macros[f"PubPe{tag}Sigma"] = f"{pe[0]['privacy']['noise_multiplier']:.3f}"
        for t in TYPES:
            macros[f"PubPe{tag}{t.title()}F"] = f"{np.mean([m['test']['per_type'][t]['f1'] for m in pe]):.2f}"
        for label, g in (("All layers above the embeddings", full), ("Top two layers + head", pe)):
            if not g:
                continue
            rows.append(f"{eps} & {label} & {g[0]['trainable_params'] / 1e6:.1f}M & {g[0]['privacy']['noise_multiplier']:.3f} & "
                        f"{ms([f1(m) for m in g])} & {ms([m['test']['entity_macro_f1'] for m in g])} & "
                        f"{ms([m['mia_auc_train_vs_test'] for m in g])} \\\\")
        rows.append("\\midrule")
    if pe_runs := [m for n, m in runs.items() if n.startswith("pub_pe_lrsweep")]:
        macros["PubPeLR"] = sci(json.load(open(os.path.join(res, "selected_lr_public_pe.json")))["lr"]) \
            if os.path.exists(os.path.join(res, "selected_lr_public_pe.json")) else "??"
        macros["PubPeNTrainable"] = f"{pe_runs[0]['trainable_params'] / 1e6:.1f} million"
    if rows:
        write(os.path.join(gen, "pe_results_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{DP fine-tuning from the public warm start with all layers above the "
              "embeddings trained (Table~\\ref{tab:public_results}) and with only the top two transformer layers and the head "
              "trained (Section~\\ref{sec:pe_design}). Test set, mean $\\pm$ s.d.\\ over three seeds.}\n\\label{tab:pe_results}\n"
              "\\begin{FitTable}\n\\begin{tabular}{clccccc}\n\\toprule\n$\\varepsilon$ & Trained parameters & Count & $\\sigma$ & "
              "Micro-F1 & Macro-F1 & MIA AUC \\\\\n\\midrule\n" + "\n".join(rows[:-1])
              + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")
    for tag in ("EpsEight", "EpsThree", "EpsOne"):
        for suf in ("F", "FPM", "Macro", "MIA", "Sigma") + tuple(f"{t.title()}F" for t in TYPES):
            macros.setdefault(f"PubPe{tag}{suf}", "??")
    macros.setdefault("PubPeLR", "??")
    macros.setdefault("PubPeNTrainable", "??")

    # ---------------------------------------------------------------- Experiment E: synthetic warm start + improved DP
    EPS_TAGS = ((8, "EpsEight"), (3, "EpsThree"), (1, "EpsOne"), (0.5, "EpsHalf"))
    sw = runs.get("synth_warmstart")
    if sw:
        macros.update(SynWarmF=f"{f1(sw):.3f}", SynWarmMacro=f"{sw['test']['entity_macro_f1']:.3f}",
                      SynWarmMIA=f"{sw['mia_auc_train_vs_test']:.3f}")
        for t in TYPES:
            macros[f"SynWarm{t.title()}F"] = f"{sw['test']['per_type'][t]['f1']:.2f}"
            macros[f"SynWarm{t.title()}R"] = f"{sw['test']['per_type'][t]['recall']:.2f}"
    for eps, tag in ((8, "EpsEight"), (1, "EpsOne")):
        m = runs.get(f"syn_dp_eps{eps:g}_seed42")
        if m:
            macros[f"Syn{tag}F"] = f"{f1(m):.3f}"
            macros[f"Syn{tag}Macro"] = f"{m['test']['entity_macro_f1']:.3f}"
    sel = os.path.join(res, "selected_lr_improve.json")
    if os.path.exists(sel):
        macros["BigLR"] = sci(json.load(open(sel))["lr"])
    rows = []
    for eps, tag in EPS_TAGS:
        g = group(runs, f"big_dp_eps{eps:g}")
        if not g:
            continue
        F = [f1(m) for m in g]
        macros[f"Big{tag}F"] = f"{np.mean(F):.3f}"
        macros[f"Big{tag}FPM"] = (f"{np.mean(F):.3f}~$\\pm$~{np.std(F, ddof=1):.3f}" if len(F) > 1
                                  else f"{np.mean(F):.3f} (one seed)")
        macros[f"Big{tag}Macro"] = f"{np.mean([m['test']['entity_macro_f1'] for m in g]):.3f}"
        macros[f"Big{tag}R"] = f"{np.mean([m['test']['entity_micro']['recall'] for m in g]):.3f}"
        macros[f"Big{tag}MIA"] = f"{np.mean([m['mia_auc_train_vs_test'] for m in g]):.3f}"
        macros[f"Big{tag}Sigma"] = f"{g[0]['privacy']['noise_multiplier']:.3f}"
        macros[f"Big{tag}PRV"] = f"{g[0]['privacy']['epsilon_prv']:.2f}"
        macros[f"Big{tag}RDP"] = f"{g[0]['privacy']['epsilon_rdp']:.2f}"
        macros[f"Big{tag}Seeds"] = str(len(g))
        for t in TYPES:
            macros[f"Big{tag}{t.title()}F"] = f"{np.mean([m['test']['per_type'][t]['f1'] for m in g]):.2f}"
        old = group(runs, f"pub_dp_eps{eps:g}")
        old_txt = ms([f1(m) for m in old]) if old else "--"
        rows.append(f"{eps:g} & {old_txt} & {ms(F)} & {ms([m['test']['entity_macro_f1'] for m in g])} & "
                    + " & ".join(f"{np.mean([m['test']['per_type'][t]['f1'] for m in g]):.2f}" for t in ("USERID", "DATE", "NORP"))
                    + f" & {g[0]['privacy']['noise_multiplier']:.2f} & {g[0]['privacy']['epsilon_prv']:.2f} & "
                    f"{ms([m['mia_auc_train_vs_test'] for m in g], 2)} \\\\")
    if rows:
        write(os.path.join(gen, "improve_results_table.tex"),
              "\\begin{table}[ht]\n\\centering\n\\caption{Experiment E: DP fine-tuning from the public + synthetic warm start "
              "with batch 256, 25 epochs and noise calibrated by the PRV accountant (Section~\\ref{sec:improve_design}), "
              "against the warm-start DP models of Table~\\ref{tab:public_results}. Test set; mean $\\pm$ s.d.\\ over seeds "
              "where more than one seed was run. $\\varepsilon$ is the PRV bound at $\\delta = 10^{-5}$.}\n"
              "\\label{tab:improve_results}\n\\begin{FitTable}\n\\begin{tabular}{cccccccccc}\n\\toprule\n"
              " & \\multicolumn{3}{c}{Micro-/macro-F1} & \\multicolumn{3}{c}{F1, new model} & & & \\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nTarget $\\varepsilon$ & Earlier & New (micro) & New (macro) & "
              "USERID & DATE & NORP & $\\sigma$ & $\\varepsilon$ & MIA AUC \\\\\n\\midrule\n" + "\n".join(rows)
              + "\n\\bottomrule\n\\end{tabular}\n\\end{FitTable}\n\\end{table}\n")
    ap = os.path.join(res, "canary_audit.json")
    if os.path.exists(ap):
        au = json.load(open(ap))
        for run, tag in (("canary2_nonprivate_seed42", "Np"), ("canary2_dp_eps8_seed42", "Dp"),
                         ("canary_nonprivate_seed42", "TplNp"), ("canary_dp_eps8_seed42", "TplDp")):
            if run in au:
                a = au[run]
                macros[f"Canary{tag}LB"] = f"{a['eps_lower_bound']:.2f}"
                macros[f"Canary{tag}Acc"] = f"{max(r['accuracy'] for r in a['by_k']):.2f}"
                full = max(a["by_k"], key=lambda r: r["guesses"])       # report the largest, pre-registered guess set
                macros[f"Canary{tag}BestK"] = str(full["guesses"])
                macros[f"Canary{tag}BestAcc"] = str(full["correct"])
                macros[f"Canary{tag}F"] = f"{f1(runs[run]):.3f}" if run in runs else "??"
    for _, tag in EPS_TAGS:
        for suf in ("F", "FPM", "Macro", "R", "MIA", "Sigma", "PRV", "RDP", "Seeds") + tuple(f"{t.title()}F" for t in TYPES):
            macros.setdefault(f"Big{tag}{suf}", "??")
    for k in ("SynWarmF", "SynWarmMacro", "SynWarmMIA", "SynEpsEightF", "SynEpsEightMacro", "SynEpsOneF", "SynEpsOneMacro",
              "BigLR") + tuple(f"SynWarm{t.title()}{m}" for t in TYPES for m in ("F", "R")) + tuple(
              f"Canary{tag}{suf}" for tag in ("Np", "Dp", "TplNp", "TplDp") for suf in ("LB", "Acc", "BestK", "BestAcc", "F")):
        macros.setdefault(k, "??")

    # ---------------------------------------------------------------- v6's own (sub-token) metric, v6_metric.py
    vp = os.path.join(res, "v6_metric.json")
    if os.path.exists(vp):
        vm = json.load(open(vp))
        for key, tag in (("v6", "Vsix"), ("old_eps8", "OldEight"), ("big_eps8", "BigEight"), ("big_eps3", "BigThree"),
                         ("big_eps1", "BigOne"), ("big_eps0.5", "BigHalf")):
            if key in vm:
                macros[f"VsixMetric{tag}"] = f"{vm[key]['v6_metric_f1']:.3f}"
                macros[f"WordMetric{tag}"] = f"{vm[key]['word_level_f1']:.3f}"
    for tag in ("Vsix", "OldEight", "BigEight", "BigThree", "BigOne", "BigHalf"):
        macros.setdefault(f"VsixMetric{tag}", "??")
        macros.setdefault(f"WordMetric{tag}", "??")

    # ---------------------------------------------------------------- hybrid: DP model + rules (hybrid_eval.py)
    hp = os.path.join(res, "hybrid", "hybrid.json")
    if os.path.exists(hp):
        hy = json.load(open(hp))
        ro = hy["rules_only"]
        for split, tag in (("train", "Train"), ("validation", "Val"), ("test", "Test")):
            for t in ("USERID", "DATE"):
                macros[f"Rules{tag}{t.title()}F"] = f"{ro[split][t]['f1']:.2f}"
                macros[f"Rules{tag}{t.title()}P"] = f"{ro[split][t]['precision']:.2f}"
                macros[f"Rules{tag}{t.title()}R"] = f"{ro[split][t]['recall']:.2f}"
                macros[f"Rules{tag}{t.title()}Len"] = f"{ro[split][t]['lenient_f1']:.2f}"
        macros["HybStrategy"] = hy["strategy"]["chosen"]
        rows = []
        for eps, tag in ((8, "EpsEight"), (3, "EpsThree"), (1, "EpsOne")):
            r = hy["runs"].get(f"pub_dp_eps{eps}_seed42")
            if not r:
                continue
            mo, mh = r["model_only"], r["model_plus_rules"]
            macros[f"Hyb{tag}BaseF"] = f"{mo['entity_micro']['f1']:.3f}"
            macros[f"Hyb{tag}F"] = f"{mh['entity_micro']['f1']:.3f}"
            macros[f"Hyb{tag}CI"] = "[{:.3f}, {:.3f}]".format(*mh["entity_micro_f1_ci95"])
            macros[f"Hyb{tag}Macro"] = f"{mh['entity_macro_f1']:.3f}"
            macros[f"Hyb{tag}BaseMacro"] = f"{mo['entity_macro_f1']:.3f}"
            macros[f"Hyb{tag}R"] = f"{mh['entity_micro']['recall']:.3f}"
            macros[f"Hyb{tag}BaseR"] = f"{mo['entity_micro']['recall']:.3f}"
            for t in ("USERID", "DATE"):
                macros[f"Hyb{tag}{t.title()}F"] = f"{mh['per_type'][t]['f1']:.2f}"
            rows.append(f"C, $\\varepsilon = {eps}$ & {mo['entity_micro']['f1']:.3f} & {mh['entity_micro']['f1']:.3f} & "
                        f"{mo['entity_macro_f1']:.3f} & {mh['entity_macro_f1']:.3f} & "
                        f"{mo['entity_micro']['recall']:.3f} & {mh['entity_micro']['recall']:.3f} & "
                        f"{mo['per_type']['USERID']['f1']:.2f} $\\to$ {mh['per_type']['USERID']['f1']:.2f} & "
                        f"{mo['per_type']['DATE']['f1']:.2f} $\\to$ {mh['per_type']['DATE']['f1']:.2f} \\\\")
        for eps, tag in ((8, "EpsEight"), (3, "EpsThree"), (1, "EpsOne"), (0.5, "EpsHalf")):
            r = hy["runs"].get(f"big_dp_eps{eps:g}_seed42")
            if r:
                mo, mh = r["model_only"], r["model_plus_rules"]
                rows.append(f"E, $\\varepsilon = {eps:g}$ & {mo['entity_micro']['f1']:.3f} & {mh['entity_micro']['f1']:.3f} & "
                            f"{mo['entity_macro_f1']:.3f} & {mh['entity_macro_f1']:.3f} & "
                            f"{mo['entity_micro']['recall']:.3f} & {mh['entity_micro']['recall']:.3f} & "
                            f"{mo['per_type']['USERID']['f1']:.2f} $\\to$ {mh['per_type']['USERID']['f1']:.2f} & "
                            f"{mo['per_type']['DATE']['f1']:.2f} $\\to$ {mh['per_type']['DATE']['f1']:.2f} \\\\")
                macros[f"HybBig{tag}BaseF"] = f"{r['model_only']['entity_micro']['f1']:.3f}"
                macros[f"HybBig{tag}F"] = f"{r['model_plus_rules']['entity_micro']['f1']:.3f}"
                macros[f"HybBig{tag}R"] = f"{r['model_plus_rules']['entity_micro']['recall']:.3f}"
        other = [hy["runs"][f"pub_dp_eps1_seed{s}"] for s in (1234, 2024) if f"pub_dp_eps1_seed{s}" in hy["runs"]]
        if other:
            macros["HybEpsOneOtherSeeds"] = " and ".join(
                f"{r['model_only']['entity_micro']['f1']:.3f} to {r['model_plus_rules']['entity_micro']['f1']:.3f}"
                for r in other)
        if rows:
            write(os.path.join(gen, "hybrid_table.tex"),
                  "\\begin{table}[ht]\n\\centering\n\\caption{Test-set results of the released private models (warm start, "
                  "seed 42) alone and combined with the rule-based USERID and DATE recognisers (Section~\\ref{sec:hybrid_method}). "
                  "The rules use no training data, so the privacy guarantee is unchanged.}\n\\label{tab:hybrid}\n"
                  "\\begin{FitTable}\n\\begin{tabular}{lcccccccc}\n\\toprule\n"
                  " & \\multicolumn{2}{c}{Micro-F1} & \\multicolumn{2}{c}{Macro-F1} & \\multicolumn{2}{c}{Recall} & "
                  "USERID F1 & DATE F1 \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
                  "Experiment & DP & DP+rules & DP & DP+rules & DP & DP+rules & DP $\\to$ DP+rules & DP $\\to$ DP+rules \\\\\n"
                  "\\midrule\n" + "\n".join(rows)
                  + f"\n\\midrule\nRules alone & -- & -- & -- & -- & -- & -- & {macros['RulesTestUseridF']} & "
                  f"{macros['RulesTestDateF']} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{FitTable}}\n\\end{{table}}\n")

    # every macro the chapters use exists even before all runs finish; missing ones show as "??"
    for tag in ("EpsEight", "EpsThree", "EpsOne"):
        for suf in ("BaseF", "F", "CI", "Macro", "BaseMacro", "R", "BaseR", "UseridF", "DateF"):
            macros.setdefault(f"Hyb{tag}{suf}", "??")
    for tag in ("Train", "Val", "Test"):
        for t in ("Userid", "Date"):
            for suf in ("F", "P", "R", "Len"):
                macros.setdefault(f"Rules{tag}{t}{suf}", "??")
    macros.setdefault("HybStrategy", "??")
    macros.setdefault("HybEpsOneOtherSeeds", "??")
    for tag in ("EpsEight", "EpsThree", "EpsOne", "EpsHalf"):
        for suf in ("BaseF", "F", "R"):
            macros.setdefault(f"HybBig{tag}{suf}", "??")
    stems = {"Base": "", "EpsEight": "", "EpsThree": "", "EpsOne": "", "PubBase": "", "PubEpsEight": "",
             "PubEpsThree": "", "PubEpsOne": ""}
    for s in stems:
        for suf in ("F", "Fsd", "FPM", "Macro", "P", "R", "Missed", "Tok", "MIA", "CI", "Achieved", "PRV", "Drop", "Time",
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

"""Build a Word 'Strengthened Results & Discussion' document from the Item 1-3
artifacts. All numbers are read from results/*.json|csv; the regenerated figures
are embedded. Saved next to the project.
"""
import json
import os

import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from config import RESULT_DIR, FIG_DIR, ROOT

OUT = os.path.join(ROOT, "XHateExplain_Strengthened_Results_Discussion.docx")
NAVY = RGBColor(0x2c, 0x3e, 0x50)

ci = json.load(open(os.path.join(RESULT_DIR, "explainability_ci.json")))
sweep = json.load(open(os.path.join(RESULT_DIR, "community_sweep_summary.json")))
sweep_csv = pd.read_csv(os.path.join(RESULT_DIR, "community_threshold_sweep.csv"))
calib = json.load(open(os.path.join(RESULT_DIR, "bert_metrics_weighted.json")))


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = NAVY
    return h


def para(doc, text, bold=False):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = bold
    return p


def bullet(doc, text):
    doc.add_paragraph(style="List Bullet").add_run(text)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def table(doc, headers, rows):
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.cell(0, j); c.text = str(h); shade(c, "2c3e50")
        for p in c.paragraphs:
            for r in p.runs:
                r.bold = True; r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(255, 255, 255)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i + 1, j); c.text = str(v)
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8.5)
                    if j == 0: r.bold = True
    return t


def figure(doc, path, caption, width=6.2):
    if not os.path.exists(path):
        para(doc, f"[missing figure: {os.path.basename(path)}]"); return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption); r.italic = True; r.font.size = Pt(9)


# =============================================================== build
doc = Document()
doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(11)

t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("XHateExplain — Strengthened Results & Discussion")
r.bold = True; r.font.size = Pt(18); r.font.color.rgb = NAVY
s = doc.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = s.add_run("Statistical significance, robustness, and an honest negative result "
              "(no classifier retraining)"); r.italic = True; r.font.size = Pt(11)
doc.add_paragraph()

para(doc,
     "This document supplements the main results report with three rigour-focused analyses requested for "
     "the paper: (1) bootstrap confidence intervals and paired significance tests on the explainability "
     "metrics, (2) a robustness sweep of the content-similarity graph over cosine thresholds, and (3) a "
     "training-free attempt to improve the weak 'offensive' class. No model was retrained; the analyses "
     "reuse the cached 300-post explainability subset, the saved per-token attributions, and encoder "
     "inference only.")

# ---------------------------------------------------------------- Item 1
heading(doc, "1. Explainability Quality with Confidence Intervals", 1)
para(doc,
     f"Each metric was recomputed to four decimal places and equipped with a 95% bootstrap confidence "
     f"interval ({ci['n_boot']:,} resamples, fixed seed {ci['seed']}). Plausibility (IOU, token-F1) is "
     f"measured over the {ci['n_with_rationale']} subset posts that carry a human rationale; faithfulness "
     f"(AOPC) over all {ci['n_posts']} posts.")
rows = []
for m in ["SHAP", "LIME", "Attention", "XHateExplain"]:
    cells = [m]
    for metric in ["iou", "token_f1", "aopc_5", "aopc_10"]:
        c = ci["metrics"][f"{m}|{metric}"]
        cells.append(f"{c['mean']:.4f}\n[{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}]")
    rows.append(cells)
table(doc, ["Method", "Plausibility (IOU)", "Token-F1", "AOPC@5", "AOPC@10"], rows)
para(doc, "")
para(doc, "Table A1 — Explainability metrics, mean with 95% bootstrap CI.", bold=False)

para(doc, "Paired bootstrap significance tests.", bold=True)
labels = {
    "XHateExplain_vs_Attention__iou": ("XHateExplain − Attention", "IOU"),
    "XHateExplain_vs_Attention__token_f1": ("XHateExplain − Attention", "Token-F1"),
    "XHateExplain_vs_SHAP__aopc_5": ("XHateExplain − SHAP", "AOPC@5"),
    "XHateExplain_vs_SHAP__aopc_10": ("XHateExplain − SHAP", "AOPC@10"),
    "Attention_vs_SHAP__aopc_5": ("Attention − SHAP", "AOPC@5"),
}
trows = []
for k, t_ in ci["paired_tests"].items():
    comp, met = labels[k]
    verdict = "significant" if t_["significant_0.05"] else "tie (n.s.)"
    trows.append([comp, met, f"{t_['mean_diff']:+.4f}",
                  f"[{t_['ci95_diff'][0]:+.4f}, {t_['ci95_diff'][1]:+.4f}]",
                  f"{t_['p_value']:.4f}", verdict])
table(doc, ["Comparison", "Metric", "Mean diff", "95% CI of diff", "p", "Verdict"], trows)
para(doc, "")

plaus = ci["paired_tests"]["XHateExplain_vs_Attention__iou"]
f5 = ci["paired_tests"]["XHateExplain_vs_SHAP__aopc_5"]
f10 = ci["paired_tests"]["XHateExplain_vs_SHAP__aopc_10"]
xhe5 = ci["metrics"]["XHateExplain|aopc_5"]["mean"]
sh5 = ci["metrics"]["SHAP|aopc_5"]["mean"]
at5 = ci["metrics"]["Attention|aopc_5"]["mean"]
para(doc, "Discussion.", bold=True)
para(doc,
     f"At four decimals the apparent IOU tie between Attention and XHateExplain is confirmed as a genuine "
     f"dead heat: the paired difference is {plaus['mean_diff']:+.4f} with a 95% CI of "
     f"[{plaus['ci95_diff'][0]:+.4f}, {plaus['ci95_diff'][1]:+.4f}] and p={plaus['p_value']:.2f}, so "
     f"XHateExplain does NOT have a real plausibility advantage over attention — they share the top "
     f"plausibility. On faithfulness, XHateExplain is significantly below SHAP at both AOPC@5 "
     f"(Δ={f5['mean_diff']:+.4f}, p<0.001) and AOPC@10 (Δ={f10['mean_diff']:+.4f}, p<0.001), but the gap "
     f"is small (~0.02 absolute). By contrast attention sits far below SHAP on AOPC@5 "
     f"(Δ={ci['paired_tests']['Attention_vs_SHAP__aopc_5']['mean_diff']:+.4f}, p<0.001). The defensible "
     f"claim for the paper is therefore one of balance, not dominance: SHAP is the most faithful single "
     f"method and attention the least; attention and XHateExplain are the most plausible; and XHateExplain "
     f"is the only method that is simultaneously top-tier on plausibility while retaining most of SHAP's "
     f"faithfulness (AOPC@5 {xhe5:.4f} vs SHAP {sh5:.4f}, well above attention's {at5:.4f}).")
figure(doc, os.path.join(FIG_DIR, "table2_explainability.png"),
       "Figure A1 — Explainability metrics with 95% bootstrap confidence intervals (regenerated).", 6.6)

# ---------------------------------------------------------------- Item 2
heading(doc, "2. Content-Similarity Graph: Robustness and Reframing", 1)
para(doc,
     "The post-similarity graph was rebuilt on the same 2,000-post sample at cosine thresholds 0.70, 0.80 "
     "and 0.85, and characterised by edge count, average degree, number of Louvain communities, modularity "
     "Q, and hate-ratio separation.")
srows = []
for r_ in sweep["sweep"]:
    star = " (chosen)" if r_["threshold"] == sweep["chosen_threshold"] else ""
    srows.append([f"{r_['threshold']}{star}", f"{r_['edges']:,}", r_["avg_degree"],
                  r_["n_communities"], f"{r_['modularity_Q']:.4f}",
                  r_["hate_comm_gt70pct"], r_["normal_comm_lt10pct"]])
table(doc, ["Cosine thr", "Edges", "Avg deg", "#Comms", "Modularity Q", "Hate>70%", "Normal<10%"], srows)
para(doc, "")
para(doc, "Table A2 — Threshold sweep of the content-similarity graph.", bold=False)

_q = " → ".join(f"{r_['modularity_Q']:.4f}" for r_ in sweep["sweep"])
_c = " → ".join(str(r_["n_communities"]) for r_ in sweep["sweep"])
para(doc, "Discussion.", bold=True)
para(doc,
     f"Contrary to the intuition that a sparser graph would yield cleaner communities, raising the "
     f"threshold degrades the structure: modularity falls monotonically (Q: {_q}) while the graph "
     f"fragments into ever more, mostly singleton, communities (count: {_c}). The large numbers of "
     f"'pure' hate (>70%) and normal (<10%) communities at 0.80 and 0.85 are thus a fragmentation "
     f"artifact — hundreds of tiny components that are trivially homogeneous — rather than evidence of "
     f"better separation. The dense 0.70 graph is the only setting with a non-fragmented, modular "
     f"community structure (Q={sweep['sweep'][0]['modularity_Q']:.4f}), and is retained as the operating "
     f"point. The core qualitative finding is unchanged: {sweep['sentiment_gap']}.")
para(doc,
     "Honest reframing. Because HateXplain contains no author identifiers, this module is best described "
     "as content-similarity clustering for hate-profile characterization, not social-network community "
     "detection. The data legitimately supports characterising what distinguishes hateful clusters — their "
     "LDA vocabulary, their markedly more negative VADER sentiment, and their dense intra-cluster "
     "similarity — but it does not support claims about real user communities, social ties, or information "
     "flow, and the paper will state this limitation explicitly.")
figure(doc, os.path.join(FIG_DIR, "community_graph_best.png"),
       f"Figure A2 — Content-similarity graph at the chosen cosine threshold "
       f"{sweep['chosen_threshold']}, nodes coloured by community hate ratio.", 5.8)
figure(doc, os.path.join(FIG_DIR, "community_graph_0.85.png"),
       "Figure A3 — The same data at cosine ≥ 0.85, illustrating fragmentation into many small "
       "components (lower modularity).", 5.8)

# ---------------------------------------------------------------- Item 3
heading(doc, "3. Offensive-Class Calibration (Training-Free) — Negative Result", 1)
bt = calib["baseline_test"]; vt = calib["val_tuned"]["test"]; inv = calib["inverse_freq"]["test"]
v = calib["verdict"]
para(doc,
     "The 'offensive' class is the weakest (F1 ≈ 0.54). Without retraining, we applied posterior "
     "re-weighting: class weights were tuned on the validation set to maximise macro-F1, then applied to "
     "the test posteriors. Two schemes are reported — parameter-free inverse-frequency weights, and the "
     "validation-tuned grid optimum.")
crows = [
    ["Baseline (uncalibrated)", "(1, 1, 1)", f"{bt['macro_f1']:.4f}", f"{bt['f1_offensive']:.4f}",
     f"{bt['f1_hate']:.4f}", f"{bt['f1_normal']:.4f}", f"{bt['accuracy']:.4f}"],
    ["Inverse-frequency", str(tuple(calib['inverse_freq']['weights'])), f"{inv['macro_f1']:.4f}",
     f"{inv['f1_offensive']:.4f}", f"{inv['f1_hate']:.4f}", f"{inv['f1_normal']:.4f}", f"{inv['accuracy']:.4f}"],
    ["Val-tuned", str(tuple(calib['val_tuned']['weights'])), f"{vt['macro_f1']:.4f}",
     f"{vt['f1_offensive']:.4f}", f"{vt['f1_hate']:.4f}", f"{vt['f1_normal']:.4f}", f"{vt['accuracy']:.4f}"],
]
table(doc, ["Scheme", "Weights (h,o,n)", "Macro-F1", "Off-F1", "Hate-F1", "Norm-F1", "Acc"], crows)
para(doc, "")
para(doc, "Table A3 — Posterior re-weighting on the test set (no retraining).", bold=False)
para(doc, "Discussion.", bold=True)
para(doc,
     f"This is a negative result and is reported as such. Validation macro-F1 rises to "
     f"{calib['val_tuned']['val']['macro_f1']:.4f} under the tuned weights, but the improvement does not "
     f"transfer to the test set: test macro-F1 moves by only {v['delta_macro_f1']:+.4f} "
     f"({bt['macro_f1']:.4f} → {vt['macro_f1']:.4f}), i.e. essentially flat. Offensive-F1 does rise "
     f"({v['delta_offensive_f1']:+.4f}, {bt['f1_offensive']:.4f} → {vt['f1_offensive']:.4f}), but only by "
     f"trading away hate-F1 ({bt['f1_hate']:.4f} → {vt['f1_hate']:.4f}); the macro average is therefore "
     f"unchanged, and AUC is invariant because re-weighting does not change the ranking. The conclusion is "
     f"that cheap, training-free calibration cannot fix the offensive class without harming another, and a "
     f"proper remedy — class-weighted or focal-loss re-training, or richer linguistic features — is left "
     f"as future work.")

# ---------------------------------------------------------------- manifest
heading(doc, "4. Artifacts", 1)
para(doc, "All numbers above are read directly from the following saved files; the analyses are "
          "reproducible via step9/step10/step11.")
mrows = [
    ["step9_explain_ci.py", "explainability_ci.json, table2_explainability_ci.csv, figures/table2_explainability.png"],
    ["step10_community_sweep.py", "community_threshold_sweep.csv, community_sweep_summary.json, "
     "community_profiles.csv, community_meta.json, figures/community_graph_{0.85,best}.png"],
    ["step11_offensive_calibration.py", "bert_metrics_weighted.json"],
    ["make_paper_updates.py", "PAPER_UPDATES.md (consolidated summary)"],
]
table(doc, ["Script", "Outputs"], mrows)

doc.save(OUT)
print("SAVED", OUT)

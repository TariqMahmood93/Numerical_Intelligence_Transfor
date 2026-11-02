"""Generates reports/methodology.md and reports/experimental_report.md from the saved results.

Every number in the experimental report is read from results/metrics.json. Findings are classified
with explicit, pre-declared decision rules (see `verdict`) rather than written by hand.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

COMPOSITE = ["equivalence.equivalence_auc_hard", "retrieval.mrr", "distance.same_template_log_distance_spearman",
             "ordering.probe_pairwise_order_acc", "ordering.triplet_betweenness_acc",
             "context.dimension_discrimination_acc"]

LABELS = {  # display names, matching the paper
    "exp1_original_minilm": "Base (original MiniLM)",
    "exp2_parser_plus_original": "Base+Parse (original + parser)",
    "ref_parser_only": "ParseOnly (parser alone)",
    "exp3_math_full": "NumBridge-Exact (exact supervision, all pairs)",
    "exp3b_math_budget": "NumBridge-Matched (exact labels on the reasoner's pairs; reasoner-free control)",
    "exp4a_teacher_raw": "NumBridge-Raw (raw LLM-reasoner labels)",
    "exp4b_teacher_verified": "NumBridge-Verified (verified LLM-reasoner labels)",
    "exp5_hybrid_finetuned_plus_parser": "NumBridge-Exact+Parse (Exact + parser)",
    "exp6_learned_projection": "NumBridge-Dual (learned numerical subspace, no parser)",
    "exp3_math_full_absdist": "NumBridge-Abs (absolute-distance target)",
}


def verdict(delta: float, noise: float, min_effect: float = 0.01) -> str:
    """Pre-declared rule: an improvement is 'supported' when it exceeds both a minimum effect size and
    2x the noise estimate (seed std, or half the bootstrap CI width); 'contradicted' when it is a
    decrease of the same magnitude; otherwise 'inconclusive'."""
    if delta is None or not np.isfinite(delta):
        return "not measured"
    thr = max(min_effect, 2 * (noise or 0))
    if delta > thr:
        return "supported"
    if delta < -thr:
        return "contradicted"
    return "inconclusive"


def fmt(x, std=None):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.3f}" + (f" ± {std:.3f}" if std is not None and np.isfinite(std) and std > 0 else "")


def table(agg: pd.DataFrame, cols: list[tuple[str, str]], rows: list[str]) -> str:
    head = "| Configuration | " + " | ".join(l for _, l in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        if r not in agg.index:
            continue
        cells = []
        for c, _ in cols:
            v = agg.loc[r].get(c, np.nan)
            s = agg.loc[r].get(c + "__std", np.nan)
            cells.append(fmt(v, s))
        body += f"| {LABELS.get(r, r)} | " + " | ".join(cells) + " |\n"
    return head + body


def g(agg, row, col):
    try:
        v = agg.loc[row, col]
        return float(v) if pd.notna(v) else None
    except KeyError:
        return None


def gs(agg, row, col):
    v = g(agg, row, col + "__std")
    return 0.0 if v is None else v


def write_methodology(cfg: dict, env: dict, audit: dict, path: str):
    tc = cfg["training"]
    txt = f"""# Methodology

## 1. Data
Synthetic statements are generated deterministically (seed {cfg['seed']}) by `src/dataset_generator.py`.
Ground truth for every relation (equivalence, order, ratio, log distance, relative absolute distance,
dimension, plausibility) is computed in Python by `src/numerical_parser.py`; no LLM output is used as
ground truth.

* Canonical units: mass → g, length → m (distance is a *context* of length), time → s, temperature → K,
  volume → L, proportion → dimensionless.
* Temperature is affine: K = C + 273.15, K = (F − 32)·5/9 + 273.15. All ratios and log distances use the
  absolute (kelvin) scale; Celsius/Fahrenheit ratios are never computed.
* Equivalence: same dimension and |a − b| ≤ 1e-9·max(|a|,|b|) on the canonical scale. 10, 10.0 and 10.00 are
  equivalent; 10.01 is not.
* Number formats: integers, decimals, negatives (temperatures), thousands separators, scientific notation
  (`2.5e6`); a test-only stress slice spells integers out in words (unreadable by the parser).

Dataset audit: {json.dumps(audit)}

### Leakage control
Statements are keyed by quantity (dimension, canonical value). All statements expressing the same quantity —
in any unit, precision, context or category — share an `equivalence_group` and the same split.
Multi-quantity sets (ordering triplets, relation sets, contradiction pairs) are placed in one split as a whole.
Templates ending in `_ho` are held out of train/validation and appear only in test. Pairs and triplets are
built strictly within each split. Audit: equivalence groups crossing splits =
{audit.get('equivalence_groups_crossing_splits')}, duplicate sentences = {audit.get('duplicate_sentences')},
held-out-template rows outside test = {audit.get('held_out_template_rows_outside_test')}.

## 2. Teacher supervision
The teacher is queried with fixed prompts (`src/teacher_supervision.py`). Yes/No and A/B/Equal questions are
answered by one forward pass, reading the next-token distribution over the option tokens; this gives a hard
label and a soft probability. The ratio question uses greedy decoding (max {cfg['teacher']['max_new_tokens']}
new tokens). Every determinable answer is checked against the Python ground truth. A pair is *verified* when all
of its determinable answers are correct.

## 3. Student objectives
Let e(x) be the L2-normalised student embedding and s(x,y) = e(x)·e(y).

* **L_eq** (weighted symmetric InfoNCE, τ = {tc['tau']}) over same-context equivalent pairs (10 kg ↔ 10000 g). Equivalent
  statements in *different* contexts are deliberately not positives; they go to L_ctx.
* **L_ord** = pairwise margin ranking on a learned direction u (r(x) = u·e(x), margin {tc['margins']['rank']}) +
  triplet betweenness on lo < mid < hi: max(0, s(lo,hi) − s(lo,mid) + m) + max(0, s(lo,hi) − s(mid,hi) + m), m = {tc['margins']['triplet']}.
* **L_dist** = MSE between s(a,b) and t(Δ) = c_min + (c_max − c_min)·exp(−Δ/σ) for same-dimension,
  non-equivalent pairs; c_min = {tc['distance']['c_min']}, c_max = {tc['distance']['c_max']}.
  Log mode: Δ = |log10(a/b)|, σ = {tc['distance']['sigma_log']}. Absolute mode: Δ = |a − b| / max(|a|,|b|), σ = {tc['distance']['sigma_abs']}.
* **L_ctx** = mean max(0, s(a,b) − m_k): different dimensions m = {tc['margins']['cross']}; same quantity in different
  contexts m = {tc['margins']['context']}; plausible vs contradictory m = {tc['margins']['plausibility']}.
* **L_sem** = mean(1 − cos(e(g), f0(g))) on unlabeled general sentences g (STS-B *train* sentences, labels unused),
  where f0 is the frozen pretrained model.

L_total = λ1 L_eq + λ2 L_ord + λ3 L_dist + λ4 L_ctx + λ5 L_sem with λ = {tc['lambdas']}.

Exp6 adds a linear projection P (384 → {tc['proj_dim']}). The numerical losses act on normalise(P·e(x)) and L_sem
on e(x), so numbers can be represented in a separate subspace. Its similarity is α·cos(e) + (1 − α)·cos(Pe),
with α tuned on validation. No parser is involved.

## 4. Training
AdamW, lr {tc['lr']}, weight decay {tc['weight_decay']}, linear warm-up ({tc['warmup_ratio']}) and decay, gradient clipping
{tc['max_grad_norm']}, {tc['epochs']} epochs × {tc['steps_per_epoch']} steps, batch {tc['batch_size']} per objective, max sequence length
{tc['max_seq_length']}. The checkpoint is selected by the *unweighted* sum of validation losses on a fixed set of
ground-truth validation batches. This criterion is identical for every run, so ablations are selected the same way.
Seeds: {cfg['seeds']} for the main configurations; ablations use seed {cfg['seed']}.

## 5. Configurations
| ID | Description |
|---|---|
| Exp1 | Original all-MiniLM-L6-v2, cosine |
| Exp2 | Original embeddings + parser channel: s = α·cos + (1−α)·s_num; probes see [e; parser features] |
| Ref | Parser only (upper reference on parser-readable text) |
| Exp3 | Fine-tuned with direct mathematical supervision on all training pairs/triplets |
| Exp3b | Fine-tuned with mathematical labels on exactly the teacher's pair budget (**no-teacher ablation**) |
| Exp4a | Same pairs, raw teacher labels (soft-weighted) |
| Exp4b | Same pairs, teacher labels kept only where they agree with ground truth |
| Exp5 | Exp3 embeddings + parser channel (hybrid) |
| Exp6 | Learned numerical projection head, no parser |
| Exp3-abs | Exp3 with the absolute-distance target |
| Abl. A–E | Semantic only → +equivalence → +ordering → +distance → +context (= Exp3, seed 42) |

## 6. Metrics
* **Equivalence**: ROC-AUC and balanced accuracy separating same-context equivalent pairs from hard negatives
  (same dimension, ≤ 10× apart). The threshold is tuned on validation. Also reported: unit conversion (cross-unit
  positives) and precision (10 vs 10.0 vs 10.01).
* **Ordering**: (i) a linear probe (ridge, per dimension, α tuned on validation) from features to log10(value),
  reporting pairwise order accuracy and Spearman ρ on test; (ii) triplet betweenness accuracy in cosine geometry.
* **Distance**: Spearman/Pearson between 1 − s and |Δlog10| or relative absolute distance, on all same-dimension
  test pairs and on same-template pairs (which controls for context).
* **Retrieval**: each test statement queries all other test statements; relevant = same equivalence group;
  Recall@1/5/10 and MRR.
* **Context**: dimension discrimination (s(a, equivalent) > s(a, different dimension)); mean similarity of the
  same quantity in different contexts; linear context probe; linear plausibility probe on category G.
* **Semantic**: STS-B test Spearman/Pearson.
* **Parser-free stress slice**: numbers written in words. Retrieval, equivalence AUC against the digit source,
  and ordering probes trained on digit statements.
* **Teacher**: accuracy against ground truth per question type.
* Uncertainty: mean ± std over seeds for trained configurations; bootstrap 95 % CIs (n = {cfg['evaluation']['bootstrap']}) on
  test items for selected metrics.

## 7. Environment
```
{json.dumps(env, indent=1)}
```
"""
    with open(path, "w") as f:
        f.write(txt)


def write_report(metrics: dict, agg: pd.DataFrame, abl: pd.DataFrame | None, teacher_acc: dict, cfg: dict,
                 env: dict, audit: dict, path: str, mock: bool):
    A = agg
    base, hyb, ft = "exp1_original_minilm", "exp2_parser_plus_original", "exp3_math_full"
    A["numerical_composite"] = A[[c for c in COMPOSITE if c in A.columns]].mean(axis=1)
    std_cols = [c + "__std" for c in COMPOSITE if c + "__std" in A.columns]
    A["numerical_composite__std"] = A[std_cols].mean(axis=1) if std_cols else 0
    banner = ("> **SMOKE TEST: NOT EXPERIMENTAL RESULTS.** This report was produced with a randomly initialised tiny "
              "encoder and a *mock* teacher, only to check that the pipeline runs end to end. Do not interpret any number "
              "below.\n\n") if mock else ""

    def d(row, col):
        a, b = g(A, row, col), g(A, base, col)
        return None if a is None or b is None else a - b

    def v(row, col):
        delta = d(row, col)
        return verdict(delta, max(gs(A, row, col), gs(A, base, col)))

    main_rows = [base, hyb, "ref_parser_only", ft, "exp3b_math_budget", "exp4a_teacher_raw", "exp4b_teacher_verified",
                 "exp5_hybrid_finetuned_plus_parser", "exp6_learned_projection", "exp3_math_full_absdist"]
    t_num = table(A, [("equivalence.equivalence_auc_hard", "Equiv. AUC (hard)"),
                      ("equivalence.equivalence_bal_acc_hard", "Equiv. bal. acc"),
                      ("equivalence.unit_conversion_bal_acc", "Unit conv. bal. acc"),
                      ("equivalence.precision_bal_acc", "Precision bal. acc"),
                      ("ordering.probe_pairwise_order_acc", "Order acc (probe)"),
                      ("ordering.probe_rank_spearman", "Rank ρ (probe)"),
                      ("ordering.triplet_betweenness_acc", "Betweenness acc")], main_rows)
    t_dist = table(A, [("distance.same_template_log_distance_spearman", "ρ log-dist (same tpl)"),
                       ("distance.same_template_rel_abs_distance_spearman", "ρ abs-dist (same tpl)"),
                       ("distance.all_same_dim_log_distance_spearman", "ρ log-dist (all)"),
                       ("distance.all_same_dim_log_distance_pearson", "r log-dist (all)"),
                       ("retrieval.recall@1", "R@1"), ("retrieval.recall@5", "R@5"), ("retrieval.recall@10", "R@10"),
                       ("retrieval.mrr", "MRR"), ("retrieval_unseen_templates.unseen_mrr", "MRR unseen tpl")], main_rows)
    t_ctx = table(A, [("context.dimension_discrimination_acc", "Dim. discrimination"),
                      ("context.same_quantity_diff_context_mean_sim", "Sim same qty / diff ctx"),
                      ("context.context_probe_acc", "Context probe acc"),
                      ("context.plausibility_probe_acc", "Plausibility probe acc"),
                      ("semantic.stsb_spearman", "STS-B ρ"), ("semantic.stsb_pearson", "STS-B r")], main_rows)
    t_words = table(A, [("stress_words.words_mrr", "Words MRR"), ("stress_words.words_recall@1", "Words R@1"),
                        ("stress_words.words_equivalence_auc", "Words equiv. AUC"),
                        ("ordering.words_probe_pairwise_order_acc", "Words order acc (probe)"),
                        ("ordering.words_probe_rank_spearman", "Words rank ρ (probe)")], main_rows)
    t_cos = ""
    prof = metrics.get(base, {}).get("seed_42", {}).get("cosine_profile") or {}
    prof_ft = metrics.get(ft, {}).get("seed_42", {}).get("cosine_profile") or {}
    if prof:
        t_cos = "| Pair type | n | Exp1 mean cos | Exp3 mean cos (seed 42) |\n|---|---|---|---|\n"
        for k, val in prof.items():
            t_cos += f"| {k} | {val['n']} | {fmt(val['mean'])} | {fmt((prof_ft.get(k) or {}).get('mean'))} |\n"

    # ------------------------------------------------ RQ verdicts
    ta = teacher_acc or {}
    eqb, orda = ta.get("equivalence_balanced_acc"), ta.get("ordering_acc")
    if eqb is None:
        rq1 = "not measured"
    elif eqb >= 0.9 and (orda or 0) >= 0.9:
        rq1 = "reliable (≥ 0.90 on equivalence and ordering)"
    elif eqb >= 0.7 and (orda or 0) >= 0.7:
        rq1 = "partially reliable (0.70–0.90); usable only after verification"
    else:
        rq1 = "unreliable (< 0.70 on equivalence or ordering)"

    def cmp(a, b, col):
        x, y = g(A, a, col), g(A, b, col)
        if x is None or y is None:
            return None, "not measured"
        return x - y, verdict(x - y, max(gs(A, a, col), gs(A, b, col)))

    comp = "numerical_composite"
    rq2_d, rq2_v = cmp("exp4b_teacher_verified", base, comp)
    kd_d, kd_v = cmp("exp4b_teacher_verified", "exp3b_math_budget", comp)
    raw_d, raw_v = cmp("exp4a_teacher_raw", "exp3b_math_budget", comp)
    rq4_d, rq4_v = cmp(ft, base, comp)
    sts_d, sts_v = cmp(ft, base, "semantic.stsb_spearman")
    ctxp_d, ctxp_v = cmp(ft, base, "context.context_probe_acc")
    w_d, w_v = cmp(ft, base, "stress_words.words_mrr")
    hy_d, hy_v = cmp(hyb, ft, comp)
    h5_d, h5_v = cmp("exp5_hybrid_finetuned_plus_parser", hyb, comp)
    p6_d, p6_v = cmp("exp6_learned_projection", ft, comp)

    rq3_lines = ""
    for col, lab in (("equivalence.equivalence_auc_hard", "equivalence (AUC, hard negatives)"),
                     ("ordering.probe_pairwise_order_acc", "ordering (linear probe)"),
                     ("ordering.triplet_betweenness_acc", "ordering (cosine betweenness)"),
                     ("distance.same_template_log_distance_spearman", "log distance (Spearman)"),
                     ("distance.same_template_rel_abs_distance_spearman", "absolute distance (Spearman)")):
        dd = d(ft, col)
        rq3_lines += f"* {lab}: Exp1 {fmt(g(A, base, col))} → Exp3 {fmt(g(A, ft, col), gs(A, ft, col))} (Δ = {fmt(dd)}; **{v(ft, col)}**)\n"

    abl_txt = ""
    if abl is not None and len(abl):
        cols = [c for c in COMPOSITE + ["semantic.stsb_spearman"] if c in abl.columns]
        abl_txt = "| Variant | " + " | ".join(c.split(".")[-1] for c in cols) + " | composite |\n|---|" + "---|" * (len(cols) + 1) + "\n"
        comp_vals = abl[[c for c in COMPOSITE if c in abl.columns]].mean(axis=1)
        for idx, r in abl.iterrows():
            abl_txt += f"| {idx} | " + " | ".join(fmt(r[c]) for c in cols) + f" | {fmt(comp_vals[idx])} |\n"
        steps = list(abl.index)
        gains = [(steps[i], comp_vals.iloc[i] - comp_vals.iloc[i - 1]) for i in range(1, len(steps))]
        if gains:
            best_step = max(gains, key=lambda x: x[1])
            abl_txt += (f"\nLargest single-step change in the numerical composite: adding **{best_step[0]}** "
                        f"(Δ = {best_step[1]:+.3f}). Step deltas: "
                        + ", ".join(f"{s}: {x:+.3f}" for s, x in gains) + ". The ablation uses one seed, so step "
                        "deltas smaller than the seed std of Exp3 (composite std "
                        f"{fmt(gs(A, ft, comp))}) should be read as inconclusive.\n")

    t_teacher = ""
    if ta:
        keys = ["same_dimension_acc", "equivalence_acc", "equivalence_balanced_acc", "equivalence_auc", "majority_baseline_equivalence",
                "ordering_acc", "unit_conversion_acc_cross_unit_equiv", "ratio_acc_within_1pct", "ratio_parse_rate",
                "compatibility_balanced_acc", "fully_verified_fraction", "n_pairs", "teacher_seconds"]
        t_teacher = "| Measure | Value |\n|---|---|\n" + "".join(f"| {k} | {ta.get(k)} |\n" for k in keys)
        bpt = ta.get("by_pair_type", {})
        if bpt:
            t_teacher += "\n| Pair type | n | equiv acc | dim acc | order acc |\n|---|---|---|---|---|\n"
            for k, val in bpt.items():
                t_teacher += f"| {k} | {val['n']} | {val['equiv_acc']} | {val['dim_acc']} | {val['order_acc']} |\n"

    # classification of experiments
    def status(row):
        if row not in A.index:
            return "failed / not run"
        dd = d(row, comp)
        vv = verdict(dd, max(gs(A, row, comp), gs(A, base, comp)))
        return {"supported": "successful (numerical composite above baseline beyond noise)",
                "contradicted": "failed (numerical composite below baseline)",
                "inconclusive": "inconclusive (difference within noise)"}.get(vv, vv)

    status_lines = "".join(f"| {LABELS.get(r, r)} | {fmt(g(A, r, comp), gs(A, r, comp))} | {status(r)} |\n"
                           for r in main_rows if r != base and r in A.index)

    fig = lambda n, cap: f"![{cap}](../results/figures/{n})\n\n*{cap}*\n\n"

    txt = f"""# Numerical Intelligence Transfer: Transferring Numerical Reasoning from a Small LLM into a Sentence-Embedding Space
## Preliminary experimental report

{banner}**Student:** {env.get('student')} · **Teacher:** {ta.get('teacher_model')} · **Seeds:** {cfg['seeds']} ·
**Hardware:** {env.get('hardware_summary')} · **Generated:** {env.get('timestamp')}

## 1. Abstract
We test whether numerical relations (equivalence across units, ordering, magnitude distance and dimensional and
contextual compatibility) can be transferred into all-MiniLM-L6-v2 without destroying its semantic space. The
supervision is either direct and mathematical or comes from a small instruction-tuned LLM teacher whose answers are
checked against Python ground truth. On a leakage-controlled synthetic benchmark of {audit.get('n_statements')} statements,
the teacher's equivalence accuracy against Python ground truth (balanced) is {fmt(eqb)} and its ordering accuracy is {fmt(orda)} (RQ1: {rq1}).
Fine-tuning with direct supervision changes the numerical composite from {fmt(g(A, base, comp))} (Exp1) to
{fmt(g(A, ft, comp), gs(A, ft, comp))} (Exp3; {rq4_v}). STS-B Spearman goes from {fmt(g(A, base, 'semantic.stsb_spearman'))} to
{fmt(g(A, ft, 'semantic.stsb_spearman'), gs(A, ft, 'semantic.stsb_spearman'))}. Teacher-verified distillation compared with
budget-matched mathematical supervision gives Δ = {fmt(kd_d)} ({kd_v}). On the parser-free slice, with numbers written in words,
fine-tuning changes equivalence-retrieval MRR by {fmt(w_d)} ({w_v}). This tests whether the gains live in the embedding
itself rather than in a parser.

## 2. Research motivation
Dense sentence encoders are the retrieval backbone of RAG, entity resolution and semantic search. They are known to
treat numbers mostly as tokens: "10 kg" and "10000 g" may be far apart, while "10 kg" and "10 m" sit close together.
Numerical mismatches in retrieval then propagate silently into downstream reasoning. Small LLMs show some numerical
competence but are too expensive to run on every retrieval comparison. If their numerical knowledge can be distilled
into a 22M-parameter encoder, numerically aware retrieval becomes cheap. Whether the encoder actually *learns* numbers,
or whether a symbolic parser is doing the work, is the central methodological question.

## 3. Research questions
* **RQ1** Can a small LLM provide reliable numerical supervision? → **{rq1}**
* **RQ2** Can numerical knowledge be transferred into all-MiniLM-L6-v2 through distillation? → Exp4b vs Exp1, composite
  Δ = {fmt(rq2_d)} (**{rq2_v}**). Distillation vs direct supervision on the same budget: Δ = {fmt(kd_d)} (**{kd_v}**).
  Raw (unverified) teacher vs direct supervision: Δ = {fmt(raw_d)} (**{raw_v}**).
* **RQ3** Can the embedding space preserve equivalence, ordering and distance? → see §9; per-property verdicts below.
* **RQ4** Does transfer improve numerical similarity over the original model? → Exp3 vs Exp1, composite Δ = {fmt(rq4_d)} (**{rq4_v}**).
* **RQ5** Does fine-tuning damage semantic understanding? → STS-B Δ = {fmt(sts_d)} (**{'damage' if sts_v == 'contradicted' else ('no measurable damage' if sts_v == 'inconclusive' else sts_v)}**);
  context-probe Δ = {fmt(ctxp_d)} ({ctxp_v}).

{'**Single seed: seed variance is not estimated, so verdicts below rely on the 0.01 minimum-effect floor only.** ' if len(cfg['seeds']) == 1 else ''}Verdicts follow a pre-declared rule (`src/report_writer.py::verdict`). A change is *supported* when it exceeds max(0.01, 2 × noise),
where noise is the seed standard deviation. It is *contradicted* when it is a decrease of that size, and *inconclusive*
otherwise.

## 4. Related research and positioning
* **Numeracy probing.** Probing work on word and contextual embeddings found that magnitude is partly linearly decodable
  but extrapolates poorly (Naik et al., ACL 2019; Wallace et al., EMNLP 2019). The same holds for physical scales
  (Zhang et al., Findings of EMNLP 2020). Thawani et al. (NAACL 2021) survey number representations.
* **Numeracy-preserving representations.** These include explicit numeric encodings such as DICE (Sundararaman et al.,
  EMNLP 2020), numeracy for language models (Spithourakis & Riedel, ACL 2018) and injecting numerical reasoning
  through synthetic data (Geva et al., ACL 2020).
* **Sentence embeddings and distillation.** This work builds on SBERT (Reimers & Gurevych, EMNLP 2019) and MiniLM
  (Wang et al., NeurIPS 2020), on knowledge distillation (Hinton et al., 2015) and on embedding-space distillation with
  a frozen-teacher regulariser (Reimers & Gurevych, EMNLP 2020). It also relates to continual-learning regularisers
  that preserve pretrained behaviour, such as EWC (Kirkpatrick et al., PNAS 2017) and L2-SP (Li et al., ICML 2018).
* **Positioning.** Prior work mostly probes numeracy or adds explicit numeric channels. This study instead (i) uses a
  small LLM as a *verified* teacher at the sentence-embedding level, (ii) separates the teacher's contribution from
  direct mathematical supervision, and (iii) separates embedding-internal learning from parser-driven gains, using a
  parser-free stress slice and a learned projection head.
  *(Check these citations against the original papers before any submission.)*

## 5. Proposed architecture
Data → Python ground truth → (optional) LLM teacher → verification filter → multi-objective student fine-tuning
(L_eq, L_ord, L_dist, L_ctx, L_sem) → three representation choices:
(a) a single cosine space (Exp3/4), (b) a learned numerical projection head beside the semantic space (Exp6),
(c) a symbolic parser channel fused with the embedding (Exp2/5). The exact formulation is in `reports/methodology.md`.

## 6. Dataset construction
{audit.get('n_statements')} statements; split {audit.get('by_split')}; categories {audit.get('by_category')};
dimensions {audit.get('by_dimension')}; parser-free word slice: {audit.get('n_stress_statements')} statements.
Leakage audit: groups crossing splits = {audit.get('equivalence_groups_crossing_splits')}, duplicates = {audit.get('duplicate_sentences')},
held-out templates outside test = {audit.get('held_out_template_rows_outside_test')}.

## 7. Experimental methodology
Ten configurations plus a five-step ablation were evaluated on identical test data (`reports/methodology.md` §5–6).
Thresholds and hybrid weights are tuned on validation only.

## 8. Implementation details
Python {env.get('python')}, PyTorch {env.get('torch')}, sentence-transformers {env.get('sentence_transformers')},
transformers {env.get('transformers')}; student revision {env.get('student_revision')}; teacher revision {ta.get('teacher_revision')}.
Batch {cfg['training']['batch_size']}, lr {cfg['training']['lr']}, {cfg['training']['epochs']} epochs × {cfg['training']['steps_per_epoch']} steps.
Training times and per-epoch training/validation losses: `results/training_logs.json` and Figure 9.

## 9. Quantitative results

### 9.1 Teacher reliability (RQ1)
{t_teacher}
{fig('07_teacher_accuracy.png', 'Teacher accuracy against Python ground truth')}

### 9.2 Baseline cosine analysis
{t_cos}
{fig('01_cosine_baseline_vs_finetuned.png', 'Figure 1. Cosine similarity by relation, baseline vs fine-tuned')}

### 9.3 Numerical equivalence and ordering
{t_num}
{fig('04_ordering_accuracy.png', 'Figure 4. Ordering accuracy')}

### 9.4 Distance and retrieval
{t_dist}
{fig('02_numerical_vs_embedding_distance.png', 'Figure 2. Numerical vs embedding distance')}
{fig('03_equivalence_retrieval.png', 'Figure 3. Equivalence retrieval')}

### 9.5 Context and semantics (RQ5)
{t_ctx}
{fig('05_semantic_preservation.png', 'Figure 5. Semantic and contextual preservation')}

### 9.6 Parser-free stress slice (numbers in words)
{t_words}
The regex parser reads none of these statements, so Exp2 and Exp5 fall back to their embedding component here. Gains on
this slice can only come from the embedding.

{fig('06_pca_embedding_space.png', 'Figure 6. PCA of the embedding space')}

## 10. Comparative analysis
**RQ3, property by property (Exp3 vs Exp1):**
{rq3_lines}
**Parser vs learned representation.** Exp2 (parser + original) vs Exp3 (fine-tuned): composite Δ = {fmt(hy_d)} ({hy_v}).
Exp5 (parser + fine-tuned) vs Exp2: Δ = {fmt(h5_d)} ({h5_v}). Learned projection (Exp6) vs single-space fine-tuning (Exp3):
Δ = {fmt(p6_d)} ({p6_v}). Exp2 gains that do not carry over to the word slice should be attributed to the parser, not to
learned numeracy.

**Teacher contribution.** With verification, the teacher's determinable labels match the ground truth by construction.
The only information distillation adds beyond direct supervision is (i) which pairs survive the filter and (ii) soft
confidence weights. Exp4b vs Exp3b isolates exactly that (Δ = {fmt(kd_d)}, {kd_v}). Exp4a shows the cost of using
unverified labels (Δ = {fmt(raw_d)} vs Exp3b, {raw_v}).

| Configuration | Numerical composite | Status |
|---|---|---|
{status_lines}
## 11. Ablation study
{abl_txt}
{fig('08_ablation_heatmap.png', 'Figure 8. Ablation A–E')}
{fig('09_training_curves.png', 'Figure 9. Training curves')}

## 12. Limitations
* The data is synthetic and template-based. Real text has free word order, implicit units, ranges, approximations and
  multiple quantities per sentence.
* Only one quantity per statement is evaluated; compositional statements (category F explicit arithmetic) are evaluated
  only through their result.
* The teacher budget is bounded by hardware. With the 0.5B fallback teacher (CPU or small GPU), teacher conclusions do
  not transfer to the 1.5B model.
* Outside category G, values are deliberately decoupled from real-world plausibility (for example a 12 m³ syrup dose) to
  isolate numerical reasoning. Plausibility is therefore only labelled in G.
* Cosine similarity is symmetric, so ordering cannot be expressed as a similarity. It is measured through a linear probe
  and through betweenness, which are indirect.

## 13. Threats to validity
* **Internal.** Hyperparameters (λ, margins, σ) were set a priori and not tuned per configuration. A different
  weighting could change the ablation ranking. Model selection uses validation losses only.
* **Construct.** The "numerical composite" averages heterogeneous metrics; per-metric tables are primary.
  Probe-based metrics measure linear decodability, not use of that information by downstream tasks.
* **External.** Templates are shared between train and test except for the held-out templates; see the
  `unseen_templates` columns. The word slice tests format shift, not new vocabulary.
* **Statistical.** {len(cfg['seeds'])} seeds for main configurations and 1 for ablations. Bootstrap CIs cover test-item
  sampling, not training variance.
* **Leakage.** Quantity-level grouping prevents identical quantities crossing splits. Templates and units are shared by
  design.

## 14. Research findings
1. Teacher reliability: {rq1}.
2. Direct numerical fine-tuning vs original (composite): {rq4_v} (Δ = {fmt(rq4_d)}).
3. Semantic preservation (STS-B): Δ = {fmt(sts_d)} ({sts_v}).
4. Distillation adds beyond budget-matched direct supervision: {kd_v} (Δ = {fmt(kd_d)}).
5. Embedding-internal gains on parser-free text: {w_v} (Δ MRR = {fmt(w_d)}).
6. Parser hybrid vs learned embedding: {hy_v} (Δ = {fmt(hy_d)}).

## 15. Future research directions
* Real-world numeric retrieval benchmarks (product specs, clinical notes, scientific tables) and entity-resolution
  settings with numeric attributes.
* Teacher signals that are *not* determinable by a parser: plausibility, implicit units, approximate quantities and
  ranges, where distillation can genuinely add information.
* Asymmetric or order-aware heads (box or order embeddings) instead of forcing order into cosine geometry.
* Larger teachers, self-consistency or logit ensembling, and calibration of teacher soft labels.
* Number-aware tokenisation or digit-level encoders inside the student.
* Multi-quantity sentences, compositional arithmetic and multilingual numerals.
"""
    with open(path, "w") as f:
        f.write(txt)

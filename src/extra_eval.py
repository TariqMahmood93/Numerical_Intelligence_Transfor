#!/usr/bin/env python3
"""Evaluation-only follow-ups requested in review (no retraining; reuses saved checkpoints).

    python src/extra_eval.py                       # after main.py has finished (models/ and data/ present)
    python src/extra_eval.py --root RUN_DIR

Writes results/extra/:
  words_nosource.csv   Q8  parser-free retrieval with each spelled-out query's digit twin removed from the gallery
  format_strata.csv    Q2  equivalence AUC, rho_log and MRR split by number format
  alpha_sweep.csv      Q4  NumBridge-Dual and NumBridge-Exact+Parse across the mixing weight alpha (incl. STS-B)
  failure_cases.md     Q10 concrete failures of NumBridge-Exact (seed 42)
  extra_metrics.json   everything above, for the paper
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from evaluation import EQ_NEG_TYPES, Space, retrieval_block, safe_auc, tune_alpha
from numerical_distillation import encode
from numerical_parser import parse_main

RUNS = {  # display name -> (run directory stem or "baseline", similarity mode)
    "Base": ("baseline", "cos"),
    "Base+Parse": ("baseline", "hybrid_parser"),
    "NumBridge-Exact": ("exp3_math_full", "cos"),
    "NumBridge-Matched": ("exp3b_math_budget", "cos"),
    "NumBridge-Raw": ("exp4a_teacher_raw", "cos"),
    "NumBridge-Verified": ("exp4b_teacher_verified", "cos"),
    "NumBridge-Exact+Parse": ("exp3_math_full", "hybrid_parser"),
    "NumBridge-Dual": ("exp6_learned_projection", "hybrid_proj"),
    "NumBridge-Abs": ("exp3_math_full_absdist", "cos"),
}
FORMATS = ["integer", "decimal", "separator", "scientific", "negative"]


def number_format(sentence: str) -> str:
    q = parse_main(sentence)
    if q is None:
        return "words"
    tok = sentence[q.span[0]:q.span[1]].strip().split()[0]
    if re.search(r"\d[eE][-+]?\d", tok):
        return "scientific"
    if re.search(r"\d,\d{3}", tok):
        return "separator"
    if tok.startswith(("-", "−")):
        return "negative"
    if "." in tok:
        return "decimal"
    return "integer"


def words_nosource(sp, stress, test, test_pairs):
    """Q8: words-slice retrieval and equivalence AUC with the digit source excluded."""
    src = test.set_index("id").sentence
    gal_grp = test.equivalence_group.values
    gids = sp.ids(test.sentence)
    rr, hits = [], {1: [], 5: [], 10: []}
    pos_a, pos_b = [], []
    for r in stress.itertuples():
        s_src = src.get(r.source_id)
        mask = test.sentence.values != s_src
        rel = (gal_grp == r.equivalence_group) & mask
        if rel.sum() == 0:
            continue
        qi = sp.idx[r.sentence]
        s = sp.sim(np.full(mask.sum(), qi), gids[mask])
        first = (-s).argsort().argsort()[rel[mask]].min() + 1
        rr.append(1.0 / first)
        for k in hits:
            hits[k].append(float(first <= k))
        for b in test.sentence.values[rel]:
            pos_a.append(r.sentence); pos_b.append(b)
    out = {f"nosrc_recall@{k}": float(np.mean(v)) if v else None for k, v in hits.items()}
    out["nosrc_mrr"] = float(np.mean(rr)) if rr else None
    out["nosrc_n_queries"] = len(rr)
    # negatives: the digit source's hard-negative partners (same dimension, <=10x apart, not equivalent)
    tp = test_pairs[test_pairs.pair_type.isin(EQ_NEG_TYPES)]
    partners = {}
    for a, b in zip(tp.a_sentence, tp.b_sentence):
        partners.setdefault(a, []).append(b); partners.setdefault(b, []).append(a)
    neg_a, neg_b = [], []
    for r in stress.itertuples():
        for n in partners.get(src.get(r.source_id), [])[:3]:
            neg_a.append(r.sentence); neg_b.append(n)
    if pos_a and neg_a:
        sc = np.concatenate([sp.sim(sp.ids(pos_a), sp.ids(pos_b)), sp.sim(sp.ids(neg_a), sp.ids(neg_b))])
        y = np.array([True] * len(pos_a) + [False] * len(neg_a))
        out["nosrc_equivalence_auc"] = safe_auc(y, sc)
        out["nosrc_n_pos"], out["nosrc_n_neg"] = len(pos_a), len(neg_a)
    return out


def format_strata(sp, test, test_pairs, fmt):
    """Q2: metrics split by the number format of the statements involved."""
    rows = []
    eq = test_pairs[test_pairs.pair_type.isin(("equivalent",) + EQ_NEG_TYPES)]
    sd = test_pairs[test_pairs.same_dimension & test_pairs.same_template]
    for f in FORMATS:
        r = {"format": f, "n_statements": int((test.sentence.map(fmt) == f).sum())}
        m = eq.a_sentence.map(fmt).eq(f) | eq.b_sentence.map(fmt).eq(f)
        g = eq[m]
        if g.equivalent.nunique() == 2:
            r["equivalence_auc"] = safe_auc(g.equivalent.values, sp.sim(sp.ids(g.a_sentence), sp.ids(g.b_sentence)))
            r["n_pairs_auc"] = len(g)
        m = sd.a_sentence.map(fmt).eq(f) | sd.b_sentence.map(fmt).eq(f)
        g = sd[m]
        if len(g) >= 10:
            r["rho_log"] = float(spearmanr(1 - sp.sim(sp.ids(g.a_sentence), sp.ids(g.b_sentence)), g.log_distance).correlation)
            r["n_pairs_rho"] = len(g)
        q = test[test.sentence.map(fmt) == f]
        if len(q):
            r["mrr"] = retrieval_block(sp, test, queries=q)["mrr"]
        rows.append(r)
    return rows


def alpha_sweep(sp, test, test_pairs, sts, alphas=np.linspace(0, 1, 11)):
    """Q4: numerical and semantic quality as a function of the mixing weight alpha."""
    eq = test_pairs[test_pairs.pair_type.isin(("equivalent",) + EQ_NEG_TYPES)]
    sd = test_pairs[test_pairs.same_dimension & test_pairs.same_template]
    rows, keep = [], sp.alpha
    for a in alphas:
        sp.alpha = float(a)
        r = {"alpha": round(float(a), 2),
             "equivalence_auc": safe_auc(eq.equivalent.values, sp.sim(sp.ids(eq.a_sentence), sp.ids(eq.b_sentence))),
             "rho_log": float(spearmanr(1 - sp.sim(sp.ids(sd.a_sentence), sp.ids(sd.b_sentence)), sd.log_distance).correlation),
             "mrr": retrieval_block(sp, test)["mrr"]}
        if sts is not None:
            r["stsb_spearman"] = float(spearmanr(sp.sim(sp.ids(sts.sentence1), sp.ids(sts.sentence2)), sts.score).correlation)
        rows.append(r)
    sp.alpha = keep
    return rows


def failure_cases(sp, test, test_pairs, k=8):
    """Q10: where NumBridge-Exact still goes wrong."""
    tp = test_pairs.assign(sim=sp.sim(sp.ids(test_pairs.a_sentence), sp.ids(test_pairs.b_sentence)))
    cat = test.set_index("sentence").category
    sections = [
        ("Equivalent pairs it scores lowest", tp[tp.pair_type == "equivalent"].nsmallest(k, "sim")),
        ("Hard negatives (same dimension, at most 10x apart) it scores highest", tp[tp.pair_type.isin(EQ_NEG_TYPES)].nlargest(k, "sim")),
        ("Different-dimension pairs it scores highest", tp[tp.pair_type == "cross_dimension"].nlargest(k, "sim")),
        ("Precision pairs (category E) it scores highest although not equal",
         tp[tp.a_sentence.map(cat).eq("E_precision") & tp.b_sentence.map(cat).eq("E_precision") & ~tp.equivalent].nlargest(k, "sim")),
    ]
    md = ["# Failure cases of NumBridge-Exact (seed 42)\n"]
    for title, g in sections:
        md.append(f"\n## {title}\n\n| similarity | A | B | ratio A/B |\n|---|---|---|---|")
        for r in g.itertuples():
            ratio = "" if pd.isna(r.ratio) else f"{r.ratio:.4g}"
            md.append(f"| {r.sim:.3f} | {r.a_sentence} | {r.b_sentence} | {ratio} |")
    return "\n".join(md) + "\n"


def load_sts(cfg):
    try:
        from datasets import load_dataset
        return load_dataset(cfg["semantic_corpus"]["stsb_dataset"], split="test").to_pandas()[["sentence1", "sentence2", "score"]]
    except Exception as e:  # noqa
        print(f"STS-B unavailable ({e}); alpha sweep runs without it")
        return None


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=here)
    ap.add_argument("--config", default=os.path.join(here, "config", "config.yaml"))
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(a.config))
    rd = lambda n: pd.read_csv(os.path.join(a.root, "data", n))
    test, test_pairs, val_pairs, stress = rd("test.csv"), rd("test_pairs.csv"), rd("validation_pairs.csv"), rd("stress_words_test.csv")
    sts = load_sts(cfg)
    sents = list(dict.fromkeys(list(test.sentence) + list(stress.sentence) + list(val_pairs.a_sentence) + list(val_pairs.b_sentence)
                               + ([] if sts is None else list(sts.sentence1) + list(sts.sentence2))))
    fmt_cache = {s: number_format(s) for s in test.sentence}
    fmt = lambda s: fmt_cache.get(s, number_format(s))
    out_dir = os.path.join(a.root, "results", "extra")
    os.makedirs(out_dir, exist_ok=True)

    words, strata, sweep, result = [], [], [], {"format_counts": pd.Series(fmt_cache).value_counts().to_dict()}
    enc_cache = {}
    for name, (stem, mode) in RUNS.items():
        seeds = [None] if stem == "baseline" else cfg["seeds"]
        for seed in seeds:
            path = os.path.join(a.root, "models", "baseline") if stem == "baseline" else \
                os.path.join(a.root, "models", "numerical_transfer", f"{stem}_s{seed}")
            if not os.path.isdir(path):
                continue
            if path not in enc_cache:
                print(f"encoding {os.path.basename(path)} on {dev}", flush=True)
                enc_cache[path] = encode(path, sents, dev)
            E, Z = enc_cache[path]
            sp = Space(sents, E, Z, mode, cfg)
            tune_alpha(sp, val_pairs)
            tag = dict(model=name, seed=seed if seed is not None else 42, alpha=getattr(sp, "alpha", None) if mode != "cos" else None)
            words.append({**tag, **words_nosource(sp, stress, test, test_pairs)})
            if seed in (None, cfg["seed"]):
                strata += [{**tag, **r} for r in format_strata(sp, test, test_pairs, fmt)]
                if mode != "cos":
                    sweep += [{"model": name, **r} for r in alpha_sweep(sp, test, test_pairs, sts)]
                if name == "NumBridge-Exact":
                    open(os.path.join(out_dir, "failure_cases.md"), "w").write(failure_cases(sp, test, test_pairs))

    W = pd.DataFrame(words); S = pd.DataFrame(strata); A = pd.DataFrame(sweep)
    W.to_csv(os.path.join(out_dir, "words_nosource.csv"), index=False)
    S.to_csv(os.path.join(out_dir, "format_strata.csv"), index=False)
    A.to_csv(os.path.join(out_dir, "alpha_sweep.csv"), index=False)
    num = W.drop(columns=["seed", "alpha"]).groupby("model")
    result["words_nosource_mean"] = num.mean().round(4).to_dict(orient="index")
    result["words_nosource_std"] = num.std().round(4).to_dict(orient="index")
    result["format_strata"] = S.round(4).to_dict(orient="records")
    result["alpha_sweep"] = A.round(4).to_dict(orient="records")
    json.dump(result, open(os.path.join(out_dir, "extra_metrics.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print("\n== Q8 words slice, digit source excluded (mean over seeds)\n", num.mean().round(3).to_string())
    print("\n== Q2 by number format (seed 42)\n", S.round(3).to_string(index=False))
    print("\n== Q4 alpha sweep\n", A.round(3).to_string(index=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

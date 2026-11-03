#!/usr/bin/env python3
"""Worked example: one numerical statement through the whole NumBridge pipeline.

    python src/demo_example.py                 # CPU is fine: only small encoders are loaded
    python src/demo_example.py --with-reasoner # also ask the LLM reasoner live (needs a GPU node)

Prints, for a few sentences about "10 kg":
  1. what Python's parser and ground truth say,
  2. the training signal each model would receive for these pairs,
  3. the cosine similarity of the ORIGINAL MiniLM and of the trained NumBridge models (seed 42).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml

from numerical_parser import CANONICAL_UNIT, equivalent, log_distance, parse_main

ANCHOR = "The package weighs 10 kilograms."
OTHERS = [
    ("The package weighs 10000 grams.", "same quantity, other unit"),
    ("The package weighs 10.5 kilograms.", "close value"),
    ("The package weighs 100 kilograms.", "10x larger"),
    ("The package weighs 10 grams.", "1000x smaller"),
    ("The package is 10 metres long.", "different dimension"),
    ("The package weighs ten kilograms.", "number in words"),
]
MODELS = [("Base (original MiniLM)", "models/baseline"),
          ("NumBridge-Exact", "models/numerical_transfer/exp3_math_full_s42"),
          ("NumBridge-Raw", "models/numerical_transfer/exp4a_teacher_raw_s42"),
          ("NumBridge-Verified", "models/numerical_transfer/exp4b_teacher_verified_s42")]


def training_signal(a, b, cfg):
    qa, qb = parse_main(a), parse_main(b)
    if qa is None or qb is None:
        return "none (the parser cannot read this number)"
    if qa.dimension != qb.dimension:
        return f"L_ctx: keep similarity below {cfg['training']['margins']['cross']} (different dimensions)"
    if equivalent(qa.dimension, qa.canonical, qb.dimension, qb.canonical):
        return "L_eq: pull together (equivalent quantities)"
    d = cfg["training"]["distance"]
    delta = log_distance(qa.canonical, qb.canonical)
    t = d["c_min"] + (d["c_max"] - d["c_min"]) * math.exp(-delta / d["sigma_log"])
    return f"L_dist: target similarity {t:.3f} (|log10 ratio| = {delta:.3f})"


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=here)
    ap.add_argument("--with-reasoner", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(here, "config", "config.yaml")))

    print("=" * 100, "\n1. PYTHON GROUND TRUTH (the referee)\n")
    for s in [ANCHOR] + [o for o, _ in OTHERS]:
        q = parse_main(s)
        print(f"  {s:42s} ->", "not parsed (number in words)" if q is None else
              f"dimension={q.dimension:8s} value={q.value:g} {q.unit:3s} -> canonical {q.canonical:g} {CANONICAL_UNIT[q.dimension]}")

    print("\n" + "=" * 100, f"\n2. TRAINING SIGNAL for pairs with \"{ANCHOR}\" (NumBridge-Exact uses these)\n")
    for o, lab in OTHERS:
        print(f"  vs {o:40s} [{lab}]\n      -> {training_signal(ANCHOR, o, cfg)}")

    print("\n" + "=" * 100, f"\n3. COSINE SIMILARITY with \"{ANCHOR}\" (seed-42 checkpoints)\n")
    from numerical_distillation import encode
    sents = [ANCHOR] + [o for o, _ in OTHERS]
    cols = []
    for name, rel in MODELS:
        path = os.path.join(a.root, rel)
        if os.path.isdir(path):
            E, _ = encode(path, sents, "cpu")
            cols.append((name, E[1:] @ E[0]))
    print("  " + f"{'compared with':42s}" + "".join(f"{n.split(' (')[0]:>20s}" for n, _ in cols))
    for k, (o, lab) in enumerate(OTHERS):
        print("  " + f"{o:42s}" + "".join(f"{c[k]:20.3f}" for _, c in cols))
    print("\n  Read across a row: how each model scores that pair. Ideal: 1st row highest, then close > 10x > 1000x,"
          "\n  different dimension lowest.")

    if a.with_reasoner:
        import torch
        from teacher_supervision import LLMTeacher, Q_TEXT
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t = LLMTeacher("Qwen/Qwen2.5-1.5B-Instruct", "bfloat16" if dev == "cuda" else "float32", dev, 4, 12)
        print("\n" + "=" * 100, "\n4. THE LLM REASONER (Qwen2.5-1.5B-Instruct), asked live\n")
        for o, lab in OTHERS[:2] + OTHERS[4:5]:
            for q in ("dim", "equiv", "order"):
                tmpl, opts = Q_TEXT[q]
                p = t.classify([tmpl.format(a=ANCHOR, b=o)], opts)[0]
                print(f"  vs {o:40s} {q:6s} -> {opts[int(np.argmax(p))]:6s} probabilities {dict(zip(opts, p.round(3)))}")


if __name__ == "__main__":
    main()

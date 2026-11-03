#!/usr/bin/env python3
"""Diagnose the LLM reasoner's answers (run on a GPU node; the 1.5B model is too large for a login node).

    python src/diagnose_teacher.py                      # Qwen/Qwen2.5-1.5B-Instruct, 150 sampled pairs
    python src/diagnose_teacher.py --model Qwen/Qwen2.5-0.5B-Instruct --n 100

It answers four questions:
  1. What does the model want to say first? (top next tokens and how much probability falls on the options).
     If little mass falls on the options, the renormalised option probabilities are not meaningful.
  2. Does batching / left padding change the answers?
  3. What does the model say in free text?
  4. Does a clearer wording of the questions change accuracy (prompt sensitivity)?
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from teacher_supervision import LLMTeacher, Q_TEXT

ALT = {
    "dim": ("Statement A: {a}\nStatement B: {b}\n\nQuestion: are A and B measurements of the same kind of quantity? "
            "The kinds are mass, length (including distance), time, temperature, volume and proportion. "
            "Answer Yes or No.", ["Yes", "No"]),
    "order": ("Statement A: {a}\nStatement B: {b}\n\nConvert both quantities to the same unit. Which quantity is "
              "larger? Reply A if A is larger, B if B is larger, or Equal if they are the same.", ["A", "B", "Equal"]),
}
FIXED = [("The package weighs 10 kilograms.", "The package weighs 10000 grams."),
         ("The package weighs 10 kilograms.", "The package is 10 metres long."),
         ("The board is 2 metres long.", "The board is 300 centimetres long.")]
EXPECT = {"dim": "Yes / No / Yes", "equiv": "Yes / No / No", "order": "Equal / (any) / B"}


@torch.no_grad()
def top_tokens(t: LLMTeacher, text: str, options, k=5):
    enc = t.tok([t._prompt(text)], return_tensors="pt").to(t.device)
    p = torch.softmax(t.model(**enc).logits[0, -1].float(), -1)
    top = torch.topk(p, k)
    mass = sum(float(p[i]) for ids in t._option_ids(options) for i in ids)
    return [(t.tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)], round(mass, 3)


def accuracy(t, tmpl, opts, df, truth):
    texts = [tmpl.format(a=r.a_sentence, b=r.b_sentence) for r in df.itertuples()]
    pred = np.array(opts)[t.classify(texts, opts).argmax(1)]
    return float((pred == truth).mean()), pd.Series(pred).value_counts().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--pairs", default="data/train_pairs.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if dev == "cuda" else "float32"
    t = LLMTeacher(a.model, dtype, dev, 8, 16)
    print(f"model {a.model} on {dev} ({dtype})")
    print("prompt ends with:", repr(t._prompt("X")[-50:]))
    print("option token ids:", {"Yes/No": t._option_ids(["Yes", "No"]), "A/B/Equal": t._option_ids(["A", "B", "Equal"])})

    print("\n=== 1-3. Fixed examples ===")
    for q in ("dim", "equiv", "order"):
        tmpl, opts = Q_TEXT[q]
        texts = [tmpl.format(a=x, b=y) for x, y in FIXED]
        t.bs = 3; batched = t.classify(texts, opts).round(3).tolist()
        t.bs = 1; single = t.classify(texts, opts).round(3).tolist()
        t.bs = 8
        print(f"\n[{q}] options {opts}; expected {EXPECT[q]}")
        print("  batched   :", batched)
        print("  one-by-one:", single)
        print("  free text :", t.generate(texts))
        for x in texts:
            tops, mass = top_tokens(t, x, opts)
            print(f"  top next tokens {tops} | probability on the options: {mass}")

    print(f"\n=== 4. Prompt sensitivity on {a.n} sampled training pairs ===")
    pairs = pd.read_csv(a.pairs)
    df = pairs.groupby("pair_type", group_keys=False).apply(lambda g: g.sample(min(len(g), a.n // 7 + 1), random_state=0))
    truth_dim = np.where(df.same_dimension, "Yes", "No")
    for name, (tmpl, opts) in (("original", Q_TEXT["dim"]), ("clearer", ALT["dim"])):
        acc, dist = accuracy(t, tmpl, opts, df, truth_dim)
        print(f"  same-dimension, {name:8s} prompt: accuracy {acc:.3f}  answers {dist}")
    sd = df[df.same_dimension]
    truth_ord = np.where(sd.order == 1, "A", np.where(sd.order == -1, "B", "Equal"))
    for name, (tmpl, opts) in (("original", Q_TEXT["order"]), ("clearer", ALT["order"])):
        acc, dist = accuracy(t, tmpl, opts, sd, truth_ord)
        print(f"  ordering,       {name:8s} prompt: accuracy {acc:.3f}  answers {dist}")
    print("\nmajority baselines: same-dim", round(max(np.mean(truth_dim == 'Yes'), np.mean(truth_dim == 'No')), 3),
          "| ordering", round(pd.Series(truth_ord).value_counts(normalize=True).max(), 3))


if __name__ == "__main__":
    main()

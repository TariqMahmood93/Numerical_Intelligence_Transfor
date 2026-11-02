"""Teacher LLM supervision and its verification against deterministic ground truth.

Classification-style questions are answered with a *single forward pass*: we read the next-token
distribution over the option tokens (e.g. "Yes"/"No", "A"/"B"/"Equal"). This yields both a hard
label and a soft probability (the distillation signal), and is far cheaper than free generation.
Only the numerical-ratio question uses short greedy generation.

Questions
    Q_dim    same physical dimension?                    Yes / No        (all pairs)
    Q_equiv  exactly the same quantity after conversion?  Yes / No        (all pairs)
    Q_order  which is larger after conversion?            A / B / Equal   (same-dimension pairs)
    Q_ratio  value of A / B after conversion (number)     generation      (same-dimension subset)
    Q_compat is the value plausible for the description?  Yes / No        (single statements, cat. G)

Every determinable answer is verified with Python (numerical_parser ground truth).
"""
from __future__ import annotations

import json
import math
import re
import time

import numpy as np
import pandas as pd

SYSTEM = ("You are a careful assistant for physical quantities. Convert units exactly before "
          "comparing. Temperatures must be compared on an absolute scale. Answer with only the "
          "requested option.")

Q_TEXT = {
    "dim": ("Statement A: {a}\nStatement B: {b}\n\nDo A and B measure the same physical dimension "
            "(for example, both a mass, both a length/distance, both a time, both a temperature, both a "
            "volume, or both a proportion)? Answer Yes or No.", ["Yes", "No"]),
    "equiv": ("Statement A: {a}\nStatement B: {b}\n\nAfter converting to a common unit, do A and B state "
              "exactly the same quantity? Answer Yes or No.", ["Yes", "No"]),
    "order": ("Statement A: {a}\nStatement B: {b}\n\nAfter converting to a common unit, which quantity is "
              "larger? Answer A, B, or Equal.", ["A", "B", "Equal"]),
    "ratio": ("Statement A: {a}\nStatement B: {b}\n\nConvert both quantities to the same unit (use kelvin "
              "for temperatures) and compute the ratio A divided by B. Reply with only the number.", None),
    "compat": ("Statement: {a}\n\nIs the numerical value physically plausible given the description in the "
               "statement? Answer Yes or No.", ["Yes", "No"]),
}


def select_teacher(cfg: dict, hw: dict) -> dict:
    """Pick the teacher model / dtype / batch size from the detected hardware."""
    t = dict(cfg["teacher"])
    if t.get("model", "auto") != "auto":
        return t
    gpu_gb = hw.get("gpu_mem_gb", 0)
    if gpu_gb >= 8:
        t.update(model=t["preferred_model"], dtype="bfloat16" if hw.get("bf16") else "float16", batch_size=32)
    elif gpu_gb >= 3:
        t.update(model=t["fallback_model"], dtype="float16", batch_size=32)
    else:
        t.update(model=t["fallback_model"], dtype="float32", batch_size=8,
                 max_pairs=min(t["max_pairs"], t.get("cpu_max_pairs", 600)))
    return t


class LLMTeacher:
    def __init__(self, model_name: str, dtype: str = "float32", device: str = "cpu", batch_size: int = 16,
                 max_new_tokens: int = 12):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype)).to(device).eval()
        self.device, self.bs, self.max_new = device, batch_size, max_new_tokens
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def _prompt(self, text: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _option_ids(self, options):
        ids = []
        for o in options:
            enc = [self.tok.encode(v, add_special_tokens=False) for v in (o, " " + o, o.lower())]
            variants = {e[0] for e in enc if e}
            if not variants:
                raise ValueError(f"option {o!r} cannot be tokenised by {self.name}")
            ids.append(sorted(variants))
        if len({tuple(x) for x in ids}) < len(ids):
            raise ValueError(f"options {options} share first tokens in {self.name}; soft labels would be ambiguous")
        return ids

    def classify(self, texts: list[str], options: list[str]) -> np.ndarray:
        """Return P(option) for each text, renormalised over the options (soft labels)."""
        torch = self.torch
        opt_ids = self._option_ids(options)
        prompts = [self._prompt(t) for t in texts]
        order_idx = np.argsort([len(p) for p in prompts])
        out = np.zeros((len(texts), len(options)), dtype=np.float32)
        for s in range(0, len(prompts), self.bs):
            idx = order_idx[s:s + self.bs]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            scores = torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in opt_ids], dim=-1)
            out[idx] = torch.softmax(scores, dim=-1).cpu().numpy()
        return out

    def generate(self, texts: list[str]) -> list[str]:
        torch = self.torch
        prompts = [self._prompt(t) for t in texts]
        order_idx = np.argsort([len(p) for p in prompts])
        res = [""] * len(texts)
        for s in range(0, len(prompts), self.bs):
            idx = order_idx[s:s + self.bs]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for k, i in enumerate(idx):
                res[i] = self.tok.decode(gen[k, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return res


class MockTeacher:
    """Smoke-test stand-in ONLY (no model download). Answers from ground truth with label noise.
    Results produced with it are flagged as MOCK and are not experimental findings."""

    def __init__(self, noise: float = 0.2, seed: int = 42):
        self.name, self.revision = f"MOCK(noise={noise})", None
        self.noise, self.rng = noise, np.random.default_rng(seed)
        self.gt = None  # set by caller: list of answers aligned with the texts

    def classify(self, texts, options):
        out = np.zeros((len(texts), len(options)), dtype=np.float32)
        for i, ans in enumerate(self.gt):
            k = options.index(ans)
            if self.rng.random() < self.noise:
                k = int(self.rng.integers(len(options)))
            p = self.rng.uniform(0.55, 0.95)
            out[i, :] = (1 - p) / (len(options) - 1)
            out[i, k] = p
        return out

    def generate(self, texts):
        return [f"{g * (1 + self.rng.normal(0, 0.05)) if self.rng.random() < self.noise else g:.6g}" for g in self.gt]


def parse_float(s: str) -> float:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace(",", ""))
    return float(m.group()) if m else float("nan")


def sample_teacher_pairs(pairs: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample over pair types so every relation is represented."""
    rng = np.random.default_rng(seed)
    types = pairs.pair_type.unique()
    per = max(1, n // len(types))
    parts = [g.sample(min(len(g), per), random_state=int(rng.integers(1e9))) for _, g in pairs.groupby("pair_type")]
    out = pd.concat(parts)
    if len(out) < n:
        rest = pairs.drop(out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    return out.reset_index(drop=True)


def run_teacher(teacher, pairs: pd.DataFrame, statements: pd.DataFrame, ratio_fraction: float, seed: int) -> tuple:
    """Query the teacher on `pairs` (and G statements) and verify against ground truth."""
    t0 = time.time()
    p = pairs.copy()
    mock = isinstance(teacher, MockTeacher)

    def ask(q, df, gt_answers=None):
        tmpl, opts = Q_TEXT[q]
        texts = [tmpl.format(a=r.a_sentence, b=r.b_sentence) for r in df.itertuples()]
        if mock:
            teacher.gt = gt_answers
        return teacher.classify(texts, opts)

    gt_dim = ["Yes" if x else "No" for x in p.same_dimension]
    gt_eq = ["Yes" if x else "No" for x in p.equivalent]
    pr = ask("dim", p, gt_dim); p["t_p_same_dim"] = pr[:, 0]
    pr = ask("equiv", p, gt_eq); p["t_p_equiv"] = pr[:, 0]
    sd = p[p.same_dimension].copy()
    gt_ord = ["A" if o == 1 else ("B" if o == -1 else "Equal") for o in sd.order]
    pr = ask("order", sd, gt_ord)
    p.loc[sd.index, ["t_p_A", "t_p_B", "t_p_Equal"]] = pr
    p["t_order"] = np.nan
    lab = np.array([1, -1, 0])[pr.argmax(1)] if len(sd) else []
    p.loc[sd.index, "t_order"] = lab
    # ratio by generation on a subset of same-dimension pairs
    rs = sd.sample(frac=ratio_fraction, random_state=seed) if len(sd) else sd
    p["t_ratio"] = np.nan
    p["ratio_asked"] = p.index.isin(rs.index)
    if len(rs):
        texts = [Q_TEXT["ratio"][0].format(a=r.a_sentence, b=r.b_sentence) for r in rs.itertuples()]
        if mock:
            teacher.gt = list(rs.ratio)
        p.loc[rs.index, "t_ratio"] = [parse_float(x) for x in teacher.generate(texts)]
    # hard labels
    p["t_same_dim"] = p.t_p_same_dim > 0.5
    p["t_equiv"] = p.t_p_equiv > 0.5
    # verification
    p["ok_dim"] = p.t_same_dim == p.same_dimension
    p["ok_equiv"] = p.t_equiv == p.equivalent
    p["ok_order"] = np.where(p.same_dimension, p.t_order == p.order, np.nan)
    p["ok_ratio"] = np.where(p.ratio_asked, np.isclose(p.t_ratio, p.ratio, rtol=0.01), np.nan)  # unparsable = wrong
    p["verified"] = p.ok_dim & p.ok_equiv & (p.ok_order.fillna(1) == 1) & (p.ok_ratio.fillna(1) == 1)

    # contextual compatibility on single statements (category G)
    g = statements[statements.category == "G_contradiction"].copy()
    if len(g):
        texts = [Q_TEXT["compat"][0].format(a=s) for s in g.sentence]
        if mock:
            teacher.gt = ["Yes" if x else "No" for x in g.plausible]
        g["t_p_plausible"] = teacher.classify(texts, Q_TEXT["compat"][1])[:, 0]
        g["t_plausible"] = g.t_p_plausible > 0.5
        g["ok_compat"] = g.t_plausible == g.plausible.astype(bool)

    acc = teacher_accuracy(p, g)
    acc["teacher_model"] = teacher.name
    acc["teacher_revision"] = teacher.revision
    acc["teacher_seconds"] = round(time.time() - t0, 1)
    acc["is_mock"] = mock
    return p, g, acc


def _mean(x):
    x = pd.Series(x).dropna().astype(float)
    return None if len(x) == 0 else round(float(x.mean()), 4)


def teacher_accuracy(p: pd.DataFrame, g: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    acc = {
        "n_pairs": len(p),
        "same_dimension_acc": _mean(p.ok_dim),
        "equivalence_acc": _mean(p.ok_equiv),
        "equivalence_balanced_acc": _mean([_mean(p[p.equivalent].ok_equiv), _mean(p[~p.equivalent].ok_equiv)]),
        "equivalence_auc": round(float(roc_auc_score(p.equivalent, p.t_p_equiv)), 4) if p.equivalent.nunique() == 2 else None,
        "ordering_acc": _mean(p.ok_order),
        "unit_conversion_acc_cross_unit_equiv": _mean(p[p.cross_unit & p.equivalent].ok_equiv),
        "ratio_acc_within_1pct": _mean(p.ok_ratio),
        "ratio_parse_rate": _mean(p.t_ratio.notna()[p.ratio_asked]),
        "fully_verified_fraction": _mean(p.verified),
        "majority_baseline_equivalence": _mean(~p.equivalent) if _mean(~p.equivalent) is not None else None,
        "by_pair_type": {t: {"n": len(d), "equiv_acc": _mean(d.ok_equiv), "dim_acc": _mean(d.ok_dim),
                             "order_acc": _mean(d.ok_order)} for t, d in p.groupby("pair_type")},
    }
    if len(g):
        acc["compatibility_acc"] = _mean(g.ok_compat)
        acc["compatibility_balanced_acc"] = _mean([_mean(g[g.plausible.astype(bool)].ok_compat),
                                                   _mean(g[~g.plausible.astype(bool)].ok_compat)])
    return acc

"""Numerical intelligence transfer: multi-objective fine-tuning of the student embedding model.

Notation: e(x) = L2-normalised student embedding, s(x,y) = e(x)·e(y) (cosine). If a numerical
projection head is used (variant "proj"), the numerical losses act on z(x) = normalise(P e(x)) and
the semantic loss on e(x), so numerical structure is learned in a separate subspace.

L_total = λ1 L_eq + λ2 L_ord + λ3 L_dist + λ4 L_ctx + λ5 L_sem

L_eq   weighted symmetric InfoNCE over same-context equivalent pairs (a_i, b_i), weights w_i:
         L_eq = -1/2 Σ_i w_i [log softmax_j(s(a_i,b_j)/τ)_i + log softmax_j(s(b_i,a_j)/τ)_i] / Σ_i w_i
L_ord  (i) pairwise ranking on a learned direction u (shared across dimensions, ||u|| free):
         r(x) = u·e(x);   L_rank = mean max(0, m_r - (r(hi) - r(lo)))
       (ii) betweenness on ordered triplets lo < mid < hi (math supervision only):
         L_betw = mean[max(0, s(lo,hi) - s(lo,mid) + m_t) + max(0, s(lo,hi) - s(mid,hi) + m_t)]
       L_ord = L_rank + L_betw
L_dist regression of cosine onto a magnitude-similarity target for same-dimension NON-equivalent pairs:
         t(Δ) = c_min + (c_max - c_min) · exp(-Δ/σ)
         Δ = |log10(a/b)|              (mode "log",  σ = sigma_log)
         Δ = |a - b| / max(|a|,|b|)    (mode "abs",  σ = sigma_abs)
         L_dist = mean (s(a,b) - t(Δ))²          (c_max < 1 keeps a gap below true equivalence)
L_ctx  hinge upper bounds that keep contexts / dimensions apart:
         L_ctx = mean max(0, s(a,b) - m_k), with m_k = m_cross (different dimension),
                 m_ctx (same quantity, different context), m_plaus (plausible vs contradictory)
L_sem  semantic preservation by self-distillation from the frozen pretrained model f0 on general text:
         L_sem = mean (1 - cos(e(g), f0(g)))
"""
from __future__ import annotations

import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ supervision tables

def dist_target(delta: np.ndarray, cfg: dict) -> np.ndarray:
    tcfg = cfg["training"]["distance"]
    sigma = tcfg["sigma_log"] if tcfg["mode"] == "log" else tcfg["sigma_abs"]
    return tcfg["c_min"] + (tcfg["c_max"] - tcfg["c_min"]) * np.exp(-np.asarray(delta) / sigma)


def math_supervision(pairs: pd.DataFrame, triplets: pd.DataFrame | None, stmts: pd.DataFrame, cfg: dict) -> dict:
    """Direct mathematical (ground-truth) supervision."""
    m = cfg["training"]["margins"]
    mode = cfg["training"]["distance"]["mode"]
    sid = stmts.set_index("id").sentence
    eqp = pairs[pairs.equivalent & pairs.same_context]
    dp = pairs[pairs.same_dimension & ~pairs.equivalent]
    delta = dp.log_distance if mode == "log" else dp.rel_abs_distance
    op = pairs[pairs.same_dimension & (pairs.order != 0)]
    lo = np.where(op.order == -1, op.a_sentence, op.b_sentence)
    hi = np.where(op.order == -1, op.b_sentence, op.a_sentence)
    ctx = [(r.a_sentence, r.b_sentence, m["cross"]) for r in pairs[~pairs.same_dimension].itertuples()]
    ctx += [(r.a_sentence, r.b_sentence, m["context"]) for r in pairs[pairs.equivalent & ~pairs.same_context].itertuples()]
    ctx += [(r.a_sentence, r.b_sentence, m["plausibility"]) for r in pairs[pairs.pair_type == "plausibility"].itertuples()]
    trip = []
    if triplets is not None and len(triplets):
        trip = [(sid[r.lo_id], sid[r.mid_id], sid[r.hi_id]) for r in triplets.itertuples()]
    return dict(eq=list(zip(eqp.a_sentence, eqp.b_sentence, np.ones(len(eqp)))),
                dist=list(zip(dp.a_sentence, dp.b_sentence, dist_target(delta.values, cfg))),
                ord=list(zip(lo, hi)), trip=trip, ctx=ctx)


def teacher_supervision(tp: pd.DataFrame, tg: pd.DataFrame, cfg: dict, verified: bool) -> dict:
    """Supervision derived from the teacher's answers. `verified=True` keeps only answers that agree
    with the Python ground truth (per relation), weighted by the teacher's confidence."""
    m = cfg["training"]["margins"]
    mode = cfg["training"]["distance"]["mode"]
    t = tp.copy()
    if verified:
        keep_eq = t.ok_equiv & t.ok_dim
    else:
        keep_eq = pd.Series(True, index=t.index)
    eqp = t[keep_eq & t.t_equiv & t.t_same_dim & t.same_context]
    w = eqp.t_p_equiv.values
    # distance targets from the teacher's numerical ratio
    dp = t[t.t_same_dim & ~t.t_equiv & t.t_ratio.notna() & (t.t_ratio > 0)]
    if verified:
        dp = dp[dp.ok_ratio == 1]
    r = dp.t_ratio.values
    delta = np.abs(np.log10(r)) if mode == "log" else np.abs(1 - r) / np.maximum(1, r)
    op = t[t.t_same_dim & t.t_order.notna() & (t.t_order != 0)]
    if verified:
        op = op[op.ok_order == 1]
    lo = np.where(op.t_order == -1, op.a_sentence, op.b_sentence)
    hi = np.where(op.t_order == -1, op.b_sentence, op.a_sentence)
    cx = t[~t.t_same_dim]
    if verified:
        cx = cx[cx.ok_dim]
    ctx = [(a, b, m["cross"]) for a, b in zip(cx.a_sentence, cx.b_sentence)]
    dc = t[t.t_equiv & ~t.same_context]
    if verified:
        dc = dc[dc.ok_equiv]
    ctx += [(a, b, m["context"]) for a, b in zip(dc.a_sentence, dc.b_sentence)]
    if tg is not None and len(tg):
        pl = tg.set_index("sentence")
        for r_ in t[t.pair_type == "plausibility"].itertuples():
            if r_.a_sentence in pl.index and r_.b_sentence in pl.index:
                pa, pb = pl.loc[r_.a_sentence], pl.loc[r_.b_sentence]
                if pa.t_plausible != pb.t_plausible and (not verified or (pa.ok_compat and pb.ok_compat)):
                    ctx.append((r_.a_sentence, r_.b_sentence, m["plausibility"]))
    return dict(eq=list(zip(eqp.a_sentence, eqp.b_sentence, w)),
                dist=list(zip(dp.a_sentence, dp.b_sentence, dist_target(delta, cfg))),
                ord=list(zip(lo, hi)), trip=[], ctx=ctx)


# ------------------------------------------------------------------ model

class Student(nn.Module):
    def __init__(self, st_model, proj_dim: int = 0):
        super().__init__()
        self.st = st_model
        d = st_model.get_sentence_embedding_dimension()
        self.proj = nn.Linear(d, proj_dim) if proj_dim else None
        self.rank_u = nn.Linear(proj_dim or d, 1, bias=False)  # ordering direction lives in the numerical space

    def forward(self, sentences: list[str]):
        dev = next(self.parameters()).device
        feats = self.st.tokenize(sentences)
        feats = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in feats.items()}
        e = F.normalize(self.st(feats)["sentence_embedding"], dim=-1)
        z = F.normalize(self.proj(e), dim=-1) if self.proj is not None else e
        return e, z


def info_nce(a, b, w, tau):
    logits = a @ b.T / tau
    lab = torch.arange(len(a), device=a.device)
    l = 0.5 * (F.cross_entropy(logits, lab, reduction="none") + F.cross_entropy(logits.T, lab, reduction="none"))
    return (l * w).sum() / w.sum().clamp_min(1e-8)


def compute_losses(model, batch, frozen_sem, cfg):
    """batch: dict of lists sampled from the supervision pools. Returns dict of scalar losses."""
    tc = cfg["training"]
    sents = set()
    for a, b, *_ in batch["eq"] + batch["dist"] + batch["ctx"]:
        sents.update((a, b))
    for a, b in batch["ord"]:
        sents.update((a, b))
    for t in batch["trip"]:
        sents.update(t)
    sents.update(batch["sem"])
    sents = list(sents)
    pos = {s: i for i, s in enumerate(sents)}
    e, z = model(sents)
    ix = lambda xs: torch.tensor([pos[x] for x in xs], device=e.device, dtype=torch.long)
    out = {}
    if batch["eq"]:
        a, b, w = zip(*batch["eq"])
        out["eq"] = info_nce(z[ix(a)], z[ix(b)], torch.tensor(w, device=e.device, dtype=torch.float), tc["tau"])
    if batch["dist"]:
        a, b, t = zip(*batch["dist"])
        s = (z[ix(a)] * z[ix(b)]).sum(-1)
        out["dist"] = F.mse_loss(s, torch.tensor(t, device=e.device, dtype=torch.float))
    o = torch.zeros((), device=e.device)
    if batch["ord"]:
        lo, hi = zip(*batch["ord"])
        r = model.rank_u(z).squeeze(-1)
        o = o + F.relu(tc["margins"]["rank"] - (r[ix(hi)] - r[ix(lo)])).mean()
    if batch["trip"]:
        lo, mid, hi = zip(*batch["trip"])
        zl, zm, zh = z[ix(lo)], z[ix(mid)], z[ix(hi)]
        slh, slm, smh = (zl * zh).sum(-1), (zl * zm).sum(-1), (zm * zh).sum(-1)
        mt = tc["margins"]["triplet"]
        o = o + (F.relu(slh - slm + mt) + F.relu(slh - smh + mt)).mean()
    if batch["ord"] or batch["trip"]:
        out["ord"] = o
    if batch["ctx"]:
        a, b, mk = zip(*batch["ctx"])
        s = (z[ix(a)] * z[ix(b)]).sum(-1)
        out["ctx"] = F.relu(s - torch.tensor(mk, device=e.device, dtype=torch.float)).mean()
    if batch["sem"]:
        tgt = torch.stack([frozen_sem[x] for x in batch["sem"]]).to(e.device)
        out["sem"] = (1 - (e[ix(batch["sem"])] * tgt).sum(-1)).mean()
    return out


def sample_batch(sup, sem_pool, rng, cfg, lambdas):
    tc = cfg["training"]
    bs = tc["batch_size"]

    def take(key, n):
        pool = sup.get(key, [])
        if not pool or lambdas.get({"trip": "ord"}.get(key, key), 0) == 0:
            return []
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=len(pool) < n)
        return [pool[i] for i in idx]

    return dict(eq=take("eq", bs), dist=take("dist", bs), ord=take("ord", bs), trip=take("trip", bs // 2),
                ctx=take("ctx", bs), sem=[sem_pool[i] for i in rng.choice(len(sem_pool), bs, replace=False)]
                if lambdas.get("sem", 0) > 0 and sem_pool else [])


def total(losses, lambdas):
    return sum(lambdas.get(k, 0.0) * v for k, v in losses.items())


def train_student(run_name: str, base_model_path: str, sup: dict, val_sup: dict, sem_sentences: list[str],
                  cfg: dict, lambdas: dict, out_dir: str, seed: int, proj_dim: int = 0, device: str = "cpu",
                  log=print) -> dict:
    from sentence_transformers import SentenceTransformer
    tc = cfg["training"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    st = SentenceTransformer(base_model_path, device=device)
    st.max_seq_length = tc["max_seq_length"]
    # frozen reference embeddings for L_sem (pretrained model, computed once)
    frozen = {}
    if sem_sentences:
        with torch.no_grad():
            emb = st.encode(sem_sentences, batch_size=128, convert_to_tensor=True, normalize_embeddings=True,
                            show_progress_bar=False).cpu()
        frozen = {s: emb[i] for i, s in enumerate(sem_sentences)}
    model = Student(st, proj_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    n_steps = tc["epochs"] * tc["steps_per_epoch"]
    warm = int(tc["warmup_ratio"] * n_steps)
    def lr_lambda(s):
        return (s + 1) / max(1, warm) if s < warm else max(0.0, (n_steps - s) / max(1, n_steps - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # fixed validation batches (ground-truth supervision, identical for every run)
    vrng = np.random.default_rng(1234)
    val_batches = [sample_batch(val_sup, sem_sentences, vrng, cfg, {k: 1 for k in ("eq", "dist", "ord", "ctx", "sem")})
                   for _ in range(tc["val_batches"])]
    hist, best, best_state = [], float("inf"), None
    sizes = {k: len(v) for k, v in sup.items()}
    log(f"[{run_name}] supervision sizes {sizes}; lambdas {lambdas}; steps {n_steps}; device {device}")
    t0 = time.time()
    step = 0
    for ep in range(tc["epochs"]):
        model.train()
        run_l, comp = [], {}
        for _ in range(tc["steps_per_epoch"]):
            batch = sample_batch(sup, sem_sentences, rng, cfg, lambdas)
            losses = compute_losses(model, batch, frozen, cfg)
            loss = total(losses, lambdas)
            if not torch.is_tensor(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
            opt.step()
            sched.step()
            step += 1
            run_l.append(loss.item())
            for k, v in losses.items():
                comp.setdefault(k, []).append(v.item())
        model.eval()
        vcomp = {}
        with torch.no_grad():
            for vb in val_batches:
                for k, v in compute_losses(model, vb, frozen, cfg).items():
                    vcomp.setdefault(k, []).append(v.item())
        vcomp = {k: float(np.mean(v)) for k, v in vcomp.items()}
        # model selection on the *unweighted* sum of numerical validation losses + semantic loss,
        # identical across runs, so ablations are selected by the same criterion
        vsel = sum(vcomp.values())
        rec = dict(epoch=ep + 1, train_loss=float(np.mean(run_l)) if run_l else None,
                   train_components={k: float(np.mean(v)) for k, v in comp.items()},
                   val_components=vcomp, val_loss_weighted=float(total({k: torch.tensor(v) for k, v in vcomp.items()}, lambdas)),
                   val_selection_loss=vsel, elapsed_s=round(time.time() - t0, 1))
        hist.append(rec)
        log(f"[{run_name}] epoch {ep + 1}: train {rec['train_loss']:.4f} | val(sel) {vsel:.4f} | {vcomp}")
        if vsel < best:
            best, best_state = vsel, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(out_dir, exist_ok=True)
    model.st.save(out_dir)
    torch.save({"rank_u": model.rank_u.state_dict(), "proj": model.proj.state_dict() if model.proj else None,
                "proj_dim": proj_dim}, os.path.join(out_dir, "numerical_heads.pt"))
    info = dict(run=run_name, seed=seed, lambdas=lambdas, proj_dim=proj_dim, supervision_sizes=sizes,
                history=hist, best_val_selection_loss=best, train_seconds=round(time.time() - t0, 1),
                batch_size=tc["batch_size"], lr=tc["lr"], epochs=tc["epochs"], steps_per_epoch=tc["steps_per_epoch"],
                distance_mode=cfg["training"]["distance"]["mode"])
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def load_student(path: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(path, device=device)
    heads = torch.load(os.path.join(path, "numerical_heads.pt"), map_location=device) \
        if os.path.exists(os.path.join(path, "numerical_heads.pt")) else {"proj_dim": 0}
    m = Student(st, heads.get("proj_dim", 0)).to(device)
    if heads.get("proj"):
        m.proj.load_state_dict(heads["proj"])
    if "rank_u" in heads:
        m.rank_u.load_state_dict(heads["rank_u"])
    return m.eval()


@torch.no_grad()
def encode(model_or_path, sentences: list[str], device="cpu", batch_size=128):
    """Return (e, z) as numpy arrays; z is None when there is no projection head."""
    m = load_student(model_or_path, device) if isinstance(model_or_path, str) else model_or_path
    E, Z = [], []
    for i in range(0, len(sentences), batch_size):
        e, z = m(sentences[i:i + batch_size])
        E.append(e.cpu().numpy())
        Z.append(z.cpu().numpy())
    E = np.concatenate(E)
    Z = np.concatenate(Z) if m.proj is not None else None
    return E, Z

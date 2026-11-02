"""Evaluation of embedding configurations on numerical, contextual and semantic criteria.

A configuration is evaluated through two interfaces only:
    sim(i, j)      -- similarity between statements i and j (cosine, or a hybrid)
    features(i)    -- representation fed to *probes* (linear models trained on the train split)

Thresholds and hybrid weights are tuned on the validation split and applied unchanged to test.
Cosine-based metrics answer "is the relation geometrically present?", probe metrics answer "is the
information linearly decodable?". The two are reported separately on purpose: higher cosine does
not by itself mean better numerical understanding.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from numerical_parser import numeric_features


class Space:
    """All sentences of one configuration, embedded once."""

    def __init__(self, sentences: list[str], E: np.ndarray, Z: np.ndarray | None, mode: str, cfg: dict):
        self.idx = {s: i for i, s in enumerate(sentences)}
        self.E, self.Z, self.mode = E, Z, mode
        self.alpha = 1.0
        ec = cfg["evaluation"]
        self.sigma, self.cmax = ec["parser_sigma"], cfg["training"]["distance"]["c_max"]
        feats, dims, logc = [], [], []
        for s in sentences:
            v, q = numeric_features(s)
            feats.append(v)
            dims.append(q.dimension if q else None)
            logc.append(math.log10(q.canonical) if q and q.canonical > 0 else np.nan)
        self.N = np.stack(feats)
        self.dim = np.array(dims, dtype=object)
        self.logc = np.array(logc)

    def ids(self, sents):
        return np.array([self.idx[s] for s in sents])

    def cos(self, A, i, j):
        return (A[i] * A[j]).sum(-1)

    def parser_sim(self, i, j):
        ok = (~np.isnan(self.logc[i])) & (~np.isnan(self.logc[j])) & (self.dim[i] == self.dim[j])
        d = np.abs(self.logc[i] - self.logc[j])
        s = np.where(d < 1e-9, 1.0, self.cmax * np.exp(-np.nan_to_num(d) / self.sigma))
        both_parsed = (~np.isnan(self.logc[i])) & (~np.isnan(self.logc[j]))
        s = np.where(ok, s, np.where(both_parsed, 0.0, np.nan))  # nan = parser failed -> fallback
        return s

    def sim(self, i, j):
        c = self.cos(self.E, i, j)
        if self.mode == "cos":
            return c
        if self.mode == "hybrid_proj":
            return self.alpha * c + (1 - self.alpha) * self.cos(self.Z, i, j)
        p = self.parser_sim(i, j)
        p = np.where(np.isnan(p), c, p)  # fall back to semantics when the parser cannot read a number
        if self.mode == "parser_only":
            return p
        return self.alpha * c + (1 - self.alpha) * p

    def features(self, i):
        if self.mode == "hybrid_parser":
            return np.hstack([self.E[i], self.N[i]])
        if self.mode == "parser_only":
            return self.N[i]
        if self.mode == "hybrid_proj":
            return np.hstack([self.E[i], self.Z[i]])
        return self.E[i]


# ------------------------------------------------------------------ helpers

def best_threshold(scores, labels):
    order = np.unique(scores)
    if len(order) > 400:
        order = np.quantile(scores, np.linspace(0, 1, 401))
    best_t, best_b = 0.5, -1
    for t in order:
        pred = scores >= t
        b = 0.5 * ((pred & labels).sum() / max(1, labels.sum()) + ((~pred) & ~labels).sum() / max(1, (~labels).sum()))
        if b > best_b:
            best_b, best_t = b, t
    return float(best_t), float(best_b)


def balanced_acc(scores, labels, t):
    pred = scores >= t
    return float(0.5 * ((pred & labels).sum() / max(1, labels.sum()) + ((~pred) & ~labels).sum() / max(1, (~labels).sum())))


def safe_auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def bootstrap_ci(fn, n_items, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        try:
            v = fn(idx)
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


# ------------------------------------------------------------------ metric blocks

EQ_NEG_TYPES = ("close", "mid")


def equivalence_block(sp: Space, val_pairs, test_pairs, stmts_test, boot):
    def subset(p):
        pos = p[(p.pair_type == "equivalent")]
        neg = p[p.pair_type.isin(EQ_NEG_TYPES)]
        return pd.concat([pos, neg])

    v, t = subset(val_pairs), subset(test_pairs)
    sv = sp.sim(sp.ids(v.a_sentence), sp.ids(v.b_sentence))
    thr, _ = best_threshold(sv, v.equivalent.values)
    st = sp.sim(sp.ids(t.a_sentence), sp.ids(t.b_sentence))
    yt = t.equivalent.values
    out = dict(threshold=r4(thr), equivalence_auc_hard=r4(safe_auc(yt, st)),
               equivalence_bal_acc_hard=r4(balanced_acc(st, yt, thr)))
    out["equivalence_auc_hard_ci95"] = bootstrap_ci(lambda ix: safe_auc(yt[ix], st[ix]), len(yt), boot)
    # all negatives (incl. distant + cross-dimension)
    ta = test_pairs[test_pairs.pair_type.isin(["equivalent", "close", "mid", "distant", "cross_dimension"])]
    sa = sp.sim(sp.ids(ta.a_sentence), sp.ids(ta.b_sentence))
    out["equivalence_bal_acc_all"] = r4(balanced_acc(sa, ta.equivalent.values, thr))
    # unit conversion: cross-unit equivalents vs hard negatives
    cu = t[(~t.equivalent) | (t.cross_unit)]
    sc = sp.sim(sp.ids(cu.a_sentence), sp.ids(cu.b_sentence))
    out["unit_conversion_bal_acc"] = r4(balanced_acc(sc, cu.equivalent.values, thr))
    out["unit_conversion_auc"] = r4(safe_auc(cu.equivalent.values, sc))
    # precision: category-E pairs (10 vs 10.0 vs 10.01)
    cat = stmts_test.set_index("sentence").category
    e = t[t.a_sentence.map(cat).eq("E_precision") & t.b_sentence.map(cat).eq("E_precision")]
    if len(e) and e.equivalent.nunique() == 2:
        se = sp.sim(sp.ids(e.a_sentence), sp.ids(e.b_sentence))
        out["precision_bal_acc"] = r4(balanced_acc(se, e.equivalent.values, thr))
        out["precision_auc"] = r4(safe_auc(e.equivalent.values, se))
    return out, dict(scores=st, labels=yt)


def cosine_profile(sp: Space, test_pairs):
    out = {}
    for ptype, g in test_pairs.groupby("pair_type"):
        s = sp.sim(sp.ids(g.a_sentence), sp.ids(g.b_sentence))
        out[ptype] = dict(mean=r4(s.mean()), std=r4(s.std()), n=len(g))
    return out


def distance_block(sp: Space, test_pairs):
    out = {}
    sd = test_pairs[test_pairs.same_dimension]
    for name, g in (("all_same_dim", sd), ("same_template", sd[sd.same_template])):
        if len(g) < 5:
            continue
        s = sp.sim(sp.ids(g.a_sentence), sp.ids(g.b_sentence))
        d_emb = 1 - s
        for dn in ("log_distance", "rel_abs_distance"):
            out[f"{name}_{dn}_spearman"] = r4(spearmanr(d_emb, g[dn]).correlation)
            out[f"{name}_{dn}_pearson"] = r4(pearsonr(d_emb, g[dn])[0])
    return out


def ordering_block(sp: Space, train, val, test, triplets_test, stress, boot):
    out, per_dim = {}, {}
    sid = test.set_index("id").sentence
    # (a) betweenness on ordered triplets (geometry)
    if len(triplets_test):
        lo, mid, hi = (sp.ids(triplets_test[c].map(sid)) for c in ("lo_id", "mid_id", "hi_id"))
        slh, slm, smh = sp.sim(lo, hi), sp.sim(lo, mid), sp.sim(mid, hi)
        both = (slm > slh) & (smh > slh)
        out["triplet_betweenness_acc"] = r4(both.mean())
        out["triplet_betweenness_acc_ci95"] = bootstrap_ci(lambda ix: both[ix].mean(), len(both), boot)
    # (b) linear probe per dimension: features -> log10(canonical)
    accs, rhos, w_rhos, w_accs, ns = [], [], [], [], []
    for dim in sorted(test.dimension.unique()):
        tr, va, te = (d[d.dimension == dim] for d in (train, val, test))
        if len(tr) < 20 or len(te) < 10:
            continue
        Xtr, Xva, Xte = (sp.features(sp.ids(d.sentence)) for d in (tr, va, te))
        ytr, yva, yte = (np.log10(d.canonical_value.values) for d in (tr, va, te))
        sc = StandardScaler().fit(Xtr)
        best = None
        for a in (0.1, 1, 10, 100, 1000):
            m = Ridge(alpha=a).fit(sc.transform(Xtr), ytr)
            rho = spearmanr(m.predict(sc.transform(Xva)), yva).correlation if len(va) > 3 else 0
            if best is None or rho > best[0]:
                best = (rho, m)
        m = best[1]
        p = m.predict(sc.transform(Xte))
        acc = pairwise_order_acc(p, yte)
        rho = spearmanr(p, yte).correlation
        per_dim[dim] = dict(probe_pairwise_order_acc=r4(acc), probe_spearman=r4(rho), n_test=len(te))
        accs.append(acc); rhos.append(rho); ns.append(len(te))
        if stress is not None and len(stress):
            ws = stress[stress.dimension == dim]
            if len(ws) >= 8:
                pw = m.predict(sc.transform(sp.features(sp.ids(ws.sentence))))
                yw = np.log10(ws.canonical_value.values)
                per_dim[dim]["words_probe_spearman"] = r4(spearmanr(pw, yw).correlation)
                per_dim[dim]["words_probe_pairwise_order_acc"] = r4(pairwise_order_acc(pw, yw))
                w_rhos.append(spearmanr(pw, yw).correlation); w_accs.append(pairwise_order_acc(pw, yw))
    out["probe_pairwise_order_acc"] = r4(np.mean(accs)) if accs else None
    out["probe_rank_spearman"] = r4(np.mean(rhos)) if rhos else None
    if w_rhos:
        out["words_probe_rank_spearman"] = r4(np.nanmean(w_rhos))
        out["words_probe_pairwise_order_acc"] = r4(np.nanmean(w_accs))
    out["per_dimension"] = per_dim
    return out


def pairwise_order_acc(pred, true):
    dp = np.sign(pred[:, None] - pred[None, :])
    dt = np.sign(true[:, None] - true[None, :])
    m = np.triu(np.abs(dt) > 0, 1)
    return float((dp[m] == dt[m]).mean()) if m.any() else float("nan")


def retrieval_block(sp: Space, test, ks=(1, 5, 10), queries=None, gallery=None, rel_fn=None, boot=0, tag=""):
    """Numerical-equivalence retrieval. Relevant items = same equivalence group."""
    gal = test if gallery is None else gallery
    q = test if queries is None else queries
    gids = sp.ids(gal.sentence)
    grp = gal.equivalence_group.values
    rr, hits = [], {k: [] for k in ks}
    for _, r in q.iterrows():
        qi = sp.idx[r.sentence]
        mask = gal.sentence.values != r.sentence
        if "source_id" in q.columns and isinstance(r.get("source_id"), str):
            pass  # word-form query: its digit source stays in the gallery as a relevant item
        rel = (grp == r.equivalence_group) & mask
        if rel.sum() == 0:
            continue
        s = sp.sim(np.full(mask.sum(), qi), gids[mask])
        relm = rel[mask]
        ranks = (-s).argsort().argsort()
        first = ranks[relm].min() + 1
        rr.append(1.0 / first)
        for k in ks:
            hits[k].append(float(first <= k))
    out = {f"{tag}recall@{k}": r4(np.mean(v)) if v else None for k, v in hits.items()}
    out[f"{tag}mrr"] = r4(np.mean(rr)) if rr else None
    out[f"{tag}n_queries"] = len(rr)
    if boot and rr:
        arr = np.array(rr)
        out[f"{tag}mrr_ci95"] = bootstrap_ci(lambda ix: arr[ix].mean(), len(arr), boot)
    return out


def context_block(sp: Space, train, test, test_pairs):
    out = {}
    # (a) dimension discrimination: s(a, equivalent same-dim) > s(a, cross-dimension partner)
    eq = test_pairs[test_pairs.pair_type == "equivalent"]
    cx = test_pairs[test_pairs.pair_type == "cross_dimension"]
    part_eq, part_cx = {}, {}
    for a, b in zip(eq.a_sentence, eq.b_sentence):
        part_eq.setdefault(a, b); part_eq.setdefault(b, a)
    for a, b in zip(cx.a_sentence, cx.b_sentence):
        part_cx.setdefault(a, b); part_cx.setdefault(b, a)
    anchors = [a for a in part_eq if a in part_cx]
    if anchors:
        A = sp.ids(anchors)
        s1 = sp.sim(A, sp.ids([part_eq[a] for a in anchors]))
        s2 = sp.sim(A, sp.ids([part_cx[a] for a in anchors]))
        out["dimension_discrimination_acc"] = r4((s1 > s2).mean())
        out["dimension_discrimination_n"] = len(anchors)
    # (b) mean similarity of the same quantity in different contexts (1.0 = context collapsed)
    dc = test_pairs[test_pairs.pair_type == "equivalent_diff_context"]
    if len(dc):
        out["same_quantity_diff_context_mean_sim"] = r4(sp.sim(sp.ids(dc.a_sentence), sp.ids(dc.b_sentence)).mean())
    # (c) context probe (is the measurement context still linearly decodable?)
    tr = train[train.context != "arithmetic"]
    te = test[test.context.isin(set(tr.context)) & (test.context != "arithmetic")]
    counts = tr.context.value_counts()
    keep = set(counts[counts >= 5].index)
    tr, te = tr[tr.context.isin(keep)], te[te.context.isin(keep)]
    sc = StandardScaler().fit(sp.features(sp.ids(tr.sentence)))
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(sp.features(sp.ids(tr.sentence))), tr.context)
    pred = clf.predict(sc.transform(sp.features(sp.ids(te.sentence))))
    out["context_probe_acc"] = r4((pred == te.context.values).mean())
    out["context_probe_macro_f1"] = r4(f1_score(te.context, pred, average="macro"))
    unseen = te[~te.template_seen.astype(bool)]
    if len(unseen):
        pu = clf.predict(sc.transform(sp.features(sp.ids(unseen.sentence))))
        out["context_probe_acc_unseen_templates"] = r4((pu == unseen.context.values).mean())
    # (d) plausibility / contradiction probe (category G)
    gtr, gte = train[train.category == "G_contradiction"], test[test.category == "G_contradiction"]
    if len(gtr) > 20 and len(gte) > 5:
        sc = StandardScaler().fit(sp.features(sp.ids(gtr.sentence)))
        clf = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(sp.features(sp.ids(gtr.sentence))),
                                                          gtr.plausible.astype(bool))
        p = clf.predict(sc.transform(sp.features(sp.ids(gte.sentence))))
        out["plausibility_probe_acc"] = r4((p == gte.plausible.astype(bool).values).mean())
        out["plausibility_majority_baseline"] = r4(max(gte.plausible.astype(bool).mean(), 1 - gte.plausible.astype(bool).mean()))
    return out


def stress_block(sp: Space, stress, test, test_pairs, thr):
    """Parser-free slice: numbers spelled out in words (the regex parser returns nothing)."""
    if stress is None or not len(stress):
        return {}
    out = retrieval_block(sp, test, queries=stress, gallery=test, tag="words_")
    # word-form query vs its digit source (positive) and vs the source's close/mid partners (negatives)
    src = test.set_index("id").sentence
    pos_a, pos_b, neg_a, neg_b = [], [], [], []
    tp = test_pairs[test_pairs.pair_type.isin(EQ_NEG_TYPES)]
    partners = {}
    for a, b in zip(tp.a_sentence, tp.b_sentence):
        partners.setdefault(a, []).append(b); partners.setdefault(b, []).append(a)
    for r in stress.itertuples():
        s = src.get(r.source_id)
        if s is None:
            continue
        pos_a.append(r.sentence); pos_b.append(s)
        for n in partners.get(s, [])[:2]:
            neg_a.append(r.sentence); neg_b.append(n)
    if pos_a and neg_a:
        sc = np.concatenate([sp.sim(sp.ids(pos_a), sp.ids(pos_b)), sp.sim(sp.ids(neg_a), sp.ids(neg_b))])
        y = np.array([True] * len(pos_a) + [False] * len(neg_a))
        out["words_equivalence_auc"] = r4(safe_auc(y, sc))
        out["words_equivalence_bal_acc"] = r4(balanced_acc(sc, y, thr))
    return out


def sts_block(sp: Space, sts):
    if sts is None:
        return {"stsb_spearman": None, "note": "STS-B unavailable"}
    s = sp.sim(sp.ids(sts.sentence1), sp.ids(sts.sentence2))
    return {"stsb_spearman": r4(spearmanr(s, sts.score).correlation), "stsb_pearson": r4(pearsonr(s, sts.score)[0])}


def tune_alpha(sp: Space, val_pairs):
    if sp.mode not in ("hybrid_parser", "hybrid_proj"):
        return None
    v = val_pairs[val_pairs.pair_type.isin(("equivalent",) + EQ_NEG_TYPES)]
    i, j = sp.ids(v.a_sentence), sp.ids(v.b_sentence)
    best = (-1, 1.0)
    for a in np.linspace(0, 1, 11):
        sp.alpha = a
        _, b = best_threshold(sp.sim(i, j), v.equivalent.values)
        sd = val_pairs[val_pairs.same_dimension]
        rho = spearmanr(1 - sp.sim(sp.ids(sd.a_sentence), sp.ids(sd.b_sentence)), sd.log_distance).correlation
        score = 0.5 * b + 0.5 * rho  # equal weight on equivalence and graded distance
        if score > best[0] + 1e-9:
            best = (score, a)
    sp.alpha = best[1]
    return float(best[1])


def evaluate_space(sp: Space, data: dict, cfg: dict) -> tuple[dict, dict]:
    boot = cfg["evaluation"]["bootstrap"]
    res = {"mode": sp.mode, "alpha": tune_alpha(sp, data["val_pairs"])}
    eq, preds = equivalence_block(sp, data["val_pairs"], data["test_pairs"], data["test"], boot)
    res["equivalence"] = eq
    res["cosine_profile"] = cosine_profile(sp, data["test_pairs"])
    res["distance"] = distance_block(sp, data["test_pairs"])
    res["ordering"] = ordering_block(sp, data["train"], data["val"], data["test"], data["test_triplets"],
                                     data.get("stress"), boot)
    res["retrieval"] = retrieval_block(sp, data["test"], boot=boot)
    unseen = data["test"][~data["test"].template_seen.astype(bool)]
    if len(unseen):
        res["retrieval_unseen_templates"] = retrieval_block(sp, data["test"], queries=unseen, tag="unseen_")
    res["context"] = context_block(sp, data["train"], data["test"], data["test_pairs"])
    res["stress_words"] = stress_block(sp, data.get("stress"), data["test"], data["test_pairs"], eq["threshold"])
    res["semantic"] = sts_block(sp, data.get("sts"))
    tp = data["test_pairs"]
    preds["pairs"] = tp.assign(similarity=sp.sim(sp.ids(tp.a_sentence), sp.ids(tp.b_sentence)))
    return res, preds

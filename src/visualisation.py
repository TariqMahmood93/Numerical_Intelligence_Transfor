"""Figures (matplotlib / seaborn), all saved to results/figures/."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PAIR_ORDER = ["equivalent", "equivalent_diff_context", "close", "mid", "distant", "cross_dimension", "plausibility"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cosine_by_pair_type(pairs_by_cfg: dict, out):
    rows = []
    for cfg, df in pairs_by_cfg.items():
        rows.append(df[["pair_type", "similarity"]].assign(config=cfg))
    d = pd.concat(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=d, x="pair_type", y="similarity", hue="config", order=[p for p in PAIR_ORDER if p in set(d.pair_type)],
                ax=ax, fliersize=1, linewidth=0.8)
    ax.set_xlabel("test pair type"); ax.set_ylabel("cosine similarity")
    ax.set_title("1. Baseline vs fine-tuned cosine similarity by numerical relation")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, os.path.join(out, "01_cosine_baseline_vs_finetuned.png"))


def fig_distance_scatter(pairs_by_cfg: dict, out):
    fig, axes = plt.subplots(1, len(pairs_by_cfg), figsize=(5 * len(pairs_by_cfg), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (cfg, df) in zip(axes, pairs_by_cfg.items()):
        g = df[df.same_dimension & df.same_template]
        from scipy.stats import spearmanr
        rho = spearmanr(g.log_distance, 1 - g.similarity).correlation
        ax.scatter(g.log_distance, 1 - g.similarity, s=6, alpha=0.35)
        ax.set_title(f"{cfg}\nSpearman ρ = {rho:.3f}")
        ax.set_xlabel("|log10(a/b)| (numerical distance)")
    axes[0].set_ylabel("1 - cosine (embedding distance)")
    fig.suptitle("2. Numerical distance vs embedding distance (same dimension, same template)", y=1.02)
    return _save(fig, os.path.join(out, "02_numerical_vs_embedding_distance.png"))


def _bar(agg: pd.DataFrame, metrics: list[str], labels: list[str], title, path, ylim=(0, 1)):
    d = []
    for m, l in zip(metrics, labels):
        if m in agg.columns:
            for cfg in agg.index:
                d.append(dict(config=cfg, metric=l, mean=agg.loc[cfg, m], std=agg.get(m + "__std", pd.Series(0, index=agg.index)).get(cfg, 0)))
    d = pd.DataFrame(d).dropna(subset=["mean"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * d.config.nunique() * len(metrics) / 2), 4.5))
    cfgs = list(dict.fromkeys(d.config))
    mets = list(dict.fromkeys(d.metric))
    w = 0.8 / len(cfgs)
    for k, c in enumerate(cfgs):
        sub = d[d.config == c].set_index("metric").reindex(mets)
        ax.bar(np.arange(len(mets)) + k * w, sub["mean"], w, yerr=sub["std"].fillna(0), label=c, capsize=2)
    ax.set_xticks(np.arange(len(mets)) + w * (len(cfgs) - 1) / 2)
    ax.set_xticklabels(mets)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path)


def fig_retrieval(agg, out):
    return _bar(agg, ["retrieval.recall@1", "retrieval.recall@5", "retrieval.recall@10", "retrieval.mrr",
                      "stress_words.words_mrr"],
                ["R@1", "R@5", "R@10", "MRR", "MRR (numbers in words)"],
                "3. Numerical-equivalence retrieval (test split)", os.path.join(out, "03_equivalence_retrieval.png"))


def fig_ordering(agg, out):
    return _bar(agg, ["ordering.probe_pairwise_order_acc", "ordering.triplet_betweenness_acc",
                      "ordering.words_probe_pairwise_order_acc"],
                ["probe pairwise order acc", "triplet betweenness acc (cosine)", "probe order acc (words)"],
                "4. Numerical ordering accuracy", os.path.join(out, "04_ordering_accuracy.png"))


def fig_semantic(agg, out):
    return _bar(agg, ["semantic.stsb_spearman", "context.context_probe_acc", "context.plausibility_probe_acc"],
                ["STS-B Spearman", "context probe acc", "plausibility probe acc"],
                "5. Semantic and contextual preservation", os.path.join(out, "05_semantic_preservation.png"))


def fig_pca(emb_by_cfg: dict, test: pd.DataFrame, out, dimension="mass"):
    sub = test[test.dimension == dimension]
    fig, axes = plt.subplots(1, len(emb_by_cfg), figsize=(5 * len(emb_by_cfg), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (cfg, (sents, E)) in zip(axes, emb_by_cfg.items()):
        idx = {s: i for i, s in enumerate(sents)}
        X = E[[idx[s] for s in sub.sentence]]
        P = PCA(2, random_state=42).fit_transform(X)
        sc = ax.scatter(P[:, 0], P[:, 1], c=np.log10(sub.canonical_value), cmap="viridis", s=12)
        ax.set_title(f"{cfg}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes.tolist(), label=f"log10 canonical {dimension}")
    fig.suptitle(f"6. PCA of test embeddings ({dimension} statements), colour = magnitude")
    fig.savefig(os.path.join(out, "06_pca_embedding_space.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_teacher(acc: dict, out):
    keys = [("same_dimension_acc", "same dimension"), ("equivalence_balanced_acc", "equivalence (bal.)"),
            ("ordering_acc", "ordering"), ("unit_conversion_acc_cross_unit_equiv", "unit conversion"),
            ("ratio_acc_within_1pct", "ratio within 1%"), ("compatibility_balanced_acc", "plausibility (bal.)")]
    k = [(l, acc.get(m)) for m, l in keys if acc.get(m) is not None]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar([a for a, _ in k], [b for _, b in k], color="#4C72B0")
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy vs Python ground truth")
    ax.set_title(f"Teacher reliability: {acc.get('teacher_model')}" + (" [MOCK]" if acc.get("is_mock") else ""))
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, os.path.join(out, "07_teacher_accuracy.png"))


def fig_ablation(abl: pd.DataFrame, out):
    cols = [c for c in ["equivalence.equivalence_auc_hard", "retrieval.mrr", "distance.same_template_log_distance_spearman",
                        "ordering.probe_pairwise_order_acc", "context.dimension_discrimination_acc", "semantic.stsb_spearman"]
            if c in abl.columns]
    d = abl[cols].copy()
    d.columns = [c.split(".")[-1] for c in cols]
    fig, ax = plt.subplots(figsize=(10, 3.2 + 0.3 * len(d)))
    sns.heatmap(d.astype(float), annot=True, fmt=".3f", cmap="mako", ax=ax, cbar=False)
    ax.set_title("Ablation A–E (seed 42)")
    return _save(fig, os.path.join(out, "08_ablation_heatmap.png"))


def fig_training_curves(histories: dict, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["train_loss"] for r in h], marker="o", label=name)
        ax[1].plot(ep, [r["val_selection_loss"] for r in h], marker="o", label=name)
    ax[0].set_title("training loss (weighted)"); ax[1].set_title("validation selection loss (unweighted, identical across runs)")
    for a in ax:
        a.set_xlabel("epoch")
    ax[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, os.path.join(out, "09_training_curves.png"))

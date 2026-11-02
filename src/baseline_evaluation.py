"""Experiment 1 (original all-MiniLM-L6-v2) and Experiment 2 (numerical parser + original embeddings).

Also produces the descriptive baseline cosine analysis requested in the protocol: cosine similarity
of the untouched model for (1) numerically equivalent, (2) numerically close, (3) numerically distant,
(4) contextually compatible and (5) contextually incompatible statement pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import Space, evaluate_space

PAIR_GROUPS = {
    "1_numerically_equivalent_same_context": ["equivalent"],
    "2_numerically_close_(<=2x)": ["close"],
    "3_numerically_distant_(>=10x)": ["distant"],
    "4_contextually_compatible_(same quantity, different context)": ["equivalent_diff_context"],
    "5_contextually_incompatible_(different dimension)": ["cross_dimension"],
    "5b_contextually_incompatible_(plausible vs contradictory)": ["plausibility"],
}


def baseline_cosine_table(sp: Space, test_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, types in PAIR_GROUPS.items():
        g = test_pairs[test_pairs.pair_type.isin(types)]
        if not len(g):
            continue
        s = sp.cos(sp.E, sp.ids(g.a_sentence), sp.ids(g.b_sentence))
        rows.append(dict(pair_group=name, n=len(g), mean_cosine=s.mean(), std=s.std(),
                         median=np.median(s), p10=np.percentile(s, 10), p90=np.percentile(s, 90)))
    return pd.DataFrame(rows).round(4)


def run_baseline_configs(sentences, E, data, cfg):
    """Returns {config_name: (metrics, predictions, Space)} for Exp1, Exp2 and a parser-only reference."""
    out = {}
    for name, mode in (("exp1_original_minilm", "cos"), ("exp2_parser_plus_original", "hybrid_parser"),
                       ("ref_parser_only", "parser_only")):
        sp = Space(sentences, E, None, mode, cfg)
        m, p = evaluate_space(sp, data, cfg)
        out[name] = (m, p, sp)
    return out

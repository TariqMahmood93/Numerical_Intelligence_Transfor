"""Synthetic numerical-statement dataset with deterministic ground truth.

Leakage control
---------------
* Every statement is keyed by its *quantity*: (dimension, canonical value). All statements that
  express the same physical quantity -- in any unit, precision, context or category -- share one
  `equivalence_group` and are forced into the same split.
* Multi-quantity sets (ordering triplets, relation sets, contradiction pairs) are placed wholly in
  one split; a set whose quantities are already registered to different splits is resampled.
* A subset of sentence templates per dimension is *held out*: they only appear in the test split
  (column `template_seen` = False), giving an unseen-phrasing evaluation slice.
* Training / evaluation pairs are built strictly within a split.

Nothing here is produced by an LLM.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from numerical_parser import (CANONICAL_UNIT, UNIT_WORDS, UNITS, equivalent, from_canonical,
                              int_to_words, log_distance, order, ratio, relative_abs_distance,
                              to_canonical)

DIM_UNITS = {
    "mass": ["mg", "g", "kg", "t", "lb"],
    "length": ["mm", "cm", "m", "km", "in", "ft", "mi"],
    "time": ["ms", "s", "min", "h", "day"],
    "temperature": ["C", "F", "K"],
    "volume": ["mL", "cL", "L", "m3"],
    "fraction": ["percent", "fraction"],
}
# sensible canonical ranges for generation (canonical units)
DIM_RANGE = {"mass": (1e-3, 1e8), "length": (1e-4, 1e7), "time": (1e-3, 1e7),
             "temperature": (173.15, 773.15), "volume": (1e-3, 1e7), "fraction": (1e-3, 2.0)}

# (template_id, dimension, context, template). The last template(s) per dimension are held out.
TEMPLATES = [
    ("mass_generic", "mass", "generic", "The weight is {q}."),
    ("mass_package", "mass", "package", "The package weighs {q}."),
    ("mass_suitcase", "mass", "luggage", "The suitcase weighs {q}."),
    ("mass_flour", "mass", "cooking", "The recipe calls for {q} of flour."),
    ("mass_patient", "mass", "medical", "The patient weighs {q}."),
    ("mass_sample", "mass", "laboratory", "The lab sample has a mass of {q}."),
    ("mass_truck", "mass", "transport", "The truck carries a load of {q}."),
    ("mass_parcel_ho", "mass", "package", "According to the label, the parcel's mass is {q}."),
    ("len_generic", "length", "generic", "The length is {q}."),
    ("len_board", "length", "construction", "The wooden board is {q} long."),
    ("len_height", "length", "body", "The person is {q} tall."),
    ("len_cable", "length", "electrical", "The cable measures {q}."),
    ("len_insect", "length", "biology", "The insect is {q} long."),
    ("len_room", "length", "building", "The room is {q} wide."),
    ("dist_trip", "length", "distance_travel", "The road trip covers {q}."),
    ("dist_run", "length", "distance_sport", "She ran {q} this morning."),
    ("dist_commute", "length", "distance_commute", "The commute is {q} each way."),
    ("dist_generic", "length", "distance_generic", "The distance is {q}."),
    ("len_pipe_ho", "length", "construction", "Engineers measured the pipe at {q}."),
    ("dist_hike_ho", "length", "distance_sport", "The hiking trail stretches for {q}."),
    ("time_generic", "time", "generic", "The duration is {q}."),
    ("time_meeting", "time", "work", "The meeting lasted {q}."),
    ("time_bake", "time", "cooking", "Bake the bread for {q}."),
    ("time_flight", "time", "travel", "The flight takes {q}."),
    ("time_reaction", "time", "laboratory", "The chemical reaction completes in {q}."),
    ("time_movie", "time", "entertainment", "The movie runs for {q}."),
    ("time_wait_ho", "time", "service", "Customers waited in the queue for {q}."),
    ("temp_generic", "temperature", "generic", "The temperature is {q}."),
    ("temp_patient", "temperature", "medical", "The patient's temperature is {q}."),
    ("temp_outside", "temperature", "weather", "The temperature outside is {q}."),
    ("temp_machine", "temperature", "industrial", "The machine operates at {q}."),
    ("temp_oven", "temperature", "cooking", "The oven is set to {q}."),
    ("temp_freezer", "temperature", "storage", "The freezer is kept at {q}."),
    ("temp_lake_ho", "temperature", "weather", "The lake water was measured at {q}."),
    ("vol_generic", "volume", "generic", "The volume is {q}."),
    ("vol_bottle", "volume", "container", "The bottle holds {q}."),
    ("vol_tank", "volume", "vehicle", "The car's fuel tank holds {q}."),
    ("vol_dose", "volume", "medical", "The syrup dose is {q}."),
    ("vol_pool", "volume", "leisure", "The swimming pool contains {q} of water."),
    ("vol_cup", "volume", "cooking", "The cup contains {q} of coffee."),
    ("vol_jug_ho", "volume", "container", "The measuring jug is filled with {q}."),
    ("frac_generic", "fraction", "generic", "The proportion is {q}."),
    ("frac_discount", "fraction", "retail", "The discount is {q}."),
    ("frac_share", "fraction", "business", "The company's market share is {q}."),
    ("frac_battery", "fraction", "device", "The battery charge level is {q}."),
    ("frac_turnout_ho", "fraction", "politics", "Voter turnout reached {q}."),
]
HELD_OUT = {t[0] for t in TEMPLATES if t[0].endswith("_ho")}
TPL = {t[0]: t for t in TEMPLATES}
TPL_BY_DIM = defaultdict(list)
for _t in TEMPLATES:
    TPL_BY_DIM[_t[1]].append(_t[0])

# Contradiction templates: (id, dimension, context, template, canonical plausible range)
K = 273.15
CONTRA = [
    ("g_fever", "temperature", "medical", "The patient has a normal body temperature of {q}.", (35 + K, 38 + K)),
    ("g_spring", "temperature", "weather", "It is a mild spring afternoon with a temperature of {q}.", (8 + K, 26 + K)),
    ("g_freezer", "temperature", "storage", "The freezer keeps the food frozen at {q}.", (-40 + K, -5 + K)),
    ("g_oven", "temperature", "cooking", "The cake is baking in a hot oven at {q}.", (140 + K, 260 + K)),
    ("g_bath", "temperature", "leisure", "The baby's bath water is comfortably warm at {q}.", (34 + K, 40 + K)),
    ("g_newborn", "mass", "medical", "The healthy newborn baby weighs {q}.", (2.2e3, 5e3)),
    ("g_letter", "mass", "postal", "The light envelope weighs {q}.", (3.0, 100.0)),
    ("g_adult", "mass", "medical", "The adult man weighs {q}.", (45e3, 150e3)),
    ("g_tall", "length", "body", "The adult woman is of average height at {q}.", (1.45, 1.85)),
    ("g_ant", "length", "biology", "The tiny ant is {q} long.", (1e-3, 2e-2)),
    ("g_walk", "length", "distance_sport", "It is a short walk of {q} to the shop.", (50.0, 2000.0)),
    ("g_movie", "time", "entertainment", "The feature film has a typical running time of {q}.", (75 * 60.0, 180 * 60.0)),
    ("g_blink", "time", "biology", "A blink of the eye lasts {q}.", (0.05, 0.5)),
    ("g_cup", "volume", "cooking", "The small espresso cup contains {q} of coffee.", (0.02, 0.12)),
    ("g_dose", "volume", "medical", "The child's syrup dose is {q}.", (0.0025, 0.02)),
    ("g_battery", "fraction", "device", "The battery is almost empty at {q}.", (0.0, 0.15)),
    ("g_discount", "fraction", "retail", "The shop offers a modest discount of {q}.", (0.05, 0.3)),
]

NICE_MANTISSA = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8]


def fmt_number(v: float, decimals: int | None = None, style: str = "plain") -> str:
    if decimals is not None:
        return f"{v:.{decimals}f}"
    if style == "sci" and v != 0:
        e = int(math.floor(math.log10(abs(v))))
        ms = f"{v / 10 ** e:.10g}"
        return f"{ms}e{e}"
    s = f"{round(v, 9):.9f}".rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    if style == "comma" and abs(v) >= 10000 and "." not in s:
        s = f"{int(round(v)):,}"
    return s


def clean_value(v: float) -> bool:
    """Value can be displayed exactly with <= 6 decimals and is in a readable range."""
    if v == 0 or not math.isfinite(v):
        return False
    if abs(v) < 1e-3 or abs(v) > 1e9:
        return False
    return abs(round(v, 6) - v) <= 1e-12 * max(1.0, abs(v)) and len(fmt_number(abs(v)).replace(".", "").lstrip("0")) <= 10


def unit_text(unit: str, v: float, abbrev: bool) -> str:
    if unit == "fraction":
        return "as a fraction"
    if abbrev and unit not in ("percent", "day", "in"):
        return {"C": "°C", "F": "°F"}.get(unit, unit)
    sing, plur = UNIT_WORDS[unit]
    return sing if abs(v) == 1 else plur


def quantity_key(dim: str, canon: float) -> tuple:
    return (dim, float(f"{canon:.12g}"))


class Registry:
    """Maps quantity keys to splits so that equal quantities never cross splits."""

    def __init__(self, rng: random.Random, fractions=(0.70, 0.15, 0.15)):
        self.rng = rng
        self.split_of: dict[tuple, str] = {}
        self.group_id: dict[tuple, int] = {}
        self.fr = fractions

    def draw_split(self) -> str:
        r = self.rng.random()
        return "train" if r < self.fr[0] else ("validation" if r < self.fr[0] + self.fr[1] else "test")

    def compatible(self, keys, split) -> bool:
        return all(self.split_of.get(k, split) == split for k in keys)

    def place(self, keys) -> str | None:
        existing = {self.split_of[k] for k in keys if k in self.split_of}
        if len(existing) > 1:
            return None
        split = existing.pop() if existing else self.draw_split()
        for k in keys:
            self.split_of[k] = split
            self.group_id.setdefault(k, len(self.group_id))
        return split


class Generator:
    def __init__(self, seed: int = 42, scale: float = 1.0):
        self.rng = random.Random(seed)
        self.reg = Registry(self.rng)
        self.rows: list[dict] = []
        self.scale = scale
        self.seen_sentences: set[str] = set()

    # -------------------------------------------------------------- helpers
    def _tick(self) -> bool:
        self._budget -= 1
        return self._budget > 0

    def template_for(self, dim: str, split: str, exclude=()) -> str:
        pool = [t for t in TPL_BY_DIM[dim] if t not in exclude]
        if split != "test":
            pool = [t for t in pool if t not in HELD_OUT] or pool
        return self.rng.choice(pool)

    def render(self, tpl: str, value: float, unit: str, decimals=None, style=None, abbrev=None) -> str:
        if style is None:
            r = self.rng.random()
            style = "comma" if r < 0.08 else ("sci" if r < 0.13 else "plain")
        if abbrev is None:
            abbrev = self.rng.random() < 0.25
        q = f"{fmt_number(value, decimals, style)} {unit_text(unit, value, abbrev)}"
        return tpl.format(q=q)

    def add(self, *, sentence, value, unit, category, relationship, context, template_id, split,
            set_id="", plausible=None, decimals=None, explicit_arith=False):
        if sentence in self.seen_sentences:
            return False
        dim = UNITS[unit][0]
        canon = to_canonical(value, unit)
        key = quantity_key(dim, canon)
        if self.reg.split_of.get(key, split) != split:
            return False  # quantity already lives in another split -> drop to avoid leakage
        if key not in self.reg.group_id:
            self.reg.split_of[key] = split
            self.reg.group_id[key] = len(self.reg.group_id)
        self.seen_sentences.add(sentence)
        self.rows.append(dict(
            sentence=sentence, numerical_value=value, unit=unit, canonical_value=canon,
            canonical_unit=CANONICAL_UNIT[dim], dimension=dim, category=category,
            equivalence_group=f"EQ{self.reg.group_id[key]:05d}", relationship=relationship,
            context=context, template_id=template_id, template_seen=template_id not in HELD_OUT,
            set_id=set_id, plausible=plausible, decimals=decimals if decimals is not None else -1,
            explicit_arithmetic=explicit_arith, split=split))
        return True

    def nice_value(self, dim: str, unit: str, lo_exp=-2, hi_exp=5) -> float | None:
        for _ in range(50):
            v = self.rng.choice(NICE_MANTISSA) * 10 ** self.rng.randint(lo_exp, hi_exp)
            v = float(f"{v:.10g}")
            c = to_canonical(v, unit)
            lo, hi = DIM_RANGE[dim]
            if lo <= c <= hi and clean_value(v):
                return v
        return None

    def temp_value(self, unit: str) -> float:
        c = 5 * self.rng.randint(-14, 60)  # -70..300 C, multiples of 5 keep F exact
        return float(from_canonical(c + K, unit)) if unit != "C" else float(c)

    # -------------------------------------------------------------- categories
    def cat_a_magnitude(self, n):
        made = 0
        self._budget = 300 * max(n, 1)
        while made < n and self._tick():
            dim = self.rng.choice(["mass", "length", "time", "volume", "temperature"])
            unit = self.rng.choice(DIM_UNITS[dim])
            v = self.temp_value(unit) if dim == "temperature" else self.nice_value(dim, unit, -1, 4)
            if v is None or not (dim == "temperature" or clean_value(v)):
                continue
            canon = to_canonical(v, unit)
            split = self.reg.place([quantity_key(dim, canon)])
            tpl = self.template_for(dim, split)
            s = self.render(TPL[tpl][3], v, unit)
            mag = int(math.floor(math.log10(canon)))
            made += self.add(sentence=s, value=v, unit=unit, category="A_magnitude",
                             relationship=f"order_of_magnitude=1e{mag} {CANONICAL_UNIT[dim]}",
                             context=TPL[tpl][2], template_id=tpl, split=split)

    def equivalent_units(self, dim, canon):
        out = []
        for u in DIM_UNITS[dim]:
            v = from_canonical(canon, u)
            v = float(f"{v:.12g}")
            if dim == "temperature":
                if abs(round(v, 2) - v) < 1e-9:
                    out.append((u, round(v, 2)))
            elif clean_value(v) and 1e-3 <= abs(v) <= 1e8:
                out.append((u, v))
        return out

    def cat_b_equivalence(self, n):
        made, gid = 0, 0
        self._budget = 300 * max(n, 1)
        while made < n and self._tick():
            dim = self.rng.choice(["mass", "length", "time", "temperature", "volume", "fraction",
                                   "length"])  # length twice: object length + distance contexts
            if dim == "temperature":
                canon = 5 * self.rng.randint(-14, 60) + K
            elif dim == "fraction":
                canon = self.rng.choice([1, 2, 5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90]) / 100
            else:
                u0 = self.rng.choice(DIM_UNITS[dim])
                v0 = self.nice_value(dim, u0, -1, 3)
                if v0 is None:
                    continue
                canon = to_canonical(v0, u0)
            members = self.equivalent_units(dim, canon)
            if len(members) < 2:
                continue
            self.rng.shuffle(members)
            members = members[:4]
            split = self.reg.place([quantity_key(dim, canon)])
            tpl = self.template_for(dim, split)
            gid += 1
            for u, v in members:
                t = tpl if self.rng.random() < 0.8 else self.template_for(dim, split)
                s = self.render(TPL[t][3], v, u)
                made += self.add(sentence=s, value=v, unit=u, category="B_unit_equivalence",
                                 relationship=f"equivalent to {fmt_number(canon)} {CANONICAL_UNIT[dim]}",
                                 context=TPL[t][2], template_id=t, split=split, set_id=f"B{gid:04d}")

    def cat_c_ordering(self, n):
        made, sid = 0, 0
        self._budget = 300 * max(n, 1)
        while made < n and self._tick():
            dim = self.rng.choice(["mass", "length", "time", "volume", "temperature"])
            items = []
            if dim == "temperature":
                cs = sorted(self.rng.sample(range(-14, 61), 3))
                for c in cs:
                    u = self.rng.choice(DIM_UNITS[dim])
                    items.append((u, float(from_canonical(5 * c + K, u)) if u != "C" else float(5 * c)))
            else:
                u = self.rng.choice(DIM_UNITS[dim])
                v = self.nice_value(dim, u, -1, 3)
                if v is None:
                    continue
                canon = to_canonical(v, u)
                for i in range(3):
                    if i:
                        canon *= self.rng.choice([1.5, 2, 3, 4, 5, 10, 20, 100])
                    opts = [(uu, from_canonical(canon, uu)) for uu in DIM_UNITS[dim]]
                    opts = [(uu, float(f"{vv:.12g}")) for uu, vv in opts if clean_value(float(f"{vv:.12g}"))]
                    if not opts:
                        break
                    same = [o for o in opts if o[0] == u]
                    items.append(same[0] if (same and self.rng.random() < 0.5) else self.rng.choice(opts))
                if len(items) < 3:
                    continue
            canons = [to_canonical(v, u) for u, v in items]
            if any(order(canons[i], canons[i + 1]) != -1 for i in range(2)):
                continue
            keys = [quantity_key(dim, c) for c in canons]
            split = self.reg.place(keys)
            if split is None:
                continue
            sid += 1
            tpl = self.template_for(dim, split)
            for rank, (u, v) in enumerate(items, 1):
                s = self.render(TPL[tpl][3], v, u)
                made += self.add(sentence=s, value=v, unit=u, category="C_ordering",
                                 relationship=f"rank {rank}/3 (ascending) in triplet C{sid:04d}",
                                 context=TPL[tpl][2], template_id=tpl, split=split, set_id=f"C{sid:04d}")

    def cat_d_context(self, n):
        made, sid = 0, 0
        self._budget = 300 * max(n, 1)
        while made < n and self._tick():
            dim = self.rng.choice(["mass", "length", "time", "temperature", "volume"])
            u = self.rng.choice(DIM_UNITS[dim])
            v = self.temp_value(u) if dim == "temperature" else self.nice_value(dim, u, 0, 3)
            if v is None:
                continue
            split = self.reg.place([quantity_key(dim, to_canonical(v, u))])
            pool = [t for t in TPL_BY_DIM[dim] if split == "test" or t not in HELD_OUT]
            tpls = self.rng.sample(pool, min(len(pool), self.rng.randint(3, 4)))
            sid += 1
            abbrev = self.rng.random() < 0.25
            for t in tpls:
                s = self.render(TPL[t][3], v, u, style="plain", abbrev=abbrev)
                made += self.add(sentence=s, value=v, unit=u, category="D_context",
                                 relationship=f"same quantity, context set D{sid:04d}",
                                 context=TPL[t][2], template_id=t, split=split, set_id=f"D{sid:04d}")

    def cat_e_precision(self, n):
        made, sid = 0, 0
        self._budget = 300 * max(n, 1)
        while made < n and self._tick():
            dim = self.rng.choice(["mass", "length", "time", "volume", "temperature"])
            u = self.rng.choice([x for x in DIM_UNITS[dim] if x != "K"])
            base = float(self.rng.choice([1, 2, 5, 10, 12, 20, 25, 37, 50, 100, 250]))
            variants = [(base, 0, "equal"), (base, 1, "equal"), (base, 2, "equal"),
                        (base + 0.01 * self.rng.choice([1, -1]), 2, "differs by 0.01"),
                        (base + 0.1 * self.rng.choice([1, -1]), 1, "differs by 0.1")]
            keys = [quantity_key(dim, to_canonical(round(v, d), u)) for v, d, _ in variants]
            split = self.reg.place(list(dict.fromkeys(keys)))
            if split is None:
                continue
            sid += 1
            tpl = self.template_for(dim, split)
            for v, d, rel in variants:
                v = round(v, d)
                s = self.render(TPL[tpl][3], v, u, decimals=d, style="plain", abbrev=False)
                made += self.add(sentence=s, value=v, unit=u, category="E_precision",
                                 relationship=f"{rel} (vs base {fmt_number(base)}) in set E{sid:04d}",
                                 context=TPL[tpl][2], template_id=tpl, split=split,
                                 set_id=f"E{sid:04d}", decimals=d)

    def cat_f_relations(self, n):
        made, sid = 0, 0
        self._budget = 300 * max(n, 1)
        while made < n and self._tick():
            dim = self.rng.choice(["mass", "length", "time", "volume", "fraction", "temperature"])
            u = self.rng.choice(DIM_UNITS[dim]) if dim != "fraction" else "percent"
            if dim == "temperature":
                base = float(5 * self.rng.randint(-4, 20))
                u = "C"
                derived = [(base + d, f"difference={d:+g} C (kelvin interval {d:+g} K)") for d in (5, 10, -5)]
            else:
                base = self.nice_value(dim, u, 0, 2) if dim != "fraction" else float(self.rng.choice([10, 20, 40, 50, 60, 80]))
                if base is None:
                    continue
                derived = [(base * 2, "ratio=2 (multiplication)"), (base * 10, "ratio=10 (order of magnitude)"),
                           (base / 2, "ratio=0.5 (50 percent)"), (base / 4, "ratio=0.25 (division by 4)"),
                           (base + base * 0.2, "difference=+20 percent")]
            items = [(base, "base")] + [(float(f"{v:.10g}"), r) for v, r in derived]
            if not all(clean_value(v) or (dim == "temperature") for v, _ in items):
                continue
            if dim == "temperature" and any(v == 0 for v, _ in items):
                continue
            keys = [quantity_key(UNITS[u][0], to_canonical(v, u)) for v, _ in items]
            split = self.reg.place(list(dict.fromkeys(keys)))
            if split is None:
                continue
            sid += 1
            tpl = self.template_for(UNITS[u][0], split)
            for v, rel in items:
                ex = rel != "base" and self.rng.random() < 0.2 and dim != "temperature"
                if ex:
                    ut = unit_text(u, base, False)
                    r = v / base
                    frame = {2: f"Twice {fmt_number(base)} {ut} is {{q}}.",
                             10: f"Ten times {fmt_number(base)} {ut} is {{q}}.",
                             0.5: f"Half of {fmt_number(base)} {ut} is {{q}}.",
                             0.25: f"{fmt_number(base)} {ut} divided by 4 is {{q}}."}.get(round(r, 6),
                             f"{fmt_number(base)} {ut} increased by 20 percent is {{q}}.")
                    s = self.render(frame, v, u, style="plain", abbrev=False)
                    t_id, ctx = "f_explicit_arith", "arithmetic"
                else:
                    s = self.render(TPL[tpl][3], v, u, style="plain")
                    t_id, ctx = tpl, TPL[tpl][2]
                made += self.add(sentence=s, value=v, unit=u, category="F_relationship",
                                 relationship=f"{rel} relative to base of F{sid:04d}", context=ctx,
                                 template_id=t_id, split=split, set_id=f"F{sid:04d}", explicit_arith=ex)

    def cat_g_contradiction(self, n):
        made, sid = 0, 0
        self._budget = 300 * max(n, 1)
        while made < n and self._tick():
            cid, dim, ctx, tpl, (lo, hi) = self.rng.choice(CONTRA)
            u = self.rng.choice(DIM_UNITS[dim])
            if dim == "temperature":
                cands = [5 * c + K for c in range(-14, 61)]
                cands += [c + 0.5 for c in cands]
            else:
                cands = []
                for uu in [u]:
                    for e in range(-4, 7):
                        for m in NICE_MANTISSA:
                            cands.append(to_canonical(float(f"{m * 10 ** e:.10g}"), uu))
            inside = [c for c in cands if lo <= c <= hi]
            outside = [c for c in cands if (c > hi * 3 or c < lo / 3) if dim != "temperature"] if dim != "temperature" \
                else [c for c in cands if c > hi + 40 or c < lo - 40]
            if not inside or not outside:
                continue
            pc, ic = self.rng.choice(inside), self.rng.choice(outside)
            vals = []
            for c in (pc, ic):
                v = float(f"{from_canonical(c, u):.10g}")
                if dim == "temperature":
                    v = round(v, 2)
                elif not clean_value(v):
                    vals = []
                    break
                vals.append(v)
            if len(vals) < 2:
                continue
            keys = [quantity_key(dim, to_canonical(v, u)) for v in vals]
            split = self.reg.place(list(dict.fromkeys(keys)))
            if split is None:
                continue
            sid += 1
            for v, ok in zip(vals, (True, False)):
                s = self.render(tpl, v, u, style="plain")
                made += self.add(sentence=s, value=v, unit=u, category="G_contradiction",
                                 relationship="plausible" if ok else "contradiction (outside plausible range)",
                                 context=ctx, template_id=cid, split=split, set_id=f"G{sid:04d}", plausible=ok)

    # -------------------------------------------------------------- run
    def generate(self, sizes: dict) -> pd.DataFrame:
        s = self.scale
        self.cat_b_equivalence(int(sizes["B"] * s))
        self.cat_e_precision(int(sizes["E"] * s))
        self.cat_c_ordering(int(sizes["C"] * s))
        self.cat_d_context(int(sizes["D"] * s))
        self.cat_f_relations(int(sizes["F"] * s))
        self.cat_g_contradiction(int(sizes["G"] * s))
        self.cat_a_magnitude(int(sizes["A"] * s))
        df = pd.DataFrame(self.rows)
        df.insert(0, "id", [f"S{i:05d}" for i in range(len(df))])
        return df


# ------------------------------------------------------------------ stress set (parser-free)

def build_stress_set(test_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Re-express test statements with spelled-out integers (the regex parser cannot read them).
    Used to test whether the *embedding itself* encodes numbers, independent of the parser."""
    rng = random.Random(seed)
    rows = []
    for _, r in test_df.iterrows():
        v = r.numerical_value
        if r.dimension == "fraction" or v != int(v) or not (0 < abs(v) <= 999999) or r.category == "F_relationship" \
                and r.explicit_arithmetic:
            continue
        digits = fmt_number(v)
        # replace the first occurrence of the digit string preceding the unit
        words = int_to_words(int(v))
        s = r.sentence
        for cand in (f"{int(v):,}", digits):
            idx = s.find(cand + " ")
            if idx >= 0:
                s = s[:idx] + words + s[idx + len(cand):]
                break
        else:
            continue
        d = r.to_dict()
        d.update(id="W" + r.id[1:], sentence=s, source_id=r.id, category="W_words_" + r.category)
        rows.append(d)
    out = pd.DataFrame(rows)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True) if len(out) else out


# ------------------------------------------------------------------ pairs and triplets

def pair_record(a, b, ptype):
    eq = equivalent(a.dimension, a.canonical_value, b.dimension, b.canonical_value)
    same_dim = a.dimension == b.dimension
    return dict(
        a_id=a.id, b_id=b.id, a_sentence=a.sentence, b_sentence=b.sentence, pair_type=ptype,
        same_dimension=same_dim, equivalent=eq,
        order=order(a.canonical_value, b.canonical_value) if same_dim else np.nan,
        log_distance=log_distance(a.canonical_value, b.canonical_value) if same_dim else np.nan,
        rel_abs_distance=relative_abs_distance(a.canonical_value, b.canonical_value) if same_dim else np.nan,
        ratio=ratio(a.canonical_value, b.canonical_value) if same_dim else np.nan,
        cross_unit=a.unit != b.unit, same_context=a.context == b.context,
        same_template=a.template_id == b.template_id,
        a_plausible=a.plausible, b_plausible=b.plausible)


def build_pairs(df: pd.DataFrame, seed: int = 42, k_close=2, k_mid=1, k_distant=2, k_cross=1,
                max_eq_per_group=6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = df.reset_index(drop=True)
    recs, seen = [], set()

    def push(i, j, t):
        key = (min(i, j), max(i, j))
        if i == j or key in seen:
            return
        seen.add(key)
        recs.append(pair_record(rows.iloc[i], rows.iloc[j], t))

    # 1) equivalent pairs (all categories)
    for _, idx in rows.groupby("equivalence_group").groups.items():
        idx = list(idx)
        cands = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
        rng.shuffle(cands)
        ctx = rows.context.values
        cands.sort(key=lambda ij: ctx[ij[0]] != ctx[ij[1]])  # same-context (pure unit/precision) pairs first
        for i, j in cands[:max_eq_per_group]:
            push(i, j, "equivalent")
    # 2) same-dimension non-equivalent pairs by log distance
    logc = np.log10(rows.canonical_value.clip(lower=1e-12).values)
    by_dim = rows.groupby("dimension").groups
    for i in range(len(rows)):
        same = np.array(by_dim[rows.dimension.iloc[i]])
        d = np.abs(logc[same] - logc[i])
        neq = rows.equivalence_group.values[same] != rows.equivalence_group.iloc[i]
        for lo, hi, k, t in ((1e-9, 0.31, k_close, "close"), (0.31, 1.0, k_mid, "mid"), (1.0, 99, k_distant, "distant")):
            pool = same[(d > lo) & (d <= hi) & neq]
            if len(pool):
                for j in rng.choice(pool, size=min(k, len(pool)), replace=False):
                    push(i, int(j), t)
    # 3) cross-dimension pairs, preferring identical surface number (10 kg vs 10 m)
    for i in range(len(rows)):
        other = rows.index[(rows.dimension != rows.dimension.iloc[i])].values
        same_num = other[rows.numerical_value.values[other] == rows.numerical_value.iloc[i]]
        pool = same_num if len(same_num) else other
        for j in rng.choice(pool, size=min(k_cross, len(pool)), replace=False):
            push(i, int(j), "cross_dimension")
    # 4) same quantity, different context (category D sets) -- already equivalent; tag them
    # 5) plausibility pairs (category G sets)
    for _, idx in rows[rows.category == "G_contradiction"].groupby("set_id").groups.items():
        idx = list(idx)
        if len(idx) == 2:
            push(idx[0], idx[1], "plausibility")
    out = pd.DataFrame(recs)
    ctx_mask = (out.pair_type == "equivalent") & (~out.same_context)
    out.loc[ctx_mask, "pair_type"] = "equivalent_diff_context"
    return out


def build_triplets(df: pd.DataFrame, seed: int = 42, n_random=3000) -> pd.DataFrame:
    """Ordering triplets (lo, mid, hi) within a dimension: Category-C sets + random draws."""
    rng = np.random.default_rng(seed)
    rows = df.reset_index(drop=True)
    trips = []
    for _, idx in rows[rows.category == "C_ordering"].groupby("set_id").groups.items():
        sub = rows.loc[list(idx)].sort_values("canonical_value")
        if len(sub) == 3:
            trips.append(tuple(sub.id) + ("C_set",))
    for dim, idx in rows.groupby("dimension").groups.items():
        idx = np.array(list(idx))
        if len(idx) < 3:
            continue
        for _ in range(int(n_random * len(idx) / len(rows))):
            pick = rows.loc[rng.choice(idx, 3, replace=False)].sort_values("canonical_value")
            c = pick.canonical_value.values
            if order(c[0], c[1]) == -1 and order(c[1], c[2]) == -1:
                trips.append(tuple(pick.id) + ("random",))
    return pd.DataFrame(trips, columns=["lo_id", "mid_id", "hi_id", "source"]).drop_duplicates()


def generate_all(cfg: dict, out_dir: str) -> dict:
    import os
    os.makedirs(out_dir, exist_ok=True)
    g = Generator(seed=cfg["seed"], scale=cfg["dataset"].get("scale", 1.0))
    df = g.generate(cfg["dataset"]["category_sizes"])
    df.to_csv(f"{out_dir}/numerical_intelligence_dataset.csv", index=False)
    parts = {}
    for split in ("train", "validation", "test"):
        part = df[df.split == split].reset_index(drop=True)
        part.to_csv(f"{out_dir}/{split}.csv", index=False)
        parts[split] = part
        pc = cfg["dataset"]["pairs"]
        build_pairs(part, seed=cfg["seed"], **pc).to_csv(f"{out_dir}/{split}_pairs.csv", index=False)
        build_triplets(part, seed=cfg["seed"], n_random=cfg["dataset"]["triplets_random"]
                       ).to_csv(f"{out_dir}/{split}_triplets.csv", index=False)
    stress = build_stress_set(parts["test"], seed=cfg["seed"])
    stress.to_csv(f"{out_dir}/stress_words_test.csv", index=False)
    # leakage audit
    audit = {
        "n_statements": len(df),
        "by_split": df.split.value_counts().to_dict(),
        "by_category": df.category.value_counts().to_dict(),
        "by_dimension": df.dimension.value_counts().to_dict(),
        "equivalence_groups_crossing_splits": int((df.groupby("equivalence_group").split.nunique() > 1).sum()),
        "duplicate_sentences": int(df.sentence.duplicated().sum()),
        "held_out_template_rows_outside_test": int(((~df.template_seen) & (df.split != "test")).sum()),
        "n_stress_statements": len(stress),
    }
    return audit

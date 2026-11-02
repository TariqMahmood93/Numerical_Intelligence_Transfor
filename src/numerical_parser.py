"""Deterministic numerical parser and unit system.

All ground truth in this project is computed here, never by an LLM.

Canonical units per physical dimension:
    mass        -> gram (g)
    length      -> metre (m)        ("distance" is a *context* of length, not a dimension)
    time        -> second (s)
    temperature -> kelvin (K)       (affine conversion: K = C + 273.15, K = (F - 32) * 5/9 + 273.15)
    volume      -> litre (L)
    fraction    -> dimensionless ratio (25 percent == 0.25)

Temperature is handled as an *affine* scale: ratios of Celsius/Fahrenheit values are physically
meaningless, so every ratio / log-distance is computed on the absolute (kelvin) scale.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

# unit key -> (dimension, multiplicative factor to canonical, additive offset applied after factor)
UNITS: dict[str, tuple[str, float, float]] = {
    # mass (canonical g)
    "mg": ("mass", 1e-3, 0.0), "g": ("mass", 1.0, 0.0), "kg": ("mass", 1e3, 0.0),
    "t": ("mass", 1e6, 0.0), "lb": ("mass", 453.59237, 0.0), "oz": ("mass", 28.349523125, 0.0),
    # length (canonical m)
    "mm": ("length", 1e-3, 0.0), "cm": ("length", 1e-2, 0.0), "m": ("length", 1.0, 0.0),
    "km": ("length", 1e3, 0.0), "in": ("length", 0.0254, 0.0), "ft": ("length", 0.3048, 0.0),
    "mi": ("length", 1609.344, 0.0),
    # time (canonical s)
    "ms": ("time", 1e-3, 0.0), "s": ("time", 1.0, 0.0), "min": ("time", 60.0, 0.0),
    "h": ("time", 3600.0, 0.0), "day": ("time", 86400.0, 0.0),
    # temperature (canonical K) -- affine
    "C": ("temperature", 1.0, 273.15), "F": ("temperature", 5.0 / 9.0, None),  # F handled specially
    "K": ("temperature", 1.0, 0.0),
    # volume (canonical L)
    "mL": ("volume", 1e-3, 0.0), "cL": ("volume", 1e-2, 0.0), "L": ("volume", 1.0, 0.0),
    "m3": ("volume", 1e3, 0.0),
    # dimensionless
    "percent": ("fraction", 1e-2, 0.0), "fraction": ("fraction", 1.0, 0.0),
}

CANONICAL_UNIT = {"mass": "g", "length": "m", "time": "s", "temperature": "K",
                  "volume": "L", "fraction": "fraction"}

# Surface forms (singular, plural) used for generation; parsing accepts all aliases below.
UNIT_WORDS: dict[str, tuple[str, str]] = {
    "mg": ("milligram", "milligrams"), "g": ("gram", "grams"), "kg": ("kilogram", "kilograms"),
    "t": ("tonne", "tonnes"), "lb": ("pound", "pounds"), "oz": ("ounce", "ounces"),
    "mm": ("millimetre", "millimetres"), "cm": ("centimetre", "centimetres"), "m": ("metre", "metres"),
    "km": ("kilometre", "kilometres"), "in": ("inch", "inches"), "ft": ("foot", "feet"),
    "mi": ("mile", "miles"),
    "ms": ("millisecond", "milliseconds"), "s": ("second", "seconds"), "min": ("minute", "minutes"),
    "h": ("hour", "hours"), "day": ("day", "days"),
    "C": ("degree Celsius", "degrees Celsius"), "F": ("degree Fahrenheit", "degrees Fahrenheit"),
    "K": ("kelvin", "kelvin"),
    "mL": ("millilitre", "millilitres"), "cL": ("centilitre", "centilitres"), "L": ("litre", "litres"),
    "m3": ("cubic metre", "cubic metres"),
    "percent": ("percent", "percent"), "fraction": ("as a fraction", "as a fraction"),
}

_ALIASES: dict[str, str] = {}
for _k, (_s, _p) in UNIT_WORDS.items():
    for form in {_s, _p, _s.replace("metre", "meter"), _p.replace("metre", "meter"),
                 _s.replace("litre", "liter"), _p.replace("litre", "liter")}:
        _ALIASES[form.lower()] = _k
_ALIASES.update({
    "celsius": "C", "fahrenheit": "F", "°c": "C", "°f": "F", "kelvins": "K",
    "%": "percent", "per cent": "percent", "hrs": "h", "hr": "h", "mins": "min", "sec": "s",
    "secs": "s", "kgs": "kg", "lbs": "lb", "ml": "mL", "l": "L", "cl": "cL",
})
for _k in UNITS:
    if _k not in ("in", "fraction", "percent"):  # bare "in" collides with the preposition
        _ALIASES.setdefault(_k.lower(), _k)

_NUM = r"[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*(?:x|×)\s*10\^[-+]?\d+)?"
_UNIT_ALT = "|".join(sorted((re.escape(a) for a in _ALIASES), key=len, reverse=True))
_PATTERN = re.compile(rf"(?P<num>{_NUM})\s*(?P<unit>{_UNIT_ALT})(?![A-Za-z])", re.IGNORECASE)


def to_canonical(value: float, unit: str) -> float:
    dim, factor, offset = UNITS[unit]
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    return value * factor + offset


def from_canonical(canon: float, unit: str) -> float:
    dim, factor, offset = UNITS[unit]
    if unit == "F":
        return (canon - 273.15) * 9.0 / 5.0 + 32.0
    return (canon - offset) / factor


def parse_number(s: str) -> float:
    s = s.replace("−", "-").replace(",", "").replace(" ", "")
    m = re.match(r"^(.*?)(?:[x×]10\^([-+]?\d+))?$", s)
    base, exp = m.group(1), m.group(2)
    v = float(base)
    if exp is not None:
        v *= 10.0 ** int(exp)
    return v


@dataclass
class Quantity:
    value: float
    unit: str
    dimension: str
    canonical: float
    span: tuple[int, int]
    decimals: int


def parse(text: str) -> list[Quantity]:
    """Return every (number, unit) quantity found in `text`, in order."""
    out = []
    for m in _PATTERN.finditer(text):
        unit_key = _ALIASES.get(m.group("unit").lower())
        if unit_key is None:
            continue
        num_s = m.group("num")
        try:
            v = parse_number(num_s)
        except ValueError:
            continue
        dec = len(num_s.split(".")[1].split("e")[0].split("E")[0]) if "." in num_s else 0
        out.append(Quantity(v, unit_key, UNITS[unit_key][0], to_canonical(v, unit_key), m.span(), dec))
    return out


def parse_main(text: str) -> Optional[Quantity]:
    """The quantity that the statement is about (by construction, the last one in our templates)."""
    q = parse(text)
    return q[-1] if q else None


# ---------------------------------------------------------------- ground-truth relations

REL_TOL = 1e-9


def equivalent(dim_a: str, ca: float, dim_b: str, cb: float, rel_tol: float = REL_TOL) -> bool:
    return dim_a == dim_b and math.isclose(ca, cb, rel_tol=rel_tol, abs_tol=1e-12)


def order(ca: float, cb: float) -> int:
    """-1 if a<b, 0 if equal, 1 if a>b (on canonical, absolute scale)."""
    if math.isclose(ca, cb, rel_tol=REL_TOL, abs_tol=1e-12):
        return 0
    return -1 if ca < cb else 1


def log_distance(ca: float, cb: float) -> float:
    """|log10(a/b)| on the canonical scale. Requires positive canonical values (true for all
    canonical scales here: kelvin is absolute, and generated masses/lengths/etc. are > 0)."""
    if ca <= 0 or cb <= 0:
        return float("nan")
    return abs(math.log10(ca) - math.log10(cb))


def relative_abs_distance(ca: float, cb: float) -> float:
    """|a-b| / max(|a|,|b|) -- a scale-free absolute difference in [0, 1]."""
    den = max(abs(ca), abs(cb))
    return 0.0 if den == 0 else abs(ca - cb) / den


def ratio(ca: float, cb: float) -> float:
    return float("nan") if cb == 0 else ca / cb


# ---------------------------------------------------------------- number words (stress test)

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        r = n % 100
        return _ONES[n // 100] + " hundred" + ("" if r == 0 else " and " + int_to_words(r))
    for div, word in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        if n >= div:
            r = n % div
            return int_to_words(n // div) + " " + word + ("" if r == 0 else (" and " if r < 100 else " ") + int_to_words(r))
    raise ValueError(n)


# ---------------------------------------------------------------- parser feature channel

DIMENSIONS = ["mass", "length", "time", "temperature", "volume", "fraction"]


def numeric_features(text: str, n_rbf: int = 16, lo: float = -6.0, hi: float = 10.0):
    """Parser-derived feature vector used by the hybrid configurations (Exp. 2 and 5).
    [one-hot dimension (6) | RBF encoding of log10(canonical) (n_rbf) | parse_ok flag]."""
    import numpy as np
    q = parse_main(text)
    v = np.zeros(len(DIMENSIONS) + n_rbf + 1, dtype=np.float32)
    if q is None or q.canonical <= 0:
        return v, None
    v[DIMENSIONS.index(q.dimension)] = 1.0
    centres = np.linspace(lo, hi, n_rbf)
    width = (hi - lo) / (n_rbf - 1)
    v[len(DIMENSIONS):len(DIMENSIONS) + n_rbf] = np.exp(-0.5 * ((math.log10(q.canonical) - centres) / width) ** 2)
    v[-1] = 1.0
    return v, q

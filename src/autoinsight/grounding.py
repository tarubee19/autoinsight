"""Number-grounding check: every figure in the written report must exist in the computed facts.

LLMs happily invent plausible statistics. After the Reporter agent writes the narrative we
extract every number from it and look for a matching value in the structured results (stats.json,
metrics.json, cleaning_log.json). Numbers that cannot be traced are listed and the Reporter is
asked to fix them (once); if it still fails, the deterministic narrative is used instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)(\s?%|x\b|×)?")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_CODE_SPAN = re.compile(r"`[^`]*`")
_URL = re.compile(r"https?://\S+")


@dataclass
class GroundingResult:
    checked: int = 0
    grounded: int = 0
    ungrounded: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return 1.0 if self.checked == 0 else self.grounded / self.checked

    def to_dict(self) -> dict:
        return {
            "numbers_checked": self.checked,
            "numbers_grounded": self.grounded,
            "ungrounded": self.ungrounded[:25],
            "grounding_rate": round(self.rate, 4),
        }


def _collect(obj, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for match in _NUMBER.finditer(obj):
            try:
                out.append(float(match.group(1).replace(",", "")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect(value, out)


def fact_numbers(facts: dict) -> list[float]:
    values: list[float] = []
    _collect(facts, values)
    return values


def _decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0


def check_numbers(text: str, facts: dict) -> GroundingResult:
    """Check each number in ``text`` against the facts (rounding-aware, % and x aware)."""
    cleaned = _URL.sub(" ", _CODE_SPAN.sub(" ", text))
    cleaned = _LIST_MARKER.sub("", cleaned)
    pool = fact_numbers(facts)
    result = GroundingResult()
    for match in _NUMBER.finditer(cleaned):
        raw = match.group(1).replace(",", "")
        suffix = (match.group(2) or "").strip()
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = _decimals(raw)
        is_year = decimals == 0 and 1900 <= abs(value) <= 2100 and not suffix
        is_structural = decimals == 0 and abs(value) <= 12 and not suffix  # "top 3", "5-fold", "2 groups"
        if is_year or is_structural:
            continue
        result.checked += 1
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        pairs = [(value, tol)]
        if suffix == "%":
            pairs.append((value / 100.0, tol / 100.0))  # facts store 0.474, the text says 47.4%
        if any(abs(c - f) <= t or abs(abs(c) - abs(f)) <= t for c, t in pairs for f in pool):
            result.grounded += 1
        else:
            result.ungrounded.append(match.group(0).strip())
    return result

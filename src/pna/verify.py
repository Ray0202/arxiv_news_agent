"""Mechanical hallucination check: every number in the summary must exist in the source.

This is the cheapest useful guardrail in the pipeline — pure string work, no model call.
Fabricated benchmark numbers are the failure mode that would quietly destroy trust in the
digest, and they are exactly the kind of thing a regex can catch.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
# A magnitude suffix immediately after a number. `4.2万` in a Chinese summary is the same
# quantity the paper writes as `42,000`; without expanding it the checker reports a
# hallucination for a correct claim.
_SCALE = {"万": 10_000, "亿": 100_000_000, "千": 1_000,
          "k": 1_000, "K": 1_000, "M": 1_000_000, "m": 1_000_000,
          "B": 1_000_000_000, "b": 1_000_000_000}
# Ignore numbers that are almost always structural rather than empirical claims.
_IGNORE = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "0.0", "1.0"}


def _scaled_variants(raw: str, suffix: str) -> set[str]:
    """Forms of `raw x scale(suffix)`, for matching `4.2万` against `42,000`."""
    scale = _SCALE.get(suffix)
    if not scale:
        return set()
    try:
        value = float(raw.replace(",", "")) * scale
    except ValueError:
        return set()
    if value != int(value):
        return set()
    whole = int(value)
    return {str(whole), f"{whole:,}", f"{whole:,}".replace(",", " ")}


def _variants(raw: str, suffix: str = "") -> set[str]:
    """Surface forms the same quantity might take in the source text."""
    core = raw.replace(",", "").lstrip("-")
    out = {raw, core, raw.replace(",", "")}
    out |= _scaled_variants(raw, suffix)
    if "." in core:
        stripped = core.rstrip("0").rstrip(".")
        out.add(stripped)
        # 0.361 in a summary is often written .361 in a results table
        if core.startswith("0."):
            out.add(core[1:])
            out.add(stripped[1:] if stripped.startswith("0.") else stripped)
    else:
        out.add(f"{core}.0")
    return {v for v in out if v}


def check_numbers(summary: dict, source_text: str) -> dict:
    """Return `{checked, unverified, ratio}` for numbers in `results` and the articles."""
    if not source_text:
        return {"checked": 0, "unverified": [], "ratio": 0.0, "skipped": "no source text"}

    haystack = source_text.replace(",", "")
    claims: list[tuple[str, str, str]] = []

    for entry in summary.get("results") or []:
        # `value` and `baseline` are quoted from the paper, so they must be findable.
        # `delta` is a relative change the model computes itself and is not expected to
        # appear anywhere in the source — checking it produces nothing but false alarms.
        for field in ("value", "baseline"):
            val = str(entry.get(field) or "")
            for m in _NUM.finditer(val):
                claims.append((m.group(0), val[m.end() : m.end() + 1],
                               f"results.{entry.get('benchmark', '?')}.{field}"))

    # Numbers inside the prose matter too, but percentage *deltas* are usually derived by
    # arithmetic rather than quoted, so they are expected to be absent from the source.
    for field in ("article_zh", "article_en"):
        text = summary.get(field) or ""
        for match in _NUM.finditer(text):
            num = match.group(0)
            trailing = text[match.end() : match.end() + 1]
            if trailing == "%":
                continue
            claims.append((num, trailing, field))

    checked = 0
    unverified: list[dict] = []
    seen: set[str] = set()
    for num, suffix, where in claims:
        key = num + (suffix if suffix in _SCALE else "")
        if num in _IGNORE or key in seen:
            continue
        seen.add(key)
        checked += 1
        if not any(v in haystack for v in _variants(num, suffix)):
            unverified.append({"number": key, "where": where})

    return {
        "checked": checked,
        "unverified": unverified,
        "ratio": round(len(unverified) / checked, 3) if checked else 0.0,
    }


def check_terms(summary: dict, source_text: str) -> list[str]:
    """Benchmark names in `results` that never appear in the source.

    Deliberately narrow after measuring the first version: 21 flags across 15 papers, all
    of them false positives, and several manufactured by the checker itself — splitting on
    commas turned "42,000 trials" into "000 trials", and splitting on slashes turned
    "Ablation: w/o F_set" into two nonsense names.

    Two things it no longer does:

    * **Tags are not checked.** A tag is the model's own classification vocabulary
      (`test-time-scaling`), not a quotation from the paper. Requiring it to appear in the
      source was a category error.
    * **Descriptive labels are skipped.** The `benchmark` field is frequently a phrase
      rather than a proper noun — `8-design average`, `葡萄牙全国ED数据集`,
      `12 Manager models x 7 Worker configs`. Only a single-token identifier can be
      checked as a name; a phrase cannot.

    What survives is the case this was built for: a benchmark named as a proper noun that
    does not exist in the paper.
    """
    if not source_text:
        return []
    low = source_text.lower()
    missing = []
    for entry in summary.get("results") or []:
        name = str(entry.get("benchmark") or "").strip()
        # Strip a parenthetical qualifier the paper never spells out verbatim, e.g.
        # "ALFWorld (avg over 6 families)".
        base = re.split(r"[(\[（【]", name)[0].strip()
        if not _is_proper_noun(base):
            continue
        if base.lower() not in low:
            missing.append(base)
    return sorted(set(missing))


def _is_proper_noun(token: str) -> bool:
    """A single ASCII identifier that could be looked up verbatim in the source."""
    if len(token) < 4 or len(token) > 40:
        return False
    if re.search(r"[\s:：、,，/]", token):
        return False           # a phrase or a joined list, not a name
    if re.search(r"[\u3000-\u9fff]", token):
        return False           # Chinese description
    return bool(re.search(r"[A-Za-z]", token))


def check_lengths(summary: dict, cfg) -> dict:
    """Measure the article/tldr lengths against the configured budget.

    The model overshoots the range in the field descriptions — measured at 673 Chinese
    characters against a 300-500 budget. Nothing here truncates (that would butcher the
    prose); the point is that overshoot lands in the data and the run log rather than
    being quietly accepted, so prompt changes can be judged against a number.
    """
    out: dict[str, dict] = {}
    limits = {
        "article_zh": (cfg.output.get("article_words_zh") or [0, 0])[1],
        "article_en": (cfg.output.get("article_words_en") or [0, 0])[1],
        "tldr_zh": 40,
        "tldr_en": 25,
    }
    for field, ceiling in limits.items():
        text = summary.get(field)
        if not text or not ceiling:
            continue
        # Chinese fields are budgeted in characters, English ones in words.
        size = len(text) if field.endswith("_zh") else len(text.split())
        out[field] = {
            "size": size,
            "max": ceiling,
            "over_by_pct": round(100 * (size - ceiling) / ceiling) if size > ceiling else 0,
        }
    return out

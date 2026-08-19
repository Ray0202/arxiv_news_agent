"""Tier-1 filter: category whitelist plus weighted keyword matching.

Free and fast, so it runs on everything. Its job is recall, not precision — anything it
lets through is judged again by the LLM in tier 3. Two details matter:

* Keywords are matched on word boundaries. Without that, `agent` matches "reagent" and
  "agentive", and a day's digest fills up with chemistry papers.
* A multi-word keyword matches across hyphens and runs of whitespace, so `time series`
  also catches "time-series" and a line-wrapped "time  series".
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..config import Config, Topic

TITLE_WEIGHT = 2.0
ABSTRACT_WEIGHT = 1.0
COMMENTS_WEIGHT = 0.5


@lru_cache(maxsize=2048)
def _pattern(keyword: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in re.split(r"[^0-9A-Za-z]+", keyword) if p]
    if not parts:
        return re.compile(r"(?!x)x")  # never matches
    body = r"[\s\-_/]*".join(parts)
    return re.compile(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", re.I)


def _hits(topic: Topic, text: str) -> list[str]:
    return [kw for kw in topic.keywords if _pattern(kw).search(text)]


def score_paper(rec: dict, cfg: Config) -> dict:
    """Return the tier-1 verdict for one record (does not mutate it)."""
    cats = set(rec.get("categories") or [])
    allowed = set(cfg.categories)
    category_match = bool(cats & allowed) if allowed else True

    title = rec.get("title") or ""
    abstract = rec.get("abstract") or ""
    comments = rec.get("comments") or ""

    total = 0.0
    hits: dict[str, dict[str, list[str]]] = {}
    matched: list[str] = []
    for topic in cfg.topics:
        t_hits = _hits(topic, title)
        a_hits = _hits(topic, abstract)
        c_hits = _hits(topic, comments)
        if not (t_hits or a_hits or c_hits):
            continue
        # Count each keyword once per field: a paper repeating "agent" 30 times is not
        # 30x more relevant than one that says it twice.
        subtotal = (
            TITLE_WEIGHT * len(t_hits)
            + ABSTRACT_WEIGHT * len(a_hits)
            + COMMENTS_WEIGHT * len(c_hits)
        ) * topic.weight
        total += subtotal
        matched.append(topic.name)
        hits[topic.name] = {"title": t_hits, "abstract": a_hits, "comments": c_hits}

    return {
        "score": round(total, 2),
        "topics": matched,
        "hits": hits,
        "category_match": category_match,
    }


def passes(verdict: dict, cfg: Config) -> bool:
    threshold = float(cfg.thresholds.get("keyword_min_score", 1.0))
    return verdict["category_match"] and verdict["score"] >= threshold

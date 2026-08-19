"""Human labels on published papers, and the analysis that decides whether to trust
the evidence audit for selection.

The point of collecting these is a single question: **does the reviewer's evidence signal
predict "worth a deep read" better than relevance alone?** Until that is answered with
real labels, `selection.evidence_influence` stays at `deep_order` — the audit reorders
deep slots but never decides who gets in.

So a label is stored together with the signals as they stood when it was given. Recording
only `{id, label}` would make the comparison impossible later: triage scores drift when
the model changes, and a re-run would overwrite the numbers the judgement was made
against.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .config import DATA_DIR

LABELS = ("deep", "weak_evidence", "skip")
LABEL_ZH = {
    "deep": "值得深读",
    "weak_evidence": "有趣但证据弱",
    "skip": "可跳过",
}
PATH = DATA_DIR / "feedback.jsonl"


def signals_for(rec: dict, priority: float | None = None) -> dict[str, Any]:
    """The signal snapshot stored alongside a label."""
    triage = rec.get("triage") or {}
    review = rec.get("review") or {}
    fulltext = rec.get("fulltext") or {}
    return {
        "relevance": triage.get("score"),
        "novelty": triage.get("novelty"),
        "read_depth": triage.get("read_depth"),
        "depth": rec.get("depth"),
        "evidence_grade": review.get("evidence_grade"),
        "method_risk": review.get("method_risk"),
        "evaluation_risk": review.get("evaluation_risk"),
        "quality_confidence": review.get("quality_confidence"),
        "paper_type": review.get("paper_type"),
        "reviewed": bool(review),
        "fulltext_source": fulltext.get("source"),
        "priority": round(priority, 4) if priority is not None else None,
    }


SOURCES = ("manual", "imported")


def append(entry: dict) -> Path:
    """Append one label. Atomic per line; concurrent writers are not expected.

    `source` records who produced the label and is what
    `memory._manual_only` filters on. Any future automated labeller must set its own
    value here — a model-generated label that reached the preference memory would let the
    system train its own retrieval on its own opinions.
    """
    if entry.get("label") not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {entry.get('label')!r}")
    if not entry.get("arxiv_id"):
        raise ValueError("arxiv_id is required")
    entry.setdefault("source", "manual")
    if entry["source"] not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {entry['source']!r}")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return PATH


def load() -> list[dict]:
    if not PATH.exists():
        return []
    out = []
    for line in PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def latest_per_paper(entries: Iterable[dict] | None = None) -> dict[str, dict]:
    """Last label wins — the log is append-only so a changed mind is a new line."""
    result: dict[str, dict] = {}
    for e in entries if entries is not None else load():
        if e.get("arxiv_id"):
            result[e["arxiv_id"]] = e
    return result


def merge_import(rows: Iterable[dict]) -> tuple[int, int]:
    """Take labels exported from a browser's localStorage. Returns (added, skipped)."""
    seen = {(e.get("arxiv_id"), e.get("at")) for e in load()}
    added = skipped = 0
    for row in rows:
        key = (row.get("arxiv_id"), row.get("at"))
        if key in seen or row.get("label") not in LABELS:
            skipped += 1
            continue
        append(row)
        seen.add(key)
        added += 1
    return added, skipped


def summarise() -> dict[str, Any]:
    """Does the evidence signal separate 'worth a deep read' from the rest?

    Deliberately descriptive. With a few dozen labels there is no statistical claim to
    make; the useful output is the contingency table, so the decision to raise
    `evidence_influence` is made by looking rather than by a p-value.
    """
    labels = latest_per_paper()
    by_label: dict[str, list[dict]] = {k: [] for k in LABELS}
    for entry in labels.values():
        by_label[entry["label"]].append(entry.get("signals") or {})

    def dist(rows: list[dict], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            v = r.get(key)
            if v is not None:
                out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items()))

    def mean(rows: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "total": len(labels),
        "counts": {k: len(v) for k, v in by_label.items()},
        "by_label": {
            k: {
                "mean_relevance": mean(v, "relevance"),
                "mean_priority": mean(v, "priority"),
                "evidence_grade": dist(v, "evidence_grade"),
                "evaluation_risk": dist(v, "evaluation_risk"),
                "reviewed": sum(1 for r in v if r.get("reviewed")),
            }
            for k, v in by_label.items()
        },
    }

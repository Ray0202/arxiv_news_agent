"""Tier-3 filter: per-paper relevance scoring with Haiku 4.5.

One request per paper rather than batching many abstracts into one call. Batching would
save a negligible amount (the shared prompt is cached, so its marginal cost is ~0.1x on
every request) and costs a lot of robustness: one malformed item in a batch of twenty
loses all twenty, and mapping array positions back to arxiv ids is a silent-corruption
waiting to happen.
"""

from __future__ import annotations

from typing import Any

from ..config import Config, read_prompt
from ..llm import ClientPool, Usage, call_structured, warm_then_parallel

TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": "0-10 relevance to this reader, per the rubric.",
        },
        "matched_topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the reader's topics this paper belongs to. Empty if none.",
        },
        "reason": {
            "type": "string",
            "description": "At most 40 Chinese characters naming what decided the score.",
        },
        "reason_en": {
            "type": "string",
            "description": (
                "The same, in English, at most 20 words. The digest has a language switch "
                "and shows one or the other; a missing value leaves a gap in that view."
            ),
        },
        "novelty": {"type": "string", "enum": ["incremental", "notable", "breakthrough"]},
        "read_depth": {"type": "string", "enum": ["deep", "shallow", "skip"]},
        "needs_figures": {
            "type": "boolean",
            "description": "True only if the contribution is unreadable without figures.",
        },
    },
    "required": [
        "score",
        "matched_topics",
        "reason",
        "reason_en",
        "novelty",
        "read_depth",
        "needs_figures",
    ],
    "additionalProperties": False,
}


def build_system(cfg: Config) -> str:
    return read_prompt("triage.md").replace("{{INTERESTS}}", cfg.render_interests())


def _user_block(rec: dict) -> str:
    parts = [
        f"Title: {rec['title']}",
        f"Categories: {' '.join(rec.get('categories') or [])}",
    ]
    if rec.get("comments"):
        parts.append(f"Comments: {rec['comments']}")
    if rec.get("journal_ref"):
        parts.append(f"Journal-ref: {rec['journal_ref']}")
    kw = rec.get("keyword") or {}
    if kw.get("topics"):
        parts.append(f"Tier-1 keyword hits in topics: {', '.join(kw['topics'])}")
    parts.append(f"\nAbstract:\n{rec['abstract']}")
    return "\n".join(parts)


def run(
    pool: ClientPool, records: list[dict], cfg: Config, usage: Usage, workers: int = 6
) -> tuple[int, list[tuple[str, str]]]:
    """Score records in place. Returns (scored_count, [(arxiv_id, error), ...])."""
    system = build_system(cfg)
    model = cfg.models.get("triage", "claude-haiku-4-5")
    by_id = {r["arxiv_id"]: r for r in records}

    def work(rec: dict) -> dict:
        verdict, meta = call_structured(
            pool,
            model=model,
            system=system,
            user=_user_block(rec),
            schema=TRIAGE_SCHEMA,
            max_tokens=1024,
            cfg=cfg,
            usage=usage,
            # Scoring an abstract against a rubric does not need a reasoning pass, and
            # thinking tokens bill as output on every one of ~130 daily calls.
            thinking=False,
            tool_description="Score this paper's relevance to the reader.",
        )
        verdict["_meta"] = meta
        return verdict

    errors: list[tuple[str, str]] = []
    scored = 0
    for rec, result in warm_then_parallel(records, work, workers=workers):
        target = by_id[rec["arxiv_id"]]
        if isinstance(result, BaseException):
            errors.append((rec["arxiv_id"], f"{type(result).__name__}: {result}"))
            target["triage"] = {"score": None, "error": str(result)}
            continue
        target["triage"] = result
        scored += 1
    return scored, errors

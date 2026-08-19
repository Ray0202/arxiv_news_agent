"""The summarize stage: deep (Opus 5, full text) and shallow (Haiku 4.5, abstract only)."""

from __future__ import annotations

from typing import Any

from ..config import Config, read_prompt
from .. import providers
from ..llm import (
    ClientPool,
    Usage,
    call_structured,
    estimate_input_tokens,
    warm_then_parallel,
)
from ..sources import fulltext
from ..verify import check_lengths, check_numbers, check_terms
from .schema import build as build_schema

# The ceiling covers reasoning *and* visible output on every backend. Measured: at
# `high` effort on a 39k-character paper, gpt-5.4-mini spent all 16,000 on reasoning and
# returned nothing. Output is billed by actual use, so headroom is free unless used.
DEEP_MAX_TOKENS = 32_000
SHALLOW_MAX_TOKENS = 4_000


def build_system(cfg: Config) -> str:
    """Render the summarize system prompt, including the reader's interest profile.

    Without the profile injected, `why_it_matters_to_me` asks the model to relate the
    paper to topics it was never told about, and it correctly answers "reader's interests
    not provided".
    """
    prompt = read_prompt("summarize_system.md")
    if "{{INTERESTS}}" not in prompt:
        raise ValueError(
            "summarize_system.md must contain {{INTERESTS}} so the reader's profile "
            "reaches the model; why_it_matters_to_me is unanswerable without it."
        )
    return prompt.replace("{{INTERESTS}}", cfg.render_interests())


def _header(rec: dict) -> str:
    lines = [
        f"arXiv: {rec['arxiv_id']}",
        f"Title: {rec['title']}",
        f"Authors: {', '.join(rec.get('authors') or []) or 'unknown'}",
        f"Categories: {' '.join(rec.get('categories') or [])}",
        f"Submitted: {rec.get('created')}",
    ]
    if rec.get("comments"):
        lines.append(f"Comments: {rec['comments']}")
    if rec.get("journal_ref"):
        lines.append(f"Journal-ref: {rec['journal_ref']}")
    return "\n".join(lines)


def _figure_inventory(figures: list[dict]) -> str:
    """List the paper's figures so the model can pick by number.

    Without the inventory it invents plausible-looking labels, and the site has nothing to
    match them against.
    """
    if not figures:
        return ""
    lines = ["\n--- FIGURE INVENTORY (choose figures_worth_seeing by number) ---"]
    for fig in figures:
        if fig.get("number") is None:
            continue
        label = fig.get("label") or f"{fig['kind']} {fig['number']}"
        lines.append(f"[{fig['kind']} {fig['number']}] {label}: {fig.get('caption', '')[:220]}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _deep_user(rec: dict, text: str, meta: dict) -> str:
    notes = []
    if meta.get("dropped_sections"):
        notes.append(
            "Sections omitted to fit the budget: " + ", ".join(meta["dropped_sections"])
        )
    if meta.get("truncated"):
        notes.append("The text below is truncated; treat late-paper claims as unseen.")
    note = ("\nNOTE: " + " ".join(notes) + "\n") if notes else ""
    inventory = _figure_inventory(meta.get("figures") or [])
    return (
        f"{_header(rec)}\n{note}{inventory}\n"
        f"\n--- FULL TEXT ({meta.get('source')}) ---\n{text}"
    )


def _shallow_user(rec: dict) -> str:
    return (
        f"{_header(rec)}\n\n--- ABSTRACT ONLY ---\n{rec['abstract']}\n\n"
        "You have only the abstract. Set confidence.level to 'low', caveat to '仅有摘要' "
        "and caveat_en to 'abstract only', leave institutions empty, and put only numbers "
        "the abstract itself states into results."
    )


def summarize_one(
    pool: ClientPool,
    rec: dict,
    cfg: Config,
    usage: Usage,
    schema: dict,
    system: str,
    depth: str,
    max_chars: int,
) -> dict:
    if depth == "deep":
        ft = fulltext.fetch(rec["arxiv_id"], max_chars=max_chars)
        if ft["source"] == "none" or ft["chars"] < 2000:
            # No usable full text: degrade to the shallow path rather than sending the
            # model an empty document and getting a confident summary of nothing.
            depth = "shallow"
            ft = {"text": "", "source": "none", "chars": 0, "truncated": False,
                  "dropped_sections": [], "figures": []}
    else:
        ft = {"text": "", "source": "abstract", "chars": 0, "truncated": False,
              "dropped_sections": [], "figures": []}

    if depth == "deep":
        model = cfg.models.get("deep", "claude-opus-5")
        effort = cfg.models.get("deep_effort", "high")
        user = _deep_user(rec, ft["text"], ft)
        max_tokens = DEEP_MAX_TOKENS
        thinking = True
    else:
        model = cfg.models.get("shallow", "claude-haiku-4-5")
        effort = None
        user = _shallow_user(rec)
        max_tokens = SHALLOW_MAX_TOKENS
        # Paraphrasing an abstract is not a reasoning task.
        thinking = False

    summary, meta = call_structured(
        pool,
        model=model,
        system=system,
        user=user,
        schema=schema,
        max_tokens=max_tokens,
        effort=effort,
        thinking=thinking,
        cfg=cfg,
        usage=usage,
        tool_description="Emit the digest record for this paper.",
    )

    source_for_check = ft["text"] or rec["abstract"]
    numbers = check_numbers(summary, source_for_check)
    return {
        "summary": summary,
        "fulltext": {k: v for k, v in ft.items() if k not in ("text", "figures")},
        "figures": ft.get("figures") or [],
        "depth": depth,
        "verify": {
            "numbers": numbers,
            "missing_terms": check_terms(summary, source_for_check),
            "lengths": check_lengths(summary, cfg),
        },
        "llm": meta,
    }


def run(
    pool: ClientPool,
    records: list[dict],
    cfg: Config,
    usage: Usage,
    depth: str,
    workers: int = 4,
    max_chars: int = 90_000,
) -> tuple[int, list[tuple[str, str]]]:
    """Summarize records in place. Returns (count, [(arxiv_id, error), ...])."""
    schema = build_schema(cfg)
    system = build_system(cfg)
    by_id = {r["arxiv_id"]: r for r in records}

    def work(rec: dict) -> dict:
        return summarize_one(
            pool, rec, cfg, usage, schema, system, depth, max_chars
        )

    errors: list[tuple[str, str]] = []
    done = 0
    for rec, result in warm_then_parallel(records, work, workers=workers):
        target = by_id[rec["arxiv_id"]]
        if isinstance(result, BaseException):
            errors.append((rec["arxiv_id"], f"{type(result).__name__}: {result}"))
            target["summary_error"] = str(result)
            continue
        target.update(result)
        target.pop("summary_error", None)
        done += 1
    return done, errors


def estimate_deep_cost(
    pool: ClientPool, rec: dict, cfg: Config, max_chars: int
) -> float:
    """Pre-flight USD estimate for one deep read, used by the daily budget gate."""
    model = cfg.models.get("deep", "claude-opus-5")
    provider = providers.for_model(model, default=cfg.provider)
    price = provider.price(model)
    ft = fulltext.fetch(rec["arxiv_id"], max_chars=max_chars)
    try:
        n_in = estimate_input_tokens(
            pool, cfg, model, build_system(cfg), _deep_user(rec, ft["text"], ft)
        )
    except Exception:
        n_in = ft["chars"] // 4 + 1200
    # Visible output plus thinking; measured to land near 4k at effort=high with two
    # languages. Revisit if the observed cost in data/runs drifts from this.
    assumed_out = 4_500
    return (
        (n_in * price.input + assumed_out * price.output)
        * provider.multiplier_now()
        / 1_000_000
    )


# evidence_grade -> a 0-1 strength. Deliberately coarse: the grade is a four-way
# judgement, and interpolating it finer would invent precision the model never had.
_EVIDENCE_STRENGTH = {"A": 1.0, "B": 0.75, "C": 0.45, "D": 0.15}


def priority(rec: dict, cfg: Config) -> float:
    """Rank *among already-relevant papers*. Never a quality score, never a gate.

    relevance dominates; the evidence audit nudges. `quality_confidence` scales the
    reviewer's contribution rather than the paper's standing — a low-confidence audit
    should move the ranking less, not push the paper down.
    """
    w = (cfg.selection.get("weights") or {})
    w_rel = float(w.get("relevance", 0.8))
    w_ev = float(w.get("evidence", 0.2))
    w_risk = float(w.get("high_risk_penalty", 0.5))

    relevance = (rec.get("triage") or {}).get("score") or 0
    base = w_rel * (relevance / 10.0)

    review = rec.get("review")
    if not review:
        # Un-reviewed papers are ranked on relevance alone rather than penalised; the
        # audit is an optional signal, and missing it must not demote a paper.
        return base

    confidence = float(review.get("quality_confidence") or 0.0)
    confidence = min(max(confidence, 0.0), 1.0)
    strength = _EVIDENCE_STRENGTH.get(review.get("evidence_grade", ""), 0.45)
    high_risk = 1.0 if review.get("evaluation_risk") == "high" else 0.0
    return base + confidence * (w_ev * strength - w_risk * high_risk)


def pick(records: list[dict], cfg: Config) -> tuple[list[dict], list[dict]]:
    """Split triaged records into (deep, shallow) honouring thresholds and daily caps."""
    min_score = int(cfg.thresholds.get("llm_min_score", 6))
    deep_cap = int(cfg.budget.get("deep_read_max_per_day", 5))
    shallow_cap = int(cfg.budget.get("shallow_max_per_day", 10))

    eligible = [
        r
        for r in records
        if isinstance((r.get("triage") or {}).get("score"), int)
        and r["triage"]["score"] >= min_score
        and r["triage"].get("read_depth") != "skip"
    ]
    eligible.sort(
        key=lambda r: (-r["triage"]["score"], -(r.get("keyword") or {}).get("score", 0))
    )

    # Deep slots: the first few go on relevance alone so a reviewer misjudgement can
    # never squeeze out a genuinely on-topic new direction; the rest are ordered by
    # priority, which folds in the evidence audit.
    influence = cfg.selection.get("evidence_influence", "deep_order")

    if influence == "slots":
        # The audit also decides who gets in at all. Only turn this on with evidence
        # from the feedback log that it predicts "worth a deep read" better than
        # relevance alone — otherwise a reviewer misjudgement silently drops papers.
        eligible.sort(key=lambda r: -priority(r, cfg))

    candidates = [r for r in eligible if r["triage"].get("read_depth") == "deep"]
    if influence == "off":
        deep = candidates[:deep_cap]
    else:
        pure = int(cfg.selection.get("pure_relevance_slots", deep_cap))
        deep = candidates[: min(pure, deep_cap)]
        remaining = [r for r in candidates if r not in deep]
        remaining.sort(key=lambda r: -priority(r, cfg))
        deep += remaining[: max(0, deep_cap - len(deep))]
    deep_ids = {r["arxiv_id"] for r in deep}
    # Anything eligible that did not make the deep cut still earns an abstract-level note.
    shallow = [r for r in eligible if r["arxiv_id"] not in deep_ids][:shallow_cap]
    return deep, shallow

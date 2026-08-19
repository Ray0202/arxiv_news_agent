"""Reviewer-lite: an evidence audit, not a quality verdict.

One call per paper on the full text. It asks what the paper's central claims are and
whether the evidence inside the paper carries them — deliberately *not* "is this good".

The load-bearing constraint is that every judgement is anchored. The extraction stage tags
each paragraph, table and figure with a stable LaTeXML id (`S2.p3`, `S3.T2`, `S1.F1`) and
the model must cite the ones it used. `validate` then checks those ids against the real
document: anything citing an id that does not exist is demoted to `unknowns` rather than
being published as a finding. A model that cannot say where it read something does not get
to assert it.

Deliberately absent: an overall score. `evidence_grade`, `evaluation_risk` and
`quality_confidence` describe different things and are meant to be read separately; a
weighted sum of them would be false precision.
"""

from __future__ import annotations

from typing import Any

from .config import Config, read_prompt
from .llm import ClientPool, Usage, call_structured, warm_then_parallel
from .sources import fulltext

MAX_TOKENS = 32_000

PAPER_TYPES = ["empirical_method", "theory", "systems", "benchmark", "survey"]
SUPPORT = ["direct", "partial", "absent", "not_applicable"]
RISK = ["low", "medium", "high"]

_BILINGUAL = {
    "text": {"type": "string", "description": "In Chinese, one sentence."},
    "text_en": {"type": "string", "description": "The same, in English."},
}


def _evidence_list(desc: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": desc,
    }


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "paper_type": {"type": "string", "enum": PAPER_TYPES},
        "claims": {
            "type": "array",
            "description": "At most 3 central claims, most important first.",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "In Chinese, one sentence."},
                    "claim_en": {"type": "string", "description": "The same, in English."},
                    "evidence_ids": _evidence_list(
                        "Anchors from the text (e.g. 'S3.T2', 'S2.p4') that carry this "
                        "claim. Ids not present in the document are discarded."
                    ),
                    "support": {"type": "string", "enum": SUPPORT},
                    "why": {
                        "type": "string",
                        "description": "In Chinese: what the cited evidence does and does "
                                       "not establish. One or two sentences.",
                    },
                    "why_en": {"type": "string", "description": "The same, in English."},
                },
                "required": ["claim", "claim_en", "evidence_ids", "support", "why", "why_en"],
                "additionalProperties": False,
            },
        },
        "risks": {
            "type": "array",
            "description": (
                "1-3 setup choices most likely to change the conclusion. Each needs "
                "evidence_ids; anything unanchored belongs in unknowns."
            ),
            "items": {
                "type": "object",
                "properties": {
                    **_BILINGUAL,
                    "kind": {
                        "type": "string",
                        "enum": ["baseline", "ablation", "generalisation", "statistics",
                                 "construction", "scale", "other"],
                    },
                    "evidence_ids": _evidence_list("Anchors this risk was read from."),
                },
                "required": ["text", "text_en", "kind", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "author_limitations": {
            "type": "array",
            "description": "Limitations the paper itself states. Cite where it says so.",
            "items": {
                "type": "object",
                "properties": {**_BILINGUAL,
                               "evidence_ids": _evidence_list("Where the paper says it.")},
                "required": ["text", "text_en", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "reader_limitations": {
            "type": "array",
            "description": "Gaps you inferred that the paper does not acknowledge.",
            "items": {
                "type": "object",
                "properties": {**_BILINGUAL,
                               "evidence_ids": _evidence_list("What you read to infer it.")},
                "required": ["text", "text_en", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "unknowns": {
            "type": "array",
            "description": (
                "Things you could not verify from the provided text — an omitted section, "
                "a claim with no locatable evidence, a subfield outside your competence. "
                "This is the correct home for anything you cannot anchor."
            ),
            "items": {
                "type": "object",
                "properties": {**_BILINGUAL},
                "required": ["text", "text_en"],
                "additionalProperties": False,
            },
        },
        "evidence_grade": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "method_risk": {"type": "string", "enum": RISK},
        "evaluation_risk": {"type": "string", "enum": RISK},
        "quality_confidence": {
            "type": "number",
            "description": "0-1: confidence in this audit, not in the paper.",
        },
    },
    "required": [
        "paper_type", "claims", "risks", "author_limitations", "reader_limitations",
        "unknowns", "evidence_grade", "method_risk", "evaluation_risk",
        "quality_confidence",
    ],
    "additionalProperties": False,
}


def build_system(cfg: Config) -> str:
    return read_prompt("reviewer.md")


def _user_block(rec: dict, ft: dict) -> str:
    anchors = ft.get("anchors") or {}
    head = [
        f"arXiv: {rec['arxiv_id']}",
        f"Title: {rec['title']}",
        f"Categories: {' '.join(rec.get('categories') or [])}",
    ]
    if rec.get("comments"):
        head.append(f"Comments: {rec['comments']}")
    notes = []
    if ft.get("dropped_sections"):
        notes.append(
            "Sections omitted from this extraction: " + ", ".join(ft["dropped_sections"])
            + " — treat anything that would live there as unknown, not absent."
        )
    if ft.get("truncated"):
        notes.append("The text is truncated; later sections are unseen.")
    note = ("\nNOTE: " + " ".join(notes)) if notes else ""
    inventory = (
        f"\nCitable anchors ({len(anchors)}): "
        + ", ".join(sorted(anchors)[:120])
        + ("…" if len(anchors) > 120 else "")
    )
    return (
        f"{'\n'.join(head)}{note}{inventory}\n\n--- FULL TEXT (anchored) ---\n{ft['text']}"
    )


def _repair(cited: str, valid: set[str]) -> str | None:
    """Resolve an over-qualified anchor to the real one, or None.

    Observed: the model cited `S2.Thmdefinition1.p1` for a block whose actual LaTeXML id
    is `Thmdefinition1.p1` — it prefixed the section the way ordinary paragraph ids are
    built. The block is real and the citation is meaningful, so rejecting it would throw
    away a good finding.

    This does not loosen the guarantee. Repair requires the cited string to *end with* a
    real anchor at a dot boundary, and takes the longest such match. A fabricated id like
    `S5.SS3.p2` — where the paper has `S5.SS3` but no `.p2` — has no candidate and is
    still rejected.
    """
    if cited in valid:
        return cited
    matches = [a for a in valid if cited.endswith("." + a)]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    # Ambiguous only if two candidates tie at maximal length, which cannot happen for
    # distinct strings that are both suffixes of the same string.
    return matches[0]


def validate(audit: dict, anchors: dict) -> dict:
    """Drop unanchored findings into `unknowns`. Returns a report of what moved.

    This is the whole point of the stage: a risk the model cannot locate in the document
    is an assertion, not a finding, and publishing it would be exactly the unfounded
    critique the design set out to avoid.
    """
    valid = set(anchors or {})
    moved: list[dict] = []
    dropped_ids: list[str] = []
    repaired: list[tuple[str, str]] = []

    def resolve(raw: list[str] | None) -> tuple[list[str], list[str]]:
        good: list[str] = []
        bad: list[str] = []
        for cited in raw or []:
            fixed = _repair(cited, valid)
            if fixed is None:
                bad.append(cited)
            else:
                if fixed != cited:
                    repaired.append((cited, fixed))
                if fixed not in good:
                    good.append(fixed)
        return good, bad

    def keep(item: dict, label: str) -> bool:
        ids, bad = resolve(item.get("evidence_ids"))
        dropped_ids.extend(bad)
        item["evidence_ids"] = ids
        if ids:
            return True
        moved.append({"from": label, "text": item.get("text") or item.get("claim", ""),
                      "text_en": item.get("text_en") or item.get("claim_en", ""),
                      "invalid_ids": bad})
        return False

    for field in ("risks", "author_limitations", "reader_limitations"):
        audit[field] = [i for i in (audit.get(field) or []) if keep(i, field)]

    # A claim with no locatable evidence is not deleted — the claim is still what the
    # paper says — but its support level cannot be trusted, so it is marked `absent`
    # and the reasoning is moved out of the published finding.
    for claim in audit.get("claims") or []:
        ids, bad = resolve(claim.get("evidence_ids"))
        dropped_ids.extend(bad)
        claim["evidence_ids"] = ids
        if not ids and claim.get("support") in ("direct", "partial"):
            claim["support"] = "absent"
            claim["unanchored"] = True

    audit.setdefault("unknowns", [])
    for m in moved:
        audit["unknowns"].append({"text": m["text"], "text_en": m["text_en"]})

    return {
        "checked_ids": sum(
            len(i.get("evidence_ids") or [])
            for f in ("claims", "risks", "author_limitations", "reader_limitations")
            for i in (audit.get(f) or [])
        ),
        "invalid_ids": sorted(set(dropped_ids)),
        "repaired_ids": sorted({f"{a}->{b}" for a, b in repaired}),
        "demoted_to_unknown": len(moved),
        "anchors_available": len(valid),
    }


def review_one(
    pool: ClientPool, rec: dict, cfg: Config, usage: Usage, system: str, max_chars: int
) -> dict:
    ft = fulltext.fetch(rec["arxiv_id"], max_chars=max_chars)
    if not ft["text"] or not ft.get("anchors"):
        # No anchors means nothing is citable, and an audit with no citable evidence is
        # exactly what this stage exists to refuse to produce.
        return {
            "review": None,
            "review_skipped": (
                "no anchored full text" if not ft["text"] else f"{ft['source']} has no anchors"
            ),
        }

    audit, meta = call_structured(
        pool,
        model=cfg.models.get("reviewer", cfg.models.get("deep")),
        system=system,
        user=_user_block(rec, ft),
        schema=SCHEMA,
        max_tokens=MAX_TOKENS,
        effort=cfg.models.get("reviewer_effort", "high"),
        cfg=cfg,
        usage=usage,
        tool_description="Audit this paper's claims against its own evidence.",
    )
    report = validate(audit, ft.get("anchors") or {})
    return {"review": audit, "review_check": report, "review_llm": meta,
            "review_skipped": None}


def run(
    pool: ClientPool, records: list[dict], cfg: Config, usage: Usage,
    workers: int = 4, max_chars: int = 90_000,
) -> tuple[int, list[tuple[str, str]]]:
    system = build_system(cfg)
    by_id = {r["arxiv_id"]: r for r in records}

    def work(rec: dict) -> dict:
        return review_one(pool, rec, cfg, usage, system, max_chars)

    errors: list[tuple[str, str]] = []
    done = 0
    for rec, result in warm_then_parallel(records, work, workers=workers):
        target = by_id[rec["arxiv_id"]]
        if isinstance(result, BaseException):
            errors.append((rec["arxiv_id"], f"{type(result).__name__}: {result}"))
            target["review_error"] = str(result)
            continue
        target.update(result)
        target.pop("review_error", None)
        if result.get("review"):
            done += 1
    return done, errors

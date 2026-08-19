"""Static site generation: one digest page per day plus an index.

Deliberately Node-free — Jinja2 renders self-contained HTML with inline CSS, so the whole
site is `git add site/` and nothing else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import feedback
from ..config import RUNS_DIR, SITE_DIR, load_config
from ..store import available_dates, load_day
from .mathfix import normalize_summary

TEMPLATES = Path(__file__).parent / "templates"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["paragraphs"] = _paragraphs
    return env


def _paragraphs(text: str | None) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _venue_badge(rec: dict) -> dict[str, str] | None:
    """Phase-1 stand-in for the enrich stage: read acceptance straight off arXiv fields."""
    if rec.get("journal_ref"):
        return {"label": "published", "detail": rec["journal_ref"][:60]}
    comments = rec.get("comments") or ""
    m = re.search(
        r"(accepted|to appear|camera[-\s]?ready|oral|spotlight|findings)[^.;]{0,60}?"
        r"(NeurIPS|ICML|ICLR|CVPR|ACL|EMNLP|AAAI|KDD|WWW|SIGIR|ICCV|ECCV|NAACL|COLM|TMLR|JMLR|IJCAI|WACV)"
        r"[^.;]{0,12}",
        comments,
        re.I,
    )
    if m:
        return {"label": m.group(2).upper(), "detail": m.group(0)[:70]}
    return None


def _code_badge(rec: dict) -> str | None:
    blob = " ".join(filter(None, [rec.get("abstract"), rec.get("comments")]))
    m = re.search(
        r"https?://(?:github\.com|gitlab\.com|huggingface\.co)/[\w.\-/]+", blob
    )
    return m.group(0).rstrip(".,);") if m else None


def _chosen_figures(rec: dict, summary: dict) -> list[dict[str, Any]]:
    """Match the model's picks to real extracted figures, by number.

    The model chooses from an inventory we hand it, so a number it returns should exist.
    Anything that does not match is dropped rather than rendered as a broken image.
    """
    by_number: dict[tuple[str, int], dict] = {}
    for fig in rec.get("figures") or []:
        if fig.get("number") is not None:
            by_number[(fig.get("kind", "figure"), fig["number"])] = fig
    out = []
    for pick in summary.get("figures_worth_seeing") or []:
        if not isinstance(pick, dict):
            continue  # older records stored these as free text
        fig = by_number.get((pick.get("kind", "figure"), pick.get("number")))
        if fig:
            out.append({
                **fig,
                "why": pick.get("why", ""),
                # Falls back to the Chinese clause so the English view is never blank on
                # records written before the bilingual schema.
                "why_en": pick.get("why_en") or pick.get("why", ""),
            })
    return out


def _confidence_note(rec: dict, summary: dict) -> dict[str, str] | None:
    """A visible reason, or None when the badge would carry no information.

    `confidence: low` on an abstract-only summary is not a warning, it is the definition of
    that tier — the prompt tells the model to set it. Rendering it next to the "仅摘要"
    badge put two chips saying the same thing on every shallow card, in alarming wording,
    with the actual reason hidden in a tooltip. So: show nothing when it is redundant, and
    name the concrete cause when it is not.
    """
    conf = summary.get("confidence") or {}
    level = conf.get("level")
    if level == "high":
        return None
    ft = rec.get("fulltext") or {}
    if rec.get("depth") != "deep":
        return None  # the depth badge already says 仅摘要 / abstract only
    if ft.get("truncated"):
        return {"zh": "正文被截断", "en": "text truncated"}
    if ft.get("source") == "pdf":
        return {"zh": "PDF 抽取，非 HTML", "en": "from PDF, not HTML"}
    if ft.get("source") == "none":
        return {"zh": "取不到全文", "en": "no full text"}
    # Fall back to the model's own caveat, per language. An absent English twin yields
    # an empty string, and the template omits that chip rather than showing Chinese.
    return {
        "zh": (conf.get("caveat") or "")[:24] or f"{level} confidence",
        "en": (conf.get("caveat_en") or "")[:34] or f"{level} confidence",
    }


def prepare(records: list[dict], cfg=None) -> list[dict[str, Any]]:
    """Keep only summarized papers and flatten what the template needs."""
    from ..summarize.runner import priority as _priority

    if cfg is None:
        try:
            cfg = load_config()
        except Exception:  # noqa: BLE001 - rendering must not depend on a readable config
            cfg = None
    out = []
    for rec in records:
        summary = rec.get("summary")
        if not summary:
            continue
        # `selected_for: None` marks a summary superseded by a later run whose triage
        # chose differently. It stays in the data; it does not belong on the page.
        if "selected_for" in rec and rec["selected_for"] is None:
            continue
        # Wrap any notation the model left undelimited so KaTeX can render it.
        summary = normalize_summary(summary)
        triage = rec.get("triage") or {}
        unverified = ((rec.get("verify") or {}).get("numbers") or {}).get("unverified", [])
        out.append(
            {
                "arxiv_id": rec["arxiv_id"],
                "title": rec["title"],
                "abs_url": rec["abs_url"],
                "authors": rec.get("authors") or [],
                "categories": rec.get("categories") or [],
                "created": rec.get("created"),
                "depth": rec.get("depth", "shallow"),
                "score": triage.get("score"),
                "novelty": triage.get("novelty"),
                "triage_reason": triage.get("reason"),
                "triage_reason_en": triage.get("reason_en"),
                "summary": summary,
                "venue": _venue_badge(rec),
                "code_url": _code_badge(rec),
                "unverified": unverified,
                "fulltext": rec.get("fulltext") or {},
                "chosen_figures": _chosen_figures(rec, summary),
                "review": rec.get("review"),
                "related_recent": rec.get("related_recent") or [],
                # Stored with the label so a later analysis compares against the numbers
                # the judgement was actually made on, not whatever a re-run produced.
                "signals_json": json.dumps(
                    feedback.signals_for(
                        rec, _priority(rec, cfg) if cfg is not None else None
                    ),
                    ensure_ascii=False,
                ),
                "review_check": rec.get("review_check"),
                "anchor_sections": (rec.get("fulltext") or {}).get("anchor_sections") or {},
                "confidence_note": _confidence_note(rec, summary),
            }
        )
    out.sort(key=lambda r: (-(r["score"] or 0), r["arxiv_id"]))
    return out


def load_run(date: str) -> dict[str, Any]:
    """That day's run log: the recorded criteria and funnel counts."""
    path = RUNS_DIR / f"{date}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _tier1_count(run: dict, records: list[dict]) -> int:
    """How many papers cleared the keyword filter.

    Prefer the number the filter stage recorded. Re-deriving it here means reimplementing
    `keyword.passes` in a second place, and the first attempt did exactly that — counting
    any non-zero score instead of `score >= threshold`, so the page claimed 128 where the
    pipeline had reported 127.
    """
    recorded = (run.get("filter") or {}).get("passed")
    if isinstance(recorded, int):
        return recorded
    threshold = float(
        ((run.get("criteria") or {}).get("thresholds") or {}).get("keyword_min_score", 1.0)
    )
    return sum(
        1
        for r in records
        if (r.get("keyword") or {}).get("category_match")
        and (r.get("keyword") or {}).get("score", 0) >= threshold
    )


def _was_judged(records: list[dict]) -> bool:
    """Did this day get as far as LLM scoring? Distinguishes "nothing was relevant" from
    "the pipeline has not run yet", which look identical from the published output."""
    return any(isinstance((r.get("triage") or {}).get("score"), int) for r in records)


def build(
    dates: list[str] | None = None,
    out_dir: Path | None = None,
    *,
    loader=load_day,
    dates_source=available_dates,
    run_loader=load_run,
    site_days: int = 0,
) -> list[Path]:
    """Render the site. `loader`/`dates_source` are injectable so tests can render
    fixture data without writing fake summaries into `data/`.

    `site_days` bounds the window to the most recent N days. It has to be applied here
    rather than by deleting directories afterwards: records outlive their pages by design,
    so a build that rendered every day it has data for would keep resurrecting the pages
    the retention policy just removed, and the archive would link to them either way.
    """
    out_dir = out_dir or SITE_DIR
    env = _env()
    all_dates = dates_source()
    if site_days > 0:
        all_dates = sorted(all_dates)[-site_days:]
    targets = [d for d in (dates or all_dates) if d in set(all_dates)]
    written: list[Path] = []

    digest_tpl = env.get_template("digest.html")
    index_tpl = env.get_template("index.html")

    summaries: list[dict[str, Any]] = []
    for date in all_dates:
        day_records = loader(date)
        papers = prepare(day_records)
        ingested = len(day_records)
        run = run_loader(date)
        if not papers and not _was_judged(day_records):
            # A day that was harvested but not yet scored is not an issue with nothing in
            # it — it is a day that has not happened yet. Listing it as "无入选论文" would
            # claim the pipeline looked and found nothing, which is a different and false
            # statement. Only days that actually reached triage get an archive entry.
            continue
        summaries.append(
            {
                "date": date,
                "count": len(papers),
                "deep": sum(1 for p in papers if p["depth"] == "deep"),
                "titles": [p["title"] for p in papers[:3]],
            }
        )
        if date not in targets:
            continue
        page_dir = out_dir / date
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / "index.html"
        path.write_text(
            digest_tpl.render(
                date=date, papers=papers, all_dates=all_dates, ingested=ingested,
                criteria=run.get("criteria"), tier1=_tier1_count(run, day_records),
                # `relate` looks back 30 days but only `site_days` of pages survive, so a
                # thread marker can name an issue that no longer has a page. Naming it is
                # still useful; linking to a 404 is not.
                linkable=set(all_dates),
            ),
            encoding="utf-8",
        )
        written.append(path)

    summaries.sort(key=lambda s: s["date"], reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = summaries[0]["date"] if summaries else None
    if latest:
        # Root page mirrors the newest digest so the site has a real landing page.
        latest_records = loader(latest)
        papers = prepare(latest_records)
        (out_dir / "index.html").write_text(
            digest_tpl.render(
                date=latest, papers=papers, all_dates=all_dates, is_root=True,
                ingested=len(latest_records),
                criteria=run_loader(latest).get("criteria"),
                tier1=_tier1_count(run_loader(latest), latest_records),
            ),
            encoding="utf-8",
        )
        written.append(out_dir / "index.html")
    (out_dir / "archive.html").write_text(
        index_tpl.render(days=summaries), encoding="utf-8"
    )
    written.append(out_dir / "archive.html")

    (out_dir / "digest.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(out_dir / "digest.json")
    return written

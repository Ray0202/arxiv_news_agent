"""Command line entry point.

Every stage is separately runnable and idempotent, keyed on `arxiv_id`. Stages record
themselves in `stages` on each record; re-running skips completed work unless `--force`.
That matters because the summarize stage is the only expensive one and a crash in the site
build must never mean paying for those tokens twice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from .config import CACHE_DIR, DATA_DIR, SITE_DIR, Config, load_config
from .filters import keyword, triage
from .llm import ClientPool, Usage, preflight
from .sources import oai
from .store import (
    available_dates,
    has_stage,
    load_day,
    mark,
    merge_day,
    save_day,
    write_run_log,
)
from . import feedback as fb
from . import memory as mem
from . import retention
from . import reviewer
from .summarize import runner as summarizer


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def resolve_date(value: str | None) -> str:
    """`auto` picks the most recent weekday, since arXiv does not announce on weekends."""
    if value and value != "auto":
        dt.date.fromisoformat(value)  # validate
        return value
    day = dt.datetime.now(dt.timezone.utc).date()
    while day.weekday() >= 5:  # 5=Sat, 6=Sun
        day -= dt.timedelta(days=1)
    return day.isoformat()


# --------------------------------------------------------------------------- ingest
def cmd_ingest(args: argparse.Namespace, cfg: Config) -> int:
    date = resolve_date(args.date)
    date_from = args.date_from or date
    window = int(cfg.ingest.get("new_submission_window_days", 5))
    sets = cfg.ingest.get("sets", ["cs"])
    base = cfg.ingest.get("oai_base", oai.DEFAULT_BASE)

    _log(f"[ingest] harvesting {date_from}..{date} sets={sets}")
    total = kept = 0
    incoming = []
    for rec in oai.harvest(date_from, date, sets, base=base):
        total += 1
        rec["is_new_submission"] = oai.is_new_submission(rec, window)
        if args.include_revisions or rec["is_new_submission"]:
            rec["harvest_date"] = date
            incoming.append(mark(rec, "ingest"))
            kept += 1
    added, updated = merge_day(date, incoming)
    _log(
        f"[ingest] {total} records from OAI, {kept} new submissions "
        f"({total - kept} were metadata revisions) -> +{added} new / {updated} updated"
    )
    write_run_log(date, {"ingest": {"oai_records": total, "kept": kept, "added": added}})
    return 0


# --------------------------------------------------------------------------- filter
def cmd_filter(args: argparse.Namespace, cfg: Config) -> int:
    date = resolve_date(args.date)
    records = load_day(date)
    if not records:
        _log(f"[filter] no records for {date}; run `pna ingest` first")
        return 1

    passed = 0
    for rec in records:
        if has_stage(rec, "filter_keyword") and not args.force:
            if keyword.passes(rec.get("keyword") or {}, cfg):
                passed += 1
            continue
        verdict = keyword.score_paper(rec, cfg)
        rec["keyword"] = verdict
        mark(rec, "filter_keyword")
        if keyword.passes(verdict, cfg):
            passed += 1
    rescued = _semantic_rescue(date, records, cfg)
    save_day(date, records)
    _log(f"[filter] {len(records)} records -> {passed} passed tier-1 keyword filter"
         + (f", +{rescued} rescued semantically" if rescued else ""))
    write_run_log(
        date,
        {
            "filter": {"total": len(records), "passed": passed},
            # Snapshot rather than reference: the page must state the criteria this run
            # used, not whatever interests.yaml says the next time it is rebuilt.
            "criteria": cfg.snapshot(),
        },
    )
    return 0


def _semantic_rescue(date: str, records: list[dict], cfg: Config) -> int:
    """Tier 2: pull back papers the keywords missed but your labels say you want.

    Only ever *adds* to the candidate pool, and only a capped handful. The keyword tier
    already has deliberately poor precision; this exists for the case it cannot reach at
    all — a paper below the keyword threshold that is nonetheless close to something you
    marked worth reading.

    Measured on 2026-07-30 with 5 labels: 14 papers rescued, of which 5 were then scored
    8-9 by triage — a better hit rate than the keyword tier's own (35.7% vs 29.9%). Two
    tempting tightenings were both tested and rejected:

    * **Raising the margin.** It does not separate good from bad. The papers triage scored
      8-9 averaged +0.062; the ones it scored 1 averaged +0.060, over the same range. The
      margin says "near something you liked", which is a recall signal, not a quality one.
    * **Requiring a primary category from `categories`.** It would have removed 4 of the 6
      junk papers, but also the day's best rescue — a cs.SE paper on adaptive failure
      taxonomies for agent feedback, scored 9. A paper filed outside your usual categories
      is precisely what the other tiers cannot reach.

    So the margin only ranks and caps; triage does the judging, and the junk dies there at
    a cost of roughly $0.001 per rescued paper.

    Silently degrades to a no-op when the encoder is not installed or no labels exist yet.
    """
    if not cfg.memory.get("enabled", True):
        return 0
    labels = fb.latest_per_paper()
    if not labels:
        return 0
    try:
        vecs, ids = mem.embed_day(date, records)
    except mem.EmbeddingUnavailable as exc:
        _log(f"[filter] semantic rescue skipped: {exc}")
        return 0

    index = {aid: vecs[i] for i, aid in enumerate(ids)}

    def lookup(aid):
        if aid in index:
            return index[aid]
        for past in available_dates():
            if past == date:
                continue
            past_vecs, past_ids = mem.load_day_vectors(past)
            if past_vecs is not None and aid in past_ids:
                return past_vecs[past_ids.index(aid)]
        return None

    pref = mem.build_preference(labels.values(), lookup)
    if not pref.usable:
        _log("[filter] semantic rescue skipped: no labelled papers have vectors yet")
        return 0

    scores, because = pref.score(vecs)
    margin = float(cfg.memory.get("rescue_min_margin", 0.05))
    cap = int(cfg.memory.get("rescue_max_per_day", 15))

    misses = [
        (scores[i], i) for i, rec in enumerate(records)
        if not keyword.passes(rec.get("keyword") or {}, cfg)
        and (rec.get("keyword") or {}).get("category_match")
        and scores[i] >= margin
    ]
    misses.sort(reverse=True)
    n = 0
    for score, i in misses[:cap]:
        rec = records[i]
        rec.setdefault("keyword", {})["rescued"] = {
            "score": round(float(score), 4),
            "because": because[i],
            "from_label": (labels.get(because[i]) or {}).get("label"),
        }
        n += 1
    if n:
        picked = [s for s, _ in misses[:cap]]
        # How many distinct labels did the work. Recall concentrated on one label means the
        # channel is one paper wide, however good that paper's neighbourhood looks.
        drivers = {because[i] for _, i in misses[:cap]}
        _log(f"[filter] rescue: {len(pref.pos_ids)} liked / {len(pref.neg_ids)} skipped "
             f"labels -> {n} papers added below the keyword threshold "
             f"(margin {min(picked):+.3f}..{max(picked):+.3f}, "
             f"{len(drivers)} of {len(pref.pos_ids)} liked papers pulled them in)")
    return n


# --------------------------------------------------------------------------- triage
def cmd_triage(args: argparse.Namespace, cfg: Config) -> int:
    date = resolve_date(args.date)
    records = load_day(date)
    candidates = [
        r
        for r in records
        # A semantically rescued paper enters the LLM scoring pool exactly like a keyword
        # hit; recall is where memory acts, judgement stays with the model.
        if (keyword.passes(r.get("keyword") or {}, cfg)
            or (r.get("keyword") or {}).get("rescued"))
        and (args.force or not has_stage(r, "triage"))
    ]
    cap = int(cfg.budget.get("triage_max_per_day", 400))
    if len(candidates) > cap:
        _log(
            f"[triage] {len(candidates)} candidates exceeds triage_max_per_day={cap}; "
            f"scoring the {cap} with the highest keyword score and leaving the rest unscored"
        )
        candidates.sort(key=lambda r: -(r.get("keyword") or {}).get("score", 0))
        candidates = candidates[:cap]
    if not candidates:
        _log("[triage] nothing to score")
        return 0

    pool, usage = ClientPool(), Usage()
    preflight(pool, cfg, [cfg.models.get("triage")])
    _log(f"[triage] scoring {len(candidates)} papers with {cfg.models.get('triage')}")
    scored, errors = triage.run(pool, candidates, cfg, usage, workers=args.workers)
    for rec in candidates:
        if (rec.get("triage") or {}).get("score") is not None:
            mark(rec, "triage")
    save_day(date, records)

    kept = sum(
        1
        for r in records
        if isinstance((r.get("triage") or {}).get("score"), int)
        and r["triage"]["score"] >= int(cfg.thresholds.get("llm_min_score", 6))
    )
    _log(f"[triage] scored {scored}, {len(errors)} errors, {kept} above threshold")
    for pid, err in errors[:5]:
        _log(f"  ! {pid}: {err}")
    _log(f"[triage] cost ${usage.usd:.4f} ({usage.as_dict()})")
    if scored == 0 and errors:
        _log("[triage] every request failed; not treating this as success")
        return 1
    write_run_log(
        date,
        {"triage": {"scored": scored, "errors": len(errors), "kept": kept,
                    "usage": usage.as_dict()}},
    )
    return 0


# --------------------------------------------------------------------------- review
def cmd_review(args: argparse.Namespace, cfg: Config) -> int:
    """Reviewer-lite: audit the top-ranked papers' claims against their own evidence."""
    date = resolve_date(args.date)
    records = load_day(date)
    min_score = int(cfg.thresholds.get("llm_min_score", 6))
    cap = int(args.limit or cfg.budget.get("reviewer_max_per_day", 8))

    eligible = [
        r for r in records
        if isinstance((r.get("triage") or {}).get("score"), int)
        and r["triage"]["score"] >= min_score
        and r["triage"].get("read_depth") != "skip"
    ]
    eligible.sort(key=lambda r: -r["triage"]["score"])
    targets = eligible[:cap]
    if not args.force:
        targets = [r for r in targets if r.get("review_error") or "review" not in r]
    if not targets:
        _log("[review] nothing to audit")
        return 0

    pool, usage = ClientPool(), Usage()
    preflight(pool, cfg, [cfg.models.get("reviewer", cfg.models.get("deep"))])
    _log(f"[review] auditing {len(targets)} papers with "
         f"{cfg.models.get('reviewer', cfg.models.get('deep'))}")
    done, errors = reviewer.run(
        pool, targets, cfg, usage, workers=args.workers, max_chars=args.max_chars
    )
    for rec in targets:
        if rec.get("review") or rec.get("review_skipped"):
            mark(rec, "review")
    save_day(date, records)

    skipped = [r["arxiv_id"] for r in targets if r.get("review_skipped")]
    checked = sum((r.get("review_check") or {}).get("checked_ids", 0) for r in targets)
    invalid = sum(len((r.get("review_check") or {}).get("invalid_ids", [])) for r in targets)
    demoted = sum((r.get("review_check") or {}).get("demoted_to_unknown", 0) for r in targets)
    _log(f"[review] audited {done}, skipped {len(skipped)}, {len(errors)} errors")
    _log(f"[review] evidence ids: {checked} valid, {invalid} invalid "
         f"({demoted} findings demoted to unknown)")
    for pid, err in errors[:5]:
        _log(f"  ! {pid}: {err}")
    _log(f"[review] cost ${usage.usd:.4f} ({usage.as_dict()})")
    write_run_log(date, {"review": {"audited": done, "skipped": len(skipped),
                                    "errors": len(errors), "valid_ids": checked,
                                    "invalid_ids": invalid, "demoted": demoted,
                                    "usage": usage.as_dict()}})
    if done == 0 and errors:
        return 1
    return 0


# ------------------------------------------------------------------------ summarize
def cmd_summarize(args: argparse.Namespace, cfg: Config) -> int:
    date = resolve_date(args.date)
    records = load_day(date)
    deep, shallow = summarizer.pick(records, cfg)
    if args.limit:
        deep, shallow = deep[: args.limit], shallow[: args.limit]

    # Record which papers this run selected, before deciding what still needs work.
    # Re-running after a triage change (a new model scores differently, a threshold moves)
    # picks a different set, and the previous run's summaries stay on records that are no
    # longer chosen. Keeping them in the file is right — they cost money and are the audit
    # trail — but rendering them would mix two models and two schema versions into one
    # digest. This runs even when there is nothing left to summarize.
    selected = {r["arxiv_id"]: "deep" for r in deep}
    selected.update({r["arxiv_id"]: "shallow" for r in shallow})
    stale = 0
    for rec in records:
        if rec["arxiv_id"] in selected:
            rec["selected_for"] = selected[rec["arxiv_id"]]
        elif rec.get("summary"):
            if rec.get("selected_for") is not None:
                stale += 1
            rec["selected_for"] = None
    if stale:
        _log(f"[summarize] {stale} summaries from an earlier selection kept but unpublished")
    save_day(date, records)

    if not args.force:
        # A paper promoted from shallow to deep (a threshold moved, a cap changed, the
        # triage model got swapped) already has a summary — an abstract-only one. Skipping
        # on "has a summary" alone leaves it advertised as a full-text read while showing
        # abstract-level content. Only an existing summary *at the required depth* counts.
        deep = [
            r for r in deep
            # A recorded failure means the stored summary predates it and is stale, so a
            # paper that errored last run must be retried rather than skipped as "done".
            if r.get("summary_error") or r.get("depth") != "deep" or not r.get("summary")
        ]
        # A demotion needs no work: a deep summary is strictly better than the shallow one
        # it would be replaced with, and it is already paid for.
        shallow = [r for r in shallow if r.get("summary_error") or not r.get("summary")]
    if not (deep or shallow):
        _log("[summarize] nothing to do (already summarized, or nothing above threshold)")
        return 0

    pool, usage = ClientPool(), Usage()
    preflight(pool, cfg, [cfg.models.get("deep"), cfg.models.get("shallow")])

    # Budget gate: estimate before spending, not after.
    cap = float(cfg.budget.get("usd_max_per_day", 0) or 0)
    if cap and deep:
        est = 0.0
        allowed = []
        for rec in deep:
            cost = summarizer.estimate_deep_cost(pool, rec, cfg, args.max_chars)
            if est + cost > cap:
                _log(
                    f"[summarize] budget gate: stopping at {len(allowed)} deep reads "
                    f"(estimated ${est:.2f}, cap ${cap:.2f}); "
                    f"{len(deep) - len(allowed)} dropped"
                )
                break
            est += cost
            allowed.append(rec)
        deep = allowed
        _log(f"[summarize] estimated deep-read cost ${est:.3f} (cap ${cap:.2f})")

    total_errors: list[tuple[str, str]] = []
    if deep:
        _log(f"[summarize] deep-reading {len(deep)} papers with {cfg.models.get('deep')}")
        n, errs = summarizer.run(
            pool, deep, cfg, usage, "deep", workers=args.workers,
            max_chars=args.max_chars,
        )
        total_errors += errs
        _log(f"[summarize] deep done: {n} ok, {len(errs)} failed")
    if shallow:
        _log(f"[summarize] abstract-summarizing {len(shallow)} with {cfg.models.get('shallow')}")
        n, errs = summarizer.run(
            pool, shallow, cfg, usage, "shallow", workers=args.workers
        )
        total_errors += errs
        _log(f"[summarize] shallow done: {n} ok, {len(errs)} failed")

    for rec in deep + shallow:
        if rec.get("summary"):
            mark(rec, "summarize")
    save_day(date, records)

    for pid, err in total_errors[:8]:
        _log(f"  ! {pid}: {err}")
    flagged = [
        r["arxiv_id"]
        for r in deep + shallow
        if ((r.get("verify") or {}).get("numbers") or {}).get("unverified")
    ]
    if flagged:
        _log(f"[summarize] papers with unverified numbers: {', '.join(flagged)}")
    _log(f"[summarize] cost ${usage.usd:.4f} ({usage.as_dict()})")
    if total_errors and not any(r.get("summary") for r in deep + shallow):
        _log("[summarize] every request failed; not treating this as success")
        return 1
    write_run_log(
        date,
        {"summarize": {"deep": len(deep), "shallow": len(shallow),
                       "errors": len(total_errors), "flagged": flagged,
                       "usage": usage.as_dict()}},
    )
    return 0


# ------------------------------------------------------------------- refresh-figures
def cmd_refresh_figures(args: argparse.Namespace, cfg: Config) -> int:
    """Re-extract figures onto already-summarized records.

    Figures are metadata, like citations: an extraction fix should not require paying for
    the summaries again. Only records that already have a summary are touched, and the
    model's `figures_worth_seeing` picks are matched by number, so they keep working.
    """
    from .sources import fulltext

    date = resolve_date(args.date)
    records = load_day(date)
    targets = [r for r in records if r.get("summary") and r.get("depth") == "deep"]
    if not targets:
        _log("[figures] no deep-read records for this date")
        return 0
    changed = 0
    for rec in targets:
        ft = fulltext.fetch(rec["arxiv_id"], use_cache=not args.no_cache)
        figures = ft.get("figures") or []
        if figures != rec.get("figures"):
            rec["figures"] = figures
            changed += 1
    save_day(date, records)
    total = sum(len(r.get("figures") or []) for r in targets)
    _log(f"[figures] refreshed {changed}/{len(targets)} records, {total} figures total")
    return 0


# --------------------------------------------------------------------------- relate
def cmd_relate(args: argparse.Namespace, cfg: Config) -> int:
    """Mark each published paper with the closest one from recent issues.

    Context only. Nothing is filtered, reordered or penalised on the strength of this —
    an important paper often arrives in the middle of a wave of similar ones, and having
    covered something adjacent last week is the worst possible reason to hide it.
    """
    date = resolve_date(args.date)
    if not cfg.memory.get("enabled", True):
        _log("[relate] memory disabled in config")
        return 0
    records = load_day(date)
    published = [r for r in records if r.get("summary") and r.get("selected_for")]
    if not published:
        _log("[relate] nothing published for this date")
        return 0

    try:
        vecs, ids = mem.embed_day(date, records)
    except mem.EmbeddingUnavailable as exc:
        _log(f"[relate] skipped: {exc}")
        return 0
    index = {aid: vecs[i] for i, aid in enumerate(ids)}

    lookback = int(cfg.memory.get("similarity_lookback_days", 30))
    floor = float(cfg.memory.get("similarity_floor", 0.72))
    today = dt.date.fromisoformat(date)
    history: list[tuple[str, str, Any]] = []
    titles: dict[str, str] = {}
    for past in available_dates():
        if past >= date:
            continue
        if (today - dt.date.fromisoformat(past)).days > lookback:
            continue
        past_vecs, past_ids = mem.load_day_vectors(past)
        if past_vecs is None:
            continue
        by_id = {a: i for i, a in enumerate(past_ids)}
        for rec in load_day(past):
            if rec.get("summary") and rec.get("selected_for") and rec["arxiv_id"] in by_id:
                history.append((rec["arxiv_id"], past, past_vecs[by_id[rec["arxiv_id"]]]))
                titles[rec["arxiv_id"]] = rec["title"]

    marked = 0
    for rec in published:
        vec = index.get(rec["arxiv_id"])
        if vec is None:
            continue
        near = mem.nearest_seen(vec, history, top_k=2, floor=floor)
        for n in near:
            n["title"] = titles.get(n["arxiv_id"], "")
        rec["related_recent"] = near
        marked += 1 if near else 0
    save_day(date, records)
    _log(f"[relate] {len(published)} published, {len(history)} papers in the last "
         f"{lookback} days, {marked} got a related-work marker")
    write_run_log(date, {"relate": {"history": len(history), "marked": marked}})
    return 0


# ----------------------------------------------------------------------- build-site
def cmd_build_site(args: argparse.Namespace, cfg: Config) -> int:
    from .site.build import build

    dates = [resolve_date(args.date)] if args.date else None
    if args.all:
        dates = None
    site_days = int(cfg.retention.get("site_days", retention.DEFAULT_SITE_DAYS))
    written = build(dates, site_days=site_days)
    _log(f"[site] wrote {len(written)} files; entry point: {SITE_DIR.name}/index.html")
    return 0


# ---------------------------------------------------------------------------- prune
def cmd_prune(args: argparse.Namespace, cfg: Config) -> int:
    """Drop old pages, and only much older data. See `retention` for why they differ."""
    plan = retention.plan(
        available_dates(),
        site_days=int(cfg.retention.get("site_days", retention.DEFAULT_SITE_DAYS)),
        data_days=int(cfg.retention.get("data_days", retention.DEFAULT_DATA_DAYS)),
        lookback_days=int(cfg.memory.get("similarity_lookback_days", 30)),
        labelled_ids={e["arxiv_id"] for e in fb.load() if e.get("arxiv_id")},
        loader=load_day,
    )
    removed = retention.apply(
        plan, site_dir=SITE_DIR, data_dir=DATA_DIR, cache_dir=CACHE_DIR,
        dry_run=args.dry_run,
    )
    verb = "would remove" if args.dry_run else "removed"
    counts = ", ".join(f"{len(v)} {k}" for k, v in removed.items() if v) or "nothing"
    _log(
        f"[prune] keeping {len(plan['site_keep'])} rendered days and "
        f"{len(plan['data_keep'])} days of records "
        f"(data floor {plan['data_days_effective']}d, raised to cover the "
        f"{cfg.memory.get('similarity_lookback_days', 30)}d similarity lookback); "
        f"{verb} {counts}"
    )
    if args.verbose:
        for kind, paths in removed.items():
            for p in paths:
                _log(f"[prune]   {verb}: {p}")
    return 0


# ---------------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace, cfg: Config) -> int:
    """Serve the built site over HTTP for local preview.

    Opening the files directly works, but `file://` is an opaque origin: `localStorage`
    is unavailable, so the language switch flips for the current page and forgets the
    choice on reload. Over HTTP it behaves exactly as it will on GitHub Pages.
    """
    import http.server
    import json as _json
    import socketserver
    import webbrowser

    from .config import SITE_DIR

    if not (SITE_DIR / "index.html").exists():
        _log(f"[serve] {SITE_DIR}/index.html does not exist — run `pna build-site` first")
        return 1

    class Handler(http.server.SimpleHTTPRequestHandler):
        """Static files, plus the one endpoint the feedback buttons post to."""

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(SITE_DIR), **kw)

        def do_POST(self):  # noqa: N802 - stdlib naming
            if self.path.rstrip("/") != "/api/feedback":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                entry = _json.loads(self.rfile.read(length) or b"{}")
                fb.append(entry)
            except Exception as exc:  # noqa: BLE001 - reported to the browser
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json.dumps({"error": str(exc)}).encode())
                return
            self.send_response(204)
            self.end_headers()

        def log_message(self, fmt, *args):
            if "api/feedback" in (args[0] if args else ""):
                _log(f"[serve] feedback: {args[0]}")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/"
        _log(f"[serve] {SITE_DIR} -> {url}  (Ctrl-C to stop)")
        if not args.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            _log("\n[serve] stopped")
    return 0


# -------------------------------------------------------------------------- feedback
def cmd_feedback(args: argparse.Namespace, cfg: Config) -> int:
    """Show the label log, or import labels exported from a browser."""
    if args.import_path:
        rows = []
        for line in Path(args.import_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        added, skipped = fb.merge_import(rows)
        _log(f"[feedback] imported {added}, skipped {skipped} (already present or invalid)")
        return 0

    report = fb.summarise()
    if not report["total"]:
        _log("[feedback] no labels yet — open the digest and use the buttons on each card")
        return 0
    print(f"{report['total']} labelled papers")
    for label, n in report["counts"].items():
        print(f"  {fb.LABEL_ZH[label]:<14} ({label:<14}) {n}")
    print("\nDoes the evidence audit separate these? (descriptive — look, do not test)")
    for label, stats in report["by_label"].items():
        if not report["counts"][label]:
            continue
        print(f"  {fb.LABEL_ZH[label]}:")
        print(f"    relevance mean {stats['mean_relevance']}   priority mean {stats['mean_priority']}")
        print(f"    evidence_grade {stats['evidence_grade']}   eval_risk {stats['evaluation_risk']}")
    print(
        f"\nselection.evidence_influence is currently "
        f"{cfg.selection.get('evidence_influence', 'deep_order')!r}. Raise it only if "
        f"'值得深读' separates from the others on the rows above."
    )
    return 0


# ------------------------------------------------------------------------------ run
def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    for step in (cmd_ingest, cmd_filter, cmd_triage, cmd_review, cmd_summarize,
                 cmd_relate, cmd_build_site, cmd_prune):
        code = step(args, cfg)
        if code != 0:
            return code
    return 0


# ---------------------------------------------------------------------------- stats
def cmd_stats(args: argparse.Namespace, cfg: Config) -> int:
    dates = [resolve_date(args.date)] if args.date else available_dates()
    for date in dates:
        records = load_day(date)
        if not records:
            continue
        kw = sum(1 for r in records if keyword.passes(r.get("keyword") or {}, cfg))
        triaged = [r for r in records if isinstance((r.get("triage") or {}).get("score"), int)]
        summarized = [r for r in records if r.get("summary")]
        deep = sum(1 for r in summarized if r.get("depth") == "deep")
        print(
            f"{date}  ingested={len(records):4d}  tier1={kw:3d}  triaged={len(triaged):3d}"
            f"  summarized={len(summarized):2d} (deep={deep})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pna", description=__doc__)
    parser.add_argument("--config", default=None, help="path to interests.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Any, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--date", default=None, help="YYYY-MM-DD, or 'auto' (last weekday)")
        p.set_defaults(fn=fn)
        return p

    p = add("ingest", cmd_ingest, "harvest arXiv metadata via OAI-PMH")
    p.add_argument("--date-from", default=None, help="start of datestamp range")
    p.add_argument("--include-revisions", action="store_true",
                   help="keep metadata-only updates to older papers")

    p = add("filter", cmd_filter, "tier-1 category + keyword filter")
    p.add_argument("--force", action="store_true")

    p = add("triage", cmd_triage, "tier-3 LLM relevance scoring")
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=6)

    p = add("review", cmd_review, "reviewer-lite evidence audit (no verdict, no score)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=90_000)

    p = add("summarize", cmd_summarize, "deep/shallow structured summaries")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cap papers per depth (testing)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=90_000)

    p = add("refresh-figures", cmd_refresh_figures,
            "re-extract paper figures onto existing records (no LLM cost)")
    p.add_argument("--no-cache", action="store_true", help="bypass the fulltext cache")

    add("relate", cmd_relate, "mark papers close to recent issues (context, not a filter)")

    p = add("build-site", cmd_build_site, "render the static site")
    p.add_argument("--all", action="store_true", help="rebuild every day, not just --date")

    p = add("prune", cmd_prune, "drop old page snapshots (and much older records)")
    p.add_argument("--dry-run", action="store_true", help="show what would go, delete nothing")
    p.add_argument("--verbose", action="store_true", help="list every path")

    p = add("run", cmd_run, "ingest -> filter -> triage -> summarize -> build-site -> prune")
    p.add_argument("--force", action="store_true")
    p.add_argument("--date-from", default=None)
    p.add_argument("--include-revisions", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=90_000)
    p.add_argument("--all", action="store_true")
    # `run` ends in prune, so it must carry prune's flags.
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")

    p = add("feedback", cmd_feedback, "show or import the human label log")
    p.add_argument("--import", dest="import_path", default=None,
                   help="a feedback.jsonl exported from the browser")

    p = add("serve", cmd_serve, "preview the built site over HTTP")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action="store_true", help="do not launch a browser")

    add("stats", cmd_stats, "per-day funnel counts")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return args.fn(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())

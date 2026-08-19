"""Per-day JSONL storage.

`data/papers/YYYY-MM-DD.jsonl` is the source of truth: one line per paper, holding the
output of every stage that has run on it. Line-oriented JSON so that git diffs stay
readable and a bad run can be inspected and re-run rather than archaeologically recovered.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .config import PAPERS_DIR, RUNS_DIR

Record = dict[str, Any]


def day_path(date: str) -> Path:
    return PAPERS_DIR / f"{date}.jsonl"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def mark(rec: Record, stage: str) -> Record:
    rec.setdefault("stages", {})[stage] = now()
    return rec


def has_stage(rec: Record, stage: str) -> bool:
    return stage in rec.get("stages", {})


def load_day(date: str) -> list[Record]:
    path = day_path(date)
    if not path.exists():
        return []
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return records


def save_day(date: str, records: Iterable[Record]) -> Path:
    """Atomic write: build a temp file in the target dir, then replace.

    A crash mid-write must not leave a half-truncated day file, because that file is the
    only copy of the day's LLM output.
    """
    path = day_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=_sort_key)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in ordered:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def _sort_key(rec: Record) -> tuple[float, str]:
    triage = (rec.get("triage") or {}).get("score")
    kw = (rec.get("keyword") or {}).get("score", 0.0)
    return (-(triage if triage is not None else -1), -kw, rec.get("arxiv_id", ""))


def merge_day(date: str, incoming: Iterable[Record]) -> tuple[int, int]:
    """Merge records into a day file by arxiv_id. Returns (added, updated)."""
    existing = {r["arxiv_id"]: r for r in load_day(date)}
    added = updated = 0
    for rec in incoming:
        key = rec["arxiv_id"]
        if key in existing:
            existing[key].update(rec)
            updated += 1
        else:
            existing[key] = rec
            added += 1
    save_day(date, existing.values())
    return added, updated


def available_dates() -> list[str]:
    if not PAPERS_DIR.exists():
        return []
    return sorted(p.stem for p in PAPERS_DIR.glob("*.jsonl"))


def write_run_log(date: str, payload: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{date}.json"
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(payload)
    existing["written_at"] = now()
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

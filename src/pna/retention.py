"""What gets deleted, and what must never be.

Rendered pages are disposable — they are a pure function of `data/papers/<date>.jsonl`
and can be rebuilt for free. Records and vectors are not: they are the only copy of a
harvest that cannot be repeated (arXiv's OAI window moves on), and three things read them
long after their page is gone.

So the two live on separate clocks. Trimming the site to ten days is what you see;
trimming the data underneath it would silently break:

* `relate`, which compares against `similarity_lookback_days` (30) of past issues;
* the preference memory, which looks up the vector of every hand-labelled paper — delete
  the day a label points at and the label stops contributing to recall, with no error;
* any re-run, re-score, or model comparison over past days.

`protected_dates` encodes the second one as a hard rule rather than a suggestion: a day
holding a paper you labelled is never deleted, however old it is.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_SITE_DAYS = 10
DEFAULT_DATA_DAYS = 60


def keep_newest(dates: Sequence[str], days: int) -> list[str]:
    """The `days` most recent dates. `days <= 0` means keep everything."""
    if days <= 0:
        return list(dates)
    return sorted(dates)[-days:]


def protected_dates(dates: Iterable[str], labelled_ids: set[str], loader) -> set[str]:
    """Days that hold at least one hand-labelled paper.

    The preference memory resolves a label to a vector by searching past days. A label
    whose day has been deleted degrades in silence — the paper simply stops pulling
    anything in — which is the worst failure mode available, so it is prevented here.
    """
    if not labelled_ids:
        return set()
    keep = set()
    for date in dates:
        for rec in loader(date):
            if rec.get("arxiv_id") in labelled_ids:
                keep.add(date)
                break
    return keep


def plan(
    all_dates: Sequence[str],
    *,
    site_days: int,
    data_days: int,
    lookback_days: int,
    labelled_ids: set[str],
    loader,
) -> dict[str, list[str]]:
    """Decide what to remove. Pure — returns the plan, deletes nothing.

    `data_days` is raised to `lookback_days` if it was set lower: a configuration that
    deletes the history `relate` is about to read is a mistake, not a preference.
    """
    dates = sorted(all_dates)
    site_keep = set(keep_newest(dates, site_days))
    effective_data_days = max(data_days, lookback_days) if data_days > 0 else 0
    data_keep = set(keep_newest(dates, effective_data_days))
    data_keep |= protected_dates(dates, labelled_ids, loader)
    return {
        "site_keep": sorted(site_keep),
        "site_drop": [d for d in dates if d not in site_keep],
        "data_keep": sorted(data_keep),
        "data_drop": [d for d in dates if d not in data_keep],
        "data_days_effective": effective_data_days,
    }


def apply(plan_: dict, *, site_dir: Path, data_dir: Path, cache_dir: Path,
          dry_run: bool = False) -> dict[str, list[str]]:
    """Carry out a plan. Returns what was (or would be) removed, by kind."""
    removed: dict[str, list[str]] = {"pages": [], "records": [], "vectors": [], "runs": []}
    for date in plan_["site_drop"]:
        page = site_dir / date
        if page.is_dir():
            removed["pages"].append(str(page))
            if not dry_run:
                shutil.rmtree(page)
    for date in plan_["data_drop"]:
        for path, kind in (
            (data_dir / "papers" / f"{date}.jsonl", "records"),
            (data_dir / "runs" / f"{date}.json", "runs"),
            (cache_dir / "embeddings" / f"{date}.npz", "vectors"),
        ):
            if path.exists():
                removed[kind].append(str(path))
                if not dry_run:
                    path.unlink()
    return removed

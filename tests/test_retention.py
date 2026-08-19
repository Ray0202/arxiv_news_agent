"""What must survive a prune. The interesting cases are all about *not* deleting."""

from pna import retention


DATES = [f"2026-07-{d:02d}" for d in range(1, 31)]  # 30 consecutive days


def _loader(_date):
    return []


def _plan(**kw):
    base = dict(site_days=10, data_days=60, lookback_days=30, labelled_ids=set(),
                loader=_loader)
    return retention.plan(DATES, **{**base, **kw})


def test_pages_are_trimmed_to_the_window_but_records_are_not():
    p = _plan()
    assert len(p["site_keep"]) == 10
    assert p["site_keep"][-1] == "2026-07-30"
    assert p["data_drop"] == [], "60-day data retention must not touch 30 days of records"


def test_data_retention_is_raised_to_cover_the_similarity_lookback():
    """A config that deletes the history `relate` is about to read is a bug, not a choice.

    Deleting it would not raise anything — `relate` would just quietly find fewer
    neighbours — so the floor is enforced rather than documented.
    """
    p = _plan(data_days=5, lookback_days=30)
    assert p["data_days_effective"] == 30
    assert len(p["data_keep"]) == 30


def test_a_day_holding_a_hand_labelled_paper_is_never_deleted():
    """The preference memory resolves a label to a vector by searching past days.

    Delete that day and the label silently stops contributing to recall — no error, just
    a slowly narrowing rescue channel. So the protection is absolute, not time-bounded.
    """
    def loader(date):
        return [{"arxiv_id": "old-fav"}] if date == "2026-07-01" else []

    p = retention.plan(DATES, site_days=3, data_days=5, lookback_days=5,
                       labelled_ids={"old-fav"}, loader=loader)
    assert "2026-07-01" in p["data_keep"]
    assert "2026-07-01" not in p["data_drop"]
    assert "2026-07-02" in p["data_drop"], "unlabelled days of the same age still go"


def test_zero_means_keep_everything():
    p = _plan(site_days=0, data_days=0)
    assert p["site_drop"] == [] and p["data_drop"] == []
    assert p["data_days_effective"] == 0


def test_apply_dry_run_reports_without_deleting(tmp_path):
    site = tmp_path / "site"
    (site / "2026-07-01").mkdir(parents=True)
    papers = tmp_path / "data" / "papers"
    papers.mkdir(parents=True)
    (papers / "2026-07-01.jsonl").write_text("{}")

    plan = {"site_drop": ["2026-07-01"], "data_drop": ["2026-07-01"]}
    removed = retention.apply(plan, site_dir=site, data_dir=tmp_path / "data",
                              cache_dir=tmp_path / "cache", dry_run=True)
    assert removed["pages"] and removed["records"]
    assert (site / "2026-07-01").exists()
    assert (papers / "2026-07-01.jsonl").exists()

    retention.apply(plan, site_dir=site, data_dir=tmp_path / "data",
                    cache_dir=tmp_path / "cache")
    assert not (site / "2026-07-01").exists()
    assert not (papers / "2026-07-01.jsonl").exists()

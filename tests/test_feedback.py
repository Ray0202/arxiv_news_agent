"""The human-label log that decides whether the evidence audit earns a say in selection."""

from __future__ import annotations

import json

import pytest

from pna import feedback


@pytest.fixture(autouse=True)
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "PATH", tmp_path / "feedback.jsonl")
    return feedback.PATH


def _entry(arxiv_id="2607.00001", label="deep", at="2026-08-18T10:00:00Z", **signals):
    base = {"relevance": 9, "evidence_grade": "B", "evaluation_risk": "medium",
            "priority": 0.85, "reviewed": True}
    base.update(signals)
    return {"arxiv_id": arxiv_id, "date": "2026-07-30", "label": label, "at": at,
            "signals": base}


def test_only_the_three_defined_labels_are_accepted():
    feedback.append(_entry(label="deep"))
    feedback.append(_entry(arxiv_id="b", label="weak_evidence"))
    feedback.append(_entry(arxiv_id="c", label="skip"))
    with pytest.raises(ValueError, match="label must be one of"):
        feedback.append(_entry(label="accept"))
    with pytest.raises(ValueError, match="arxiv_id is required"):
        feedback.append({"label": "deep"})


def test_the_log_is_append_only_and_the_last_label_wins():
    """Changing your mind is a new line, not an edit — the history stays auditable."""
    feedback.append(_entry(label="deep", at="2026-08-18T10:00:00Z"))
    feedback.append(_entry(label="skip", at="2026-08-18T11:00:00Z"))
    assert len(feedback.load()) == 2
    latest = feedback.latest_per_paper()
    assert len(latest) == 1
    assert latest["2607.00001"]["label"] == "skip"


def test_signals_are_stored_with_the_label():
    """Storing only {id, label} would make the two-week comparison impossible: a re-run
    overwrites the very numbers the judgement was made against."""
    rec = {
        "triage": {"score": 9, "novelty": "notable", "read_depth": "deep"},
        "review": {"evidence_grade": "A", "method_risk": "low",
                   "evaluation_risk": "high", "quality_confidence": 0.9,
                   "paper_type": "empirical_method"},
        "fulltext": {"source": "html"},
        "depth": "deep",
    }
    sig = feedback.signals_for(rec, priority=0.7123)
    assert sig["relevance"] == 9
    assert sig["evidence_grade"] == "A"
    assert sig["evaluation_risk"] == "high"
    assert sig["reviewed"] is True
    assert sig["priority"] == 0.7123


def test_signals_for_an_unreviewed_paper_are_marked_not_guessed():
    sig = feedback.signals_for({"triage": {"score": 8}}, priority=0.64)
    assert sig["reviewed"] is False
    assert sig["evidence_grade"] is None


def test_import_is_idempotent():
    rows = [_entry(at="2026-08-18T10:00:00Z"), _entry(arxiv_id="b", at="2026-08-18T10:01:00Z")]
    added, skipped = feedback.merge_import(rows)
    assert (added, skipped) == (2, 0)
    added, skipped = feedback.merge_import(rows)
    assert (added, skipped) == (0, 2), "re-importing the same export must not duplicate"


def test_import_rejects_invalid_rows_without_aborting_the_batch():
    rows = [_entry(), {"arxiv_id": "x", "label": "nonsense"}, _entry(arxiv_id="c")]
    added, skipped = feedback.merge_import(rows)
    assert (added, skipped) == (2, 1)


def test_summary_is_a_contingency_table_not_a_verdict():
    feedback.append(_entry(arxiv_id="a", label="deep", evidence_grade="A", priority=0.9))
    feedback.append(_entry(arxiv_id="b", label="deep", evidence_grade="A", priority=0.88))
    feedback.append(_entry(arxiv_id="c", label="skip", evidence_grade="C", priority=0.5))
    out = feedback.summarise()
    assert out["total"] == 3
    assert out["counts"] == {"deep": 2, "weak_evidence": 0, "skip": 1}
    assert out["by_label"]["deep"]["evidence_grade"] == {"A": 2}
    assert out["by_label"]["skip"]["evidence_grade"] == {"C": 1}
    assert out["by_label"]["deep"]["mean_priority"] == 0.89
    # No pass/fail, no significance claim — the decision is made by looking.
    assert "recommendation" not in out and "significant" not in out

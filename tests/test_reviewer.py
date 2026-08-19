"""Reviewer-lite: evidence anchoring, validation, and ranking.

The stage's whole value is that a finding it cannot locate in the document does not get
published. These tests are mostly about that boundary.
"""

from __future__ import annotations

import pytest

from pna.config import Config, Topic
from pna.reviewer import SCHEMA, validate
from pna.summarize.runner import priority

ANCHORS = {
    "S2.p1": {"kind": "paragraph", "section": "2 Method"},
    "S3.T2": {"kind": "table", "section": "3 Experiments"},
    "S3.p4": {"kind": "paragraph", "section": "3 Experiments"},
}


def _cfg(**selection) -> Config:
    base = {"pure_relevance_slots": 2,
            "weights": {"relevance": 0.8, "evidence": 0.2, "high_risk_penalty": 0.5}}
    base.update(selection)
    return Config(categories=[], topics=[Topic(name="t")], thresholds={}, budget={},
                  output={}, models={}, ingest={}, raw={}, selection=base)


def _audit(**kw):
    base = {
        "paper_type": "empirical_method",
        "claims": [], "risks": [], "author_limitations": [], "reader_limitations": [],
        "unknowns": [], "evidence_grade": "B", "method_risk": "low",
        "evaluation_risk": "low", "quality_confidence": 0.8,
    }
    base.update(kw)
    return base


# ------------------------------------------------------------- evidence validation
def test_a_risk_with_a_fabricated_anchor_is_demoted_not_published():
    audit = _audit(risks=[
        {"text": "基线未同等调参", "text_en": "baselines were not tuned equally",
         "kind": "baseline", "evidence_ids": ["S9.T7"]},
    ])
    report = validate(audit, ANCHORS)
    assert audit["risks"] == [], "an unlocatable risk is an assertion, not a finding"
    assert [u["text"] for u in audit["unknowns"]] == ["基线未同等调参"]
    assert report["invalid_ids"] == ["S9.T7"]
    assert report["demoted_to_unknown"] == 1


def test_partially_valid_anchors_keep_the_finding_and_drop_the_bad_ids():
    audit = _audit(reader_limitations=[
        {"text": "只在两个数据集验证", "text_en": "only two datasets",
         "evidence_ids": ["S3.T2", "S9.T7"]},
    ])
    report = validate(audit, ANCHORS)
    assert len(audit["reader_limitations"]) == 1
    assert audit["reader_limitations"][0]["evidence_ids"] == ["S3.T2"]
    assert report["invalid_ids"] == ["S9.T7"]
    assert audit["unknowns"] == []


def test_an_unanchored_claim_is_kept_but_its_support_is_downgraded():
    """The claim is still what the paper says; the *support level* is what we cannot trust."""
    audit = _audit(claims=[
        {"claim": "方法优于所有基线", "claim_en": "beats all baselines",
         "evidence_ids": ["S9.T9"], "support": "direct", "why": "…", "why_en": "…"},
    ])
    validate(audit, ANCHORS)
    claim = audit["claims"][0]
    assert claim["support"] == "absent"
    assert claim["unanchored"] is True
    assert claim["evidence_ids"] == []


def test_not_applicable_support_survives_without_anchors():
    """A definitional claim has no empirical evidence to cite, and that is correct."""
    audit = _audit(claims=[
        {"claim": "本文把工具检索形式化为超边预测", "claim_en": "framed as hyperedge prediction",
         "evidence_ids": [], "support": "not_applicable", "why": "…", "why_en": "…"},
    ])
    validate(audit, ANCHORS)
    assert audit["claims"][0]["support"] == "not_applicable"
    assert "unanchored" not in audit["claims"][0]


def test_validation_counts_what_it_checked():
    audit = _audit(
        claims=[{"claim": "c", "claim_en": "c", "evidence_ids": ["S2.p1"],
                 "support": "direct", "why": "w", "why_en": "w"}],
        risks=[{"text": "r", "text_en": "r", "kind": "baseline",
                "evidence_ids": ["S3.T2", "S3.p4"]}],
    )
    report = validate(audit, ANCHORS)
    assert report["checked_ids"] == 3
    assert report["anchors_available"] == 3
    assert report["invalid_ids"] == []


# --------------------------------------------------------------------- ranking
def test_priority_is_dominated_by_relevance():
    cfg = _cfg()
    strong_evidence_low_relevance = {"triage": {"score": 6},
                                     "review": _audit(evidence_grade="A", quality_confidence=1.0)}
    weak_evidence_high_relevance = {"triage": {"score": 9},
                                    "review": _audit(evidence_grade="D", quality_confidence=1.0)}
    assert priority(weak_evidence_high_relevance, cfg) > priority(strong_evidence_low_relevance, cfg)


def test_an_unreviewed_paper_is_not_penalised():
    cfg = _cfg()
    same_score = {"triage": {"score": 8}}
    assert priority(same_score, cfg) == pytest.approx(0.8 * 0.8)


def test_low_confidence_shrinks_the_reviewer_s_influence_both_ways():
    cfg = _cfg()
    base = 0.8 * 0.9
    confident = priority({"triage": {"score": 9},
                          "review": _audit(evidence_grade="A", quality_confidence=1.0)}, cfg)
    unsure = priority({"triage": {"score": 9},
                       "review": _audit(evidence_grade="A", quality_confidence=0.1)}, cfg)
    assert confident > unsure > base - 1e-9
    # A high-risk evaluation pulls down, but only in proportion to confidence.
    risky = priority({"triage": {"score": 9},
                      "review": _audit(evaluation_risk="high", quality_confidence=1.0)}, cfg)
    risky_unsure = priority({"triage": {"score": 9},
                             "review": _audit(evaluation_risk="high", quality_confidence=0.1)}, cfg)
    assert risky < risky_unsure < base


# ------------------------------------------------------------------------ schema
def test_schema_is_structured_output_compatible():
    def check(node, path="$"):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path
            assert set(node.get("required", [])) == set(node.get("properties", {})), path
            for k, v in node.get("properties", {}).items():
                check(v, f"{path}.{k}")
        elif node.get("type") == "array":
            check(node.get("items", {}), f"{path}[]")

    check(SCHEMA)


def test_schema_has_no_overall_verdict_field():
    """The design forbids an accept/reject or aggregate quality number."""
    banned = {"accept", "reject", "recommendation", "verdict", "overall", "quality",
              "score", "rating"}
    assert not (banned & set(SCHEMA["properties"]))
    # quality_confidence is about the audit, and its description must say so.
    assert "not in the paper" in SCHEMA["properties"]["quality_confidence"]["description"]


# --------------------------------------------------------------- anchor repair
def test_an_over_prefixed_anchor_is_repaired_not_rejected():
    """Observed: the model wrote `S2.Thmdefinition1.p1` for the real id
    `Thmdefinition1.p1`, prefixing the section the way ordinary paragraph ids are built.
    The block is real, so the citation is meaningful."""
    anchors = {"Thmdefinition1.p1": {"kind": "paragraph", "section": "Definition 1"},
               "S2.p1": {"kind": "paragraph", "section": "2 Method"}}
    audit = _audit(risks=[
        {"text": "定义只覆盖有限情形", "text_en": "the definition is narrow",
         "kind": "construction", "evidence_ids": ["S2.Thmdefinition1.p1"]},
    ])
    report = validate(audit, anchors)
    assert audit["risks"][0]["evidence_ids"] == ["Thmdefinition1.p1"]
    assert report["invalid_ids"] == []
    assert report["repaired_ids"] == ["S2.Thmdefinition1.p1->Thmdefinition1.p1"]


def test_a_fabricated_anchor_is_still_rejected_after_repair_is_available():
    """`S5.SS3.p2` where the paper has `S5.SS3` but no `.p2` — no suffix candidate."""
    anchors = {"S5.SS3": {"kind": "section", "section": "5.3"},
               "S5.SS3.p1": {"kind": "paragraph", "section": "5.3"}}
    audit = _audit(risks=[
        {"text": "编造的", "text_en": "fabricated", "kind": "other",
         "evidence_ids": ["S5.SS3.p2"]},
    ])
    report = validate(audit, anchors)
    assert audit["risks"] == []
    assert report["invalid_ids"] == ["S5.SS3.p2"]
    assert report["repaired_ids"] == []


def test_repair_prefers_the_longest_match():
    from pna.reviewer import _repair

    valid = {"p1", "Thmdefinition1.p1"}
    assert _repair("S2.Thmdefinition1.p1", valid) == "Thmdefinition1.p1"
    assert _repair("S2.p1", valid) == "p1"
    assert _repair("totally.made.up", valid) is None


def test_repair_does_not_match_across_a_partial_segment():
    from pna.reviewer import _repair

    # `S5.SS30.p1` must not resolve to `S5.SS3.p1` — the dot boundary is required.
    assert _repair("S5.SS30.p1", {"S5.SS3.p1"}) is None

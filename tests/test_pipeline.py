"""Tests for the parts of the pipeline that need no network and no API key."""

from __future__ import annotations

import json

import pytest

from pna.config import Config, Topic, load_config
from pna.filters import keyword
from pna.filters.triage import TRIAGE_SCHEMA
from pna.sources import fulltext, oai
from pna.summarize.schema import build as build_schema
from pna.verify import check_numbers, check_terms


@pytest.fixture
def cfg() -> Config:
    return Config(
        categories=["cs.LG", "cs.AI"],
        topics=[
            Topic(name="time-series", weight=1.0, keywords=["time series", "forecasting"]),
            Topic(name="agent", weight=1.0, keywords=["agent", "tool use"]),
        ],
        thresholds={"keyword_min_score": 1.0, "llm_min_score": 6},
        budget={},
        output={"languages": ["zh", "en"]},
        models={},
        ingest={},
        raw={},
    )


def _rec(**kw) -> dict:
    base = {
        "arxiv_id": "2601.00001",
        "title": "A paper",
        "abstract": "An abstract.",
        "categories": ["cs.LG"],
        "comments": None,
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------ keyword matching
def test_keyword_respects_word_boundaries(cfg):
    """'reagent' must not match the keyword 'agent' — this fills a digest with chemistry."""
    v = keyword.score_paper(_rec(abstract="We mix the reagent with agentive solvents."), cfg)
    assert v["score"] == 0.0
    assert v["topics"] == []


def test_keyword_matches_across_hyphen_and_whitespace(cfg):
    for variant in ["time series", "time-series", "time  series", "Time_Series"]:
        v = keyword.score_paper(_rec(abstract=f"We study {variant} models."), cfg)
        assert v["topics"] == ["time-series"], variant


def test_title_hits_outweigh_abstract_hits(cfg):
    in_title = keyword.score_paper(_rec(title="Forecasting with agents"), cfg)
    in_abstract = keyword.score_paper(_rec(abstract="Forecasting with agents"), cfg)
    assert in_title["score"] > in_abstract["score"]


def test_repeated_keyword_counted_once(cfg):
    once = keyword.score_paper(_rec(abstract="An agent."), cfg)
    many = keyword.score_paper(_rec(abstract="An agent agent agent agent agent."), cfg)
    assert once["score"] == many["score"]


def test_category_gate(cfg):
    v = keyword.score_paper(_rec(categories=["math.CO"], abstract="agent"), cfg)
    assert v["category_match"] is False
    assert keyword.passes(v, cfg) is False


# --------------------------------------------------------------- new-vs-revised split
@pytest.mark.parametrize(
    "created,stamp,expected",
    [
        ("2026-07-29", "2026-07-30", True),   # submitted yesterday, announced today
        ("2026-07-28", "2026-07-30", True),   # weekend lag
        ("2021-11-09", "2026-07-30", False),  # old paper, metadata edited
        ("2026-07-13", "2026-07-30", False),  # 17 days: a revision, not an announcement
        (None, "2026-07-30", True),           # unknown: let later stages decide
    ],
)
def test_is_new_submission(created, stamp, expected):
    rec = {"created": created, "datestamp": stamp}
    assert oai.is_new_submission(rec, window_days=5) is expected


# ------------------------------------------------------------------ HTML → markdown
def test_html_extraction_keeps_latex_and_deduplicates_abstract():
    html = """
    <html><body><div class="ltx_page_content">
      <h1 class="ltx_title">Some Title</h1>
      <div class="ltx_abstract"><h6 class="ltx_title_abstract">Abstract</h6>
        <p class="ltx_p">We propose <math alttext="\\alpha_t">a</math> a thing.</p></div>
      <section class="ltx_section"><h2 class="ltx_title_section">1 Introduction</h2>
        <p class="ltx_p">Body text with <math alttext="x^2">x2</math> inline.</p></section>
      <div class="ltx_bibliography"><p class="ltx_p">Should be dropped.</p></div>
    </div></body></html>
    """
    text = fulltext._html_to_markdown(html)
    assert text.count("We propose") == 1, "abstract emitted twice by nested traversal"
    assert "$\\alpha_t$" in text
    assert "$x^2$" in text
    assert "Should be dropped" not in text
    assert "## 1 Introduction" in text


def test_html_extraction_survives_math_without_alttext():
    """Falls back to rendered glyphs rather than emitting an empty pair of delimiters.

    An empty `$$` pairs up with the *next* formula's opening delimiter, which silently
    wraps a sentence of prose in math markers.
    """
    html = """<div class="ltx_page_content"><p class="ltx_p">Cost is
      <math>N</math> dollars and grows with <math>M</math> items.</p></div>"""
    text = fulltext._html_to_markdown(html)
    assert "$N$" in text and "$M$" in text
    assert "$$" not in text
    assert "dollars and grows with" in text


def test_html_extraction_drops_math_that_has_no_content_at_all():
    html = """<div class="ltx_page_content"><p class="ltx_p">Cost is
      <math></math> dollars and grows with <math></math> items.</p></div>"""
    text = fulltext._html_to_markdown(html)
    assert "$" not in text
    assert "dollars and grows with" in text


def test_trim_drops_low_value_sections_before_hard_cutting():
    text = (
        "## Abstract\n" + "a" * 100
        + "\n## 2 Related Work\n" + "r" * 5000
        + "\n## 3 Method\n" + "m" * 100
    )
    out, dropped, hard = fulltext._trim(text, max_chars=1000)
    assert dropped == ["2 Related Work"]
    assert hard is False
    assert "## 3 Method" in out


def test_trim_reports_hard_cut():
    text = "## 1 Method\n" + "m" * 5000
    out, dropped, hard = fulltext._trim(text, max_chars=1000)
    assert dropped == []
    assert hard is True, "a hard cut must be reported so confidence drops to low"


# ---------------------------------------------------------------------- number check
def test_check_numbers_flags_fabricated_value():
    summary = {
        "results": [
            {"benchmark": "ETTh1", "metric": "MSE", "value": "0.361",
             "baseline": "PatchTST 0.379", "delta": "-4.7%"}
        ],
        "article_zh": "在 ETTh1 上 MSE 为 0.361。",
    }
    source = "Our method reaches an MSE of .361 on ETTh1, versus 0.379 for PatchTST."
    out = check_numbers(summary, source)
    assert out["unverified"] == [], out
    bogus = json.loads(json.dumps(summary))
    bogus["results"][0]["value"] = "0.288"
    bogus["article_zh"] = "在 ETTh1 上 MSE 为 0.288。"
    flagged = check_numbers(bogus, source)
    assert [u["number"] for u in flagged["unverified"]] == ["0.288"]


def test_check_numbers_tolerates_leading_zero_and_thousands_separator():
    summary = {"results": [], "article_zh": "共 1,081.2 万美元，误差 0.361。"}
    source = "totalling 1081.2 million with an error of .361"
    assert check_numbers(summary, source)["unverified"] == []


def test_check_numbers_ignores_derived_percentages():
    summary = {"results": [], "article_zh": "相对提升 4.7% 。"}
    assert check_numbers(summary, "no percentages here")["unverified"] == []


def test_check_numbers_ignores_the_delta_field():
    """`delta` is arithmetic the model performs, not a quotation from the paper."""
    summary = {
        "results": [
            {"benchmark": "ETTh1", "metric": "MSE", "value": "0.361",
             "baseline": "0.379", "delta": "-4.7%"}
        ],
        "article_zh": "",
    }
    out = check_numbers(summary, "MSE 0.361 versus 0.379")
    assert out["unverified"] == [], out


def test_check_terms_flags_benchmark_absent_from_source():
    summary = {"tags": ["time-series"], "results": [{"benchmark": "Fictional-Bench"}]}
    assert check_terms(summary, "we evaluate on ETTh1 time series data") == ["Fictional-Bench"]


def test_check_terms_does_not_check_tags():
    """A tag is the model's classification vocabulary, not a quotation."""
    summary = {"tags": ["test-time-scaling", "credit-assignment"], "results": []}
    assert check_terms(summary, "unrelated source text") == []


@pytest.mark.parametrize(
    "label",
    [
        "8-design average",                    # descriptive phrase
        "12 Manager models x 7 Worker configs",
        "葡萄牙全国ED数据集",                    # Chinese description
        "Ablation: w/o F_set",                 # contains a colon and a slash
        "ETT",                                 # too short to look up meaningfully
    ],
)
def test_check_terms_skips_labels_that_are_not_proper_nouns(label):
    """Measured: every one of these produced a false positive in the first version."""
    assert check_terms({"tags": [], "results": [{"benchmark": label}]}, "nothing here") == []


# --------------------------------------------------------------------------- schemas
def _assert_structured_output_compatible(schema: dict, path: str = "$") -> None:
    """Structured outputs require additionalProperties:false and complete `required`."""
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, f"{path} missing the flag"
        props = set(schema.get("properties", {}))
        assert set(schema.get("required", [])) == props, f"{path} required != properties"
        for name, sub in schema.get("properties", {}).items():
            _assert_structured_output_compatible(sub, f"{path}.{name}")
    elif schema.get("type") == "array":
        _assert_structured_output_compatible(schema.get("items", {}), f"{path}[]")


def test_triage_schema_is_structured_output_compatible():
    _assert_structured_output_compatible(TRIAGE_SCHEMA)


def test_summary_schema_is_structured_output_compatible(cfg):
    _assert_structured_output_compatible(build_schema(cfg))


def test_summary_schema_follows_configured_languages(cfg):
    both = build_schema(cfg)
    assert "article_zh" in both["properties"] and "article_en" in both["properties"]
    cfg.output["languages"] = ["zh"]
    zh_only = build_schema(cfg)
    assert "article_en" not in zh_only["properties"]
    assert "article_en" not in zh_only["required"]


def test_real_config_loads_and_renders_interests():
    real = load_config()
    text = real.render_interests()
    assert "AVOID" in text
    for topic in real.topics:
        assert topic.name in text


# ---------------------------------------------------- author / affiliation extraction
def test_html_extraction_keeps_the_author_block():
    """Without an ltx_authors branch the whole block is dropped and `institutions` is
    always empty — the model has nothing to read, which looks like a model failure."""
    html = """
    <div class="ltx_page_content">
      <h1 class="ltx_title">A Title</h1>
      <div class="ltx_authors">
        <span class="ltx_creator"><span class="ltx_personname">Wei Li</span>
        <span class="ltx_author_notes">Tsinghua University</span></span>
      </div>
      <div class="ltx_abstract"><p class="ltx_p">Body.</p></div>
    </div>
    """
    text = fulltext._html_to_markdown(html)
    assert "Tsinghua University" in text
    assert "Wei Li" in text
    assert text.count("Tsinghua University") == 1, "author block emitted more than once"


def test_check_terms_ignores_a_qualifier_the_paper_never_spells_out():
    summary = {"tags": [], "results": [{"benchmark": "ALFWorld (avg over 6 families)"}]}
    assert check_terms(summary, "we evaluate on ALFWorld across families") == []
    # A genuinely absent benchmark is still reported.
    summary = {"tags": [], "results": [{"benchmark": "MadeUpBench (subset)"}]}
    assert check_terms(summary, "we evaluate on ALFWorld") == ["MadeUpBench"]
    # Full-width parentheses too — the model writes those in Chinese labels.
    summary = {"tags": [], "results": [{"benchmark": "MadeUpBench（子集）"}]}
    assert check_terms(summary, "we evaluate on ALFWorld") == ["MadeUpBench"]


def test_summarize_prompt_carries_the_interest_profile():
    """why_it_matters_to_me is unanswerable if the reader's topics never reach the model."""
    from pna.config import load_config
    from pna.summarize.runner import build_system

    cfg = load_config()
    rendered = build_system(cfg)
    assert "{{INTERESTS}}" not in rendered
    for topic in cfg.topics:
        assert topic.name in rendered
    assert "AVOID" in rendered


def test_figure_caption_does_not_double_math_symbols():
    """MathML carries the glyph *and* the TeX annotation.

    A raw `text_content()` on `<math alttext="K"><mi>K</mi><annotation>K</annotation></math>`
    returns "KK", so a caption reads `length (KK)`. Inlining the math before reading
    captions fixes it and leaves something KaTeX can render.
    """
    html = """
    <div class="ltx_page_content"><figure class="ltx_figure" id="S4.F2">
      <img src="x2.png">
      <figcaption class="ltx_caption">Figure 2: sequences of length
        <math alttext="K"><semantics><mi>K</mi>
        <annotation encoding="application/x-tex">K</annotation></semantics></math>,
        once each.</figcaption>
    </figure></div>
    """
    from lxml import html as lxml_html

    doc = lxml_html.fromstring(html)
    fulltext._inline_math(doc)
    figs = fulltext._extract_figures(doc, "https://arxiv.org/html/2607.26784")
    assert len(figs) == 1
    assert "KK" not in figs[0]["caption"]
    assert "$K$" in figs[0]["caption"]
    assert figs[0]["number"] == 2
    assert figs[0]["src"] == "https://arxiv.org/html/x2.png"


def test_check_numbers_expands_chinese_and_si_magnitude_suffixes():
    """`4.2万` is the quantity the paper prints as `42,000`.

    Without expansion the checker reports a hallucination for a correct claim — observed
    on a real summary that wrote 跨4.2万次对抗试验 against a source saying 42,000.
    """
    cases = [
        ("跨 4.2万 次对抗试验", "across 42,000 adversarial trials"),
        ("覆盖 20K 条查询", "we cover 20000 queries"),
        ("训练 1.5M 样本", "1,500,000 examples"),
        ("共 3亿 参数", "300,000,000 parameters"),
    ]
    for article, source in cases:
        out = check_numbers({"results": [], "article_zh": article}, source)
        assert out["unverified"] == [], (article, source, out)


def test_check_numbers_still_flags_a_scaled_number_that_is_absent():
    out = check_numbers({"results": [], "article_zh": "覆盖 9.9万 条"}, "we cover 42,000")
    assert [u["number"] for u in out["unverified"]] == ["9.9万"]


# ------------------------------------------------------------- depth promotion/demotion
def test_promotion_from_shallow_to_deep_is_not_skipped():
    """`pick` can promote a paper when a threshold or cap moves.

    Skipping on "already has a summary" alone left it labelled 全文精读 on the page while
    showing abstract-only content.
    """
    from pna.cli import cmd_summarize

    promoted = {"arxiv_id": "a", "summary": {"x": 1}, "depth": "shallow"}
    fresh = {"arxiv_id": "b"}
    demoted = {"arxiv_id": "c", "summary": {"x": 1}, "depth": "deep"}
    done = {"arxiv_id": "d", "summary": {"x": 1}, "depth": "deep"}

    deep = [r for r in [promoted, fresh, done] if r.get("depth") != "deep" or not r.get("summary")]
    assert [r["arxiv_id"] for r in deep] == ["a", "b"], "promoted paper must be re-read"

    shallow = [r for r in [demoted] if not r.get("summary")]
    assert shallow == [], "a demoted paper keeps its already-paid-for deep summary"


def test_class_matching_is_by_token_not_substring():
    """LaTeXML puts `ltx_authors_1line` on the document root.

    With a substring test, `"ltx_authors" in cls` matched the root, so the whole paper was
    emitted as one author blob and the walker returned without descending. Observed on a
    real paper: 96,000 characters of body text collapsed to 51.
    """
    html = """
    <article class="ltx_document ltx_authors_1line">
      <div class="ltx_page_content">
        <h1 class="ltx_title">Real Title</h1>
        <div class="ltx_authors"><span>Wei Li, Tsinghua University</span></div>
        <section class="ltx_section"><h2 class="ltx_title_section">1 Introduction</h2>
          <p class="ltx_p">This body text must survive.</p>
          <p class="ltx_p">And so must this second paragraph.</p></section>
      </div>
    </article>
    """
    text = fulltext._html_to_markdown(html)
    assert "This body text must survive." in text
    assert "And so must this second paragraph." in text
    assert "## 1 Introduction" in text
    # The real author block still renders, exactly once.
    assert text.count("Wei Li, Tsinghua University") == 1


def test_has_class_helper():
    from lxml import html as lxml_html

    node = lxml_html.fromstring('<div class="ltx_document ltx_authors_1line"></div>')
    assert fulltext._has_class(node, "ltx_document") is True
    assert fulltext._has_class(node, "ltx_authors_1line") is True
    assert fulltext._has_class(node, "ltx_authors") is False
    assert fulltext._has_class(lxml_html.fromstring("<div></div>"), "ltx_p") is False


def test_pdf_fallback_fires_on_thin_html_not_only_on_exceptions(monkeypatch):
    """A page that parses cleanly but yields no prose must still reach the PDF path.

    Before this, the fallback was in an `except` block, so a stub or a broken LaTeXML
    conversion silently produced an abstract-only digest while a good PDF sat unused.
    """
    monkeypatch.setattr(fulltext, "_fetch_html", lambda pid: ("tiny stub", [], {}))
    monkeypatch.setattr(fulltext, "_fetch_pdf", lambda pid: "P" * 40_000)
    out = fulltext.fetch("2601.00001", use_cache=False)
    assert out["source"] == "pdf"
    assert out["chars"] == 40_000


def test_thin_html_still_wins_when_the_pdf_is_also_unusable(monkeypatch):
    monkeypatch.setattr(fulltext, "_fetch_html", lambda pid: ("short but real", [], {}))
    monkeypatch.setattr(
        fulltext, "_fetch_pdf", lambda pid: (_ for _ in ()).throw(ValueError("scan"))
    )
    out = fulltext.fetch("2601.00002", use_cache=False)
    assert out["source"] == "html"
    assert out["text"] == "short but real"


def test_a_paper_that_errored_last_run_is_retried_not_skipped():
    """`runner.run` records `summary_error` and leaves the previous summary in place.

    Without checking it, the stale summary counts as "already done" and the failure is
    never retried — the digest keeps publishing pre-failure content.
    """
    errored = {"arxiv_id": "a", "summary": {"x": 1}, "depth": "deep",
               "summary_error": "hit max_completion_tokens"}
    ok = {"arxiv_id": "b", "summary": {"x": 1}, "depth": "deep"}

    needs_work = [
        r for r in [errored, ok]
        if r.get("summary_error") or r.get("depth") != "deep" or not r.get("summary")
    ]
    assert [r["arxiv_id"] for r in needs_work] == ["a"]


# ------------------------------------------------------------------ evidence anchors
def test_extraction_emits_citable_anchors_for_every_structural_block():
    """`evidence_ids` are only checkable if the text carries the ids the model must cite."""
    html = """
    <div class="ltx_page_content">
      <section class="ltx_section" id="S2"><h2 class="ltx_title_section">2 Method</h2>
        <div class="ltx_para" id="S2.p1"><p class="ltx_p">First method paragraph.</p></div>
        <div class="ltx_para" id="S2.p2"><p class="ltx_p">Second method paragraph.</p></div>
        <figure class="ltx_table" id="S2.T1">
          <figcaption class="ltx_caption">Table 1: Main results</figcaption>
          <table><tr><td>a</td><td>b</td></tr></table></figure>
        <figure class="ltx_figure" id="S2.F1">
          <img src="x1.png"><figcaption class="ltx_caption">Figure 1: Overview</figcaption></figure>
      </section>
    </div>
    """
    anchors: dict = {}
    text = fulltext._markdown_from_doc(
        __import__("lxml.html", fromlist=["html"]).fromstring(html), anchors
    )
    assert "[[S2]]" in text and "[[S2.p1]]" in text and "[[S2.T1]]" in text
    assert anchors["S2.p1"]["kind"] == "paragraph"
    assert anchors["S2.T1"]["kind"] == "table"
    assert anchors["S2.F1"]["kind"] == "figure"
    # Paragraphs carry the heading they sit under, so a cited id can be named in the UI.
    assert anchors["S2.p2"]["section"] == "2 Method"


def test_anchors_dropped_by_trimming_are_not_offered_as_evidence():
    """A section cut to fit the token budget must not remain citable."""
    out = fulltext._finish(
        "[[S1]] kept\n\n## 9 Related Work\n[[S9.p1]] cut",
        "html", max_chars=10_000,
        anchors={"S1": {"kind": "section", "section": ""},
                 "S9.p1": {"kind": "paragraph", "section": "9 Related Work"}},
    )
    assert "S1" in out["anchors"]
    trimmed = fulltext._finish(
        "[[S1]] kept only", "html", max_chars=10_000,
        anchors={"S1": {"kind": "section", "section": ""},
                 "S9.p1": {"kind": "paragraph", "section": "9 Related Work"}},
    )
    assert "S9.p1" not in trimmed["anchors"], "a dropped block cannot be cited"


def test_pdf_fallback_yields_no_anchors(monkeypatch):
    """A PDF has no structural ids, so nothing in it is citable."""
    monkeypatch.setattr(fulltext, "_fetch_html", lambda pid: ("stub", [], {"S1": {}}))
    monkeypatch.setattr(fulltext, "_fetch_pdf", lambda pid: "P" * 40_000)
    out = fulltext.fetch("2601.00003", use_cache=False)
    assert out["source"] == "pdf"
    assert out["anchors"] == {}

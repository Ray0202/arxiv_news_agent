"""Template rendering, exercised with a fixture record.

Kept separate from the pipeline tests because this is the one place a bad template can
ship a broken page while every other stage reports success.
"""

from __future__ import annotations

import pytest

from pna.site.build import _code_badge, _paragraphs, _venue_badge, build, prepare

FIXTURE = {
    "arxiv_id": "2607.12345",
    "title": "Mixture-of-Recursions: Dynamic Recursive Depths",
    "abs_url": "https://arxiv.org/abs/2607.12345",
    "authors": ["Sangmin Bae", "Yujin Kim"],
    "categories": ["cs.LG", "cs.CL"],
    "created": "2026-07-29",
    "comments": "Accepted at NeurIPS 2026 (Spotlight). Code: https://github.com/raymin0223/mor",
    "journal_ref": None,
    "depth": "deep",
    "triage": {"score": 9, "novelty": "notable", "reason": "参数共享与自适应计算首次统一",
               "reason_en": "First to unify parameter sharing with adaptive computation"},
    "fulltext": {"source": "html", "chars": 55157, "truncated": False,
                 "dropped_sections": ["6 Related Work"]},
    "figures": [
        {"kind": "figure", "number": 3, "label": "Figure 3",
         "src": "https://arxiv.org/html/2607.12345v1/x3.png",
         "caption": "Figure 3: recursion depth distribution", "id": "S4.F3",
         "extra_srcs": []},
    ],
    "verify": {"numbers": {"checked": 6, "unverified": [{"number": "0.288",
                                                         "where": "results.ETTh1.value"}]}},
    "summary": {
        "tldr_zh": "在递归 Transformer 里用轻量 router 给每个 token 分配递归深度，同等训练 FLOPs 下降低困惑度。",
        "tldr_en": "Per-token recursion depth via lightweight routers beats vanilla at equal FLOPs.",
        "article_zh": "第一段：问题与背景。\n\n第二段：方法核心。\n\n第三段：实验证据。\n\n第四段：意义与局限。",
        "article_en": "Para one.\n\nPara two.\n\nPara three.\n\nPara four.",
        "key_contributions": ["统一参数共享与自适应计算", "KV 共享变体进一步降低显存"],
        "key_contributions_en": ["Unifies parameter sharing with adaptive computation",
                                 "A KV-sharing variant cuts memory further"],
        "method": {"core_idea": "共享层堆叠 + token 级 router 决定递归步数。",
                   "core_idea_en": "A shared layer stack with a per-token router.",
                   "architecture": "Recursive Transformer, 135M-1.7B",
                   "training_data": "FineWeb-Edu 子集", "compute": ""},
        "results": [{"benchmark": "ETTh1", "metric": "MSE", "value": "0.288",
                     "baseline": "0.379", "delta": "-24%"}],
        "limitations": [{"text": "只在 1.7B 以下验证。",
                         "text_en": "Only validated below 1.7B.", "source": "author"},
                        {"text": "对照基线未做同等超参搜索。",
                         "text_en": "Baselines got no equivalent hyperparameter search.",
                         "source": "reader"}],
        "why_it_matters_to_me": "自适应计算与 agent 的推理预算分配直接相关。",
        "why_it_matters_to_me_en": "Adaptive computation bears directly on agent inference budgets.",
        "tags": ["adaptive-computation", "parameter-sharing"],
        "institutions": ["KAIST", "Google DeepMind"],
        "figures_worth_seeing": [
            {"number": 3, "kind": "figure", "why": "递归深度随 token 难度分化，正文说不清",
             "why_en": "Recursion depth diverges with token difficulty"},
            {"number": 9, "kind": "figure", "why": "编号不存在，必须被丢弃"},
        ],
        "confidence": {"level": "medium", "caveat": "省略了 related work",
                       "caveat_en": "related work omitted"},
    },
}


def test_paragraphs_splits_on_blank_lines():
    assert _paragraphs("a\n\nb\n\n\nc") == ["a", "b", "c"]
    assert _paragraphs(None) == []
    assert _paragraphs("") == []


def test_venue_badge_reads_acceptance_from_comments():
    badge = _venue_badge(FIXTURE)
    assert badge["label"] == "NEURIPS"


def test_venue_badge_prefers_journal_ref():
    badge = _venue_badge({**FIXTURE, "journal_ref": "TMLR 2026"})
    assert badge["label"] == "published"


def test_venue_badge_absent_for_plain_preprint():
    assert _venue_badge({"comments": "12 pages, 4 figures", "journal_ref": None}) is None


def test_venue_badge_ignores_a_bare_conference_mention():
    """'submitted to NeurIPS' or a cited venue is not an acceptance."""
    assert _venue_badge({"comments": "Submitted for review", "journal_ref": None}) is None


def test_code_badge_extracts_and_strips_trailing_punctuation():
    assert _code_badge(FIXTURE) == "https://github.com/raymin0223/mor"
    assert _code_badge({"abstract": "see https://github.com/a/b."}) == "https://github.com/a/b"


def test_prepare_skips_unsummarized_records():
    out = prepare([FIXTURE, {"arxiv_id": "x", "title": "t", "abs_url": "u"}])
    assert [p["arxiv_id"] for p in out] == ["2607.12345"]


def test_build_renders_digest_archive_and_root(tmp_path):
    written = build(
        out_dir=tmp_path,
        loader=lambda date: [FIXTURE],
        dates_source=lambda: ["2026-07-29", "2026-07-30"],
    )
    names = {p.relative_to(tmp_path).as_posix() for p in written}
    assert {"index.html", "archive.html", "digest.json",
            "2026-07-30/index.html", "2026-07-29/index.html"} <= names

    page = (tmp_path / "2026-07-30" / "index.html").read_text(encoding="utf-8")
    assert "Mixture-of-Recursions" in page
    assert "9/10" in page
    assert "NEURIPS" in page
    assert "全文精读" in page
    # Every summary field the template promises to show
    assert "统一参数共享与自适应计算" in page
    assert "对照基线未做同等超参搜索" in page
    assert "KAIST" in page
    # The unverified-number warning must be visible, not silently swallowed
    assert "1 个数字未核到" in page
    # The badge names the model's own caveat rather than the opaque word "medium".
    assert "省略了 related work" in page
    # article_zh split into four <p>, not dumped as one blob with literal \n\n
    assert page.count("<p>第") == 4
    assert "\\n\\n" not in page


def test_build_escapes_html_in_paper_text(tmp_path):
    hostile = {**FIXTURE, "title": "Attention <script>alert(1)</script> Is All You Need"}
    build(out_dir=tmp_path, loader=lambda d: [hostile], dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_build_handles_a_day_that_was_judged_and_selected_nothing(tmp_path):
    judged = [{"arxiv_id": "1", "title": "T", "triage": {"score": 2}}]
    build(out_dir=tmp_path, loader=lambda d: judged, dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "这一天没有通过筛选的论文" in page


def test_a_harvested_but_unscored_day_is_not_published_as_an_empty_issue(tmp_path):
    """"Nothing was relevant" and "the pipeline has not run yet" render identically.

    Only the first is true of a finished day, so the second must not reach the archive —
    otherwise ingesting tomorrow's papers silently publishes tomorrow as a blank issue.
    """
    unscored = [{"arxiv_id": "1", "title": "T", "keyword": {"score": 3.0}}]
    build(out_dir=tmp_path, loader=lambda d: unscored, dates_source=lambda: ["2026-07-31"])
    assert not (tmp_path / "2026-07-31").exists()
    assert "2026-07-31" not in (tmp_path / "archive.html").read_text(encoding="utf-8")


def test_build_with_no_days_at_all_still_writes_archive(tmp_path):
    build(out_dir=tmp_path, loader=lambda d: [], dates_source=lambda: [])
    assert (tmp_path / "archive.html").exists()
    assert not (tmp_path / "index.html").exists()


# ---------------------------------------------------------------------- figures
def test_chosen_figures_matches_by_number_and_drops_invented_ones():
    from pna.site.build import _chosen_figures

    picked = _chosen_figures(FIXTURE, FIXTURE["summary"])
    assert [f["number"] for f in picked] == [3], "figure 9 does not exist and must be dropped"
    assert picked[0]["src"].endswith("x3.png")
    assert picked[0]["why"].startswith("递归深度")


def test_chosen_figures_tolerates_the_old_free_text_shape():
    """Records written before the schema change stored plain strings."""
    from pna.site.build import _chosen_figures

    legacy = {**FIXTURE["summary"], "figures_worth_seeing": ["Fig.3 递归深度分布"]}
    assert _chosen_figures(FIXTURE, legacy) == []


def test_build_renders_the_real_image_and_its_reason(tmp_path):
    build(out_dir=tmp_path, loader=lambda d: [FIXTURE], dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'src="https://arxiv.org/html/2607.12345v1/x3.png"' in page
    assert 'loading="lazy"' in page
    assert "递归深度随 token 难度分化" in page
    assert "x9.png" not in page


def test_build_normalizes_undelimited_math(tmp_path):
    hostile = {**FIXTURE}
    hostile["summary"] = {**FIXTURE["summary"], "article_zh": "以S_{i-1}为条件。\n\n第二段。"}
    build(out_dir=tmp_path, loader=lambda d: [hostile], dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "$S_{i-1}$" in page, "KaTeX has nothing to render without the delimiters"


def test_katex_is_loaded_with_integrity_hashes(tmp_path):
    build(out_dir=tmp_path, loader=lambda d: [FIXTURE], dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert page.count("integrity=\"sha384-") == 3
    assert "renderMathInElement" in page
    assert "throwOnError:false" in page


# ------------------------------------------------------------------- confidence badge
def test_no_confidence_badge_on_an_abstract_only_summary():
    """`low` there is the tier's definition, and the "仅摘要" badge already says it."""
    from pna.site.build import _confidence_note

    rec = {"depth": "shallow", "fulltext": {"source": "abstract", "truncated": False}}
    summary = {"confidence": {"level": "low", "caveat": "abstract only"}}
    assert _confidence_note(rec, summary) is None


def test_no_confidence_badge_when_high():
    from pna.site.build import _confidence_note

    rec = {"depth": "deep", "fulltext": {"source": "html", "truncated": False}}
    assert _confidence_note(rec, {"confidence": {"level": "high", "caveat": ""}}) is None


@pytest.mark.parametrize(
    "fulltext,zh,en",
    [
        ({"source": "pdf", "truncated": True}, "正文被截断", "text truncated"),
        ({"source": "pdf", "truncated": False}, "PDF 抽取，非 HTML", "from PDF, not HTML"),
        ({"source": "none", "truncated": False}, "取不到全文", "no full text"),
    ],
)
def test_deep_read_badge_names_the_concrete_cause(fulltext, zh, en):
    """The badge is generated here, so both languages come for free."""
    from pna.site.build import _confidence_note

    rec = {"depth": "deep", "fulltext": fulltext}
    summary = {"confidence": {"level": "medium", "caveat": "文本在 Section 6.6 处截断"}}
    assert _confidence_note(rec, summary) == {"zh": zh, "en": en}


def test_confidence_badge_falls_back_per_language():
    from pna.site.build import _confidence_note

    rec = {"depth": "deep", "fulltext": {"source": "html", "truncated": False}}
    note = _confidence_note(
        rec, {"confidence": {"level": "medium", "caveat": "省略了 related work",
                             "caveat_en": "related work omitted"}}
    )
    assert note == {"zh": "省略了 related work", "en": "related work omitted"}
    # No English twin -> a neutral label, never the Chinese text under an English view.
    note = _confidence_note(
        rec, {"confidence": {"level": "medium", "caveat": "省略了 related work"}}
    )
    assert note["en"] == "medium confidence"


def test_shallow_card_shows_one_badge_not_two_saying_the_same_thing(tmp_path):
    shallow = {
        **FIXTURE,
        "depth": "shallow",
        "fulltext": {"source": "abstract", "truncated": False, "dropped_sections": []},
        "summary": {**FIXTURE["summary"],
                    "confidence": {"level": "low", "caveat": "abstract only"}},
    }
    build(out_dir=tmp_path, loader=lambda d: [shallow], dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "仅摘要" in page
    assert "low confidence" not in page
    # The reason is still available as visible text where the detail belongs.
    assert "abstract only" in page


# --------------------------------------------------------------------- language switch
def test_both_languages_are_rendered_into_the_page(tmp_path):
    """The switch is CSS-only, so both languages must be present in the HTML."""
    build(out_dir=tmp_path, loader=lambda d: [FIXTURE], dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    for zh, en in [
        ("第一段：问题与背景。", "Para one."),
        ("统一参数共享与自适应计算", "Unifies parameter sharing with adaptive computation"),
        ("共享层堆叠", "A shared layer stack with a per-token router."),
        ("对照基线未做同等超参搜索。", "Baselines got no equivalent hyperparameter search."),
        ("自适应计算与 agent", "Adaptive computation bears directly on agent inference budgets."),
        ("递归深度随 token 难度分化", "Recursion depth diverges with token difficulty"),
        ("核心贡献", "Key contributions"),
        ("全文精读", "full text"),
    ]:
        assert zh in page, zh
        assert en in page, en


def test_language_switch_markup_and_default(tmp_path):
    build(out_dir=tmp_path, loader=lambda d: [FIXTURE], dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'data-set="zh"' in page and 'data-set="en"' in page
    assert 'html[data-lang="zh"] [lang="en"] { display: none; }' in page
    assert 'html[data-lang="en"] [lang="zh"] { display: none; }' in page
    # Set before first paint, so switching never flashes the other language.
    assert 'localStorage.getItem("pna-lang")' in page
    assert page.index("data-lang") < page.index("<body")


def test_english_view_falls_back_when_a_translation_is_missing(tmp_path):
    """Older records have no `_en` twins; the EN view must not render blanks."""
    legacy = {**FIXTURE}
    legacy["summary"] = {k: v for k, v in FIXTURE["summary"].items()
                         if not k.endswith("_en") or k in ("tldr_en", "article_en")}
    legacy["summary"]["limitations"] = [{"text": "只在 1.7B 以下验证。", "source": "author"}]
    legacy["summary"]["figures_worth_seeing"] = [
        {"number": 3, "kind": "figure", "why": "递归深度分化"}
    ]
    build(out_dir=tmp_path, loader=lambda d: [legacy], dates_source=lambda: ["2026-07-30"])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    # Chinese text appears twice: once for each view, because EN falls back to it.
    assert page.count("只在 1.7B 以下验证。") == 2
    assert page.count("递归深度分化") == 2


def test_superseded_summaries_are_kept_in_data_but_left_off_the_page():
    """A re-run whose triage picks a different set leaves stale summaries behind.

    They must not render: the digest would mix two models' output, and older records
    predate the bilingual schema so half the English view would be blank.
    """
    current = {**FIXTURE, "selected_for": "deep"}
    stale = {**FIXTURE, "arxiv_id": "2601.99999", "selected_for": None}
    legacy = {**FIXTURE, "arxiv_id": "2601.00001"}  # no key at all -> still shown
    out = prepare([current, stale, legacy])
    assert sorted(p["arxiv_id"] for p in out) == ["2601.00001", "2607.12345"]


# ------------------------------------------------------------------- criteria block
CRITERIA_RUN = {
    "filter": {"total": 881, "passed": 127},
    "criteria": {
        "categories": ["cs.LG", "cs.AI"],
        "topics": [{"name": "agent", "weight": 1.0,
                    "keywords": ["LLM agent", "tool use"],
                    "description": "工具调用与长程规划。",
                    "avoid": "不涉及语言模型的 multi-agent RL。"}],
        "thresholds": {"keyword_min_score": 1.0, "llm_min_score": 8},
        "budget": {"deep_read_max_per_day": 3, "shallow_max_per_day": 5},
        "models": {"triage": "gpt-5.4-mini", "deep": "gpt-5.4-mini"},
    },
}


def test_criteria_block_states_the_recorded_keywords_and_funnel(tmp_path):
    build(out_dir=tmp_path, loader=lambda d: [FIXTURE], dates_source=lambda: ["2026-07-30"],
          run_loader=lambda d: CRITERIA_RUN)
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "本期筛选条件" in page and "How these were selected" in page
    assert "LLM agent · tool use" in page
    assert "cs.LG · cs.AI" in page
    assert "工具调用与长程规划。" in page
    assert "不涉及语言模型的 multi-agent RL。" in page
    assert "≥ 8/10" in page
    # ingested is the number of records rendered (1 fixture); tier1 comes from the run log.
    assert "1 → 127 → 1" in page


def test_funnel_uses_the_number_the_pipeline_recorded(tmp_path):
    """Re-deriving it in the template drifted from `keyword.passes` and printed 128/127."""
    from pna.site.build import _tier1_count

    records = [
        {"keyword": {"score": 0.5, "category_match": True}},   # below threshold
        {"keyword": {"score": 3.0, "category_match": True}},
        {"keyword": {"score": 9.0, "category_match": False}},  # wrong category
    ]
    assert _tier1_count(CRITERIA_RUN, records) == 127, "the recorded count wins"
    # Only when nothing was recorded does it fall back — and then it must apply the
    # same threshold comparison the filter stage uses.
    no_record = {"criteria": CRITERIA_RUN["criteria"]}
    assert _tier1_count(no_record, records) == 1


def test_page_without_a_run_log_omits_the_criteria_block(tmp_path):
    build(out_dir=tmp_path, loader=lambda d: [FIXTURE], dates_source=lambda: ["2026-07-30"],
          run_loader=lambda d: {})
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "本期筛选条件" not in page
    assert "Mixture-of-Recursions" in page, "the digest itself must still render"

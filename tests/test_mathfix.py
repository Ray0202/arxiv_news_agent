"""Math normalisation: wrap notation the model left undelimited, touch nothing else."""

from __future__ import annotations

import pytest

from pna.site.mathfix import normalize, normalize_summary


@pytest.mark.parametrize(
    "raw,expect_in",
    [
        # The exact fragments the first real run produced, undelimited.
        ("以技能文档S_{i-1}为条件", "$S_{i-1}$"),
        ("后续任务折扣回报γ^{j-i}r_j", "$\\gamma^{j-i}$"),
        ("投影矩阵∈R^{d_k×d_v}", "$R^{d_k\\times d_v}$"),
        ("状态M_{t+1}=λM_t", "$M_{t+1}$"),
    ],
)
def test_wraps_bare_subscripts_and_superscripts(raw, expect_in):
    assert expect_in in normalize(raw)


def test_already_delimited_text_is_untouched():
    for text in ("已有 $S_{i-1}$ 与 $\\gamma^{j-i}$", "$\\mathbf{h}_t^{\\ell+1} = f(x)$"):
        assert normalize(text) == text


def test_fully_unicode_expressions_are_left_alone_by_design():
    """A half-wrapped formula reads worse than an unwrapped one.

    `λ·Σ||M_m||_F²` has no braced script, so nothing here recognises it. Wrapping just the
    `M_m` would produce `λ·Σ||$M_m$||_F²`. The prompt is responsible for these.
    """
    raw = "正则项λ·Σ||M_m||_F²"
    assert normalize(raw) == raw


def test_prose_arrows_and_punctuation_are_left_alone():
    """`抽取→存储→检索` is prose. Wrapping it would be worse than leaving it."""
    for text in (
        "流水线把抽取→存储→检索→执行拆成多阶段",
        "准确率提升 4.7%，成本下降 1/6",
        "Pass@1 达 85.9%，超过 GiGPO 2.3pp",
        "使用 8×H800 训练",
    ):
        assert normalize(text) == text, text


def test_no_double_wrapping_on_repeat_application():
    once = normalize("条件是S_{i-1}和τ_i")
    assert normalize(once) == once
    assert "$$" not in once


def test_underscore_in_plain_identifiers_is_not_math():
    """Snake_case names and file paths must not become formulas."""
    for text in ("参数 max_tokens 设为 16000", "写入 data_dir 目录", "字段 tldr_zh 为空"):
        assert normalize(text) == text, text


def test_greek_letters_only_converted_inside_wrapped_notation():
    # A bare Greek letter with no script attached stays as prose text.
    assert normalize("折扣因子设为 0.6") == "折扣因子设为 0.6"
    # But one carrying a subscript is notation and gets texified.
    assert "\\tau" in normalize("阈值τ_{rb}=0.10")


def test_normalize_summary_covers_every_free_text_field():
    summary = {
        "tldr_zh": "用S_{i-1}做条件",
        "article_zh": "回报γ^{j-i}监督",
        "key_contributions": ["共享M_{t}"],
        "method": {"core_idea": "投影W^{Q}", "architecture": "", "training_data": "",
                   "compute": ""},
        "limitations": [{"text": "仅验证≤4B", "source": "author"}],
        "why_it_matters_to_me": "与R^{d}相关",
        "results": [{"benchmark": "ETTh1", "metric": "MSE", "value": "0.361",
                     "baseline": "", "delta": ""}],
    }
    out = normalize_summary(summary)
    assert "$S_{i-1}$" in out["tldr_zh"]
    assert "$\\gamma^{j-i}$" in out["article_zh"]
    assert "$M_{t}$" in out["key_contributions"][0]
    assert "$W^{Q}$" in out["method"]["core_idea"]
    assert "$R^{d}$" in out["why_it_matters_to_me"]
    # results values are plain numbers rendered in a table, not math
    assert out["results"][0]["value"] == "0.361"
    # the original is not mutated
    assert summary["tldr_zh"] == "用S_{i-1}做条件"


# ----------------------------------------------------------------- CJK/Latin spacing
@pytest.mark.parametrize(
    "raw,expect",
    [
        ("跨任务test-time scaling持续提升", "跨任务 test-time scaling 持续提升"),
        ("其运行时间仅为RetroAgent的1/6。", "其运行时间仅为 RetroAgent 的 1/6。"),
        ("比GiGPO高出2.3个百分点", "比 GiGPO 高出 2.3 个百分点"),
        ("文档$S_{i-1}$解题", "文档 $S_{i-1}$ 解题"),
        ("Pass@1达85.9%、84.4%", "Pass@1 达 85.9%、84.4%"),
    ],
)
def test_cjk_latin_boundaries_get_a_space(raw, expect):
    from pna.site.mathfix import add_cjk_spacing

    assert add_cjk_spacing(raw) == expect


def test_spacing_is_idempotent():
    from pna.site.mathfix import add_cjk_spacing

    once = add_cjk_spacing("跨任务test-time scaling")
    assert add_cjk_spacing(once) == once
    assert "  " not in once


def test_spacing_leaves_full_width_punctuation_alone():
    from pna.site.mathfix import add_cjk_spacing

    for text in ("（Qwen3-4B）", "在ALFWorld、WebShop上", "结论：可行。"):
        out = add_cjk_spacing(text)
        assert "（ " not in out and " ）" not in out
        assert "、 " not in out


def test_spacing_does_not_split_math_internals():
    from pna.site.mathfix import add_cjk_spacing

    text = "回报 $\\sum_{j=i+1}^K \\gamma^{j-i} r_j$ 监督"
    assert add_cjk_spacing(text) == text


def test_polish_runs_math_then_spacing():
    from pna.site.mathfix import polish

    # Undelimited notation is wrapped first, then spaced away from the Chinese around it.
    assert polish("文档S_{i-1}解题") == "文档 $S_{i-1}$ 解题"

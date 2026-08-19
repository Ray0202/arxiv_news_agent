"""Build-time math normalisation — a safety net, not the primary mechanism.

The prompt tells the model to wrap every expression in `$...$`. Measured on the first real
run it wrapped **none** of them: the summaries carried bare fragments like `S_{i-1}`,
`γ^{j-i}`, `R^{d_k×d_v}` and `λ·Σ||M_m||_F²`, which render as literal typos. The prompt is
now explicit about delimiters, but a model that forgets on one field should not put a
broken formula on the page, so anything that is unmistakably notation gets wrapped here.

Deliberately conservative: only fragments carrying a *braced* LaTeX script (`S_{i-1}`) or
a backslash command are touched. Two things are deliberately out of scope.

Bare Unicode symbols in prose stay: `抽取→存储→检索` is a sentence, and `8×H800` is a spec.
Wrapping either would be worse than leaving it.

Unbraced scripts stay too, which means a fully Unicode expression like `λ·Σ||M_m||_F²`
survives unimproved. Wrapping only the recognisable piece would yield
`λ·Σ||$M_m$||_F²` — a formula half in math mode reads worse than one consistently not in
it. Getting these right is the prompt's job; this module only prevents the easy cases from
reaching the page broken.
"""

from __future__ import annotations

import re

# A token that is unambiguously notation: an identifier followed by at least one braced
# sub/superscript, e.g. `S_{i-1}`, `R^{d_k}`, `M_{t+1}`.
# The lookbehind must exclude ASCII word characters only. `\w` is Unicode-aware in
# Python, so it matches CJK — and `档S_{i-1}` then fails to match because the preceding
# Chinese character counts as "mid-word".
_BRACED = re.compile(
    r"(?<![$A-Za-z0-9_\\])"             # not already inside math, not mid-identifier
    r"("
    # Identifiers may be Greek: the model writes `\u03b3^{j-i}` and `\u03c4_{rb}` as often as
    # `S_{i-1}`, and an ASCII-only class silently skips every one of them.
    r"\\?[A-Za-z\u0391-\u03a9\u03b1-\u03c9][A-Za-z0-9]*"
    r"(?:[_^]\{[^}{]{1,40}\})+"         # at least one *braced* script
    r"(?:[_^]\{[^}{]{1,40}\}|[_^][A-Za-z0-9])*"
    r")"
)

# A lone LaTeX command outside math mode, e.g. `\gamma`, `\times`, `\leq`.
_COMMAND = re.compile(r"(?<![$A-Za-z0-9_])(\\[A-Za-z]{2,12})(?![A-Za-z])")

# Unicode operators the model uses where the prompt asked for LaTeX. Only mapped when they
# sit next to notation we are already wrapping (handled by _UNICODE_IN_MATH below).
_UNICODE_TO_TEX = {
    "×": r"\times",
    "≤": r"\leq",
    "≥": r"\geq",
    "∈": r"\in",
    "∑": r"\sum",
    "∏": r"\prod",
    "√": r"\sqrt",
    "≈": r"\approx",
    "≠": r"\neq",
    "·": r"\cdot",
    "∇": r"\nabla",
    "∂": r"\partial",
    "‖": r"\|",
}
_GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
    "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ι": r"\iota", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi",
    "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}


def _texify_inside(fragment: str) -> str:
    """Translate Unicode look-alikes to LaTeX, for text already known to be math.

    A LaTeX command name is terminated by a non-letter, so the separating space is only
    needed when a letter follows. Adding it unconditionally yields `\\gamma ^{j-i}`, which
    renders correctly but reads as a typo in the page source.
    """
    for uni, tex in {**_UNICODE_TO_TEX, **_GREEK}.items():
        # A lambda replacement, not a string: `re.sub` would read the backslash in
        # `\lambda` as an escape sequence and raise `bad escape \l`.
        fragment = re.sub(
            re.escape(uni) + r"(?=[A-Za-z])", lambda _m, t=tex: t + " ", fragment
        )
        fragment = fragment.replace(uni, tex)
    return fragment.strip()


def normalize(text: str | None) -> str:
    """Return `text` with undelimited notation wrapped in `$...$`.

    Content already inside `$...$` is passed through untouched, so a correctly formatted
    summary is a no-op.
    """
    if not text:
        return text or ""

    # Split on existing math spans and only rewrite the prose between them.
    parts = re.split(r"(\$[^$\n]*\$)", text)
    out: list[str] = []
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) > 2:
            out.append(part)
            continue
        part = _BRACED.sub(lambda m: "$" + _texify_inside(m.group(1)) + "$", part)
        part = _COMMAND.sub(lambda m: "$" + m.group(1) + "$", part)
        out.append(part)
    return "".join(out)


# CJK adjacent to Latin/digits, in either order. Chinese has no inter-word space, so
# `跨任务test-time scaling` renders as one run of glyphs and the eye has no boundary to
# latch onto. Inserting a space is the standard fix (what pangu.js does for the web).
_CJK = r"\u2e80-\u2eff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_THEN_LATIN = re.compile(rf"([{_CJK}])([A-Za-z0-9$\\(])")
_LATIN_THEN_CJK = re.compile(rf"([A-Za-z0-9%)\]$])([{_CJK}])")


def add_cjk_spacing(text: str | None) -> str:
    """Insert a space at every CJK/Latin boundary, leaving math spans alone.

    Applied after `normalize`, so `$...$` spans already exist and are skipped: adding a
    space inside `$\\gamma_{i}$` would not change the rendering, but adding one between the
    closing `$` and the next Chinese character does help, which is why `$` appears in the
    boundary classes rather than being excluded.
    """
    if not text:
        return text or ""
    parts = re.split(r"(\$[^$\n]*\$)", text)
    out: list[str] = []
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) > 2:
            out.append(part)
            continue
        part = _CJK_THEN_LATIN.sub(r"\1 \2", part)
        part = _LATIN_THEN_CJK.sub(r"\1 \2", part)
        out.append(part)
    joined = "".join(out)
    # The split boundary itself: `务$x$` and `$x$的` need the same treatment.
    joined = _CJK_THEN_LATIN.sub(r"\1 \2", joined)
    joined = _LATIN_THEN_CJK.sub(r"\1 \2", joined)
    return joined


def polish(text: str | None) -> str:
    """Everything the page needs done to a free-text field: math, then spacing."""
    return add_cjk_spacing(normalize(text))


def normalize_summary(summary: dict) -> dict:
    """Apply `polish` to every free-text field of a summary record."""
    fields = ("tldr_zh", "tldr_en", "article_zh", "article_en",
              "why_it_matters_to_me", "why_it_matters_to_me_en")
    fixed = dict(summary)
    for field in fields:
        if isinstance(fixed.get(field), str):
            fixed[field] = polish(fixed[field])
    for key in ("key_contributions", "key_contributions_en"):
        if isinstance(fixed.get(key), list):
            fixed[key] = [polish(c) for c in fixed[key]]
    if isinstance(fixed.get("method"), dict):
        fixed["method"] = {k: polish(v) for k, v in fixed["method"].items()}
    if isinstance(fixed.get("limitations"), list):
        fixed["limitations"] = [
            {**item, "text": polish(item.get("text")),
             "text_en": polish(item.get("text_en")) if item.get("text_en") else None}
            for item in fixed["limitations"]
        ]
    if isinstance(fixed.get("figures_worth_seeing"), list):
        fixed["figures_worth_seeing"] = [
            {**f, "why": polish(f.get("why")), "why_en": polish(f.get("why_en"))}
            if isinstance(f, dict) else f
            for f in fixed["figures_worth_seeing"]
        ]
    return fixed

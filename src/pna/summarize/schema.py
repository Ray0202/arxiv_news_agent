"""The structured-output schema for a paper digest.

Structured outputs reject `minLength`/`maxLength`/numeric constraints, so length limits
live in the field descriptions — which is where the model reads them anyway.
"""

from __future__ import annotations

from typing import Any

from ..config import Config


def build(cfg: Config) -> dict[str, Any]:
    langs = cfg.languages
    zh_lo, zh_hi = cfg.output.get("article_words_zh", [300, 500])
    en_lo, en_hi = cfg.output.get("article_words_en", [220, 380])

    props: dict[str, Any] = {}
    required: list[str] = []

    if "zh" in langs:
        props["tldr_zh"] = {
            "type": "string",
            "description": (
                "One Chinese sentence. HARD CEILING 40 characters — count them. State the "
                "actual finding "
                "with its concrete mechanism or number. Not the topic."
            ),
        }
        props["article_zh"] = {
            "type": "string",
            "description": (
                f"Chinese prose. HARD CEILING {zh_hi} characters — going over is a "
                f"failure, not a style choice; aim for {zh_lo}-{zh_hi}. Exactly four paragraphs "
                "separated by blank lines: (1) problem and gap in prior work, (2) the "
                "method's core, (3) what the experiments show, (4) significance and where "
                "it breaks. No headings, no bullet lists. Keep field-standard English "
                "terms in English. Wrap every formula and symbol in $...$ with LaTeX "
                "inside ($\\gamma=0.6$, $S_{i-1}$) — never bare Unicode or bare "
                "subscripts."
            ),
        }
        required += ["tldr_zh", "article_zh"]

    if "en" in langs:
        props["tldr_en"] = {
            "type": "string",
            "description": "One English sentence, at most 25 words, stating the finding.",
        }
        props["article_en"] = {
            "type": "string",
            "description": (
                f"English prose. HARD CEILING {en_hi} words; aim for {en_lo}-{en_hi}. "
                f"Same four-paragraph structure as "
                "article_zh. Not a translation — write it natively."
            ),
        }
        required += ["tldr_en", "article_en"]

    props.update(
        {
            "key_contributions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "At most 3 items, one sentence each, in Chinese.",
            },
            "key_contributions_en": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The same contributions in English, same order and count. The site "
                    "shows this instead when the reader switches language, so a missing "
                    "entry leaves a gap on the page."
                ),
            },
            "method": {
                "type": "object",
                "properties": {
                    "core_idea": {
                        "type": "string",
                        "description": (
                            "Technical description in Chinese. Formulas and jargon are "
                            "welcome here. 2-4 sentences."
                        ),
                    },
                    "core_idea_en": {
                        "type": "string",
                        "description": "The same, in English. 2-4 sentences.",
                    },
                    "architecture": {
                        "type": "string",
                        "description": (
                            "Concrete architecture / algorithm, written as language-neutral "
                            "technical notation — model names, layer counts, symbols. This "
                            "field is shown unchanged in both language views, so avoid "
                            "Chinese connective prose. '' if not stated."
                        ),
                    },
                    "training_data": {
                        "type": "string",
                        "description": (
                            "Datasets and scale, as names and numbers. Shown in both "
                            "language views. '' if not stated."
                        ),
                    },
                    "compute": {
                        "type": "string",
                        "description": (
                            "e.g. '8xA100, 72h'. Shown in both language views. '' if the "
                            "paper does not say."
                        ),
                    },
                },
                "required": ["core_idea", "core_idea_en", "architecture",
                             "training_data", "compute"],
                "additionalProperties": False,
            },
            "results": {
                "type": "array",
                "description": (
                    "Headline numbers only, each verifiable in the source text. Empty "
                    "array if the source gives no numbers."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "benchmark": {"type": "string"},
                        "metric": {"type": "string"},
                        "value": {"type": "string"},
                        "baseline": {
                            "type": "string",
                            "description": "Comparison point, e.g. 'PatchTST 0.379'. '' if none.",
                        },
                        "delta": {
                            "type": "string",
                            "description": "Relative change, e.g. '-4.7%'. '' if not derivable.",
                        },
                    },
                    "required": ["benchmark", "metric", "value", "baseline", "delta"],
                    "additionalProperties": False,
                },
            },
            "limitations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "In Chinese, one sentence."},
                        "text_en": {"type": "string", "description": "The same, in English."},
                        "source": {"type": "string", "enum": ["author", "reader"]},
                    },
                    "required": ["text", "text_en", "source"],
                    "additionalProperties": False,
                },
            },
            "why_it_matters_to_me": {
                "type": "string",
                "description": (
                    "At most 80 Chinese characters tying the paper to the reader's topics, "
                    "or plainly saying the connection is thin."
                ),
            },
            "why_it_matters_to_me_en": {
                "type": "string",
                "description": "The same, in English, at most 45 words.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-6 lowercase-hyphenated English tags.",
            },
            "institutions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Author affiliations as written in the paper. Empty if the text does "
                    "not show them (abstract-only input)."
                ),
            },
            "figures_worth_seeing": {
                "type": "array",
                "description": (
                    "At most 3, chosen by number from the figure inventory in the user "
                    "turn. Empty array if no inventory was given. Numbers must exist in "
                    "the inventory — the digest renders the real image by number, so an "
                    "invented number shows nothing."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "description": "The figure number as printed in the paper.",
                        },
                        "kind": {"type": "string", "enum": ["figure", "table"]},
                        "why": {
                            "type": "string",
                            "description": (
                                "One clause in Chinese: what this shows that the prose "
                                "cannot."
                            ),
                        },
                        "why_en": {
                            "type": "string",
                            "description": "The same clause in English.",
                        },
                    },
                    "required": ["number", "kind", "why", "why_en"],
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["high", "medium", "low"]},
                    "caveat": {
                        "type": "string",
                        "description": (
                            "Why below high, in Chinese, e.g. '仅有摘要' or '正文被截断'. "
                            "'' when high."
                        ),
                    },
                    "caveat_en": {
                        "type": "string",
                        "description": "The same in English. '' when high.",
                    },
                },
                "required": ["level", "caveat", "caveat_en"],
                "additionalProperties": False,
            },
        }
    )
    required += [
        "key_contributions",
        "key_contributions_en",
        "why_it_matters_to_me_en",
        "method",
        "results",
        "limitations",
        "why_it_matters_to_me",
        "tags",
        "institutions",
        "figures_worth_seeing",
        "confidence",
    ]

    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }

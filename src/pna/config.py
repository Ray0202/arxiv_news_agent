"""Configuration loading and the interest-profile rendering used by prompts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
PROMPT_DIR = CONFIG_DIR / "prompts"
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
RUNS_DIR = DATA_DIR / "runs"
CACHE_DIR = ROOT / "cache"
# `docs/` rather than `site/` because GitHub Pages' branch deploy offers exactly two
# folders — the repository root or `/docs` — and putting rendered HTML at the root mixes
# it with the source tree. This directory is committed; it is the published site.
SITE_DIR = ROOT / "docs"


@dataclass
class Topic:
    name: str
    weight: float = 1.0
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    avoid: str = ""
    # Shown in the published "how these were selected" block, which has a language
    # switch. Optional: without them the English view simply omits the prose, rather
    # than printing Chinese under an English heading.
    description_en: str = ""
    avoid_en: str = ""


@dataclass
class Config:
    categories: list[str]
    topics: list[Topic]
    thresholds: dict[str, Any]
    budget: dict[str, Any]
    output: dict[str, Any]
    models: dict[str, Any]
    ingest: dict[str, Any]
    raw: dict[str, Any]
    provider: str = "anthropic"
    selection: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    retention: dict[str, Any] = field(default_factory=dict)

    @property
    def languages(self) -> list[str]:
        return list(self.output.get("languages", ["zh"]))

    def snapshot(self) -> dict[str, Any]:
        """The filter criteria as they stood for one run.

        Written into that day's run log so the published page can state what it actually
        selected on. Reading the live config at build time instead would silently rewrite
        the stated criteria of every past digest the next time a keyword changes.
        """
        return {
            "categories": list(self.categories),
            "topics": [
                {
                    "name": t.name,
                    "weight": t.weight,
                    "keywords": list(t.keywords),
                    "description": " ".join((t.description or "").split()),
                    "avoid": " ".join((t.avoid or "").split()),
                    "description_en": " ".join((t.description_en or "").split()),
                    "avoid_en": " ".join((t.avoid_en or "").split()),
                }
                for t in self.topics
            ],
            "thresholds": dict(self.thresholds),
            "budget": {
                k: v for k, v in self.budget.items()
                if k in ("deep_read_max_per_day", "shallow_max_per_day",
                         "reviewer_max_per_day")
            },
            "selection": dict(self.selection),
            "models": dict(self.models),
        }

    def render_interests(self) -> str:
        """The interest profile, as injected into the triage prompt.

        Stable across a run, so it lives inside the cached prompt prefix.
        """
        out: list[str] = []
        for t in self.topics:
            out.append(f"### {t.name} (weight {t.weight})")
            if t.description:
                out.append(t.description.strip())
            if t.keywords:
                out.append("Keywords: " + ", ".join(t.keywords))
            if t.avoid:
                out.append("AVOID: " + t.avoid.strip())
            out.append("")
        out.append("In-scope arXiv categories: " + ", ".join(self.categories))
        return "\n".join(out).strip()


def load_config(path: Path | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    path = path or CONFIG_DIR / "interests.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    topics = [Topic(**t) for t in data.get("topics", [])]
    if not topics:
        raise ValueError(f"{path} defines no topics; nothing to filter on.")
    return Config(
        categories=data.get("categories", []),
        topics=topics,
        thresholds=data.get("thresholds", {}),
        budget=data.get("budget", {}),
        output=data.get("output", {}),
        models=data.get("models", {}),
        ingest=data.get("ingest", {}),
        raw=data,
        provider=data.get("provider", "anthropic"),
        selection=data.get("selection", {}),
        memory=data.get("memory", {}),
        retention=data.get("retention", {}),
    )


def read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def require_key(env_name: str, provider: str) -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get(env_name)
    if not key:
        raise SystemExit(
            f"{env_name} is not set, but the pipeline is configured to call the "
            f"{provider!r} provider. Copy .env.example to .env and fill it in, or export "
            f"the variable in your shell."
        )
    return key

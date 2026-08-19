"""Provider registry: what each backend supports, and what it costs.

Both backends speak the Anthropic Messages API, so there is one SDK and one call path.
What differs is capability, and the differences are the silent-failure kind — sending
`output_config.format` to DeepSeek is *accepted and ignored*, which would return prose
where the pipeline expects JSON. Hence an explicit table rather than feature-sniffing.

DeepSeek facts verified against api-docs.deepseek.com (2026-08):

* Anthropic-format base URL `https://api.deepseek.com/anthropic`, `x-api-key` auth.
* `output_config`: only `effort` is honoured; structured outputs are NOT enabled.
* `tools` / `input_schema` / `tool_choice: {"type": "tool"}` ARE fully supported — which
  is how we still get schema-constrained output.
* `cache_control` is ignored (caching happens automatically server-side), as are
  `anthropic-beta`, `top_k`, and image/document content blocks.
* `thinking` is supported; `budget_tokens` is ignored.
* Peak pricing: 2x during 09:00-12:00 and 14:00-18:00 Beijing time.

OpenAI facts verified against the live API with `gpt-5.4-mini` (2026-08):

* `max_tokens` is rejected — `max_completion_tokens` only.
* `reasoning_effort` accepts `none`/`low`/`medium`/`high`/`xhigh`; `minimal` is rejected.
* **`max_completion_tokens` covers reasoning *and* visible output.** At `xhigh` a 6000
  ceiling was consumed entirely by reasoning: `finish_reason="length"` and `content=""` —
  an empty string, not partial JSON. Every call must check `finish_reason` before parsing.
* `response_format={"type": "json_schema", ..., "strict": True}` gives real server-side
  schema enforcement; `temperature` is accepted.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

StructuredStrategy = Literal["output_config", "forced_tool", "json_schema_strict"]
# Which wire protocol the backend speaks. Anthropic and DeepSeek share one (DeepSeek
# exposes an Anthropic-compatible endpoint); OpenAI is a genuinely different shape.
WireAPI = Literal["anthropic", "openai"]


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float | None = None      # None -> derive as 0.1x input
    cache_write_mult: float = 1.25       # Anthropic 5-minute cache write premium

    @property
    def read_price(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input * 0.1


@dataclass(frozen=True)
class Provider:
    name: str
    api_key_env: str
    base_url: str | None
    structured: StructuredStrategy
    supports_effort: bool
    supports_cache_control: bool
    prices: dict[str, Price]
    api: WireAPI = "anthropic"
    # Refusal fallbacks are per-model, not per-provider: only models with safety
    # classifiers accept the parameter, and passing it for e.g. Haiku is a 400 because
    # the model is not in any allowed_fallback_models list.
    fallback_models: frozenset[str] = frozenset()
    peak_multiplier: float = 1.0
    peak_windows_beijing: tuple[tuple[int, int], ...] = ()

    def supports_fallbacks(self, model: str) -> bool:
        return model in self.fallback_models

    def price(self, model: str) -> Price:
        if model not in self.prices:
            raise KeyError(
                f"No price entry for {model!r} on provider {self.name!r}. Add it to "
                f"providers.py so the daily budget gate keeps working."
            )
        return self.prices[model]

    def multiplier_now(self, when: dt.datetime | None = None) -> float:
        """Peak-hour price multiplier. 1.0 for providers with flat pricing."""
        if self.peak_multiplier == 1.0 or not self.peak_windows_beijing:
            return 1.0
        when = when or dt.datetime.now(dt.timezone.utc)
        beijing_hour = (when.astimezone(dt.timezone(dt.timedelta(hours=8)))).hour
        for start, end in self.peak_windows_beijing:
            if start <= beijing_hour < end:
                return self.peak_multiplier
        return 1.0


ANTHROPIC = Provider(
    name="anthropic",
    api_key_env="ANTHROPIC_API_KEY",
    base_url=None,
    structured="output_config",
    supports_effort=True,
    supports_cache_control=True,
    prices={
        "claude-opus-5": Price(5.0, 25.0),
        "claude-opus-4-8": Price(5.0, 25.0),
        "claude-sonnet-5": Price(3.0, 15.0),
        "claude-haiku-4-5": Price(1.0, 5.0),
    },
    fallback_models=frozenset({"claude-opus-5"}),
)

OPENAI = Provider(
    name="openai",
    api_key_env="OPENAI_API_KEY",
    base_url=None,
    api="openai",
    # Real JSON-Schema enforcement (`strict: true`), stronger than either of the other
    # two paths: the server rejects a non-conforming generation rather than us catching it.
    structured="json_schema_strict",
    supports_effort=True,
    # Prompt caching is automatic and discounted; there is no cache_control to send.
    supports_cache_control=False,
    prices={
        # developers.openai.com/api/docs/pricing, checked 2026-08.
        "gpt-5.4-mini": Price(0.75, 4.50, cache_read=0.075, cache_write_mult=1.0),
        "gpt-5.4": Price(2.50, 15.00, cache_read=0.25, cache_write_mult=1.0),
        "gpt-5.4-nano": Price(0.15, 0.90, cache_read=0.015, cache_write_mult=1.0),
    },
)

DEEPSEEK = Provider(
    name="deepseek",
    api_key_env="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com/anthropic",
    # output_config.format is accepted and ignored here, so a forced tool call is the
    # only way to actually constrain the shape of the output.
    structured="forced_tool",
    supports_effort=True,
    supports_cache_control=False,
    prices={
        "deepseek-v4-pro": Price(0.435, 0.87, cache_read=0.003625, cache_write_mult=1.0),
        "deepseek-v4-flash": Price(0.14, 0.28, cache_read=0.0028, cache_write_mult=1.0),
    },
    peak_multiplier=2.0,
    peak_windows_beijing=((9, 12), (14, 18)),
)

PROVIDERS = {p.name: p for p in (ANTHROPIC, DEEPSEEK, OPENAI)}

# Models that only exist on one provider, so `provider:` in config is optional.
_MODEL_OWNER = {m: p.name for p in PROVIDERS.values() for m in p.prices}


def get(name: str) -> Provider:
    if name not in PROVIDERS:
        raise KeyError(f"Unknown provider {name!r}; known: {sorted(PROVIDERS)}")
    return PROVIDERS[name]


def for_model(model: str, default: str | None = None) -> Provider:
    """Resolve the provider that owns `model`, falling back to `default`."""
    owner = _MODEL_OWNER.get(model)
    if owner:
        return PROVIDERS[owner]
    if default:
        return get(default)
    raise KeyError(
        f"Cannot tell which provider serves model {model!r}. Either use a known model "
        f"id or set `provider:` in interests.yaml. Known: {sorted(_MODEL_OWNER)}"
    )

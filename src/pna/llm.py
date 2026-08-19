"""One call path for every backend, with per-provider capability handling.

Both Anthropic and DeepSeek speak the Messages API, so there is a single SDK and a single
`call_structured`. Two things justify the machinery around it:

* **Schema enforcement differs.** Anthropic has native structured outputs. DeepSeek
  accepts `output_config.format` and silently ignores it, so we constrain the shape with a
  forced tool call (`tools` + `input_schema` + `tool_choice: {"type": "tool"}`) instead.
  Sending the wrong one would return prose where the pipeline expects JSON — a failure
  that only shows up as a JSON parse error three stages later.
* **Neither guarantee is worth trusting blindly.** Every payload is validated against the
  schema client-side, and a failure retries with the validator's complaint fed back to the
  model. DeepSeek's docs also warn that JSON responses are occasionally empty.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence, TypeVar

import anthropic
import openai
from jsonschema import Draft202012Validator

from . import providers
from .config import Config, require_key
from .providers import Provider

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
TOOL_NAME = "emit_result"
MAX_ATTEMPTS = 3


class LLMError(RuntimeError):
    pass


class Refusal(LLMError):
    def __init__(self, category: str | None, explanation: str | None):
        super().__init__(f"model declined the request (category={category}): {explanation}")
        self.category = category
        self.explanation = explanation


class SchemaViolation(LLMError):
    pass


class Truncated(LLMError):
    """The token ceiling was consumed before any usable content came back.

    Distinct from a schema violation because the fix is different: retrying the same
    request unchanged will truncate again. The loop steps `effort` down instead.
    """

    def __init__(self, message: str, effort: str | None):
        super().__init__(message)
        self.effort = effort


# Ordered so a truncated call can step down one tier at a time.
_EFFORT_LADDER = ["xhigh", "high", "medium", "low", "none"]


def _lower_effort(effort: str | None) -> str | None:
    if effort not in _EFFORT_LADDER:
        return "medium"
    i = _EFFORT_LADDER.index(effort)
    return _EFFORT_LADDER[i + 1] if i + 1 < len(_EFFORT_LADDER) else None


# --------------------------------------------------------------------------- clients
class ClientPool:
    """One SDK client per provider, created lazily so an unused provider needs no key."""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, provider: Provider):
        with self._lock:
            if provider.name not in self._clients:
                key = require_key(provider.api_key_env, provider.name)
                if provider.api == "openai":
                    self._clients[provider.name] = openai.OpenAI(
                        api_key=key, max_retries=4, timeout=600.0
                    )
                else:
                    kwargs: dict[str, Any] = {
                        "api_key": key, "max_retries": 4, "timeout": 600.0
                    }
                    if provider.base_url:
                        kwargs["base_url"] = provider.base_url
                    self._clients[provider.name] = anthropic.Anthropic(**kwargs)
            return self._clients[provider.name]


# ----------------------------------------------------------------------------- usage
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0
    retries: int = 0
    usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, provider: Provider, model: str, usage: Any) -> float:
        price = provider.price(model)
        mult = provider.multiplier_now()
        if provider.api == "openai":
            # OpenAI reports cached tokens *inside* prompt_tokens, so the uncached part is
            # the difference. Counting prompt_tokens whole would double-bill the cache hit.
            read = getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
            ) or 0
            plain = max((getattr(usage, "prompt_tokens", 0) or 0) - read, 0)
            write = 0
            out = getattr(usage, "completion_tokens", 0) or 0
        else:
            plain = getattr(usage, "input_tokens", 0) or 0
            write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            read = getattr(usage, "cache_read_input_tokens", 0) or 0
            out = getattr(usage, "output_tokens", 0) or 0
        usd = (
            plain * price.input
            + write * price.input * price.cache_write_mult
            + read * price.read_price
            + out * price.output
        ) * mult / 1_000_000
        with self._lock:
            self.input_tokens += plain
            self.output_tokens += out
            self.cache_creation_input_tokens += write
            self.cache_read_input_tokens += read
            self.calls += 1
            self.usd += usd
            self.by_model[model] = round(self.by_model.get(model, 0.0) + usd, 6)
        return usd

    def note_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "usd": round(self.usd, 4),
            "by_model": self.by_model,
        }


# ------------------------------------------------------------------- structured call
def coerce_scalars(payload: Any, schema: dict) -> Any:
    """Losslessly normalise scalars that a model stringified.

    DeepSeek returns tool-call arguments with numbers as JSON strings — `{"score": "8"}`
    where the schema says integer. Measured on the triage stage, this happened on every
    single call, so every paper burned a second request just to be told its own type was
    wrong. Coercing here removes that retry entirely.

    Only exact round-trips are converted (`"8"` -> 8 because `str(8) == "8"`). `"8.5"` for
    an integer field, or `"abc"`, is left alone so validation still fails and the retry
    path — with the validator's complaint — handles a genuinely wrong answer.
    """
    if not isinstance(schema, dict):
        return payload

    kind = schema.get("type")

    if kind == "object" and isinstance(payload, dict):
        props = schema.get("properties") or {}
        return {
            key: coerce_scalars(val, props[key]) if key in props else val
            for key, val in payload.items()
        }
    if kind == "array" and isinstance(payload, list):
        items = schema.get("items") or {}
        return [coerce_scalars(item, items) for item in payload]

    if kind == "integer" and isinstance(payload, str):
        try:
            as_int = int(payload.strip())
        except ValueError:
            return payload
        return as_int if str(as_int) == payload.strip() else payload
    if kind == "integer" and isinstance(payload, float) and payload.is_integer():
        return int(payload)
    if kind == "number" and isinstance(payload, str):
        try:
            return float(payload.strip())
        except ValueError:
            return payload
    if kind == "boolean" and isinstance(payload, str):
        lowered = payload.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    return payload


def _validate(payload: Any, schema: dict) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload), key=lambda e: list(e.path)
    )
    if errors:
        joined = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors[:6]
        )
        raise SchemaViolation(joined)


def _extract(resp: Any, strategy: str) -> Any:
    if strategy == "forced_tool":
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return block.input
        text = next(
            (b.text for b in resp.content if getattr(b, "type", None) == "text"), ""
        )
        raise LLMError(
            f"expected a {TOOL_NAME} tool call, got none "
            f"(stop_reason={resp.stop_reason}, text={text[:160]!r})"
        )
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if not text:
        raise LLMError(f"no text block in response (stop_reason={resp.stop_reason})")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"structured output was not valid JSON: {exc}") from exc


def _call_anthropic(
    client, provider: Provider, *, model, system, user, schema, max_tokens, effort,
    thinking, tool_description, messages,
) -> tuple[Any, Any, Any]:
    """Returns (raw_payload, usage, meta_extra). Raises on refusal/truncation."""
    system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
    if provider.supports_cache_control:
        system_blocks[0]["cache_control"] = {"type": "ephemeral"}

    kwargs: dict[str, Any] = dict(
        model=model, max_tokens=max_tokens, system=system_blocks, messages=messages
    )
    output_config: dict[str, Any] = {}
    if effort and provider.supports_effort:
        output_config["effort"] = effort
    if not thinking and effort not in ("xhigh", "max"):
        # Opus 5 rejects disabled thinking above `high` effort.
        kwargs["thinking"] = {"type": "disabled"}

    if provider.structured == "forced_tool":
        kwargs["tools"] = [
            {"name": TOOL_NAME, "description": tool_description, "input_schema": schema}
        ]
        # `any`, not the named form: DeepSeek rejects `{"type": "tool", ...}` whenever
        # thinking is active. With exactly one tool offered they pin the same tool.
        kwargs["tool_choice"] = {"type": "any"}
    else:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    if output_config:
        kwargs["output_config"] = output_config

    if provider.supports_fallbacks(model):
        resp = client.beta.messages.create(
            betas=[_FALLBACK_BETA], fallbacks="default", **kwargs
        )
    else:
        resp = client.messages.create(**kwargs)

    served = getattr(resp, "model", model) or model
    if resp.stop_reason == "refusal":
        d = getattr(resp, "stop_details", None)
        raise Refusal(getattr(d, "category", None), getattr(d, "explanation", None))
    if resp.stop_reason == "max_tokens":
        raise Truncated(
            f"hit max_tokens={max_tokens} before finishing; output is truncated", effort
        )
    return _extract(resp, provider.structured), resp.usage, {
        "model": served, "stop_reason": resp.stop_reason
    }


def _call_openai(
    client, provider: Provider, *, model, system, user, schema, max_tokens, effort,
    thinking, tool_description, messages,
) -> tuple[Any, Any, Any]:
    """Chat Completions with strict JSON-Schema output.

    `max_completion_tokens` budgets reasoning *and* visible output together. When the
    reasoning pass consumes it all the API returns `finish_reason="length"` with
    `content=""` — an empty string, not partial JSON — so the check below is what stands
    between a truncated call and a bare `json.loads("")` crash three frames away.
    """
    chat = [{"role": "system", "content": system}]
    for m in messages:
        chat.append({"role": m["role"], "content": m["content"]})

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=chat,
        max_completion_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "digest", "strict": True, "schema": schema},
        },
    )
    if provider.supports_effort:
        # `thinking=False` maps to the model's own no-reasoning tier rather than a
        # separate parameter; `minimal` is not a valid value on this family.
        kwargs["reasoning_effort"] = (effort or "medium") if thinking else "none"

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        used = getattr(
            getattr(resp.usage, "completion_tokens_details", None), "reasoning_tokens", 0
        )
        raise Truncated(
            f"hit max_completion_tokens={max_tokens} ({used} of them spent on reasoning) "
            f"and returned no usable content",
            effort,
        )
    if choice.finish_reason == "content_filter":
        raise Refusal("content_filter", "the request was filtered by the provider")

    text = choice.message.content
    if not text:
        raise LLMError(
            f"empty content with finish_reason={choice.finish_reason!r}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"structured output was not valid JSON: {exc}") from exc
    return payload, resp.usage, {
        "model": getattr(resp, "model", model), "stop_reason": choice.finish_reason
    }


_TRANSPORTS = {"anthropic": _call_anthropic, "openai": _call_openai}


def call_structured(
    pool: ClientPool,
    *,
    model: str,
    system: str,
    user: str,
    schema: dict,
    max_tokens: int,
    cfg: Config,
    usage: Usage,
    effort: str | None = None,
    thinking: bool = True,
    tool_description: str = "Return the result for this paper.",
) -> tuple[dict, dict]:
    """One schema-constrained call, validated and retried. Returns (payload, meta)."""
    provider = providers.for_model(model, default=cfg.provider)
    client = pool.get(provider)
    transport = _TRANSPORTS[provider.api]

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    last_error: Exception | None = None
    meta: dict[str, Any] = {}

    for attempt in range(MAX_ATTEMPTS):
        try:
            raw, raw_usage, extra = transport(
                client, provider, model=model, system=system, user=user, schema=schema,
                max_tokens=max_tokens, effort=effort, thinking=thinking,
                tool_description=tool_description, messages=messages,
            )
        except Truncated as exc:
            # Reasoning ate the whole ceiling. Re-sending the same request would do it
            # again, so drop a tier and try once more rather than losing the paper.
            nxt = _lower_effort(exc.effort)
            if attempt == MAX_ATTEMPTS - 1 or nxt is None:
                raise LLMError(
                    f"{model}: {exc}. Raise max_tokens ({max_tokens}) or lower "
                    f"reasoning effort in config."
                ) from exc
            usage.note_retry()
            effort = nxt
            continue
        served = extra["model"]
        priced = served if served in provider.prices else model
        usd = usage.add(provider, priced, raw_usage)
        meta = {
            "provider": provider.name,
            "model": served,
            "stop_reason": extra["stop_reason"],
            "usd": round(usd, 6),
            "attempts": attempt + 1,
        }

        try:
            payload = coerce_scalars(raw, schema)
            _validate(payload, schema)
            return payload, meta
        except (LLMError, SchemaViolation) as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            usage.note_retry()
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": "(previous attempt was rejected)"},
                {
                    "role": "user",
                    "content": (
                        f"That response did not satisfy the required json schema: {exc}\n"
                        f"Return the result again, using the schema exactly. Every "
                        f"required field must be present; use \"\" or [] where you have "
                        f"nothing to report."
                    ),
                },
            ]

    raise LLMError(
        f"{model} failed to produce schema-valid output in {MAX_ATTEMPTS} attempts; "
        f"last error: {last_error}"
    )


def preflight(pool: ClientPool, cfg: Config, models: Iterable[str]) -> None:
    """Construct a client for every model a stage will use, before doing any work.

    Without this a missing key surfaces once per paper inside the thread pool and the
    command still exits 0.
    """
    for model in {m for m in models if m}:
        pool.get(providers.for_model(model, default=cfg.provider))


def estimate_input_tokens(
    pool: ClientPool, cfg: Config, model: str, system: str, user: str
) -> int:
    """Token count for the budget gate. OpenAI has no counting endpoint, so estimate."""
    provider = providers.for_model(model, default=cfg.provider)
    if provider.api == "openai":
        # ~3.3 chars/token is a reasonable blend for the mixed English-paper +
        # Chinese-instruction prompts this pipeline sends. Only feeds the budget gate,
        # which compares against a cap with headroom.
        return int(len(system) + len(user)) // 3
    client = pool.get(provider)
    resp = client.messages.count_tokens(
        model=model,
        system=[{"type": "text", "text": system}],
        messages=[{"role": "user", "content": user}],
    )
    return resp.input_tokens


# ------------------------------------------------------------------------- fan-out
T = TypeVar("T")
R = TypeVar("R")


def warm_then_parallel(
    items: Sequence[T], work: Callable[[T], R], workers: int = 6
) -> Iterable[tuple[T, R | BaseException]]:
    """Run `work` over `items`: first item alone, then the rest concurrently.

    Only `Exception` is caught per item. `SystemExit` and `KeyboardInterrupt` propagate —
    a missing API key is one config error, not N identical per-paper failures.

    On Anthropic a cache entry only becomes readable once the first response starts
    coming back, so firing everything at once makes every request miss the cache and pay
    the write premium. Harmless on DeepSeek, which caches server-side regardless.
    """
    results: list[tuple[T, R | BaseException]] = []
    if not items:
        return results
    head, tail = items[0], list(items[1:])
    try:
        results.append((head, work(head)))
    except Exception as exc:  # noqa: BLE001 - reported per-item by the caller
        results.append((head, exc))
    if tail:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(work, item): item for item in tail}
            for future in futures:
                item = futures[future]
                try:
                    results.append((item, future.result()))
                except Exception as exc:  # noqa: BLE001
                    results.append((item, exc))
    return results

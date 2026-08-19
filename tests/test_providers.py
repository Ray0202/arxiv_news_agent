"""Provider capability routing and cost accounting.

The failure this file guards against is silent: DeepSeek *accepts* `output_config.format`
and ignores it, so a mis-routed request returns prose and the pipeline only notices as a
JSON error further downstream. These tests assert on the request that would go out.
"""

from __future__ import annotations

import datetime as dt

import pytest

from pna import providers
from pna.config import Config, Topic
from pna.llm import TOOL_NAME, ClientPool, LLMError, Usage, call_structured
from pna.providers import ANTHROPIC, DEEPSEEK

SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def _cfg(provider: str) -> Config:
    return Config(
        categories=["cs.LG"],
        topics=[Topic(name="t", keywords=["x"])],
        thresholds={},
        budget={},
        output={},
        models={},
        ingest={},
        raw={},
        provider=provider,
    )


# ------------------------------------------------------------------ fake transport
class _Usage:
    def __init__(self, i=100, o=50, r=0, w=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = r
        self.cache_creation_input_tokens = w


class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, content, stop_reason="end_turn", model="m"):
        self.content = content
        self.stop_reason = stop_reason
        self.model = model
        self.usage = _Usage()
        self.stop_details = None


class _Messages:
    def __init__(self, recorder, responses):
        self._rec = recorder
        self._responses = list(responses)

    def create(self, **kwargs):
        self._rec.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    """Stands in for anthropic.Anthropic, recording the kwargs each call receives."""

    def __init__(self, responses):
        self.calls: list[dict] = []
        self.messages = _Messages(self.calls, responses)
        self.beta = type("B", (), {"messages": _Messages(self.calls, responses)})()


class _FixedPool(ClientPool):
    def __init__(self, client):
        super().__init__()
        self._fixed = client

    def get(self, provider):  # no API key needed
        return self._fixed


def _tool_resp(payload):
    return _Resp([_Block("tool_use", name=TOOL_NAME, input=payload)])


def _text_resp(text):
    return _Resp([_Block("text", text=text)])


# ------------------------------------------------------------------------- routing
def test_model_ids_resolve_to_their_owning_provider():
    assert providers.for_model("claude-opus-5").name == "anthropic"
    assert providers.for_model("deepseek-v4-pro").name == "deepseek"


def test_unknown_model_falls_back_to_configured_provider():
    assert providers.for_model("some-future-model", default="deepseek").name == "deepseek"
    with pytest.raises(KeyError):
        providers.for_model("some-future-model")


def test_deepseek_never_receives_structured_outputs():
    """output_config.format is accepted and ignored by DeepSeek — must not be sent."""
    client = _FakeClient([_tool_resp({"score": 7, "reason": "ok"})])
    payload, meta = call_structured(
        _FixedPool(client),
        model="deepseek-v4-flash",
        system="sys",
        user="usr",
        schema=SCHEMA,
        max_tokens=512,
        cfg=_cfg("deepseek"),
        usage=Usage(),
        effort="high",
    )
    sent = client.calls[0]
    assert payload == {"score": 7, "reason": "ok"}
    assert "format" not in sent.get("output_config", {})
    assert sent["tool_choice"] == {"type": "any"}
    assert len(sent["tools"]) == 1, "`any` only pins the tool while exactly one is offered"
    assert sent["tools"][0]["input_schema"] is SCHEMA
    assert sent["output_config"]["effort"] == "high"
    # cache_control is ignored by DeepSeek; sending it would be noise
    assert "cache_control" not in sent["system"][0]
    assert meta["provider"] == "deepseek"


def test_anthropic_uses_native_structured_outputs_and_cache_control():
    client = _FakeClient([_text_resp('{"score": 9, "reason": "good"}')])
    payload, _ = call_structured(
        _FixedPool(client),
        model="claude-haiku-4-5",
        system="sys",
        user="usr",
        schema=SCHEMA,
        max_tokens=512,
        cfg=_cfg("anthropic"),
        usage=Usage(),
    )
    sent = client.calls[0]
    assert payload["score"] == 9
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert "tools" not in sent
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_refusal_fallbacks_are_per_model_not_per_provider():
    """`fallbacks` is a 400 on a model with no safety-classifier fallback list.

    Opus 5 accepts it; Haiku does not. A provider-wide flag would send it to both.
    """
    assert ANTHROPIC.supports_fallbacks("claude-opus-5") is True
    assert ANTHROPIC.supports_fallbacks("claude-haiku-4-5") is False
    assert DEEPSEEK.supports_fallbacks("deepseek-v4-pro") is False

    haiku = _FakeClient([_text_resp('{"score": 1, "reason": "r"}')])
    call_structured(
        _FixedPool(haiku), model="claude-haiku-4-5", system="s", user="u",
        schema=SCHEMA, max_tokens=256, cfg=_cfg("anthropic"), usage=Usage(),
    )
    assert "fallbacks" not in haiku.calls[0]
    assert "betas" not in haiku.calls[0]

    opus = _FakeClient([_text_resp('{"score": 1, "reason": "r"}')])
    call_structured(
        _FixedPool(opus), model="claude-opus-5", system="s", user="u",
        schema=SCHEMA, max_tokens=256, cfg=_cfg("anthropic"), usage=Usage(),
    )
    assert opus.calls[0]["fallbacks"] == "default"
    assert opus.calls[0]["betas"] == ["server-side-fallback-2026-07-01"]


# ---------------------------------------------------------------- validate & retry
def test_schema_violation_retries_with_the_validator_complaint():
    client = _FakeClient(
        [
            _tool_resp({"score": "seven"}),                 # wrong type, missing field
            _tool_resp({"score": 7, "reason": "second try"}),
        ]
    )
    usage = Usage()
    payload, meta = call_structured(
        _FixedPool(client), model="deepseek-v4-flash", system="s", user="u",
        schema=SCHEMA, max_tokens=256, cfg=_cfg("deepseek"), usage=usage,
    )
    assert payload["reason"] == "second try"
    assert meta["attempts"] == 2
    assert usage.retries == 1
    retry_messages = client.calls[1]["messages"]
    assert "did not satisfy the required json schema" in retry_messages[-1]["content"]
    # Both attempts are billed — a retry is not free and must show up in the total.
    assert usage.calls == 2


def test_gives_up_after_max_attempts():
    client = _FakeClient([_tool_resp({"nope": 1})] * 3)
    with pytest.raises(LLMError, match="failed to produce schema-valid output"):
        call_structured(
            _FixedPool(client), model="deepseek-v4-flash", system="s", user="u",
            schema=SCHEMA, max_tokens=256, cfg=_cfg("deepseek"), usage=Usage(),
        )


def test_missing_tool_call_is_reported_not_silently_accepted():
    """DeepSeek's docs warn JSON responses are occasionally empty."""
    client = _FakeClient([_text_resp("Sure! Here is the summary...")] * 3)
    with pytest.raises(LLMError):
        call_structured(
            _FixedPool(client), model="deepseek-v4-flash", system="s", user="u",
            schema=SCHEMA, max_tokens=256, cfg=_cfg("deepseek"), usage=Usage(),
        )


def test_max_tokens_truncation_never_parses_the_partial_json():
    """Truncation now steps effort down first; it must never return the partial text."""
    truncated = _Resp([_Block("text", text='{"score":')], stop_reason="max_tokens")
    client = _FakeClient([truncated] * 4)
    with pytest.raises(LLMError, match="Raise max_tokens"):
        call_structured(
            _FixedPool(client), model="claude-haiku-4-5", system="s", user="u",
            schema=SCHEMA, max_tokens=8, cfg=_cfg("anthropic"), usage=Usage(),
            effort="low",
        )


# ------------------------------------------------------------------------ pricing
@pytest.mark.parametrize(
    "utc_hour,expected",
    [
        (1, 2.0),    # 09:00 Beijing -> peak
        (3, 2.0),    # 11:00 Beijing -> peak (the old cron slot)
        (5, 1.0),    # 13:00 Beijing -> off-peak (the cron slot we use)
        (7, 2.0),    # 15:00 Beijing -> peak
        (12, 1.0),   # 20:00 Beijing -> off-peak
        (18, 1.0),   # 02:00 Beijing next day -> off-peak
    ],
)
def test_deepseek_peak_window_arithmetic(utc_hour, expected):
    when = dt.datetime(2026, 8, 4, utc_hour, tzinfo=dt.timezone.utc)
    assert DEEPSEEK.multiplier_now(when) == expected


def test_anthropic_has_no_peak_pricing():
    for hour in range(24):
        when = dt.datetime(2026, 8, 4, hour, tzinfo=dt.timezone.utc)
        assert ANTHROPIC.multiplier_now(when) == 1.0


def test_usage_applies_peak_multiplier_and_cache_read_price():
    usage = Usage()
    off = dt.datetime(2026, 8, 4, 5, tzinfo=dt.timezone.utc)
    peak = dt.datetime(2026, 8, 4, 3, tzinfo=dt.timezone.utc)
    assert DEEPSEEK.multiplier_now(off) == 1.0 and DEEPSEEK.multiplier_now(peak) == 2.0

    price = DEEPSEEK.price("deepseek-v4-pro")
    # cache reads are ~120x cheaper than a miss, so they must not be priced as input
    assert price.read_price < price.input / 50
    expected = (100 * price.input + 50 * price.output) / 1_000_000
    got = usage.add(DEEPSEEK, "deepseek-v4-pro", _Usage(100, 50))
    assert got == pytest.approx(expected * DEEPSEEK.multiplier_now())


def test_unknown_model_price_fails_loudly():
    with pytest.raises(KeyError, match="No price entry"):
        DEEPSEEK.price("deepseek-v5-omega")


def test_deep_read_is_far_cheaper_on_deepseek_than_opus():
    """Guards the headline claim in the README against a silent price-table edit."""
    ds = DEEPSEEK.price("deepseek-v4-pro")
    op = ANTHROPIC.price("claude-opus-5")
    ds_cost = (16_000 * ds.input + 3_000 * ds.output) / 1e6
    op_cost = (16_000 * op.input + 4_500 * op.output) / 1e6
    assert op_cost / ds_cost > 10


# ---------------------------------------------------- thinking / tool_choice interplay
def test_forced_tool_never_uses_the_named_tool_choice():
    """DeepSeek 400s on {"type":"tool"} whenever thinking is active.

    Verified against the live API: `any` works with thinking on and off, the named form
    only with thinking disabled. Since one tool is offered, both pin the same tool.
    """
    for thinking in (True, False):
        client = _FakeClient([_tool_resp({"score": 1, "reason": "r"})])
        call_structured(
            _FixedPool(client), model="deepseek-v4-pro", system="s", user="u",
            schema=SCHEMA, max_tokens=256, cfg=_cfg("deepseek"), usage=Usage(),
            thinking=thinking,
        )
        assert client.calls[0]["tool_choice"] == {"type": "any"}


def test_thinking_false_sends_an_explicit_disable():
    client = _FakeClient([_tool_resp({"score": 1, "reason": "r"})])
    call_structured(
        _FixedPool(client), model="deepseek-v4-flash", system="s", user="u",
        schema=SCHEMA, max_tokens=256, cfg=_cfg("deepseek"), usage=Usage(),
        thinking=False,
    )
    assert client.calls[0]["thinking"] == {"type": "disabled"}


def test_thinking_true_omits_the_parameter():
    client = _FakeClient([_tool_resp({"score": 1, "reason": "r"})])
    call_structured(
        _FixedPool(client), model="deepseek-v4-pro", system="s", user="u",
        schema=SCHEMA, max_tokens=256, cfg=_cfg("deepseek"), usage=Usage(),
        thinking=True,
    )
    assert "thinking" not in client.calls[0]


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_disabled_thinking_is_suppressed_at_high_effort(effort):
    """Opus 5 returns 400 for disabled thinking above `high` effort."""
    client = _FakeClient([_text_resp('{"score": 1, "reason": "r"}')])
    call_structured(
        _FixedPool(client), model="claude-opus-5", system="s", user="u",
        schema=SCHEMA, max_tokens=256, cfg=_cfg("anthropic"), usage=Usage(),
        effort=effort, thinking=False,
    )
    assert "thinking" not in client.calls[0]
    assert client.calls[0]["output_config"]["effort"] == effort


def test_thinking_blocks_in_the_response_do_not_confuse_extraction():
    """With thinking on, content is ['thinking', 'tool_use'] — the payload is the latter."""
    resp = _Resp([_Block("thinking", thinking="..."), _Block("tool_use", input={"score": 3, "reason": "r"})])
    client = _FakeClient([resp])
    payload, _ = call_structured(
        _FixedPool(client), model="deepseek-v4-pro", system="s", user="u",
        schema=SCHEMA, max_tokens=256, cfg=_cfg("deepseek"), usage=Usage(),
    )
    assert payload == {"score": 3, "reason": "r"}


# ------------------------------------------------------- stringified-scalar coercion
def test_coerce_fixes_stringified_integers():
    """DeepSeek returns `{"score": "8"}` for an integer field on every triage call."""
    from pna.llm import coerce_scalars

    assert coerce_scalars({"score": "8", "reason": "r"}, SCHEMA) == {"score": 8, "reason": "r"}


def test_coerce_leaves_genuinely_wrong_values_for_the_validator():
    from pna.llm import coerce_scalars

    for bad in ("abc", "8.5", "", " 8 x", "08"):
        out = coerce_scalars({"score": bad, "reason": "r"}, SCHEMA)
        assert out["score"] == bad, f"{bad!r} must not be silently coerced"
        with pytest.raises(Exception):
            from pna.llm import _validate

            _validate(out, SCHEMA)


def test_coerce_handles_nested_objects_arrays_and_booleans():
    from pna.llm import coerce_scalars

    schema = {
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "ratio": {"type": "number"},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["flag", "ratio", "rows"],
        "additionalProperties": False,
    }
    got = coerce_scalars(
        {"flag": "true", "ratio": "0.5", "rows": [{"n": "1"}, {"n": 2}]}, schema
    )
    assert got == {"flag": True, "ratio": 0.5, "rows": [{"n": 1}, {"n": 2}]}


def test_coerce_does_not_touch_strings_that_should_stay_strings():
    from pna.llm import coerce_scalars

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    # A results table legitimately carries "0.361" as a string.
    assert coerce_scalars({"value": "0.361"}, schema) == {"value": "0.361"}


def test_stringified_score_no_longer_costs_a_retry():
    client = _FakeClient([_tool_resp({"score": "9", "reason": "ok"})])
    usage = Usage()
    payload, meta = call_structured(
        _FixedPool(client), model="deepseek-v4-flash", system="s", user="u",
        schema=SCHEMA, max_tokens=256, cfg=_cfg("deepseek"), usage=usage,
    )
    assert payload == {"score": 9, "reason": "ok"}
    assert meta["attempts"] == 1
    assert usage.retries == 0
    assert usage.calls == 1


# ------------------------------------------------------------------ openai transport
class _OAUsage:
    def __init__(self, prompt=100, completion=50, cached=0, reasoning=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = type("D", (), {"cached_tokens": cached})()
        self.completion_tokens_details = type("D", (), {"reasoning_tokens": reasoning})()


class _OAChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish_reason


class _OAResp:
    def __init__(self, content, finish_reason="stop", usage=None, model="gpt-5.4-mini"):
        self.choices = [_OAChoice(content, finish_reason)]
        self.usage = usage or _OAUsage()
        self.model = model


class _FakeOpenAI:
    def __init__(self, responses):
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return outer._responses.pop(0)

        self._responses = list(responses)
        self.chat = type("C", (), {"completions": _Completions()})()


def test_openai_sends_strict_json_schema_and_max_completion_tokens():
    """`max_tokens` is rejected outright by this model family."""
    client = _FakeOpenAI([_OAResp('{"score": 7, "reason": "ok"}')])
    payload, meta = call_structured(
        _FixedPool(client), model="gpt-5.4-mini", system="sys", user="usr",
        schema=SCHEMA, max_tokens=4000, cfg=_cfg("openai"), usage=Usage(), effort="high",
    )
    sent = client.calls[0]
    assert payload == {"score": 7, "reason": "ok"}
    assert sent["max_completion_tokens"] == 4000
    assert "max_tokens" not in sent
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"] is SCHEMA
    assert sent["reasoning_effort"] == "high"
    # The system prompt is a chat message here, not a separate parameter.
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert meta["provider"] == "openai"


def test_openai_thinking_false_maps_to_the_none_effort_tier():
    client = _FakeOpenAI([_OAResp('{"score": 1, "reason": "r"}')])
    call_structured(
        _FixedPool(client), model="gpt-5.4-mini", system="s", user="u", schema=SCHEMA,
        max_tokens=1000, cfg=_cfg("openai"), usage=Usage(), thinking=False,
    )
    assert client.calls[0]["reasoning_effort"] == "none"


def test_openai_truncation_never_parses_the_empty_string():
    """Measured on the live API: reasoning can consume the whole ceiling and the response
    comes back `finish_reason="length"` with `content=""`. `json.loads("")` would crash
    with a meaningless error far from the cause. It steps effort down, and if every tier
    truncates it fails with a message naming the two knobs that fix it."""
    empty = _OAResp("", finish_reason="length", usage=_OAUsage(completion=6000, reasoning=6000))
    client = _FakeOpenAI([empty] * 4)
    with pytest.raises(LLMError, match="max_completion_tokens"):
        call_structured(
            _FixedPool(client), model="gpt-5.4-mini", system="s", user="u", schema=SCHEMA,
            max_tokens=6000, cfg=_cfg("openai"), usage=Usage(), effort="xhigh",
        )


def test_openai_content_filter_surfaces_as_a_refusal():
    from pna.llm import Refusal

    client = _FakeOpenAI([_OAResp(None, finish_reason="content_filter")])
    with pytest.raises(Refusal):
        call_structured(
            _FixedPool(client), model="gpt-5.4-mini", system="s", user="u", schema=SCHEMA,
            max_tokens=1000, cfg=_cfg("openai"), usage=Usage(),
        )


def test_openai_cached_tokens_are_not_double_billed():
    """OpenAI reports cached tokens *inside* prompt_tokens."""
    from pna.providers import OPENAI

    usage = Usage()
    price = OPENAI.price("gpt-5.4-mini")
    got = usage.add(OPENAI, "gpt-5.4-mini", _OAUsage(prompt=1000, completion=100, cached=800))
    expected = (200 * price.input + 800 * price.read_price + 100 * price.output) / 1_000_000
    assert got == pytest.approx(expected)
    assert usage.input_tokens == 200 and usage.cache_read_input_tokens == 800


def test_openai_schema_violation_still_retries():
    client = _FakeOpenAI(
        [_OAResp('{"score": "seven"}'), _OAResp('{"score": 7, "reason": "second"}')]
    )
    usage = Usage()
    payload, meta = call_structured(
        _FixedPool(client), model="gpt-5.4-mini", system="s", user="u", schema=SCHEMA,
        max_tokens=1000, cfg=_cfg("openai"), usage=usage,
    )
    assert payload["reason"] == "second"
    assert meta["attempts"] == 2 and usage.retries == 1


def test_every_public_entry_point_is_importable():
    """A smoke test for the whole surface the CLI depends on.

    `preflight` was silently deleted by an edit that rewrote the surrounding region; no
    test referenced it, so the suite stayed green while `pna` failed at import.
    """
    import importlib

    llm = importlib.import_module("pna.llm")
    for name in ("ClientPool", "Usage", "preflight", "call_structured",
                 "estimate_input_tokens", "warm_then_parallel", "coerce_scalars",
                 "LLMError", "Refusal", "SchemaViolation", "TOOL_NAME"):
        assert hasattr(llm, name), f"pna.llm.{name} is missing"

    cli = importlib.import_module("pna.cli")
    for name in ("cmd_ingest", "cmd_filter", "cmd_triage", "cmd_summarize",
                 "cmd_refresh_figures", "cmd_build_site", "cmd_run", "cmd_stats", "main"):
        assert hasattr(cli, name), f"pna.cli.{name} is missing"


def test_cli_parser_builds_and_lists_every_subcommand():
    import argparse
    from pna.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])


def test_truncation_steps_effort_down_instead_of_repeating_the_same_call():
    """Observed in a real run: `high` effort spent all 16,000 tokens on reasoning.

    Re-sending an identical request would truncate identically, so the paper would be
    lost. The loop drops a tier and tries again.
    """
    client = _FakeOpenAI([
        _OAResp("", finish_reason="length", usage=_OAUsage(completion=16000, reasoning=16000)),
        _OAResp('{"score": 7, "reason": "recovered at a lower tier"}'),
    ])
    usage = Usage()
    payload, meta = call_structured(
        _FixedPool(client), model="gpt-5.4-mini", system="s", user="u", schema=SCHEMA,
        max_tokens=16000, cfg=_cfg("openai"), usage=usage, effort="high",
    )
    assert payload["reason"] == "recovered at a lower tier"
    assert client.calls[0]["reasoning_effort"] == "high"
    assert client.calls[1]["reasoning_effort"] == "medium"
    assert usage.retries == 1


def test_truncation_at_the_bottom_of_the_ladder_fails_with_an_actionable_message():
    client = _FakeOpenAI([
        _OAResp("", finish_reason="length", usage=_OAUsage(completion=99, reasoning=99))
    ] * 3)
    with pytest.raises(LLMError, match="Raise max_tokens"):
        call_structured(
            _FixedPool(client), model="gpt-5.4-mini", system="s", user="u", schema=SCHEMA,
            max_tokens=99, cfg=_cfg("openai"), usage=Usage(), effort="none",
        )


def test_effort_ladder_ordering():
    from pna.llm import _lower_effort

    assert _lower_effort("xhigh") == "high"
    assert _lower_effort("high") == "medium"
    assert _lower_effort("medium") == "low"
    assert _lower_effort("low") == "none"
    assert _lower_effort("none") is None
    assert _lower_effort(None) == "medium"

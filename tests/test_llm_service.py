import time

import pytest

from llm_service import LLMService, RateLimited, ToolCall


@pytest.fixture
def service():
    s = LLMService.__new__(LLMService)
    s.models, s._cooldown_until = ["primary", "fallback"], {}
    s._cancelled = lambda: False
    return s


def attempt_where(behaviour, calls):
    def attempt(model):
        calls.append(model)
        outcome = behaviour.get(model, "ok")
        if outcome == "429":
            raise RateLimited(30)
        return None if outcome == "fail" else ("answer", [])

    return attempt


def test_bad_response_retries_primary_once_then_falls_back(service):
    calls = []
    result = service._with_fallback(
        attempt_where({"primary": "fail"}, calls), "x", primary_attempts=2
    )

    assert result == ("answer", [])
    assert calls == ["primary", "primary", "fallback"]


def test_rate_limit_skips_retry_and_cools_primary_down(service):
    calls = []
    attempt = attempt_where({"primary": "429"}, calls)

    service._with_fallback(attempt, "x", primary_attempts=2)
    service._with_fallback(attempt, "x", primary_attempts=2)

    assert calls == ["primary", "fallback", "fallback"]
    assert service._cooldown_until["primary"] > time.time() + 25


def test_primary_is_used_again_after_cooldown(service):
    service._cooldown_until["primary"] = time.time() - 1
    calls = []

    service._with_fallback(attempt_where({}, calls), "x")

    assert calls == ["primary"]


def test_both_rate_limited_returns_none(service):
    calls = []
    result = service._with_fallback(
        attempt_where({"primary": "429", "fallback": "429"}, calls), "x"
    )

    assert result is None


def test_failed_round_after_a_tool_call_is_reported_incomplete(service):
    rounds = iter(
        [
            (
                "I'll search for that.",
                [ToolCall("call_1", "web_search", '{"query": "x"}')],
            ),
            None,
        ]
    )
    service._stream_step = lambda *args, **kwargs: next(rounds, None)
    executed = []

    result = service.run_agent(
        [],
        [{"type": "function"}],
        lambda name, args: executed.append(name) or "{}",
        print,
    )

    assert executed == ["web_search"]
    assert not result.completed
    assert result.text == "I'll search for that."


def test_answer_without_tools_is_complete(service):
    service._stream_step = lambda *args, **kwargs: ("Lisbon.", [])

    result = service.run_agent([], [], lambda *a: "", print)

    assert result.completed and result.text == "Lisbon."


def test_chain_continues_to_third_model(service):
    service.models = ["primary", "fallback", "local"]
    calls = []

    result = service._with_fallback(
        attempt_where({"primary": "429", "fallback": "429"}, calls), "x"
    )

    assert result == ("answer", [])
    assert calls == ["primary", "fallback", "local"]


@pytest.mark.parametrize(
    "model, provider, level, expected",
    [
        ("openai/gpt-oss-120b", "groq", "low", "low"),
        ("openai/gpt-oss-120b", "groq", "medium", "medium"),
        ("qwen/qwen3.8-27b", "groq", "low", "none"),
        ("qwen/qwen3.8-27b", "groq", "medium", "default"),
    ],
)
def test_reasoning_level_per_model(service, model, provider, level, expected):
    service._provider = {model: provider}

    assert service._reasoning_kwargs(model, level) == {"reasoning_effort": expected}


@pytest.mark.parametrize("level, think", [("low", False), ("medium", True)])
def test_llamacpp_thinking_switch(service, level, think):
    service._provider = {"qwythos-9b": "llamacpp"}

    kwargs = service._reasoning_kwargs("qwythos-9b", level)

    assert kwargs == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": think}}
    }


def test_malformed_chain_entry_is_rejected(monkeypatch):
    import llm_service

    monkeypatch.setattr(llm_service, "LLM_CHAIN", ["openai/gpt-oss-120b"])

    with pytest.raises(ValueError, match="provider one of groq, llamacpp"):
        LLMService()


def test_system_messages_are_folded_into_one_at_the_start():
    from llm_service import _single_system_message

    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "date"},
        {"role": "user", "content": "weather?"},
    ]

    assert _single_system_message(messages) == [
        {"role": "system", "content": "prompt\n\ndate"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "weather?"},
    ]


def test_cancelling_stops_before_the_requested_tool_runs(service):
    cancelled = []
    executed = []

    def step(messages, model, on_sentence, **kwargs):
        cancelled.append(True)  # the user says "stop" while this round streams
        return "Let me check.", [ToolCall("1", "create_event", "{}")]

    service._stream_step = step
    result = service.run_agent(
        [],
        [{}],
        lambda name, args: executed.append(name),
        lambda s: None,
        cancelled=lambda: bool(cancelled),
    )

    assert executed == []
    assert result.text == "Let me check."
    assert service._cancelled() is False

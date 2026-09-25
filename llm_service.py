import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from groq import Groq, RateLimitError
from openai import OpenAI

from config import (
    GROQ_API_KEY,
    LLM_CHAIN,
    LLM_MAX_TOKENS,
    LLM_REASONING_VOICE,
    LLM_TEMPERATURE,
    LLAMACPP_URL,
)
from llamacpp_server import LlamaCppServer
from utils import stop_event

logger = logging.getLogger(__name__)

SENTENCE = re.compile(r"[^.!?\n]*[.!?\n]")
# llama-server and gpt-oss return thinking separately (reasoning_content), but
# Groq's Qwen streams it inline in <think> tags; spoken, it would be read out.
THINKING = re.compile(r"<think>.*?(?:</think>|$)\s*", re.DOTALL)
MAX_AGENT_STEPS = 5
PROVIDERS = ("groq", "llamacpp")
DEFAULT_RETRY_AFTER = 10


class RateLimited(Exception):
    """The model hit Groq's rate limit; retrying it soon is pointless."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate-limited for {retry_after:.0f}s")
        self.retry_after = retry_after


def _rate_limited(error: RateLimitError, model_name: str) -> RateLimited:
    try:
        retry_after = float(
            error.response.headers.get("retry-after", DEFAULT_RETRY_AFTER)
        )
    except (TypeError, ValueError):
        retry_after = DEFAULT_RETRY_AFTER
    logger.warning(f"⏳ {model_name} rate-limited for {retry_after:.0f}s")
    return RateLimited(retry_after)


def _single_system_message(messages: List[dict]) -> List[dict]:
    """Fold every system message into one at the start.

    The output style follows the history. Groq tolerates that, but Qwen's chat
    template (llama.cpp) rejects any system message that isn't first.
    """
    system = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    return (
        [{"role": "system", "content": "\n\n".join(system)}] if system else []
    ) + rest


@dataclass
class ToolCall:
    id: str
    name: str = ""
    arguments: str = ""


@dataclass
class AgentResult:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    # False when a round failed (e.g. every model rate-limited): `text` may then
    # be only an announcement like "I'll search for that".
    completed: bool = True
    # The tool-call rounds of a completed turn (assistant tool_calls + tool
    # results), in request order, for the caller to keep in its history.
    trail: List[dict] = field(default_factory=list)


class LLMService:
    def __init__(self):
        self._provider = {}
        for entry in LLM_CHAIN:
            provider, _, model = entry.partition(":")
            if provider not in PROVIDERS or not model:
                raise ValueError(
                    f"LLM_CHAIN entry {entry!r}: expected <provider>:<model>, "
                    f"provider one of {', '.join(PROVIDERS)}"
                )
            self._provider[model] = provider
        self.models = list(self._provider)

        clients = {}
        if "groq" in self._provider.values():
            # No SDK retries: on a 429 it silently sleeps 20-40 s, which is dead
            # air for a voice assistant; the next model in the chain is the
            # better retry.
            clients["groq"] = Groq(api_key=GROQ_API_KEY, max_retries=0)
        self._llamacpp = LlamaCppServer()
        if "llamacpp" in self._provider.values():
            clients["llamacpp"] = OpenAI(
                base_url=f"{LLAMACPP_URL}/v1", api_key="none", max_retries=0
            )
            alias = next(m for m, p in self._provider.items() if p == "llamacpp")
            self._llamacpp.ensure_running(alias)
        self._clients = {model: clients[p] for model, p in self._provider.items()}
        # While a model is rate-limited it is skipped instead of costing a
        # failed request (and time) every turn.
        self._cooldown_until = {}

        self.max_retries = 2
        self.retry_delay = 1
        # The current turn's cancel check (the user said "stop"); see run_agent.
        self._cancelled: Callable[[], bool] = lambda: False

        logger.info(f"🤖 LLM chain: {' → '.join(self.models)}")

    def shutdown(self):
        self._llamacpp.stop()

    def _reasoning_kwargs(self, model, level):
        if "gpt-oss" in model:
            return {"reasoning_effort": level}
        # The local model thinks unless told "none": without it, Qwythos narrated
        # its reasoning as the answer ("The user said yes, I should…"). It thinks
        # briefly (~1.5 s), capped by llama-server's --reasoning-budget.
        if self._provider[model] == "llamacpp":
            think = level != "none"
            return {"extra_body": {"chat_template_kwargs": {"enable_thinking": think}}}
        # Hybrid Qwen on Groq only thinks or doesn't, and even "low" costs 10+ s.
        think = level in ("medium", "high")
        return {"reasoning_effort": "default" if think else "none"}

    def _halted(self) -> bool:
        return stop_event.is_set() or self._cancelled()

    def _with_fallback(self, attempt, label, primary_attempts=1):
        """Run attempt(model) down the chain until one returns a result."""
        for position, model in enumerate(self.models):
            if time.time() < self._cooldown_until.get(model, 0):
                continue
            for _ in range(primary_attempts if position == 0 else 1):
                try:
                    result = attempt(model)
                except RateLimited as limited:
                    self._cooldown_until[model] = time.time() + limited.retry_after
                    break
                if result or self._halted():
                    return result
            if position < len(self.models) - 1:
                logger.warning(
                    f"⚠️ {model} {label} failed, trying {self.models[position + 1]}"
                )
        return None

    def _create(
        self,
        messages,
        model_name,
        max_tokens=LLM_MAX_TOKENS,
        reasoning=LLM_REASONING_VOICE,
        **kwargs,
    ):
        kwargs.update(self._reasoning_kwargs(model_name, reasoning))
        return self._clients[model_name].chat.completions.create(
            model=model_name,
            messages=_single_system_message(messages),
            temperature=LLM_TEMPERATURE,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_completion(self, messages):
        result = self._with_fallback(
            lambda model: self._try_model_completion(messages, model), "completion"
        )
        if result:
            return result

        logger.error("❌ Every model in the chain failed")
        return "I'm having trouble with the model. Please try again."

    def _try_model_completion(self, messages, model_name):
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    logger.info(
                        f"🔄 Retry {attempt}/{self.max_retries - 1} on {model_name}..."
                    )

                response = self._create(messages, model_name)

                content = response.choices[0].message.content
                if content and content.strip():
                    return content
                logger.warning(f"⚠️ {model_name} returned empty content")

            except RateLimitError as e:
                raise _rate_limited(e, model_name) from e
            except Exception as e:
                logger.exception(
                    f"❌ Attempt {attempt + 1} on {model_name} failed: {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def run_agent(
        self,
        messages: List[dict],
        tools: List[dict],
        execute: Callable[[str, str], str],
        on_sentence: Callable[[str], None],
        max_tokens: int = LLM_MAX_TOKENS,
        reasoning: str = LLM_REASONING_VOICE,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> AgentResult:
        """Stream answers, running requested tools between rounds until the model
        replies without calling one. The last round offers no tools, forcing an answer.
        Once `cancelled()` is true, streaming stops and no further tools run.
        """
        self._cancelled = cancelled
        try:
            return self._agent_loop(
                list(messages), tools, execute, on_sentence, max_tokens, reasoning
            )
        finally:
            self._cancelled = lambda: False

    def _agent_loop(
        self, messages, tools, execute, on_sentence, max_tokens, reasoning
    ) -> AgentResult:
        made: List[ToolCall] = []
        text = ""
        start = len(messages)

        for step in range(MAX_AGENT_STEPS):
            offer = {"max_tokens": max_tokens, "reasoning": reasoning}
            if tools and step < MAX_AGENT_STEPS - 1:
                offer["tools"] = tools
            # gpt-oss occasionally emits a tool call Groq rejects against the
            # schema; one retry on the same model usually clears it.
            result = self._with_fallback(
                lambda model: self._stream_step(messages, model, on_sentence, **offer),
                "streaming",
                primary_attempts=2,
            )
            if result is None:
                return AgentResult(text, made, completed=False)
            text, calls = result
            if not calls or self._halted():
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                if self._halted():
                    return AgentResult(text, made)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": execute(call.name, call.arguments),
                    }
                )
                made.append(call)

        return AgentResult(text, made, trail=messages[start:])

    def _stream_step(
        self, messages, model_name, on_sentence, **kwargs
    ) -> Optional[Tuple[str, List[ToolCall]]]:
        try:
            stream = self._create(messages, model_name, stream=True, **kwargs)

            raw = ""
            text = ""
            spoken = 0
            calls = {}

            for chunk in stream:
                if self._halted():
                    break
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                for part in delta.tool_calls or []:
                    call = calls.setdefault(part.index, ToolCall(id=part.id or ""))
                    call.id = part.id or call.id
                    if part.function:
                        call.name += part.function.name or ""
                        call.arguments += part.function.arguments or ""

                if not delta.content:
                    continue
                raw += delta.content
                # Re-derived from the whole stream: a tag split across chunks is
                # only removable once complete, and it holds no sentence end
                # until then, so none of it is spoken early.
                text = THINKING.sub("", raw)

                while match := SENTENCE.match(text, spoken):
                    spoken = match.end()
                    sentence = match.group().strip()
                    if sentence:
                        on_sentence(sentence)

            if text[spoken:].strip():
                on_sentence(text[spoken:].strip())

            if not text.strip() and not calls:
                return None
            return text, list(calls.values())

        except RateLimitError as e:
            raise _rate_limited(e, model_name) from e
        except Exception as e:
            logger.error(f"Streaming error with {model_name}: {e}")
            return None

import json
import logging
import time
from datetime import datetime
from typing import Dict, List

from commands import get_formatted_commands
from config import (
    ALLOWED_LANGUAGES,
    FOLLOW_UP_SECONDS,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_TEXT,
    LLM_REASONING_TEXT,
    LLM_REASONING_VOICE,
    VOICE_OUTPUT,
    WAKE_WORD,
    WAKE_WORD_DURATION,
)
from embeddings import Embedder
from github_tools import github_tools
from google_tools import google_tools
from intent_detector import IntentDetector
from language_guard import spoken_language, unexpected_language
from memory_manager import FACT_EVENT, MemoryManager
from performance_monitor import performance_monitor
from timers import TimerService
from tool_manager import ToolManager
from tools import Tool, ToolRegistry, core_tools, memory_tools, params
from utils import stop_event

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Alfred, a helpful personal assistant.

Reply in the language the user spoke (a Romanian question gets a Romanian answer), even when tool results are in another language.

Requests come from speech recognition and can be garbled. If one doesn't make sense, ask the user to repeat it instead of guessing.

Tools: use them whenever they help, e.g. web_search for current events or facts you're unsure of, set_timer for timers. Never invent a tool result; if a tool fails, say so briefly.
Never say you did something (set a timer, saved a fact, searched, created an event) unless you called the tool for it in this turn and it succeeded. To do it, call the tool; don't describe doing it.
Search once per question. Only search again if the first results clearly don't answer it, and then with a genuinely different query, not a rephrasing.

When the user asks you to search, look something up or check, always call the tool, even if you think you know the answer or an earlier conversation covered it. If the user disputes a fact, verify it with a search rather than repeating yourself.

Memory: save with remember only when the user explicitly asks you to remember something. For "what do you know about me" or anything about their past requests, use recall before answering. Memory holds the user's facts and past questions, not answers: never treat it as a source for facts about the world."""

# Sent with each request rather than stored in the history, so switching mode
# mid-conversation takes effect immediately.
VOICE_STYLE = """Output mode: VOICE. Everything you write is spoken aloud.
- Keep each reply to two or three short spoken sentences.
- Only if there is clearly more worth saying, stop and ask "Want more?". If the answer is already complete, don't ask. If the user says yes, continue with the next part, again briefly.
- Plain speech only: no markdown, lists, tables, code, URLs or file paths (say just the file or folder name). Summarise tool output (file listings, search results, emails) instead of reading it out; give counts and highlights."""

TEXT_STYLE = """Output mode: TEXT. Your reply is shown on screen, not spoken.
Give complete answers. Markdown, lists, code blocks and URLs are fine, and you may show tool output in full when it's useful."""

SUMMARY_PROMPT = """Please summarize the following conversation, focusing on:
1. Key topics discussed
2. Important information shared
3. User preferences or patterns observed
4. Any ongoing tasks or commitments

Conversation:
{conversation}

Provide a concise summary that preserves the essential context for future interactions."""

MAX_CONTEXT_TOKENS = 4000
RECENT_MESSAGES_KEPT = 10
# Past tool calls stay in the history so the model keeps seeing that actions
# take a tool call; without them, a local model answered "I've opened it" with
# no call at all. Their results are trimmed so old output isn't resent in full.
HISTORY_TOOL_RESULT_CHARS = 300


def _estimate_token_count(messages: List[Dict[str, str]]) -> int:
    return sum(len(msg.get("content") or "") for msg in messages) // 4


def _trimmed(message: Dict) -> Dict:
    if (
        message["role"] != "tool"
        or len(message["content"]) <= HISTORY_TOOL_RESULT_CHARS
    ):
        return message
    return {**message, "content": message["content"][:HISTORY_TOOL_RESULT_CHARS] + "…"}


def _fresh_history():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


class ConversationHandler:
    def __init__(self, audio_processor, llm_service, search_service, tts_service):
        self.audio_processor = audio_processor
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.memory_manager = MemoryManager(Embedder())
        self.intent_detector = IntentDetector()
        self.timer_service = TimerService()
        self.tools = ToolRegistry()
        self.tools.register(
            *core_tools(
                ToolManager(safe_mode=True), search_service, self.timer_service
            ),
            *memory_tools(self.memory_manager),
            *github_tools(),
            *google_tools(),
            Tool(
                "set_output_mode",
                "Switch between VOICE (short spoken answers) and TEXT (full answers shown on "
                "screen, nothing spoken). Use when the user asks to read, see or stop hearing "
                "the output.",
                params(["mode"], mode={"type": "string", "enum": ["voice", "text"]}),
                self._set_output_mode,
            ),
        )
        self.conversation_history = _fresh_history()
        self._awaiting_reply = False
        # Transcript held while Alfred asks whether its language was intended.
        self._pending_foreign_text = None
        self.voice_output = VOICE_OUTPUT

    def announce_timers(self):
        for message in self.timer_service.pending_announcements():
            print(f"⏰ {message}")
            if self.voice_output:
                self.tts_service.reset()
                self.tts_service.speak(message)

    def _set_output_mode(self, mode: str):
        self.voice_output = mode == "voice"
        logger.info(f"🔈 Output mode: {mode}")
        return {"mode": mode}

    def shutdown(self):
        self.timer_service.cancel_all()

    def detect_wake_word(self, transcript):
        return WAKE_WORD.lower() in transcript.lower()

    def process_wake_word_detection(self):
        logger.info("\nListening for your calling, master...")
        temp_wake_path = None

        try:
            with self.audio_processor.create_temp_audio_file() as tmpfile:
                temp_wake_path = tmpfile.name

            heard_speech = self.audio_processor.record_audio(
                temp_wake_path, WAKE_WORD_DURATION
            )
            # Silent windows never reach Whisper: they cost GPU time and it
            # hallucinates text ("Thank you.") for them.
            if stop_event.is_set() or not heard_speech:
                return False

            # Timed from here: the fixed-length recording above isn't a slowdown.
            op_id = performance_monitor.start_operation("wake_word_transcription")
            transcript = self.audio_processor.transcribe_audio(temp_wake_path)
            performance_monitor.end_operation(op_id, success=bool(transcript))

            if not (transcript and transcript.strip()):
                return False
            logger.info(f'Heard: "{transcript}"')
            return self.detect_wake_word(transcript)

        except Exception as e:
            logger.error(f"Error during wake word processing: {e}")
            return False
        finally:
            self.audio_processor.cleanup_temp_file(temp_wake_path)

    def process_command(self):
        logger.info("✅ Wake word detected! Listening for command...")
        keep_running = self._listen_and_handle()
        # After Alfred asks something ("Want more?"), take the reply without
        # requiring the wake word again.
        while keep_running and self._awaiting_reply and not stop_event.is_set():
            logger.info("👂 Listening for your reply...")
            keep_running = self._listen_and_handle(start_timeout=FOLLOW_UP_SECONDS)
        return keep_running

    def _listen_and_handle(self, start_timeout=None):
        self._awaiting_reply = False
        temp_cmd_path = None

        try:
            with self.audio_processor.create_temp_audio_file() as cmdfile:
                temp_cmd_path = cmdfile.name

            heard_speech = self.audio_processor.record_until_silence(
                temp_cmd_path, start_timeout=start_timeout
            )
            if stop_event.is_set():
                return False
            # Whisper invents text ("Thank you.") for silent audio, so only
            # transcribe when the voice detector actually heard someone.
            if not heard_speech:
                logger.info("🤷 No speech heard.")
                return True
            command_text = self.audio_processor.transcribe_audio(temp_cmd_path)

            if not command_text.strip():
                logger.info("🤷 Command was empty or just silence.")
                return True

            logger.info(f'Command heard: "{command_text}"')
            return self._handle_command(command_text)

        except Exception as e:
            logger.error(f"Error during command processing: {e}")
            return True
        finally:
            self.audio_processor.cleanup_temp_file(temp_cmd_path)

    def _say(self, text):
        print(f"🔥 Alfred: {text}")
        if self.voice_output:
            self.tts_service.reset()
            self.tts_service.speak(text)

    def _handle_command(self, command_text):
        intent = self.intent_detector.detect(command_text)

        if self._pending_foreign_text:
            pending, self._pending_foreign_text = self._pending_foreign_text, None
            if intent == "affirm":
                return self._dispatch_command(pending)
            if intent == "decline":
                self._say("Okay, please say it again.")
                self._awaiting_reply = True
                return True

        if intent == "terminate":
            logger.info("🛑 Shutdown command detected. Shutting down Alfred.")
            return False

        if intent == "clear_context":
            logger.info("🧠 Context cleared. Starting fresh conversation.")
            self.conversation_history = _fresh_history()
            return True

        if intent == "show_commands":
            logger.info("📋 Displaying command reference...")
            print("\n" + get_formatted_commands())
            return True

        if intent == "performance":
            performance_monitor.print_stats()
            return True

        if intent == "decline":
            logger.info("👍 Okay, done.")
            return True

        if intent in ("voice_on", "voice_off"):
            self._set_output_mode("voice" if intent == "voice_on" else "text")
            if self.voice_output:
                self.tts_service.reset()
                self.tts_service.speak("Voice mode on.")
            return True

        language = unexpected_language(command_text, ALLOWED_LANGUAGES)
        if language:
            logger.info(f"🌐 Transcript looks like {language}: {command_text!r}")
            self._pending_foreign_text = command_text
            self._say(
                f"That sounded like {language}. Did you mean to speak {language}?"
            )
            self._awaiting_reply = True
            return True

        return self._dispatch_command(command_text)

    def _dispatch_command(self, command_text):
        # Retrieved memories and tool results go out with this request only; the
        # history keeps just the exchange, so old tool output isn't resent every turn.
        self.auto_summarize_context()

        # The prompt's start (system prompt, style, tools, history) stays
        # identical between turns so servers can reuse their prompt cache instead
        # of re-reading thousands of tokens. What changes every
        # turn (time, recalled memories) rides on the new user message instead.
        user_message = {"role": "user", "content": command_text}
        now = datetime.now().astimezone()
        notes = [f"Current date and time: {now:%A %d %B %Y, %H:%M %Z (UTC%z)}"]
        # The prompt rule alone isn't enough: after a Romanian exchange, models
        # kept answering English questions in Romanian.
        language = spoken_language(command_text)
        if language:
            notes.append(f"Reply in {language}.")
        relevant_context = self.get_relevant_context(command_text)
        if relevant_context:
            notes.append(relevant_context)
        request = list(self.conversation_history) + [
            {
                "role": "system",
                "content": VOICE_STYLE if self.voice_output else TEXT_STYLE,
            },
            {
                "role": "user",
                "content": "[Context for this request, not said by the user]\n"
                + "\n\n".join(notes)
                + f"\n\n[The user said]\n{command_text}",
            },
        ]

        print("🔥 Alfred: ", end="", flush=True)
        self.tts_service.reset()

        # Voice mode streams sentence by sentence so speech starts early. Text
        # mode prints the finished answer: the sentence splitter drops the line
        # breaks that markdown lists and code blocks depend on.
        first_sentence_op = performance_monitor.start_operation(
            "time_to_first_sentence"
        )

        spoken = []

        def on_sentence(sentence):
            nonlocal first_sentence_op
            if first_sentence_op:
                performance_monitor.end_operation(first_sentence_op)
                first_sentence_op = None
            if self.voice_output and not self.tts_service.interrupted:
                print(sentence, end=" ", flush=True)
                spoken.append(sentence)
                self.tts_service.say(sentence)

        voice = self.voice_output
        op_id = performance_monitor.start_operation("agent_turn")
        result = self.llm_service.run_agent(
            request,
            self.tools.schemas(),
            self.tools.call,
            on_sentence,
            max_tokens=LLM_MAX_TOKENS if voice else LLM_MAX_TOKENS_TEXT,
            reasoning=LLM_REASONING_VOICE if voice else LLM_REASONING_TEXT,
            cancelled=lambda: self.tts_service.interrupted,
        )
        if voice:
            self.tts_service.wait_until_done()
        performance_monitor.end_operation(
            op_id, success=bool(result.text), details={"tools": len(result.tool_calls)}
        )
        if first_sentence_op:
            performance_monitor.end_operation(first_sentence_op, success=False)
        if voice:
            print()
        else:
            print(result.text)

        # Interrupted with a key press: keep what was said, and listen for
        # the next request straight away, without the wake word.
        if voice and self.tts_service.interrupted:
            if spoken:
                self.conversation_history += [
                    user_message,
                    {"role": "assistant", "content": " ".join(spoken)},
                ]
            self._awaiting_reply = True
            return True

        if not result.completed and not stop_event.is_set():
            apology = (
                "Sorry, I couldn't finish that; the language model isn't reachable right now."
                if result.text
                else "Sorry, I can't reach the language model right now. Try again in a minute."
            )
            print(apology)
            if self.voice_output:
                self.tts_service.speak(apology)

        if result.completed and result.text:
            self._awaiting_reply = result.text.rstrip().endswith("?")
            logger.info(f"Alfred: {result.text}", extra={"file_only": True})
            self.conversation_history += [
                user_message,
                *map(_trimmed, result.trail),
                {"role": "assistant", "content": result.text},
            ]
            self._store_interaction_memory(
                command_text, result.text, self._tool_context(result.tool_calls)
            )

        return True

    @staticmethod
    def _tool_context(tool_calls) -> Dict:
        context = {"tools_used": [call.name for call in tool_calls]}
        searches = [call for call in tool_calls if call.name == "web_search"]
        if searches:
            context["search_performed"] = True
            context["search_query"] = json.loads(searches[-1].arguments or "{}").get(
                "query", ""
            )
        return context

    def auto_summarize_context(self) -> bool:
        try:
            if _estimate_token_count(self.conversation_history) <= MAX_CONTEXT_TOKENS:
                return False

            system_messages = [
                m for m in self.conversation_history if m["role"] == "system"
            ]
            other_messages = [
                m for m in self.conversation_history if m["role"] != "system"
            ]
            # Cut at a user message: a tool result separated from its call is
            # rejected by the API.
            cut = max(len(other_messages) - RECENT_MESSAGES_KEPT, 0)
            while cut > 0 and other_messages[cut]["role"] != "user":
                cut -= 1
            recent_messages = other_messages[cut:]
            older_messages = other_messages[:cut]

            if not older_messages:
                return False

            logger.info(
                "🧠 Context approaching limit, summarizing older conversations..."
            )

            conversation = "\n".join(
                f"{m['role'].title()}: {m['content']}"
                for m in older_messages
                if m["content"]
            )
            summary = self.llm_service.get_completion(
                [
                    {
                        "role": "user",
                        "content": SUMMARY_PROMPT.format(conversation=conversation),
                    }
                ]
            )
            if not summary:
                return False

            self.conversation_history = (
                system_messages
                + [
                    {
                        "role": "system",
                        "content": f"Previous conversation summary: {summary}",
                    }
                ]
                + recent_messages
            )

            self.memory_manager.episodic_memory(
                event_type="context_summarization",
                content=summary,
                context={"original_messages_count": len(older_messages)},
            )

            logger.info(f"✅ Summarized {len(older_messages)} older messages")
            return True

        except Exception as e:
            logger.error(f"Error during context summarization: {e}")
            return False

    def get_relevant_context(self, query: str) -> str:
        try:
            # Exchanges already in this conversation are skipped: recalled as an
            # "earlier question", the model answered the previous turn again
            # instead of the new one ("Okay." got the mayor answer repeated).
            in_conversation = {
                m["content"] for m in self.conversation_history if m["role"] == "user"
            }
            facts, topics = [], []
            for memory, _ in self.memory_manager.search_memories(query, limit=3):
                if memory.event_type == FACT_EVENT:
                    facts.append(f"- {memory.content}")
                elif memory.user_input not in in_conversation:
                    topics.append(f"- {memory.user_input[:120]}")
            # Only the user's past questions, never Alfred's past answers: fed
            # back automatically, a wrong answer was repeated verbatim instead
            # of searching, reinforcing itself on every turn.
            sections = []
            if facts:
                sections.append(
                    "Facts the user asked you to remember:\n" + "\n".join(facts)
                )
            if topics:
                sections.append(
                    "The user asked about related things before (use recall for "
                    "details):\n" + "\n".join(topics)
                )
            return "\n\n".join(sections)

        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""

    def _store_interaction_memory(
        self, user_input: str, assistant_response: str, context: Dict = None
    ):
        try:
            truncated_response = assistant_response[:500]
            interaction_data = {
                "user_input": user_input,
                "assistant_response": truncated_response,
                "search_performed": False,
                "search_query": "",
                "timestamp": time.time(),
                **(context or {}),
            }

            self.memory_manager.episodic_memory(
                event_type="conversation_interaction",
                content=f"User: {user_input[:200]}\nAssistant: {assistant_response[:200]}",
                user_input=user_input,
                assistant_response=truncated_response,
                context=interaction_data,
            )

            self.memory_manager.user_preference_learning(interaction_data)

        except Exception as e:
            logger.error(f"Error storing interaction memory: {e}")

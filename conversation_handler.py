import time
from typing import Any, Dict, List

from commands import get_formatted_commands
from config import WAKE_WORD, WAKE_WORD_DURATION
from intent_detector import IntentDetector
from memory_manager import MemoryManager
from performance_monitor import performance_monitor
from tool_manager import ToolManager

SYSTEM_PROMPT = """You are Alfred, a helpful voice assistant. Answer the user's questions directly and conversationally. Use information from previous interactions for context.

Important: Always respond with a direct answer. Never attempt to call tools, functions, or commands. Never output structured/JSON responses. Just speak naturally to the user.

If the user asks you to search, the search results will be provided to you as system messages — just summarize them for the user."""

SUMMARY_PROMPT = """Please summarize the following conversation, focusing on:
1. Key topics discussed
2. Important information shared
3. User preferences or patterns observed
4. Any ongoing tasks or commitments

Conversation:
{conversation}

Provide a concise summary that preserves the essential context for future interactions."""


def _truncate(text: str, limit: int) -> str:
    return text[:limit] + "..." if len(text) > limit else text


class ConversationHandler:
    def __init__(self, audio_processor, llm_service, search_service, tts_service):
        self.audio_processor = audio_processor
        self.llm_service = llm_service
        self.search_service = search_service
        self.tts_service = tts_service
        self.memory_manager = MemoryManager()
        self.tool_manager = ToolManager(safe_mode=True)
        self.intent_detector = IntentDetector()
        self.conversation_history = self._initialize_conversation_history()
        self.max_context_tokens = 4000
        self._last_memory_search = {}
        self._memory_search_cache_time = 60

    def _initialize_conversation_history(self):
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def detect_wake_word(self, transcript):
        return WAKE_WORD.lower() in transcript.lower()

    def process_wake_word_detection(self):
        print("\nListening for your calling, master...")
        temp_wake_path = None

        op_id = performance_monitor.start_operation("wake_word_detection")
        try:
            with self.audio_processor.create_temp_audio_file() as tmpfile:
                temp_wake_path = tmpfile.name

            self.audio_processor.record_audio(temp_wake_path, WAKE_WORD_DURATION)
            transcript = self.audio_processor.transcribe_audio(temp_wake_path)

            result = False
            if transcript and transcript.strip():
                print(f'Heard: "{transcript}"')
                result = self.detect_wake_word(transcript)

            performance_monitor.end_operation(op_id, success=True)
            return result

        except Exception as e:
            print(f"Error during wake word processing: {e}")
            performance_monitor.end_operation(
                op_id, success=False, details={"error": str(e)}
            )
            return False
        finally:
            if temp_wake_path:
                self.audio_processor.cleanup_temp_file(temp_wake_path)

    def process_command(self):
        print("✅ Wake word detected! Listening for command...")
        temp_cmd_path = None

        try:
            with self.audio_processor.create_temp_audio_file() as cmdfile:
                temp_cmd_path = cmdfile.name

            self.audio_processor.record_until_silence(temp_cmd_path)
            command_text = self.audio_processor.transcribe_audio(temp_cmd_path)

            if not command_text.strip():
                print("🤷 Command was empty or just silence.")
                return True

            print(f'Command heard: "{command_text}"')
            return self._handle_command(command_text)

        except Exception as e:
            print(f"Error during command processing: {e}")
            return True
        finally:
            self.audio_processor.cleanup_temp_file(temp_cmd_path)

    def _handle_command(self, command_text):
        intent, query = self.intent_detector.detect(command_text)

        if intent == "terminate":
            print("🛑 Shutdown command detected. Shutting down Alfred.")
            return False

        if intent == "clear_context":
            print("🧠 Context cleared. Starting fresh conversation.")
            self.conversation_history = self._initialize_conversation_history()
            return True

        if intent == "show_commands":
            print("📋 Displaying command reference...")
            print("\n" + get_formatted_commands())
            return True

        if intent == "performance":
            performance_monitor.print_stats()
            return True

        if intent == "search":
            return self._handle_search_command(command_text, query)

        if intent == "system_command":
            return self._handle_system_command(command_text, query)

        if intent in ("read_file", "write_file"):
            return self._handle_file_operation(command_text, intent, query)

        if intent == "scrape":
            return self._handle_web_scraping(command_text, query)

        return self._dispatch_command(command_text)

    def _dispatch_command(self, command_text, system_message=None, memory_context=None):
        """Summarize context, inject system message, stream the LLM response with TTS."""
        self.auto_summarize_context()

        relevant_context = self.get_relevant_context(command_text)
        if relevant_context:
            self.conversation_history.append(
                {"role": "system", "content": relevant_context}
            )

        if system_message:
            self.conversation_history.append(
                {"role": "system", "content": system_message}
            )

        self.conversation_history.append({"role": "user", "content": command_text})

        print("🔥 Alfred: ", end="", flush=True)
        self.tts_service.reset()

        def on_sentence(sentence):
            print(sentence, end=" ", flush=True)
            if not self.tts_service.interrupted:
                self.tts_service.speak(sentence)

        full_response = self.llm_service.get_completion_streaming(
            self.conversation_history, on_sentence=on_sentence
        )
        print()

        if full_response:
            self.conversation_history.append(
                {"role": "assistant", "content": full_response}
            )
            self._store_interaction_memory(command_text, full_response, memory_context)
        elif self.conversation_history[-1]["role"] == "user":
            self.conversation_history.pop()

        return True

    @staticmethod
    def _tool_system_message(label, fields, result, success_summary):
        """Build the system message describing a tool result for the LLM."""
        lines = [f"{label} result:"]
        lines += [f"{name}: {value}" for name, value in fields.items()]
        lines.append(f"Success: {result['success']}")
        lines.append(
            success_summary(result) if result["success"] else f"Error: {result['error']}"
        )
        return "\n".join(lines)

    def _handle_search_command(self, command_text, query):
        print(f"🔎 Searching for: {query}")

        op_id = performance_monitor.start_operation("search_command")

        try:
            search_system_msg = None
            search_results = self.search_service.multi_source_search(
                query, num_results=10
            )
            if search_results and search_results.get("combined_results"):
                search_system_msg = self.search_service.format_multi_source_results(
                    search_results, query
                )

            self._dispatch_command(
                command_text,
                system_message=search_system_msg,
                memory_context={"search_performed": True, "search_query": query},
            )

            performance_monitor.end_operation(op_id, success=True)

        except Exception as e:
            performance_monitor.end_operation(
                op_id, success=False, details={"error": str(e)}
            )
            print(f"Error in search command: {e}")

        return True

    def _handle_system_command(self, command_text, command):
        print(f"💻 Executing command: {command}")

        result = self.tool_manager.execute_system_commands(command, safety_check=True)

        return self._dispatch_command(
            command_text,
            system_message=self._tool_system_message(
                "Command execution",
                {"Command": command},
                result,
                lambda r: f"Output: {r['output']}",
            ),
            memory_context={"tool_used": "system_command", "command": command},
        )

    def _handle_file_operation(self, command_text, intent, query):
        content = None
        if intent == "write_file":
            action = "write"
            path, _, content = query.partition(" with content ")
            path = path.strip()
        else:
            action = "read"
            path = query.strip()

        if not path:
            print("❌ No file path given")
            return True

        print(f"📁 File operation: {action} on {path}")

        result = self.tool_manager.file_operations(action, path, content)

        def summarize(r):
            if "content" in r:
                return f"Content: {_truncate(r['content'], 500)}"
            if "items" in r:
                return f"Items found: {len(r['items'])}"
            return f"Result: {r.get('message', 'Operation completed')}"

        return self._dispatch_command(
            command_text,
            system_message=self._tool_system_message(
                "File operation", {"Action": action, "Path": path}, result, summarize
            ),
            memory_context={
                "tool_used": "file_operation",
                "action": action,
                "path": path,
            },
        )

    def _handle_web_scraping(self, command_text, query):
        url, _, extract_type = query.partition(" for ")
        url = url.strip()
        extract_type = extract_type.strip() or "text"

        if not url:
            print("❌ No URL given to scrape")
            return True

        print(f"🌐 Scraping {url} for {extract_type}")

        result = self.tool_manager.web_scraping(url, extract_type)

        def summarize(r):
            if "text" in r:
                return f"Text content: {_truncate(r['text'], 1000)}"
            if "links" in r:
                return f"Found {len(r['links'])} links"
            if "images" in r:
                return f"Found {len(r['images'])} images"
            return "Content extracted successfully"

        return self._dispatch_command(
            command_text,
            system_message=self._tool_system_message(
                "Web scraping",
                {"URL": url, "Extract type": extract_type},
                result,
                summarize,
            ),
            memory_context={
                "tool_used": "web_scraping",
                "url": url,
                "extract_type": extract_type,
            },
        )

    def auto_summarize_context(self) -> bool:
        try:
            if self._estimate_token_count(self.conversation_history) <= self.max_context_tokens:
                return False

            print("🧠 Context approaching limit, summarizing older conversations...")

            system_messages = [
                m for m in self.conversation_history if m["role"] == "system"
            ]
            other_messages = [
                m for m in self.conversation_history if m["role"] != "system"
            ]

            recent_messages = other_messages[-10:]
            older_messages = other_messages[:-10]

            if not older_messages:
                return False

            conversation = "\n".join(
                f"{m['role'].title()}: {m['content']}" for m in older_messages
            )
            summary = self.llm_service.get_completion(
                [{"role": "user", "content": SUMMARY_PROMPT.format(conversation=conversation)}]
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

            print(f"✅ Summarized {len(older_messages)} older messages")
            return True

        except Exception as e:
            print(f"Error during context summarization: {e}")
            return False

    def semantic_memory_search(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            cache_key = f"{query}_{limit}"
            current_time = time.time()

            cached = self._last_memory_search.get(cache_key)
            if cached and current_time - cached["time"] < self._memory_search_cache_time:
                return cached["results"]

            query_words = set(query.lower().split())
            if not query_words:
                return []

            memories = self.memory_manager.recall_episodic_memories(limit=20)

            scored = []
            for memory in memories:
                words = {w.lower() for w in memory.content.split()[:50]}
                words |= {w.lower() for w in memory.user_input.split()[:20]}

                overlap = len(query_words & words)
                if overlap:
                    scored.append(
                        {
                            "memory": memory,
                            "relevance_score": overlap / len(query_words),
                            "word_overlap": overlap,
                        }
                    )

            scored.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = scored[:limit]

            # Evict oldest entry instead of clearing entire cache
            if len(self._last_memory_search) > 100:
                oldest_key = min(
                    self._last_memory_search,
                    key=lambda k: self._last_memory_search[k]["time"],
                )
                del self._last_memory_search[oldest_key]
            self._last_memory_search[cache_key] = {
                "time": current_time,
                "results": results,
            }

            return results

        except Exception as e:
            print(f"Error during semantic memory search: {e}")
            return []

    def _estimate_token_count(self, messages: List[Dict[str, str]]) -> int:
        return sum(len(msg.get("content", "")) for msg in messages) // 4

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
            print(f"Error storing interaction memory: {e}")

    def get_relevant_context(self, query: str) -> str:
        try:
            if len(query.split()) < 2:
                return ""

            context_parts = [
                f"Previous: {m['memory'].user_input[:100]} -> "
                f"{m['memory'].assistant_response[:150]}"
                for m in self.semantic_memory_search(query, limit=2)
                if m["relevance_score"] > 0.3
            ]

            return "Context: " + " | ".join(context_parts) if context_parts else ""

        except Exception as e:
            print(f"Error getting relevant context: {e}")
            return ""

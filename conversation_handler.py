from config import (
    WAKE_WORD,
    TERMINATION_PHRASE,
    CLEAR_CONTEXT_PHRASE,
    WAKE_WORD_DURATION,
    COMMAND_DURATION,
)
from memory_manager import MemoryManager
from tool_manager import ToolManager
from commands import get_formatted_commands, search_commands
from intent_detector import IntentDetector
from performance_monitor import performance_monitor
from typing import List, Dict, Any
import time


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
        system_prompt = """You are Alfred, a helpful voice assistant. Answer the user's questions directly and conversationally. Use information from previous interactions for context.

Important: Always respond with a direct answer. Never attempt to call tools, functions, or commands. Never output structured/JSON responses. Just speak naturally to the user.

If the user asks you to search, the search results will be provided to you as system messages — just summarize them for the user."""
        return [{"role": "system", "content": system_prompt}]

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

            if transcript and transcript.strip():
                print(f'Heard: "{transcript}"')
                result = self.detect_wake_word(transcript)
            else:
                result = False

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

            if command_text.strip():
                print(f'Command heard: "{command_text}"')
                return self._handle_command(command_text)
            else:
                print("🤷 Command was empty or just silence.")
                return True

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
            return self._handle_show_commands()

        if intent == "performance":
            performance_monitor.print_stats()
            return True

        if intent == "search":
            return self._handle_search_command(command_text)

        if intent == "system_command":
            return self._handle_system_command(command_text)

        if intent in ("read_file", "write_file"):
            return self._handle_file_operation(command_text)

        if intent == "scrape":
            return self._handle_web_scraping(command_text)

        show_thinking = "show thinking" in command_text.lower()
        return self._handle_general_command(command_text, show_thinking)

    def _dispatch_command(
        self,
        command_text,
        system_message=None,
        show_thinking=False,
        memory_context=None,
    ):
        """Shared logic: summarize context, inject system message, stream LLM response with TTS."""
        self.auto_summarize_context(self.max_context_tokens)

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
            self._store_interaction_memory(
                user_input=command_text,
                assistant_response=full_response[:500],
                context=memory_context,
            )
        else:
            if (
                self.conversation_history
                and self.conversation_history[-1]["role"] == "user"
            ):
                self.conversation_history.pop()

        return True

    def _handle_show_commands(self):
        print("📋 Displaying command reference...")
        print("\n" + get_formatted_commands())
        return True

    def _handle_search_commands(self, command_text):
        query = command_text[len("search commands ") :].strip()
        print(f"🔍 Searching commands for: {query}")

        matching_commands = search_commands(query)

        if matching_commands:
            print(f"\n📋 Found {len(matching_commands)} matching commands:")
            print("=" * 50)

            for cmd in matching_commands:
                print(f"\n{cmd['category']}")
                print(f"Command: {cmd['command']}")
                print(f"Description: {cmd['description']}")
                print(f"Example: {cmd['example']}")
                print("-" * 30)
        else:
            print(f"❌ No commands found matching '{query}'")
            print("\n💡 Try 'show commands' for the full reference")

        return True

    def _handle_search_command(self, command_text):
        query = command_text[len("search for ") :].strip()
        print(f"🔎 Searching for: {query}")

        op_id = performance_monitor.start_operation("search_command")

        try:
            normalized_command = command_text.lower().strip()
            show_thinking = (
                "show thinking" in normalized_command
                or "show thoughts" in normalized_command
            )

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
                show_thinking=show_thinking,
                memory_context={"search_performed": True, "search_query": query},
            )

            performance_monitor.end_operation(op_id, success=True)
            return True

        except Exception as e:
            performance_monitor.end_operation(
                op_id, success=False, details={"error": str(e)}
            )
            print(f"Error in search command: {e}")
            return True

    def _handle_general_command(self, command_text, show_thinking=False):
        return self._dispatch_command(command_text, show_thinking=show_thinking)

    def clear_context(self):
        self.conversation_history = self._initialize_conversation_history()

    def auto_summarize_context(self, token_limit: int = 4000) -> bool:
        try:
            estimated_tokens = self._estimate_token_count(self.conversation_history)

            if estimated_tokens <= token_limit:
                return False

            print("🧠 Context approaching limit, summarizing older conversations...")

            system_messages = []
            non_system_messages = []
            for msg in self.conversation_history:
                if msg["role"] == "system":
                    system_messages.append(msg)
                else:
                    non_system_messages.append(msg)

            recent_messages = non_system_messages[-10:]
            older_messages = non_system_messages[:-10]

            if not older_messages:
                return False

            summary_prompt = self._create_summary_prompt(older_messages)
            summary_messages = [{"role": "user", "content": summary_prompt}]

            summary_response = self.llm_service.get_completion(summary_messages)

            if summary_response:
                summary_message = {
                    "role": "system",
                    "content": f"Previous conversation summary: {summary_response}",
                }

                self.conversation_history = (
                    system_messages + [summary_message] + recent_messages
                )

                self.memory_manager.episodic_memory(
                    event_type="context_summarization",
                    content=summary_response,
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

            if (
                cache_key in self._last_memory_search
                and current_time - self._last_memory_search[cache_key]["time"]
                < self._memory_search_cache_time
            ):
                return self._last_memory_search[cache_key]["results"]

            memories = self.memory_manager.recall_episodic_memories(limit=20)

            if not memories:
                return []

            query_words = set(query.lower().split())
            if not query_words:
                return []

            relevant_memories = []

            for memory in memories:
                content_words = set(memory.content.lower().split()[:50])
                user_input_words = set(memory.user_input.lower().split()[:20])

                all_words = content_words | user_input_words
                overlap = len(query_words & all_words)

                if overlap > 0:
                    relevance_score = overlap / len(query_words)
                    relevant_memories.append(
                        {
                            "memory": memory,
                            "relevance_score": relevance_score,
                            "word_overlap": overlap,
                        }
                    )

            relevant_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = relevant_memories[:limit]

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
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars / 4)

    def _create_summary_prompt(self, messages: List[Dict[str, str]]) -> str:
        conversation_text = "\n".join(
            [f"{msg['role'].title()}: {msg['content']}" for msg in messages]
        )

        return f"""Please summarize the following conversation, focusing on:
1. Key topics discussed
2. Important information shared
3. User preferences or patterns observed
4. Any ongoing tasks or commitments

Conversation:
{conversation_text}

Provide a concise summary that preserves the essential context for future interactions."""

    def _store_interaction_memory(
        self,
        user_input: str,
        assistant_response: str,
        search_performed: bool = False,
        search_query: str = "",
        context: Dict = None,
    ):
        try:
            truncated_response = assistant_response[:500]
            interaction_data = {
                "user_input": user_input,
                "assistant_response": truncated_response,
                "search_performed": search_performed,
                "search_query": search_query,
                "timestamp": time.time(),
            }

            if context:
                interaction_data.update(context)

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

            relevant_memories = self.semantic_memory_search(query, limit=2)

            if not relevant_memories:
                return ""

            context_parts = []
            for mem_data in relevant_memories:
                memory = mem_data["memory"]
                if mem_data["relevance_score"] > 0.3:
                    context_parts.append(
                        f"Previous: {memory.user_input[:100]} -> {memory.assistant_response[:150]}"
                    )

            if context_parts:
                return "Context: " + " | ".join(context_parts)
            return ""

        except Exception as e:
            print(f"Error getting relevant context: {e}")
            return ""

    def _handle_system_command(self, command_text):
        command = command_text[len("run command ") :].strip()
        print(f"💻 Executing command: {command}")

        result = self.tool_manager.execute_system_commands(command, safety_check=True)

        system_message = f"Command execution result:\nCommand: {command}\nSuccess: {result['success']}\n"
        if result["success"]:
            system_message += f"Output: {result['output']}"
        else:
            system_message += f"Error: {result['error']}"

        return self._dispatch_command(
            command_text,
            system_message=system_message,
            memory_context={"tool_used": "system_command", "command": command},
        )

    def _handle_file_operation(self, command_text):
        normalized = command_text.lower().strip()
        content = None

        if normalized.startswith("read file "):
            path = command_text[len("read file ") :].strip()
            action = "read"
        elif normalized.startswith("write file "):
            parts = (
                command_text[len("write file ") :].strip().split(" with content ", 1)
            )
            path = parts[0]
            content = parts[1] if len(parts) > 1 else ""
            action = "write"
        elif normalized.startswith("file "):
            parts = command_text[len("file ") :].strip().split(" ", 1)
            action = parts[0] if parts else "list"
            path = parts[1] if len(parts) > 1 else "."
        else:
            print("❌ Invalid file operation format")
            return True

        print(f"📁 File operation: {action} on {path}")

        result = self.tool_manager.file_operations(action, path, content)

        system_message = f"File operation result:\nAction: {action}\nPath: {path}\nSuccess: {result['success']}\n"
        if result["success"]:
            if "content" in result:
                system_message += f"Content: {result['content'][:500]}{'...' if len(result['content']) > 500 else ''}"
            elif "items" in result:
                system_message += f"Items found: {len(result['items'])}"
            else:
                system_message += (
                    f"Result: {result.get('message', 'Operation completed')}"
                )
        else:
            system_message += f"Error: {result['error']}"

        return self._dispatch_command(
            command_text,
            system_message=system_message,
            memory_context={
                "tool_used": "file_operation",
                "action": action,
                "path": path,
            },
        )

    def _handle_web_scraping(self, command_text):
        normalized = command_text.lower().strip()

        for prefix in ("web scrape ", "scrape "):
            if normalized.startswith(prefix):
                parts = command_text[len(prefix) :].strip().split(" for ", 1)
                url = parts[0]
                extract_type = parts[1] if len(parts) > 1 else "text"
                break
        else:
            print("❌ Invalid web scraping format")
            return True

        print(f"🌐 Scraping {url} for {extract_type}")

        result = self.tool_manager.web_scraping(url, extract_type)

        system_message = f"Web scraping result:\nURL: {url}\nExtract type: {extract_type}\nSuccess: {result['success']}\n"
        if result["success"]:
            if "text" in result:
                system_message += f"Text content: {result['text'][:1000]}{'...' if len(result['text']) > 1000 else ''}"
            elif "links" in result:
                system_message += f"Found {len(result['links'])} links"
            elif "images" in result:
                system_message += f"Found {len(result['images'])} images"
            else:
                system_message += f"Content extracted successfully"
        else:
            system_message += f"Error: {result['error']}"

        return self._dispatch_command(
            command_text,
            system_message=system_message,
            memory_context={
                "tool_used": "web_scraping",
                "url": url,
                "extract_type": extract_type,
            },
        )

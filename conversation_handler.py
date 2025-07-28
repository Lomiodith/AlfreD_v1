from config import (
    WAKE_WORD, TERMINATION_PHRASE, CLEAR_CONTEXT_PHRASE,
    WAKE_WORD_DURATION, COMMAND_DURATION
)


class ConversationHandler:
    def __init__(self, audio_processor, llm_service, search_service):
        self.audio_processor = audio_processor
        self.llm_service = llm_service
        self.search_service = search_service
        self.conversation_history = self._initialize_conversation_history()

    def _initialize_conversation_history(self):
        system_prompt = '''
Remember and use information from our previous interactions (my questions and your answers) to make your answers better and have some context from where to pick up the discussion.
        '''
        return [{"role": "system", "content": system_prompt}]

    def detect_wake_word(self, transcript):
        return WAKE_WORD.lower() in transcript.lower()

    def process_wake_word_detection(self):
        print("\nListening for your calling, master...")
        temp_wake_path = None
        
        try:
            with self.audio_processor.create_temp_audio_file() as tmpfile:
                temp_wake_path = tmpfile.name
            
            self.audio_processor.record_audio(temp_wake_path, WAKE_WORD_DURATION)
            transcript = self.audio_processor.transcribe_audio(temp_wake_path)
            
            if transcript.strip():
                print(f"Heard: \"{transcript}\"")
                
            return self.detect_wake_word(transcript)
            
        except Exception as e:
            print(f"Error during wake word processing: {e}")
            return False
        finally:
            self.audio_processor.cleanup_temp_file(temp_wake_path)

    def process_command(self):
        print("✅ Wake word detected! Listening for command...")
        temp_cmd_path = None
        
        try:
            with self.audio_processor.create_temp_audio_file() as cmdfile:
                temp_cmd_path = cmdfile.name
            
            self.audio_processor.record_audio(temp_cmd_path, COMMAND_DURATION)
            command_text = self.audio_processor.transcribe_audio(temp_cmd_path)
            
            if command_text.strip():
                print(f"Command heard: \"{command_text}\"")
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
        normalized_command = command_text.lower().strip()

        if TERMINATION_PHRASE.lower() in normalized_command:
            print(f"🛑 '{TERMINATION_PHRASE}' command detected. Shutting down Alfred.")
            return False

        elif CLEAR_CONTEXT_PHRASE.lower() in normalized_command:
            print("🧠 Context cleared by command. Starting fresh conversation.")
            self.conversation_history = self._initialize_conversation_history()
            return True

        elif normalized_command.startswith("search for "):
            return self._handle_search_command(command_text)

        else:
            show_thinking = "show thinking" in normalized_command or "show thoughts" in normalized_command
            return self._handle_general_command(command_text, show_thinking)

    def _handle_search_command(self, command_text):
        query = command_text[len("search for "):].strip()
        print(f"🔎 Searching for: {query}")
        
        normalized_command = command_text.lower().strip()
        show_thinking = "show thinking" in normalized_command or "show thoughts" in normalized_command
        
        search_results = self.search_service.multi_source_search(query, num_results=5)
        if search_results and search_results.get('combined_results'):
            formatted_results = self.search_service.format_multi_source_results(search_results, query)
            self.conversation_history.append({
                "role": "system",
                "content": formatted_results
            })

        self.conversation_history.append({"role": "user", "content": command_text})
        
        processed_response = self.llm_service.get_completion_with_processing(
            self.conversation_history, 
            show_thinking=show_thinking
        )
        
        if processed_response:
            self.conversation_history.append({
                "role": "assistant", 
                "content": processed_response['full_response']
            })
            print(f"🔥 Alfred: {processed_response['display']}")
        
        return True

    def _handle_general_command(self, command_text, show_thinking=False):
        self.conversation_history.append({"role": "user", "content": command_text})
        
        processed_response = self.llm_service.get_completion_with_processing(
            self.conversation_history, 
            show_thinking=show_thinking
        )
        
        if processed_response:
            self.conversation_history.append({
                "role": "assistant", 
                "content": processed_response['full_response']
            })
            print(f"🔥 Alfred: {processed_response['display']}")
        else:
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
        
        return True

    def clear_context(self):
        self.conversation_history = self._initialize_conversation_history()
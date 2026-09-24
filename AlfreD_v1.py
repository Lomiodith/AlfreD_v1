import os

from audio_processor import AudioProcessor
from conversation_handler import ConversationHandler
from llm_service import LLMService
from search_service import SearchService
from tts_service import TTSService
from utils import clean_temp_audio, stop_event


def main():
    clean_temp_audio()
    tts_service = None

    try:
        audio_processor = AudioProcessor()
        llm_service = LLMService()
        search_service = SearchService()
        tts_service = TTSService(default_voice="en-GB-RyanNeural")
        conversation_handler = ConversationHandler(
            audio_processor, llm_service, search_service, tts_service
        )

        while not stop_event.is_set():
            wake_word_detected = conversation_handler.process_wake_word_detection()
            if wake_word_detected:
                should_continue = conversation_handler.process_command()
                if not should_continue:
                    break

    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        print(f"Error initializing services: {e}")
    finally:
        print("Performing final cleanup...")
        if tts_service:
            tts_service.shutdown()
        clean_temp_audio()
        print("Alfred is now offline. Goodbye, master!")


if __name__ == "__main__":
    main()
    os._exit(0)

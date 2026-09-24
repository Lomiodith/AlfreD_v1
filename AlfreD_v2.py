# First: points the model caches at MODELS_DIR before any Hugging Face library loads.
import config  # noqa: F401  isort:skip

import logging
import os

from audio_processor import AudioProcessor
from conversation_handler import ConversationHandler
from llm_service import LLMService
from config import TTS_VOICE
from search_service import SearchService
from tts_service import TTSService
from utils import clean_temp_audio, setup_logging, stop_event

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    clean_temp_audio()
    tts_service = None
    llm_service = None
    conversation_handler = None

    try:
        audio_processor = AudioProcessor()
        llm_service = LLMService()
        search_service = SearchService()
        tts_service = TTSService(default_voice=TTS_VOICE)
        conversation_handler = ConversationHandler(
            audio_processor, llm_service, search_service, tts_service
        )

        while not stop_event.is_set():
            conversation_handler.announce_timers()
            wake_word_detected = conversation_handler.process_wake_word_detection()
            if wake_word_detected:
                should_continue = conversation_handler.process_command()
                if not should_continue:
                    break

    except KeyboardInterrupt:
        logger.info("\n🛑 Script interrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        logger.exception(f"AlfreD stopped on an error: {e}")
    finally:
        logger.info("Performing final cleanup...")
        if conversation_handler:
            conversation_handler.shutdown()
        if tts_service:
            tts_service.shutdown()
        if llm_service:
            llm_service.shutdown()
        clean_temp_audio()
        logger.info("Alfred is now offline. Goodbye, master!")


if __name__ == "__main__":
    main()
    os._exit(0)

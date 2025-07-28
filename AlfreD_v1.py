import sys
from audio_processor import AudioProcessor
from llm_service import LLMService
from search_service import SearchService
from conversation_handler import ConversationHandler
from utils import clean_temp_audio
from config import GOOGLE_API_KEY, GOOGLE_CSE_ID

def main():
    clean_temp_audio()
    
    try:
        audio_processor = AudioProcessor()
        llm_service = LLMService()
        search_service = SearchService(GOOGLE_API_KEY, GOOGLE_CSE_ID)
        conversation_handler = ConversationHandler(audio_processor, llm_service, search_service)
        
        while True:
            wake_word_detected = conversation_handler.process_wake_word_detection()
            
            if wake_word_detected:
                should_continue = conversation_handler.process_command()
                if not should_continue:
                    break

    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        print(f"Error initializing services: {e}")
        sys.exit(1)
    finally:
        print("Performing final cleanup...")
        clean_temp_audio()
        print("Alfred is now offline. Goodbye, master!")


if __name__ == "__main__":
    main()
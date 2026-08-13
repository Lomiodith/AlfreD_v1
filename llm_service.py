import re
import time
import traceback

from groq import Groq

from config import TEMPERATURE, GROQ_API_KEY

SENTENCE = re.compile(r"[^.!?\n]*[.!?\n]")


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "openai/gpt-oss-120b"
        self.fallback_model = "llama-3.3-70b-versatile"

        self.max_retries = 2
        self.retry_delay = 1

        print(f"🤖 LLM Service initialized with Groq model: {self.model}")

    def _with_fallback(self, attempt, label):
        """Run attempt(model) on the primary model, then the fallback."""
        result = attempt(self.model)
        if result:
            return result

        print(f"⚠️ {self.model} {label} failed, switching to fallback: {self.fallback_model}")
        return attempt(self.fallback_model)

    def _create(self, messages, model_name, **kwargs):
        return self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=1024,
            tool_choice="none",
            **kwargs,
        )

    def get_completion(self, messages):
        result = self._with_fallback(
            lambda model: self._try_model_completion(messages, model), "completion"
        )
        if result:
            return result

        print("❌ Both models failed after all retries")
        return "I'm having trouble with the model. Please try again."

    def _try_model_completion(self, messages, model_name):
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"🔄 Retry {attempt}/{self.max_retries - 1} on {model_name}...")

                response = self._create(messages, model_name)

                content = response.choices[0].message.content
                if content and content.strip():
                    return content
                print(f"⚠️ {model_name} returned empty content")

            except Exception as e:
                print(f"❌ Attempt {attempt + 1} on {model_name} failed: {e}")
                print(f"🔍 Exception details: {traceback.format_exc()}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def get_completion_streaming(self, messages, on_sentence):
        return self._with_fallback(
            lambda model: self._try_streaming(messages, model, on_sentence), "streaming"
        )

    def _try_streaming(self, messages, model_name, on_sentence):
        try:
            stream = self._create(messages, model_name, stream=True)

            full_response = ""
            buffer = ""

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue

                buffer += delta
                full_response += delta

                while match := SENTENCE.match(buffer):
                    buffer = buffer[match.end():]
                    sentence = match.group().strip()
                    if sentence:
                        on_sentence(sentence)

            if buffer.strip():
                on_sentence(buffer.strip())

            return full_response if full_response.strip() else None

        except Exception as e:
            print(f"Streaming error with {model_name}: {e}")
            return None

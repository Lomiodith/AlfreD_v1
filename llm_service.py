import re
import time
import traceback
from typing import Dict, Optional, Tuple

from groq import Groq
# import torch
# from transformers import AutoModelForCausalLM, AutoProcessor

from config import TEMPERATURE, GROQ_API_KEY


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "openai/gpt-oss-120b"
        self.fallback_model = "llama-3.3-70b-versatile"

        self.max_retries = 2
        self.retry_delay = 1

        print(f"🤖 LLM Service initialized with Groq model: {self.model}")

        # --- Local Gemma 4 setup (disabled; re-enable on desktop if desired) ---
        # self.MODEL_ID = "google/gemma-4-E4B-it"
        # print(f"🤖 Loading local model: {self.MODEL_ID} ...")
        #
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        #     dtype = torch.bfloat16
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     dtype = torch.float16
        # else:
        #     self.device = torch.device("cpu")
        #     dtype = torch.float32
        #
        # self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.MODEL_ID,
        #     torch_dtype=dtype,
        #     low_cpu_mem_usage=False,
        # ).to(self.device)
        # self.model.eval()
        # print(f"🤖 LLM Service initialized with local model: {self.MODEL_ID}")

    def get_completion(self, messages):
        result = self._try_model_completion(messages, self.model)
        if result is not None and result.strip():
            return result

        print(f"⚠️ {self.model} failed, switching to fallback: {self.fallback_model}")
        result = self._try_model_completion(messages, self.fallback_model)
        if result is not None and result.strip():
            return result

        print(f"❌ Both models failed after all retries")
        return "I'm having trouble with the model. Please try again."

    def _try_model_completion(self, messages, model_name):
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"🔄 Retry {attempt}/{self.max_retries - 1} on {model_name}...")

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=1024,
                    tool_choice="none",
                )

                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    print(f"⚠️ {model_name} returned empty content")

            except Exception as e:
                print(f"❌ Attempt {attempt + 1} on {model_name} failed: {e}")
                print(f"🔍 Exception details: {traceback.format_exc()}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def get_completion_streaming(self, messages, on_sentence):
        result = self._try_streaming(messages, self.model, on_sentence)
        if result is not None:
            return result

        print(f"⚠️ {self.model} streaming failed, switching to fallback: {self.fallback_model}")
        result = self._try_streaming(messages, self.fallback_model, on_sentence)
        if result is not None:
            return result

        return None

    def _try_streaming(self, messages, model_name, on_sentence):
        try:
            stream = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=1024,
                tool_choice="none",
                stream=True,
            )

            full_response = ""
            buffer = ""

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    buffer += delta.content
                    full_response += delta.content

                    while any(p in buffer for p in ".!?\n"):
                        end_idx = len(buffer)
                        for punct in ".!?\n":
                            idx = buffer.find(punct)
                            if idx != -1 and idx < end_idx:
                                end_idx = idx

                        sentence = buffer[: end_idx + 1].strip()
                        buffer = buffer[end_idx + 1:]

                        if sentence:
                            on_sentence(sentence)

            if buffer.strip():
                on_sentence(buffer.strip())

            if full_response.strip():
                return full_response
            return None

        except Exception as e:
            print(f"Streaming error with {model_name}: {e}")
            return None

    def extract_thinking_parts(self, response: str) -> Tuple[str, str]:
        if not response or not isinstance(response, str):
            return "", ""

        think_pattern = r"<think>(.*?)</think>"
        think_matches = re.findall(think_pattern, response, re.DOTALL)

        if think_matches:
            thinking_content = "\n".join(think_matches)
            actual_response = re.sub(
                think_pattern, "", response, flags=re.DOTALL
            ).strip()
            return thinking_content.strip(), actual_response

        return "", response.strip()

    def response_post_processing(
        self, response: str, show_thinking: bool = False
    ) -> Dict[str, str]:
        if not response:
            error_msg = "Sorry, I encountered an error processing your request."
            return {
                "thinking": "",
                "response": error_msg,
                "full_response": response or "",
                "display": error_msg,
            }

        thinking_content, actual_response = self.extract_thinking_parts(response)
        cleaned_response = self._clean_response_content(actual_response)

        if show_thinking and thinking_content:
            display = (
                f"**Thinking:** {thinking_content}\n\n**Response:** {cleaned_response}"
            )
        else:
            display = cleaned_response

        return {
            "thinking": thinking_content,
            "response": cleaned_response,
            "full_response": response,
            "display": display,
        }

    def _clean_response_content(self, content: str) -> str:
        if not content or not isinstance(content, str):
            return "I understand your request, but I don't have a specific response to provide."

        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
        content = content.strip()

        if not content:
            return "I understand your request, but I don't have a specific response to provide."

        return content

    def get_completion_with_processing(
        self, messages, show_thinking: bool = False
    ) -> Optional[Dict[str, str]]:
        raw_response = self.get_completion(messages)
        return self.response_post_processing(raw_response, show_thinking)
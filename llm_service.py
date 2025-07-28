from groq import Groq
from config import TEMPERATURE
import re
from typing import Dict, Optional, Tuple


class LLMService:
    def __init__(self):
        self.client = Groq()
        self.model = "deepseek-r1-distill-llama-70b"

    def get_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return None

    def extract_thinking_parts(self, response: str) -> Tuple[str, str]:
        thinking_content = ""
        actual_response = response
        
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, response, re.DOTALL)
        
        if think_matches:
            thinking_content = "\n".join(think_matches)
            actual_response = re.sub(think_pattern, '', response, flags=re.DOTALL)
        
        actual_response = actual_response.strip()
        return thinking_content.strip(), actual_response

    def extract_thinking_response(self, response: str) -> str:
        _, actual_response = self.extract_thinking_parts(response)
        return actual_response

    def get_thinking_content(self, response: str) -> str:
        thinking_content, _ = self.extract_thinking_parts(response)
        return thinking_content

    def response_post_processing(self, response: str, show_thinking: bool = False) -> Dict[str, str]:
        if not response:
            return {
                'thinking': '',
                'response': 'Sorry, I encountered an error processing your request.',
                'full_response': response or ''
            }
        
        thinking_content, actual_response = self.extract_thinking_parts(response)
        
        actual_response = self._clean_response_content(actual_response)
        
        processed_response = {
            'thinking': thinking_content,
            'response': actual_response,
            'full_response': response
        }
        
        if show_thinking and thinking_content:
            processed_response['display'] = f"**Thinking:** {thinking_content}\n\n**Response:** {actual_response}"
        else:
            processed_response['display'] = actual_response
        
        return processed_response

    def _clean_response_content(self, content: str) -> str:
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'^\s+|\s+$', '', content)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.strip()
        
        if not content:
            return "I understand your request, but I don't have a specific response to provide."
        
        return content

    def get_completion_with_processing(self, messages, show_thinking: bool = False) -> Optional[Dict[str, str]]:
        raw_response = self.get_completion(messages)
        if not raw_response:
            return None
        
        return self.response_post_processing(raw_response, show_thinking)
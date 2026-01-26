
import time
import requests
import json
from .config import PipelineConfig, TEMPERATURE, MAX_TOKENS, TIMEOUT, MAX_RETRIES

class InferenceEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

    def call_model(self, prompt: str) -> str:
        """
        State 7: Model Inference
        Supports multiple providers: DeepSeek, OpenAI, Qwen (Alibaba), SiliconFlow
        """
        # O1 models and newer GPT versions often require max_completion_tokens
        token_param = "max_tokens"
        if self.config.model.startswith("o1-") or "gpt-5" in self.config.model:
            token_param = "max_completion_tokens"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": TEMPERATURE,
            token_param: MAX_TOKENS,
            "stream": False
        }

        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=TIMEOUT
                )
                if response.status_code != 200:
                    # Check for max_tokens error (OpenAI o1/newer models)
                    if response.status_code == 400 and "max_completion_tokens" in response.text and "max_tokens" in payload:
                         print("API Error: max_tokens not supported. Retrying with max_completion_tokens...")
                         del payload["max_tokens"]
                         payload["max_completion_tokens"] = MAX_TOKENS
                         # Retry immediately
                         response = requests.post(
                             f"{self.config.base_url}/chat/completions",
                             headers=self.headers,
                             json=payload,
                             timeout=TIMEOUT
                         )
                    
                    if response.status_code != 200:
                        err_msg = f"API Error {response.status_code}: {response.text}\nProvider: {self.config.provider}\nModel: {self.config.model}\nBase URL: {self.config.base_url}\n"
                        print(err_msg)
                        with open("api_error.log", "a") as f:
                            f.write(err_msg + "\n" + "-"*50 + "\n")
                        response.raise_for_status()
                
                data = response.json()
                content = data['choices'][0]['message']['content']
                return content.strip()

            except Exception as e:
                # Exponential backoff
                sleep_time = 2 ** attempt
                print(f"Attempt {attempt+1} failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        # If all retries fail
        print("All retries failed.")
        return "Error: Timeout or API Failure"

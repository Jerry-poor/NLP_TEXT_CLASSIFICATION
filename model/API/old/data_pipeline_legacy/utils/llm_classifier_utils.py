#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM classification utilities with provider-agnostic clients."""

import os
import json
import re
from typing import Dict, List, Optional, Tuple

import openai
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# Load environment variables once

# Load environment variables once
# Try to find .env in current or parent directories
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    # Fallback: try to look 2 levels up (model/API/.env) from model/API/method1/scripts
    potential_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if os.path.exists(potential_path):
        dotenv_path = potential_path

if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("Warning: .env file not found.")



# --------- LLM Clients ---------
class BaseLLMClient:
    """Base LLM client interface."""

    def _uses_max_completion_tokens(self) -> bool:
        """
        Some OpenAI models (e.g. gpt-5.* / gpt-4.1* / o1 / o3) require
        `max_completion_tokens` instead of `max_tokens`.
        """
        model_name = str(getattr(self, "model", "") or getattr(self, "model_name", "") or "").lower()
        return any(tag in model_name for tag in ("gpt-5", "gpt-4.1", "o1", "o3"))

    def classify_text_with_confidence(self, prompt: str, max_tokens: int = 200) -> List[Tuple[str, float]]:
        raise NotImplementedError

    def classify_text_with_logprobs(
        self, prompt: str, max_tokens: int = 256, candidate_codes: Optional[List[str]] = None
    ) -> Tuple[str, Optional[float], Dict[str, float]]:
        """
        Return single-label prediction, first-token probability, and an optional
        probability distribution over provided candidate_codes (when logprobs available).
        """
        raise NotImplementedError


class DeepseekClient(BaseLLMClient):
    """Deepseek API client."""

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY_1")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not self.api_key:
            raise ValueError("Deepseek API key not found in environment variables")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def classify_text_with_confidence(self, prompt: str, max_tokens: int = 200) -> List[Tuple[str, float]]:
        """Use Deepseek for text classification with pseudo confidence."""
        try:
            token_param = {
                ("max_completion_tokens" if self._uses_max_completion_tokens() else "max_tokens"): max_tokens
            }
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional text classification assistant, capable of accurately identifying text topic categories and providing confidence scores.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                **token_param,
            )

            result_text = response.choices[0].message.content.strip()
            return parse_confidence_output(result_text)
        except Exception as e:
            print(f"Deepseek API call error: {e}")
            return []

    def classify_text_with_logprobs(
        self, prompt: str, max_tokens: int = 256, candidate_codes: Optional[List[str]] = None
    ) -> Tuple[str, Optional[float], Dict[str, float]]:
        try:
            logit_bias = build_logit_bias(candidate_codes)
            top_k = max(10, len(logit_bias) if logit_bias else 0)
            token_param = {
                ("max_completion_tokens" if self._uses_max_completion_tokens() else "max_tokens"): max_tokens
            }
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only one category code from the provided list."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                logprobs=True,
                top_logprobs=top_k,
                logit_bias=logit_bias,
                **token_param,
            )
            choice = response.choices[0]
            text = choice.message.content.strip()
            prob = extract_first_token_prob(choice)
            prob_dist = extract_label_probs(choice, candidate_codes) if candidate_codes else {}
            return text, prob, prob_dist
        except Exception as e:
            print(f"Deepseek API call error (logprobs): {e}")
            return "", None, {}


class ChatGPTClient(BaseLLMClient):
    """OpenAI/ChatGPT API client."""

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables (OPENAI_API_KEY)")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _uses_max_completion_tokens(self) -> bool:
        model_name = (self.model or "").lower()
        return any(tag in model_name for tag in ("gpt-5", "gpt-4.1", "o1", "o3"))

    def classify_text_with_confidence(self, prompt: str, max_tokens: int = 200) -> List[Tuple[str, float]]:
        try:
            token_param = {
                ("max_completion_tokens" if self._uses_max_completion_tokens() else "max_tokens"): max_tokens
            }
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional text classification assistant, capable of accurately identifying text topic categories and providing confidence scores.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                **token_param,
            )
            result_text = response.choices[0].message.content.strip()
            return parse_confidence_output(result_text)
        except Exception as e:
            print(f"ChatGPT API call error: {e}")
            return []

    def classify_text_with_logprobs(
        self, prompt: str, max_tokens: int = 256, candidate_codes: Optional[List[str]] = None
    ) -> Tuple[str, Optional[float], Dict[str, float]]:
        try:
            logit_bias = build_logit_bias(candidate_codes)
            top_k = max(10, len(logit_bias) if logit_bias else 0)
            token_param = {
                ("max_completion_tokens" if self._uses_max_completion_tokens() else "max_tokens"): max_tokens
            }
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only one category code from the provided list."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                logprobs=True,
                top_logprobs=top_k,
                logit_bias=logit_bias,
                **token_param,
            )
            choice = response.choices[0]
            text = choice.message.content.strip()
            prob = extract_first_token_prob(choice)
            prob_dist = extract_label_probs(choice, candidate_codes) if candidate_codes else {}
            return text, prob, prob_dist
        except Exception as e:
            print(f"ChatGPT API call error (logprobs): {e}")
            return "", None, {}


class GeminiClient(BaseLLMClient):
    """Gemini API client."""

    def __init__(self, model: Optional[str] = None):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("google-generativeai is required for Gemini provider") from exc

        self._genai = genai
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found in environment variables (GEMINI_API_KEY)")
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        self._genai.configure(api_key=self.api_key)
        self.model = self._genai.GenerativeModel(self.model_name)

    def classify_text_with_confidence(self, prompt: str, max_tokens: int = 200) -> List[Tuple[str, float]]:
        try:
            response = self.model.generate_content(
                [{"role": "user", "parts": [prompt]}],
                generation_config={"temperature": 0.0, "top_p": 1.0, "max_output_tokens": max_tokens},
            )
            result_text = response.text or ""
            return parse_confidence_output(result_text)
        except Exception as e:
            print(f"Gemini API call error: {e}")
            return []

    def classify_text_with_logprobs(
        self, prompt: str, max_tokens: int = 50, candidate_codes: Optional[List[str]] = None
    ) -> Tuple[str, Optional[float], Dict[str, float]]:
        try:
            response = self.model.generate_content(
                [{"role": "user", "parts": [prompt]}],
                generation_config={"temperature": 0.0, "max_output_tokens": max_tokens},
            )
            text = response.text or ""
            # Gemini SDK may not expose token logprobs in this harness; return None when absent.
            prob = None
            return text.strip(), prob, {}
        except Exception as e:
            print(f"Gemini API call error (logprobs): {e}")
            return "", None, {}


class QwenClient(BaseLLMClient):
    """Qwen (Dashscope) API client."""

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("Qwen API key not found in environment variables (QWEN_API_KEY)")
        self.base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("QWEN_MODEL", "qwen-plus")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def classify_text_with_confidence(self, prompt: str, max_tokens: int = 200) -> List[Tuple[str, float]]:
        """Use Qwen for text classification with pseudo confidence."""
        try:
            # Qwen supports standard OpenAI format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional text classification assistant, capable of accurately identifying text topic categories and providing confidence scores.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )

            result_text = response.choices[0].message.content.strip()
            return parse_confidence_output(result_text)
        except Exception as e:
            print(f"Qwen API call error: {e}")
            return []

    def classify_text_with_logprobs(
        self, prompt: str, max_tokens: int = 256, candidate_codes: Optional[List[str]] = None
    ) -> Tuple[str, Optional[float], Dict[str, float]]:
        try:
            logit_bias = build_logit_bias(candidate_codes)
            top_k = max(10, len(logit_bias) if logit_bias else 0)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only one category code from the provided list."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                logprobs=True,
                top_logprobs=top_k,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            text = choice.message.content.strip()
            prob = extract_first_token_prob(choice)
            prob_dist = extract_label_probs(choice, candidate_codes) if candidate_codes else {}
            return text, prob, prob_dist
        except Exception as e:
            print(f"Qwen API call error (logprobs): {e}")
            return "", None, {}


class SiliconFlowClient(BaseLLMClient):
    """SiliconFlow API client."""

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("SiliconFlow API key not found in environment variables (SILICONFLOW_API_KEY)")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def classify_text_with_confidence(self, prompt: str, max_tokens: int = 200) -> List[Tuple[str, float]]:
        """Use SiliconFlow for text classification."""
        import time
        import random
        
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional text classification assistant, capable of accurately identifying text topic categories and providing confidence scores.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=max_tokens,
                )

                result_text = response.choices[0].message.content.strip()
                return parse_confidence_output(result_text)
            except Exception as e:
                err_str = str(e).lower()
                # Check for rate limit or 403/429
                if "403" in err_str or "429" in err_str or "rate limit" in err_str:
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"SiliconFlow rate limit (attempt {attempt+1}/{max_retries}), retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        continue
                print(f"SiliconFlow API call error: {e}")
                return []
        return []

    def classify_text_with_logprobs(
        self, prompt: str, max_tokens: int = 256, candidate_codes: Optional[List[str]] = None
    ) -> Tuple[str, Optional[float], Dict[str, float]]:
        try:
            logit_bias = build_logit_bias(candidate_codes)
            top_k = max(10, len(logit_bias) if logit_bias else 0)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only one category code from the provided list."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                logprobs=True,
                top_logprobs=top_k,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            text = choice.message.content.strip()
            prob = extract_first_token_prob(choice)
            prob_dist = extract_label_probs(choice, candidate_codes) if candidate_codes else {}
            return text, prob, prob_dist
        except Exception as e:
            print(f"SiliconFlow API call error (logprobs): {e}")
            return "", None, {}


def get_client(provider: str, model: Optional[str] = None) -> BaseLLMClient:
    """Return an LLM client by provider key."""
    key = (provider or "deepseek").lower()
    if key in ("deepseek", "deepseek-chat", "ds"):
        return DeepseekClient(model)
    if key in ("chatgpt", "openai", "gpt"):
        return ChatGPTClient(model)
    if key in ("gemini", "google"):
        return GeminiClient(model)
    if key in ("qwen", "dashscope", "aliyun"):
        return QwenClient(model)
    if key in ("siliconflow", "silicon"):
        return SiliconFlowClient(model)
    raise ValueError(f"Unsupported provider: {provider}")


# --------- Dataset helpers ---------
def load_dataset(file_path: str):
    """Load dataset from CSV/XLSX."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file format, please use CSV or Excel files")


def load_results(file_path: str):
    """Load classification results and parse string representations back to lists."""
    df = load_dataset(file_path)
    results = []

    for _, row in df.iterrows():
        result = row.to_dict()

        # Parse predicted_categories string back to list
        if "predicted_categories" in result and isinstance(result["predicted_categories"], str):
            try:
                if result["predicted_categories"].startswith("["):
                    result["predicted_categories"] = eval(result["predicted_categories"])
                else:
                    result["predicted_categories"] = []
            except Exception:
                result["predicted_categories"] = []

        # Parse predicted_categories_with_codes string back to list
        if "predicted_categories_with_codes" in result and isinstance(result["predicted_categories_with_codes"], str):
            try:
                if result["predicted_categories_with_codes"].startswith("["):
                    result["predicted_categories_with_codes"] = eval(result["predicted_categories_with_codes"])
                else:
                    result["predicted_categories_with_codes"] = []
            except Exception:
                result["predicted_categories_with_codes"] = []

        results.append(result)

    return results


def save_results(results: List[Dict], output_file: str):
    """Save classification results to CSV/XLSX."""
    processed_results = []
    for result in results:
        processed_result = result.copy()

        if "predicted_categories" in processed_result and isinstance(processed_result["predicted_categories"], list):
            processed_result["predicted_categories"] = str(processed_result["predicted_categories"])

        if "predicted_categories_with_codes" in processed_result and isinstance(processed_result["predicted_categories_with_codes"], list):
            processed_result["predicted_categories_with_codes"] = str(
                processed_result["predicted_categories_with_codes"]
            )

        processed_results.append(processed_result)

    df = pd.DataFrame(processed_results)
    if output_file.endswith(".csv"):
        df.to_csv(output_file, index=False, encoding="utf-8")
    elif output_file.endswith(".xlsx"):
        df.to_excel(output_file, index=False)
    else:
        raise ValueError("Unsupported file format, please use CSV or Excel files")

    print(f"Results saved to: {output_file}")


def calculate_top_k_accuracy(results: List[Dict]) -> Dict[str, float]:
    """Calculate top-k accuracy."""
    if not results:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "counts": {"total": 0, "top1_correct": 0, "top3_correct": 0, "top5_correct": 0},
        }

    ddc_categories = get_ddc_categories()
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_labeled = 0

    for result in results:
        true_category = result.get("true_category") or result.get("true_ddc_name")
        if not true_category and "true_ddc_code" in result:
            true_category = ddc_categories.get(str(result.get("true_ddc_code")))

        predicted_categories = result.get("predicted_categories", [])

        if not true_category:
            continue
        total_labeled += 1

        if isinstance(predicted_categories, str):
            try:
                if predicted_categories.startswith("["):
                    predicted_categories = eval(predicted_categories)
                else:
                    predicted_categories = []
            except Exception:
                predicted_categories = []

        if not predicted_categories:
            continue

        predicted_names = []
        for item in predicted_categories:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                predicted_names.append(str(item[0]))
            else:
                predicted_names.append(str(item))

        if not predicted_names:
            continue

        if predicted_names[0] == true_category:
            top1_correct += 1

        top3_predicted = predicted_names[:3]
        if true_category in top3_predicted:
            top3_correct += 1

        top5_predicted = predicted_names[:5]
        if true_category in top5_predicted:
            top5_correct += 1

    if total_labeled == 0:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "counts": {"total": 0, "top1_correct": 0, "top3_correct": 0, "top5_correct": 0},
        }

    return {
        "top1": top1_correct / total_labeled,
        "top3": top3_correct / total_labeled,
        "top5": top5_correct / total_labeled,
        "counts": {
            "total": total_labeled,
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "top5_correct": top5_correct,
        },
    }


def calculate_top_k_accuracy_by_class(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Calculate top-k accuracy per class (true category name as key)."""
    ddc_categories = get_ddc_categories()
    per_class_counts: Dict[str, Dict[str, int]] = {}

    for result in results:
        true_category = result.get("true_category") or result.get("true_ddc_name")
        if not true_category and "true_ddc_code" in result:
            true_category = ddc_categories.get(str(result.get("true_ddc_code")))

        predicted_categories = result.get("predicted_categories", [])

        if not true_category:
            continue

        if isinstance(predicted_categories, str):
            try:
                if predicted_categories.startswith("["):
                    predicted_categories = eval(predicted_categories)
                else:
                    predicted_categories = []
            except Exception:
                predicted_categories = []

        predicted_names = []
        for item in predicted_categories:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                predicted_names.append(str(item[0]))
            else:
                predicted_names.append(str(item))

        if true_category not in per_class_counts:
            per_class_counts[true_category] = {
                "labeled": 0,
                "top1_correct": 0,
                "top3_correct": 0,
                "top5_correct": 0,
            }

        per_class_counts[true_category]["labeled"] += 1

        if not predicted_names:
            continue

        if predicted_names[0] == true_category:
            per_class_counts[true_category]["top1_correct"] += 1
        if true_category in predicted_names[:3]:
            per_class_counts[true_category]["top3_correct"] += 1
        if true_category in predicted_names[:5]:
            per_class_counts[true_category]["top5_correct"] += 1

    per_class_accuracy: Dict[str, Dict[str, float]] = {}
    for cls, counts in per_class_counts.items():
        labeled = counts["labeled"]
        per_class_accuracy[cls] = {
            "top1": (counts["top1_correct"] / labeled) if labeled else 0.0,
            "top3": (counts["top3_correct"] / labeled) if labeled else 0.0,
            "top5": (counts["top5_correct"] / labeled) if labeled else 0.0,
            "support": labeled,
            "top1_correct": counts["top1_correct"],
            "top3_correct": counts["top3_correct"],
            "top5_correct": counts["top5_correct"],
        }

    for code, name in ddc_categories.items():
        if name not in per_class_accuracy:
            per_class_accuracy[name] = {
                "top1": 0.0,
                "top3": 0.0,
                "top5": 0.0,
                "support": 0,
                "top1_correct": 0,
                "top3_correct": 0,
                "top5_correct": 0,
            }

    return per_class_accuracy


def get_ddc_categories() -> Dict[str, str]:
    """Get DDC classification categories."""
    return {
        "0": "Computer science, information & general works",
        "100": "Philosophy & psychology",
        "200": "Religion",
        "300": "Social sciences",
        "400": "Language",
        "500": "Science",
        "600": "Technology",
        "700": "Arts & recreation",
        "800": "Literature",
        "900": "History & geography",
    }


def parse_confidence_output(text: str) -> List[Tuple[str, float]]:
    """Parse confidence output, extract categories and confidence scores."""
    results: List[Tuple[str, float]] = []
    pattern = r"([^:]+):\s*(\d+(?:\.\d+)?)%?"
    matches = re.findall(pattern, text)

    for category, confidence in matches:
        category = category.strip()
        try:
            conf_value = float(confidence) / 100.0
            results.append((category, conf_value))
        except ValueError:
            continue

    if not results:
        categories = get_ddc_categories().values()
        for category in categories:
            if category.lower() in text.lower():
                results.append((category, 0.5))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]


def extract_first_token_prob(choice) -> Optional[float]:
    """Extract probability of the first generated token if logprobs are available."""
    logprobs = getattr(choice, "logprobs", None)
    if not logprobs:
        return None

    # OpenAI chat logprobs: choice.logprobs.content is a list of tokens with logprob
    content = getattr(logprobs, "content", None)
    if content and isinstance(content, list) and content:
        token_info = content[0]
        # prefer top_logprobs (includes alternatives)
        top_list = getattr(token_info, "top_logprobs", None)
        if top_list and isinstance(top_list, list) and top_list:
            for item in top_list:
                lp = getattr(item, "logprob", None)
                if lp is None:
                    continue
                try:
                    import math
                    p = float(math.exp(lp))
                except Exception:
                    continue
                # ignore degenerate 1.0 probs returned by some backends
                if p >= 0.9999:
                    continue
                return p
        # fallback to primary token logprob
        lp = getattr(token_info, "logprob", None)
        if lp is not None:
            try:
                import math
                p = float(math.exp(lp))
                if p >= 0.9999:
                    return None
                return p
            except Exception:
                return None

    # Fallback: check if choice.logprobs is a dict-like containing "token_logprobs"
    if isinstance(logprobs, dict):
        token_lps = logprobs.get("token_logprobs") or []
        if token_lps:
            lp = token_lps[0]
            if lp is not None:
                try:
                    import math
                    return float(math.exp(lp))
                except Exception:
                    return None
    return None


def build_logit_bias(candidate_codes: Optional[List[str]], encoding: str = "o200k_base") -> Optional[Dict[int, int]]:
    """
    Build a logit_bias dict to constrain outputs to given codes.
    Returns None if tiktoken unavailable or no single-token codes found.
    """
    if not candidate_codes:
        return None
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None

    try:
        enc = tiktoken.get_encoding(encoding)
    except Exception:
        return None

    bias: Dict[int, int] = {}
    for code in candidate_codes:
        tokens = enc.encode(str(code))
        if len(tokens) == 1:
            bias[tokens[0]] = 100
    return bias or None


def extract_label_probs(choice, candidate_codes: Optional[List[str]]) -> Dict[str, float]:
    """Extract probability mass for candidate codes from top_logprobs of the first token."""
    if not candidate_codes:
        return {}
    candidate_set = {str(c) for c in candidate_codes}
    logprobs = getattr(choice, "logprobs", None)
    prob_map: Dict[str, float] = {}

    def _normalize_token(tok: str) -> str:
        return tok.strip().strip('"').strip("'").rstrip(".,;:").strip()

    if logprobs:
        content = getattr(logprobs, "content", None)
        if content and isinstance(content, list) and content:
            token_info = content[0]
            top_list = getattr(token_info, "top_logprobs", None)
            if top_list and isinstance(top_list, list):
                for item in top_list:
                    lp = getattr(item, "logprob", None)
                    tok = getattr(item, "token", "")
                    norm = _normalize_token(str(tok))
                    if lp is None or norm not in candidate_set:
                        continue
                    try:
                        import math
                        prob_map[norm] = float(math.exp(lp))
                    except Exception:
                        continue
        elif isinstance(logprobs, dict):
            tokens = logprobs.get("tokens") or []
            token_lps = logprobs.get("token_logprobs") or []
            if tokens and token_lps:
                tok = str(tokens[0]) if tokens else ""
                lp = token_lps[0] if token_lps else None
                norm = _normalize_token(tok)
                if lp is not None and norm in candidate_set:
                    try:
                        import math
                        prob_map[norm] = float(math.exp(lp))
                    except Exception:
                        pass

    return prob_map

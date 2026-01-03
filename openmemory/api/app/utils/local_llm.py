import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from ollama import Client
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("OPENMEMORY_LOCAL_LLM_MODEL", "qwen3:8b")
DEFAULT_TEMPERATURE = float(os.getenv("OPENMEMORY_LOCAL_LLM_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("OPENMEMORY_LOCAL_LLM_MAX_TOKENS", "2000"))
DEFAULT_TOP_P = float(os.getenv("OPENMEMORY_LOCAL_LLM_TOP_P", "0.1"))
DEFAULT_TOP_K = int(os.getenv("OPENMEMORY_LOCAL_LLM_TOP_K", "1"))


def resolve_ollama_base_url() -> str:
    env_base_url = os.getenv("OLLAMA_BASE_URL")
    if env_base_url:
        return env_base_url
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return "http://localhost:11434"


def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _parse_json(text: str) -> Dict[str, Any] | List[Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    message = getattr(response, "message", None)
    return getattr(message, "content", "") if message else ""


class LocalLlmClient:
    """Lightweight Ollama client that enforces JSON schema output."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        client: Optional[Client] = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or resolve_ollama_base_url()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.client = client or Client(host=self.base_url)

    def _chat(self, messages: Sequence[Dict[str, str]], schema: Optional[dict]) -> Any:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
            },
        }
        if schema:
            params["format"] = schema
        return self.client.chat(**params)

    def generate_json(
        self,
        messages: Sequence[Dict[str, str]],
        schema: dict,
        retries: int = 1,
    ) -> Dict[str, Any] | List[Any]:
        last_error: Optional[Exception] = None
        prompt = list(messages)

        for attempt in range(retries + 1):
            response = self._chat(prompt, schema)
            content = _extract_content(response)
            payload = _extract_json(content)
            parsed = _parse_json(payload)
            if parsed is not None:
                return parsed

            last_error = ValueError(f"Invalid JSON response: {content}")
            if attempt < retries:
                prompt = list(messages) + [
                    {
                        "role": "user",
                        "content": "Return only valid JSON that matches the schema. No extra text.",
                    }
                ]

        raise last_error or ValueError("Failed to parse JSON response")

    def generate_pydantic(
        self,
        messages: Sequence[Dict[str, str]],
        schema: Type[T],
        retries: int = 1,
    ) -> T:
        last_error: Optional[Exception] = None
        prompt = list(messages)
        schema_json = schema.model_json_schema()

        for attempt in range(retries + 1):
            try:
                parsed = self.generate_json(prompt, schema_json, retries=0)
                return schema.model_validate(parsed)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                if attempt < retries:
                    prompt = list(messages) + [
                        {
                            "role": "user",
                            "content": "Return JSON that matches the schema exactly. No extra text.",
                        }
                    ]

        raise last_error or ValueError("Failed to validate JSON response")

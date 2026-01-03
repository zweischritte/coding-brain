import logging
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from app.utils.local_llm import LocalLlmClient
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

_llm_client: LocalLlmClient | None = None


def _get_llm_client() -> LocalLlmClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLlmClient(temperature=0.0)
    return _llm_client


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        parsed = _get_llm_client().generate_pydantic(
            messages=messages,
            schema=MemoryCategories,
            retries=1,
        )
        return [cat.strip().lower() for cat in parsed.categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        raise

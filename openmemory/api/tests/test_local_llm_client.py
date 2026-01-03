"""
Tests for the local Ollama JSON LLM client.
"""

from types import SimpleNamespace

from pydantic import BaseModel


class SampleSchema(BaseModel):
    items: list[str]


class DummyOllamaClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, **params):
        self.calls.append(params)
        content = self._responses.pop(0)
        return SimpleNamespace(message=SimpleNamespace(content=content))


def test_local_llm_parses_json_schema():
    from app.utils.local_llm import LocalLlmClient

    client = DummyOllamaClient(['{"items": ["one", "two"]}'])
    llm = LocalLlmClient(model="qwen3:8b", client=client, base_url="http://localhost:11434")

    result = llm.generate_pydantic(
        messages=[{"role": "user", "content": "List items."}],
        schema=SampleSchema,
    )

    assert result.items == ["one", "two"]
    assert client.calls[0]["format"]["type"] == "object"


def test_local_llm_retries_on_invalid_json():
    from app.utils.local_llm import LocalLlmClient

    client = DummyOllamaClient([
        '{"items": ["one", "two"]',  # missing closing brace
        '{"items": ["one", "two"]}',
    ])
    llm = LocalLlmClient(model="qwen3:8b", client=client, base_url="http://localhost:11434")

    result = llm.generate_pydantic(
        messages=[{"role": "user", "content": "List items."}],
        schema=SampleSchema,
        retries=1,
    )

    assert result.items == ["one", "two"]
    assert len(client.calls) == 2

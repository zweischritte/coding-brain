"""
Tests for Ollama tool-call compatibility in Mem0 LLM wrapper.
"""

from types import SimpleNamespace


class DummyOllamaClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, **params):
        self.calls.append(params)
        content = self._responses.pop(0)
        return {"message": {"content": content}}


def test_ollama_llm_emits_tool_calls(monkeypatch):
    from mem0.graphs.tools import EXTRACT_ENTITIES_TOOL
    from mem0.llms.ollama import OllamaLLM

    client = DummyOllamaClient(['{"entities": [{"entity": "Alice", "entity_type": "person"}]}'])
    monkeypatch.setattr("mem0.llms.ollama.Client", lambda host=None: client)

    llm = OllamaLLM({"model": "qwen3:8b", "ollama_base_url": "http://localhost:11434"})
    result = llm.generate_response(
        messages=[{"role": "user", "content": "Alice met Bob."}],
        tools=[EXTRACT_ENTITIES_TOOL],
    )

    assert result["tool_calls"][0]["name"] == "extract_entities"
    assert result["tool_calls"][0]["arguments"]["entities"][0]["entity"] == "Alice"
    assert client.calls[0]["format"]["type"] == "object"


def test_ollama_llm_splits_list_for_delete_tool(monkeypatch):
    from mem0.graphs.tools import DELETE_MEMORY_TOOL_GRAPH
    from mem0.llms.ollama import OllamaLLM

    client = DummyOllamaClient([
        '[{"source": "alice", "relationship": "knows", "destination": "bob"},'
        ' {"source": "bob", "relationship": "likes", "destination": "coffee"}]'
    ])
    monkeypatch.setattr("mem0.llms.ollama.Client", lambda host=None: client)

    llm = OllamaLLM({"model": "qwen3:8b", "ollama_base_url": "http://localhost:11434"})
    result = llm.generate_response(
        messages=[{"role": "user", "content": "Remove relations."}],
        tools=[DELETE_MEMORY_TOOL_GRAPH],
    )

    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["arguments"]["source"] == "alice"


def test_ollama_llm_retries_on_invalid_json(monkeypatch):
    from mem0.graphs.tools import EXTRACT_ENTITIES_TOOL
    from mem0.llms.ollama import OllamaLLM

    client = DummyOllamaClient([
        '{"entities": [{"entity": "Alice", "entity_type": "person"}]',  # invalid
        '{"entities": [{"entity": "Alice", "entity_type": "person"}]}',
    ])
    monkeypatch.setattr("mem0.llms.ollama.Client", lambda host=None: client)

    llm = OllamaLLM({"model": "qwen3:8b", "ollama_base_url": "http://localhost:11434"})
    result = llm.generate_response(
        messages=[{"role": "user", "content": "Alice met Bob."}],
        tools=[EXTRACT_ENTITIES_TOOL],
    )

    assert result["tool_calls"][0]["arguments"]["entities"][0]["entity"] == "Alice"
    assert len(client.calls) == 2
